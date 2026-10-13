// Minimal in-memory vector store with versioned JSON persistence.
// Fine for a few thousand documents; once this stops being fine,
// swap the backend — the `insert`/`search` interface here is the
// contract to preserve.
//
// The on-disk format is a versioned JSON envelope holding each
// entry's path, content, and embedding plus the embedding model id
// and dimension. It lives in the app-specific Local AppData directory
// as `index-v1.json` and contains full document contents.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub const INDEX_SCHEMA_VERSION: u32 = 2;
/// Stable identifier for the current local embedding model
/// (`fastembed::EmbeddingModel::AllMiniLML6V2`).
pub const INDEX_MODEL_ID: &str = "all-MiniLM-L6-v2";
/// Embedding dimension emitted by `AllMiniLML6V2`.
pub const INDEX_DIM: usize = 384;
pub const INDEX_FILE_NAME: &str = "index-v1.json";

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Serialize, Deserialize)]
struct Entry {
    path: String,
    content: String,
    vector: Vec<f32>,
}

pub struct VectorStore {
    entries: Vec<Entry>,
    folders: Vec<String>,
}

#[derive(Serialize, Clone)]
pub struct FolderInfo {
    pub path: String,
    pub file_count: usize,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub path: String,
    pub snippet: String,
    pub score: f32,
}

/// Staged indexing result: scan output plus its embedding. Applied to
/// the live store only via `reconcile` after the full scan+embed pass
/// succeeds.
pub struct StagedDoc {
    pub path: String,
    pub content: String,
    pub vector: Vec<f32>,
}

/// Normalized identity for an absolute Windows path, or `None` when the
/// input cannot be scoped safely (relative, drive-relative,
/// rooted-relative, device-namespace, or otherwise ambiguous). Never
/// resolves relative inputs against the current working directory.
pub fn path_key(raw: &str) -> Option<String> {
    let mut s = raw.trim();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        s = s[1..s.len() - 1].trim();
    }
    if s.is_empty() {
        return None;
    }
    let unified = s.replace('/', "\\");
    let mut l = unified.to_lowercase();
    if l.starts_with("\\\\?\\unc\\") {
        let rest = &l["\\\\?\\unc\\".len()..];
        if rest.is_empty() {
            return None;
        }
        l = format!("\\\\{rest}");
    } else if l.starts_with("\\\\?\\") {
        let rest = &l["\\\\?\\".len()..];
        if rest.len() >= 3
            && rest.as_bytes()[0].is_ascii_alphabetic()
            && rest.as_bytes()[1] == b':'
            && rest.as_bytes()[2] == b'\\'
        {
            l = rest.to_string();
        } else if rest.len() == 2
            && rest.as_bytes()[0].is_ascii_alphabetic()
            && rest.as_bytes()[1] == b':'
        {
            l = rest.to_string();
        } else {
            return None;
        }
    } else if l.starts_with("\\\\.\\") {
        return None;
    }
    if l.starts_with("\\\\") {
        normalize_unc(&l)
    } else if l.len() >= 3
        && l.as_bytes()[0].is_ascii_alphabetic()
        && l.as_bytes()[1] == b':'
        && l.as_bytes()[2] == b'\\'
    {
        normalize_drive(&l)
    } else {
        None
    }
}

/// Root-boundary check on normalized keys. Equality counts as inside.
/// A root already ending in `\` (drive root, UNC share root) matches
/// descendants directly; otherwise descendants require `root + "\\"`,
/// so `D:\Docs-old` is never inside `D:\Docs`.
pub fn is_in_root(entry_key: &str, root_key: &str) -> bool {
    if entry_key == root_key {
        true
    } else if root_key.ends_with('\\') {
        entry_key.starts_with(root_key)
    } else {
        entry_key.len() > root_key.len()
            && entry_key.starts_with(root_key)
            && entry_key.as_bytes()[root_key.len()] == b'\\'
    }
}

fn normalize_drive(l: &str) -> Option<String> {
    let prefix = &l[..3];
    let rest = &l[3..];
    let mut stack: Vec<&str> = Vec::new();
    for comp in rest.split('\\') {
        if comp.is_empty() || comp == "." {
            continue;
        } else if comp == ".." {
            stack.pop();
        } else {
            stack.push(comp);
        }
    }
    if stack.is_empty() {
        Some(prefix.to_string())
    } else {
        Some(format!("{prefix}{}", stack.join("\\")))
    }
}

fn normalize_unc(l: &str) -> Option<String> {
    let without = &l[2..];
    let mut stack: Vec<&str> = Vec::new();
    for comp in without.split('\\') {
        if comp.is_empty() || comp == "." {
            continue;
        } else if comp == ".." {
            if stack.len() > 2 {
                stack.pop();
            } else {
                return None;
            }
        } else {
            stack.push(comp);
        }
    }
    if stack.len() < 2 {
        return None;
    }
    if stack.len() == 2 {
        Some(format!("\\\\{}\\{}\\", stack[0], stack[1]))
    } else {
        Some(format!("\\\\{}", stack.join("\\")))
    }
}

fn paths_match(a: &str, b: &str) -> bool {
    match (path_key(a), path_key(b)) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

#[derive(Serialize, Deserialize)]
struct SavedEntry {
    path: String,
    content: String,
    vector: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct SavedIndex {
    version: u32,
    model: String,
    dim: usize,
    entries: Vec<SavedEntry>,
    #[serde(default)]
    folders: Vec<String>,
}

pub enum LoadOutcome {
    /// No saved index exists yet. Quiet startup.
    Empty,
    Loaded(VectorStore),
    Quarantined {
        store: VectorStore,
        notice: String,
        /// Whether the bad file was moved aside. Writes may resume only
        /// when this is true; otherwise the original must be left alone.
        quarantine_succeeded: bool,
    },
}

impl VectorStore {
    pub fn new() -> Self {
        VectorStore {
            entries: Vec::new(),
            folders: Vec::new(),
        }
    }

    /// Number of indexed documents (used by tests and diagnostics).
    #[allow(dead_code)]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Registered folder display paths in stored order.
    #[allow(dead_code)]
    pub fn registered_folders(&self) -> Vec<String> {
        self.folders.clone()
    }

    /// Registered folders with live per-folder file counts, sorted
    /// deterministically by normalized key. Counts are derived from
    /// entries, never persisted.
    pub fn list_folders(&self) -> Vec<FolderInfo> {
        let mut keyed: Vec<(String, String)> = Vec::with_capacity(self.folders.len());
        for display in &self.folders {
            if let Some(key) = path_key(display) {
                keyed.push((key, display.clone()));
            }
        }
        keyed.sort_by(|a, b| a.0.cmp(&b.0));
        keyed
            .into_iter()
            .map(|(key, display)| FolderInfo {
                path: display,
                file_count: self.count_in_root(&key),
            })
            .collect()
    }

    fn count_in_root(&self, root_key: &str) -> usize {
        self.entries
            .iter()
            .filter_map(|e| path_key(&e.path))
            .filter(|k| is_in_root(k, root_key))
            .count()
    }

    /// True when the given root (any spelling) is registered.
    pub fn is_registered(&self, root: &Path) -> bool {
        let key = match path_key(&root.to_string_lossy()) {
            Some(k) => k,
            None => return false,
        };
        self.folders.iter().any(|f| path_key(f).as_deref() == Some(&key))
    }

    /// Upsert a folder registration by normalized identity, refreshing
    /// the display spelling. Returns true when a new entry was added.
    /// Never matches ambiguous paths. Call only after the indexing pass
    /// validated; fails without mutating on ambiguous input.
    fn register_display(&mut self, display: &str) -> Result<bool, String> {
        let key = path_key(display).ok_or_else(|| {
            format!("folder has ambiguous identity ({display}); refusing to register")
        })?;
        if let Some(existing) = self
            .folders
            .iter_mut()
            .find(|f| path_key(f).as_deref() == Some(&key))
        {
            *existing = display.to_string();
            Ok(false)
        } else {
            self.folders.push(display.to_string());
            Ok(true)
        }
    }

    /// Validate root, display, and staged paths, then atomically apply
    /// entry reconciliation plus folder registration. Any validation
    /// failure leaves both entries and registrations unchanged.
    pub fn reconcile_and_register(
        &mut self,
        root: &Path,
        display: &str,
        staged: Vec<StagedDoc>,
    ) -> Result<usize, String> {
        let root_key = path_key(&root.to_string_lossy()).ok_or_else(|| {
            format!(
                "selected folder has ambiguous identity ({}); refusing to reconcile",
                root.display()
            )
        })?;
        let display_key = path_key(display).ok_or_else(|| {
            format!("folder has ambiguous identity ({display}); refusing to reconcile")
        })?;
        // Display must denote the same folder as the canonical root.
        if display_key != root_key {
            return Err(format!(
                "folder display ({display}) does not match scanned folder ({}); refusing to reconcile",
                root.display()
            ));
        }
        let n = self.reconcile(root, staged)?;
        // Registration cannot fail after the pre-validation above.
        self.register_display(display)
            .map_err(|e| format!("could not register folder: {e}"))?;
        Ok(n)
    }

    /// Remove a folder registration and all in-root entries. Files on
    /// disk are never touched. Ambiguous roots are rejected without
    /// mutation; unknown-but-valid roots return 0 removals.
    pub fn remove_folder(&mut self, root: &Path) -> Result<usize, String> {
        let root_key = path_key(&root.to_string_lossy()).ok_or_else(|| {
            format!(
                "folder has ambiguous identity ({}); refusing to remove",
                root.display()
            )
        })?;
        let before = self.entries.len();
        self.entries.retain(|e| match path_key(&e.path) {
            None => true,
            Some(k) => !is_in_root(&k, &root_key),
        });
        self.folders
            .retain(|f| path_key(f).as_deref() != Some(&root_key));
        Ok(before - self.entries.len())
    }

    /// Path-based upsert using the shared normalized identity. Only two
    /// valid absolute keys that are equal may match; ambiguous paths
    /// never match anything and are preserved as distinct entries.
    #[allow(dead_code)]
    pub fn insert(&mut self, path: String, content: String, vector: Vec<f32>) {
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|e| paths_match(&e.path, &path))
        {
            existing.content = content;
            existing.vector = vector;
            existing.path = path;
        } else {
            self.entries.push(Entry { path, content, vector });
        }
    }

    /// Atomically reconcile the selected absolute root with a completely
    /// staged set: upsert every staged doc by normalized identity, then
    /// drop in-root entries absent from the staged set. Unrelated roots
    /// and ambiguous legacy entries are always retained. Validates all
    /// keys before mutating, so an ambiguous root or staged path aborts
    /// without changing the store.
    pub fn reconcile(&mut self, root: &Path, staged: Vec<StagedDoc>) -> Result<usize, String> {
        let root_key = path_key(&root.to_string_lossy()).ok_or_else(|| {
            format!(
                "selected folder has ambiguous identity ({}); refusing to reconcile",
                root.display()
            )
        })?;
        let mut staged_keys: HashSet<String> = HashSet::with_capacity(staged.len());
        let mut staged_with_keys: Vec<(String, StagedDoc)> = Vec::with_capacity(staged.len());
        for doc in staged {
            let key = path_key(&doc.path).ok_or_else(|| {
                format!(
                    "scanned file has ambiguous identity ({}); refusing to reconcile",
                    doc.path
                )
            })?;
            staged_keys.insert(key.clone());
            staged_with_keys.push((key, doc));
        }
        for (key, doc) in staged_with_keys {
            if let Some(existing) = self.entries.iter_mut().find(|e| match path_key(&e.path) {
                Some(existing_key) => existing_key == key,
                None => false,
            }) {
                existing.content = doc.content;
                existing.vector = doc.vector;
                existing.path = doc.path;
            } else {
                self.entries.push(Entry {
                    path: doc.path,
                    content: doc.content,
                    vector: doc.vector,
                });
            }
        }
        let mut seen: HashSet<String> = HashSet::new();
        self.entries.retain_mut(|e| match path_key(&e.path) {
            None => true,
            Some(entry_key) => {
                if is_in_root(&entry_key, &root_key) {
                    if !staged_keys.contains(&entry_key) {
                        return false;
                    }
                    // Deduplicate previously equivalent paths inside the
                    // root: the upsert loop updates the first occurrence,
                    // so keep the first staged key seen and drop the rest.
                    if !seen.insert(entry_key) {
                        return false;
                    }
                    true
                } else {
                    true
                }
            }
        });
        Ok(staged_keys.len())
    }

    pub fn search(&self, query_vector: &[f32], limit: usize) -> Vec<SearchResult> {
        let mut scored: Vec<SearchResult> = self
            .entries
            .iter()
            .map(|e| SearchResult {
                path: e.path.clone(),
                snippet: snippet(&e.content),
                score: cosine_similarity(query_vector, &e.vector),
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scored.truncate(limit);
        scored
    }

    /// Serialize the full index envelope. Pure; holds no locks itself.
    pub fn serialize(&self) -> Result<String, String> {
        let saved = SavedIndex {
            version: INDEX_SCHEMA_VERSION,
            model: INDEX_MODEL_ID.to_string(),
            dim: INDEX_DIM,
            entries: self
                .entries
                .iter()
                .map(|e| SavedEntry {
                    path: e.path.clone(),
                    content: e.content.clone(),
                    vector: e.vector.clone(),
                })
                .collect(),
            folders: self.folders.clone(),
        };
        serde_json::to_string(&saved).map_err(|e| format!("could not serialize index: {e}"))
    }

    /// Validate a parsed envelope and convert it. v1 envelopes (no
    /// folders field) load entries with an empty registry. v2 envelopes
    /// load entries strictly and sanitize duplicate or ambiguous folder
    /// paths without discarding valid entries. Anything else quarantines.
    fn from_saved(saved: SavedIndex) -> Result<Self, String> {
        if saved.version != 1 && saved.version != INDEX_SCHEMA_VERSION {
            return Err(format!(
                "unsupported index version {} (expected {})",
                saved.version, INDEX_SCHEMA_VERSION
            ));
        }
        if saved.model != INDEX_MODEL_ID {
            return Err(format!(
                "index model '{}' does not match current model '{INDEX_MODEL_ID}'",
                saved.model
            ));
        }
        if saved.dim != INDEX_DIM {
            return Err(format!(
                "index dimension {} does not match expected {INDEX_DIM}",
                saved.dim
            ));
        }
        let mut store = VectorStore::new();
        for e in saved.entries {
            if e.path.is_empty() {
                return Err("index entry has an empty path".to_string());
            }
            if e.vector.len() != INDEX_DIM {
                return Err(format!(
                    "entry '{}' has dimension {} (expected {INDEX_DIM})",
                    e.path,
                    e.vector.len()
                ));
            }
            if !e.vector.iter().all(|v| v.is_finite()) {
                return Err(format!("entry '{}' has a non-finite embedding value", e.path));
            }
            store.entries.push(Entry {
                path: e.path,
                content: e.content,
                vector: e.vector,
            });
        }
        if saved.version == 1 {
            return Ok(store);
        }
        let mut seen: HashSet<String> = HashSet::new();
        for folder in saved.folders {
            if folder.trim().is_empty() {
                continue;
            }
            let key = match path_key(&folder) {
                Some(k) => k,
                None => continue,
            };
            if seen.insert(key) {
                store.folders.push(folder);
            }
        }
        Ok(store)
    }

    /// Convenience for tests and single-threaded callers: serialize and
    /// commit atomically with the real filesystem rename.
    #[allow(dead_code)]
    pub fn save_to_path(&self, dest: &Path) -> Result<(), String> {
        self.save_to_path_with_rename(dest, fs_rename)
    }

    #[allow(dead_code)]
    fn save_to_path_with_rename(
        &self,
        dest: &Path,
        rename: fn(&Path, &Path) -> std::io::Result<()>,
    ) -> Result<(), String> {
        let json = self.serialize()?;
        write_serialized_atomic_with_rename(dest, json.as_bytes(), rename)
    }

    /// Load and fully validate. Missing file is quiet-empty. Unreadable
    /// (non-NotFound) IO is an `Err` so callers do not discard an index
    /// whose state is unknown. Corrupt/incompatible data is quarantined
    /// beside the original and yields an empty store plus a notice.
    pub fn load_from_path(path: &Path) -> Result<LoadOutcome, String> {
        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(LoadOutcome::Empty),
            Err(e) => {
                return Err(format!("could not read saved index {}: {e}", path.display()));
            }
        };
        let text = match String::from_utf8(bytes) {
            Ok(t) => t,
            Err(e) => {
                return Ok(quarantine_and_empty(
                    path,
                    format!("saved search index was corrupt (invalid text: {e}); starting fresh — please re-index."),
                ));
            }
        };
        let saved: SavedIndex = match serde_json::from_str(&text) {
            Ok(s) => s,
            Err(e) => {
                return Ok(quarantine_and_empty(
                    path,
                    format!("saved search index was corrupt ({e}); starting fresh — please re-index."),
                ));
            }
        };
        match VectorStore::from_saved(saved) {
            Ok(store) => Ok(LoadOutcome::Loaded(store)),
            Err(reason) => Ok(quarantine_and_empty(
                path,
                format!("saved search index was incompatible ({reason}); starting fresh — please re-index."),
            )),
        }
    }
}

/// Serialize-free atomic commit of already-serialized bytes. `main.rs`
/// snapshots under the store lock, then calls this without holding it.
pub fn write_serialized_atomic(dest: &Path, json: &str) -> Result<(), String> {
    write_serialized_atomic_with_rename(dest, json.as_bytes(), fs_rename)
}

fn fs_rename(from: &Path, to: &Path) -> std::io::Result<()> {
    fs::rename(from, to)
}

fn write_serialized_atomic_with_rename(
    dest: &Path,
    bytes: &[u8],
    rename: fn(&Path, &Path) -> std::io::Result<()>,
) -> Result<(), String> {
    write_serialized_atomic_full(dest, bytes, fs_create_tmp, rename)
}

fn fs_create_tmp(path: &Path) -> std::io::Result<fs::File> {
    fs::File::create(path)
}

fn write_serialized_atomic_full(
    dest: &Path,
    bytes: &[u8],
    create_tmp: fn(&Path) -> std::io::Result<fs::File>,
    rename: fn(&Path, &Path) -> std::io::Result<()>,
) -> Result<(), String> {
    let parent = dest
        .parent()
        .ok_or_else(|| format!("index path {} has no parent directory", dest.display()))?;
    // Failure here leaves any existing destination untouched.
    fs::create_dir_all(parent)
        .map_err(|e| format!("could not create index directory {}: {e}", parent.display()))?;

    let tmp = unique_tmp_path(dest);
    // Any failure below removes only the temp file, never the destination.
    let mut file = create_tmp(&tmp)
        .map_err(|e| format!("could not write temporary index {}: {e}", tmp.display()))?;
    let write_result: std::io::Result<()> = (|| {
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    })();
    drop(file);
    if let Err(e) = write_result {
        let _ = fs::remove_file(&tmp);
        return Err(format!("could not write temporary index {}: {e}", tmp.display()));
    }
    match rename(&tmp, dest) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(format!("could not replace saved index {}: {e}", dest.display()))
        }
    }
}

fn unique_tmp_path(dest: &Path) -> PathBuf {
    let count = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let file_name = dest
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| INDEX_FILE_NAME.to_string());
    dest.with_file_name(format!(
        ".{file_name}.tmp.{}-{count}-{millis}",
        std::process::id()
    ))
}

fn quarantine_and_empty(path: &Path, notice: String) -> LoadOutcome {
    let quarantined_to = quarantine_path(path);
    let (notice, quarantine_succeeded) = match quarantined_to {
        Ok(q) => (
            format!("{notice} The previous file was moved to {}.", q.display()),
            true,
        ),
        Err(e) => (
            format!("{notice} Could not move the previous file aside ({e}). Persistence remains disabled this session; no files were changed."),
            false,
        ),
    };
    LoadOutcome::Quarantined {
        store: VectorStore::new(),
        notice,
        quarantine_succeeded,
    }
}

fn quarantine_path(path: &Path) -> Result<PathBuf, String> {
    let count = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let file_name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| INDEX_FILE_NAME.to_string());
    let target = path.with_file_name(format!(
        "{}.corrupt-{}-{count}-{millis}.json",
        file_name.trim_end_matches(".json"),
        std::process::id()
    ));
    fs::rename(path, &target)
        .map(|_| target)
        .map_err(|e| format!("could not quarantine {}: {e}", path.display()))
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn snippet(content: &str) -> String {
    content.chars().take(200).collect()
}

// TODO next: this re-scores every document on every query, which is
// fine at small scale but is exactly the kind of thing that motivates
// moving to a real vector index (HNSW, IVF, etc.) once the collection
// grows past a few thousand documents.

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn vec384(fill: f32) -> Vec<f32> {
        vec![fill; INDEX_DIM]
    }

    fn test_dir(name: &str) -> PathBuf {
        let count = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "searchiee-test-{}-{}-{name}-{count}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("test dir");
        dir
    }

    fn corrupt_files_in(dir: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .expect("read test dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .map(|n| n.to_string_lossy().contains("corrupt"))
                    .unwrap_or(false)
            })
            .collect()
    }

    #[test]
    fn reinserting_same_path_leaves_single_entry() {
        let mut store = VectorStore::new();
        store.insert(
            "D:\\Docs\\a.txt".to_string(),
            "hello".to_string(),
            vec![1.0, 0.0],
        );
        store.insert(
            "D:\\Docs\\a.txt".to_string(),
            "hello".to_string(),
            vec![1.0, 0.0],
        );

        assert_eq!(store.entries.len(), 1);
        assert_eq!(store.search(&[1.0, 0.0], 10).len(), 1);
    }

    #[test]
    fn reinserting_same_path_updates_content_and_embedding() {
        let mut store = VectorStore::new();
        store.insert(
            "D:\\Docs\\a.txt".to_string(),
            "old".to_string(),
            vec![1.0, 0.0],
        );
        store.insert(
            "D:\\Docs\\a.txt".to_string(),
            "new".to_string(),
            vec![0.0, 1.0],
        );

        assert_eq!(store.entries.len(), 1);
        assert_eq!(store.entries[0].content, "new");
        assert_eq!(store.entries[0].vector, vec![0.0, 1.0]);
    }

    #[test]
    fn different_paths_remain_distinct_entries() {
        let mut store = VectorStore::new();
        store.insert(
            "D:\\Docs\\a.txt".to_string(),
            "same text".to_string(),
            vec![1.0, 0.0],
        );
        store.insert(
            "D:\\Docs\\b.txt".to_string(),
            "same text".to_string(),
            vec![1.0, 0.0],
        );

        assert_eq!(store.entries.len(), 2);
        assert_eq!(store.search(&[1.0, 0.0], 10).len(), 2);
    }

    #[test]
    fn round_trip_preserves_entries_and_search() {
        let dir = test_dir("roundtrip");
        let dest = dir.join(INDEX_FILE_NAME);
        let mut store = VectorStore::new();
        store.insert("a.txt".to_string(), "astronomy stars".to_string(), vec384(1.0));
        let mut other = vec384(0.0);
        other[0] = 1.0;
        store.insert("b.txt".to_string(), "gardening soil".to_string(), other);

        store.save_to_path(&dest).expect("save");
        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };

        assert_eq!(loaded.entries.len(), 2);
        let results = loaded.search(&vec384(1.0), 10);
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.path == "a.txt"));
        assert!(results.iter().any(|r| r.path == "b.txt"));
    }

    #[test]
    fn repeated_save_replaces_and_loads_latest() {
        let dir = test_dir("repeated");
        let dest = dir.join(INDEX_FILE_NAME);
        let mut first = VectorStore::new();
        first.insert("a.txt".to_string(), "v1".to_string(), vec384(1.0));
        first.save_to_path(&dest).expect("first save");

        let mut second = VectorStore::new();
        second.insert("a.txt".to_string(), "v2".to_string(), vec384(2.0));
        second.insert("b.txt".to_string(), "new".to_string(), vec384(3.0));
        second.save_to_path(&dest).expect("second save");

        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(loaded.entries.len(), 2);
        let entry = loaded.entries.iter().find(|e| e.path == "a.txt").expect("a.txt");
        assert_eq!(entry.content, "v2");
        assert_eq!(entry.vector, vec384(2.0));
    }

    #[test]
    fn restart_and_upsert_stays_idempotent() {
        let dir = test_dir("restart");
        let dest = dir.join(INDEX_FILE_NAME);
        let mut store = VectorStore::new();
        store.insert(
            "D:\\Docs\\a.txt".to_string(),
            "hello".to_string(),
            vec384(1.0),
        );
        store.save_to_path(&dest).expect("save");

        let mut restarted = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        restarted.insert(
            "D:\\Docs\\a.txt".to_string(),
            "hello updated".to_string(),
            vec384(5.0),
        );
        assert_eq!(restarted.entries.len(), 1);
        assert_eq!(restarted.entries[0].content, "hello updated");
        assert_eq!(restarted.entries[0].vector, vec384(5.0));
    }

    #[test]
    fn failed_temp_write_preserves_prior_index() {
        let dir = test_dir("tempfail");
        let dest = dir.join(INDEX_FILE_NAME);
        let mut original = VectorStore::new();
        original.insert("a.txt".to_string(), "original".to_string(), vec384(1.0));
        original.save_to_path(&dest).expect("seed valid index");

        // Force the temporary-write stage to fail while saving to the
        // same destination that already holds the valid index.
        let mut other = VectorStore::new();
        other.insert("b.txt".to_string(), "other".to_string(), vec384(2.0));
        let json = other.serialize().expect("serialize");
        let result = write_serialized_atomic_full(
            &dest,
            json.as_bytes(),
            |_| Err(std::io::Error::other("injected temp failure")),
            fs_rename,
        );
        assert!(result.is_err());

        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].content, "original");
        // No temp debris left beside the destination.
        let strays: Vec<_> = std::fs::read_dir(&dir)
            .expect("read dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().contains(".tmp."))
            .collect();
        assert!(strays.is_empty());
    }

    #[test]
    fn failed_replacement_preserves_prior_index() {
        let dir = test_dir("replacefail");
        let dest = dir.join(INDEX_FILE_NAME);
        let mut original = VectorStore::new();
        original.insert("a.txt".to_string(), "original".to_string(), vec384(1.0));
        original.save_to_path(&dest).expect("seed valid index");

        let mut other = VectorStore::new();
        other.insert("b.txt".to_string(), "other".to_string(), vec384(2.0));
        let result = other.save_to_path_with_rename(&dest, |_, _| {
            Err(std::io::Error::other("injected rename failure"))
        });
        assert!(result.is_err());

        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].path, "a.txt");
        // No stray temp files left beside the destination.
        let strays: Vec<_> = std::fs::read_dir(&dir)
            .expect("read dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().contains(".tmp."))
            .collect();
        assert!(strays.is_empty());
    }

    #[test]
    fn corrupt_json_quarantines_and_loads_empty() {
        let dir = test_dir("corrupt");
        let dest = dir.join(INDEX_FILE_NAME);
        std::fs::write(&dest, b"{ not valid json").expect("seed corrupt");

        match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Quarantined {
                store,
                notice,
                quarantine_succeeded: true,
            } => {
                assert_eq!(store.entries.len(), 0);
                assert!(notice.contains("re-index"));
            }
            other => panic!("expected Quarantined, got {}", outcome_name(&other)),
        }
        assert!(!dest.exists());
        assert_eq!(corrupt_files_in(&dir).len(), 1);
    }

    #[test]
    fn unsupported_schema_rejected_safely() {
        let dir = test_dir("schema");
        let dest = dir.join(INDEX_FILE_NAME);
        let bad = serde_json::json!({
            "version": 999,
            "model": INDEX_MODEL_ID,
            "dim": INDEX_DIM,
            "entries": []
        });
        std::fs::write(&dest, serde_json::to_string(&bad).unwrap()).expect("seed");

        match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Quarantined { store, .. } => assert_eq!(store.entries.len(), 0),
            other => panic!("expected Quarantined, got {}", outcome_name(&other)),
        }
        assert!(!dest.exists());
        assert_eq!(corrupt_files_in(&dir).len(), 1);
    }

    #[test]
    fn model_mismatch_rejected_safely() {
        let dir = test_dir("model");
        let dest = dir.join(INDEX_FILE_NAME);
        let bad = serde_json::json!({
            "version": INDEX_SCHEMA_VERSION,
            "model": "other-model",
            "dim": INDEX_DIM,
            "entries": []
        });
        std::fs::write(&dest, serde_json::to_string(&bad).unwrap()).expect("seed");

        match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Quarantined { store, .. } => assert_eq!(store.entries.len(), 0),
            other => panic!("expected Quarantined, got {}", outcome_name(&other)),
        }
        assert_eq!(corrupt_files_in(&dir).len(), 1);
    }

    #[test]
    fn dimension_mismatch_rejected_safely() {
        let dir = test_dir("dim");
        let dest = dir.join(INDEX_FILE_NAME);
        let bad = serde_json::json!({
            "version": INDEX_SCHEMA_VERSION,
            "model": INDEX_MODEL_ID,
            "dim": 999,
            "entries": []
        });
        std::fs::write(&dest, serde_json::to_string(&bad).unwrap()).expect("seed");

        match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Quarantined { store, .. } => assert_eq!(store.entries.len(), 0),
            other => panic!("expected Quarantined, got {}", outcome_name(&other)),
        }
        assert_eq!(corrupt_files_in(&dir).len(), 1);
    }

    #[test]
    fn invalid_vector_rejected_safely() {
        for (name, vector) in [
            ("short", vec![1.0; 2]),
            ("nonfinite", {
                let mut v = vec384(1.0);
                v[0] = f32::NAN;
                v
            }),
        ] {
            let dir = test_dir(name);
            let dest = dir.join(INDEX_FILE_NAME);
            let bad = serde_json::json!({
                "version": INDEX_SCHEMA_VERSION,
                "model": INDEX_MODEL_ID,
                "dim": INDEX_DIM,
                "entries": [{ "path": "a.txt", "content": "x", "vector": vector }]
            });
            std::fs::write(&dest, serde_json::to_string(&bad).unwrap()).expect("seed");

            match VectorStore::load_from_path(&dest).expect("load") {
                LoadOutcome::Quarantined { store, .. } => assert_eq!(store.entries.len(), 0),
                other => panic!("expected Quarantined for {name}, got {}", outcome_name(&other)),
            }
            assert_eq!(corrupt_files_in(&dir).len(), 1);
        }
    }

    #[test]
    fn missing_file_loads_quiet_empty() {
        let dir = test_dir("missing");
        let dest = dir.join(INDEX_FILE_NAME);
        match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Empty => {}
            other => panic!("expected Empty, got {}", outcome_name(&other)),
        }
    }

    fn staged(path: &str, content: &str, fill: f32) -> StagedDoc {
        StagedDoc {
            path: path.to_string(),
            content: content.to_string(),
            vector: vec384(fill),
        }
    }

    fn root(path: &str) -> PathBuf {
        PathBuf::from(path)
    }

    #[test]
    fn reconcile_updates_changed_file_without_duplicating() {
        let mut store = VectorStore::new();
        store.insert(
            "D:\\Docs\\a.txt".to_string(),
            "v1".to_string(),
            vec384(1.0),
        );
        let n = store
            .reconcile(
                &root("D:\\Docs"),
                vec![staged("D:\\Docs\\a.txt", "v2", 2.0)],
            )
            .expect("reconcile");
        assert_eq!(n, 1);
        assert_eq!(store.entries.len(), 1);
        assert_eq!(store.entries[0].content, "v2");
        assert_eq!(store.entries[0].vector, vec384(2.0));
    }

    #[test]
    fn reconcile_deduplicates_previously_equivalent_paths() {
        let mut store = VectorStore::new();
        // Two spellings of the same absolute path, as could accumulate
        // before normalized matching existed. Push directly to bypass
        // upsert collapsing and simulate the legacy duplication.
        store.entries.push(Entry {
            path: "D:\\Docs\\Note.md".to_string(),
            content: "stale one".to_string(),
            vector: vec384(1.0),
        });
        store.entries.push(Entry {
            path: "d:/docs/note.md".to_string(),
            content: "stale two".to_string(),
            vector: vec384(1.0),
        });
        // An unrelated entry must survive untouched.
        store.entries.push(Entry {
            path: "E:\\Other\\keep.txt".to_string(),
            content: "keep".to_string(),
            vector: vec384(7.0),
        });
        assert_eq!(store.entries.len(), 3);
        store
            .reconcile(
                &root("D:\\Docs"),
                vec![staged("D:\\DOCS\\NOTE.md", "fresh", 4.0)],
            )
            .expect("reconcile");
        let matching: Vec<_> = store
            .entries
            .iter()
            .filter(|e| path_key(&e.path).as_deref() == Some("d:\\docs\\note.md"))
            .collect();
        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0].content, "fresh");
        assert_eq!(matching[0].vector, vec384(4.0));
        assert_eq!(matching[0].path, "D:\\DOCS\\NOTE.md");
        assert_eq!(store.entries.len(), 2);
        assert!(store.entries.iter().any(|e| e.path == "E:\\Other\\keep.txt"));
    }

    #[test]
    fn reconcile_handles_rename() {
        let mut store = VectorStore::new();
        store.insert(
            "D:\\Docs\\old.txt".to_string(),
            "body".to_string(),
            vec384(1.0),
        );
        store
            .reconcile(
                &root("D:\\Docs"),
                vec![staged("D:\\Docs\\new.txt", "body", 1.0)],
            )
            .expect("reconcile");
        assert_eq!(store.entries.len(), 1);
        assert_eq!(store.entries[0].path, "D:\\Docs\\new.txt");
    }

    #[test]
    fn reconcile_removes_deleted_file() {
        let mut store = VectorStore::new();
        store.insert("D:\\Docs\\a.txt".to_string(), "a".to_string(), vec384(1.0));
        store.insert("D:\\Docs\\b.txt".to_string(), "b".to_string(), vec384(2.0));
        store
            .reconcile(&root("D:\\Docs"), vec![staged("D:\\Docs\\a.txt", "a", 1.0)])
            .expect("reconcile");
        assert_eq!(store.entries.len(), 1);
        assert_eq!(store.entries[0].path, "D:\\Docs\\a.txt");
    }

    #[test]
    fn reconcile_leaves_unrelated_roots_untouched() {
        let mut store = VectorStore::new();
        store.insert("D:\\Docs\\a.txt".to_string(), "a".to_string(), vec384(1.0));
        store.insert(
            "E:\\Other\\b.txt".to_string(),
            "b".to_string(),
            vec384(2.0),
        );
        store
            .reconcile(
                &root("D:\\Docs"),
                vec![staged("D:\\Docs\\a.txt", "a2", 3.0)],
            )
            .expect("reconcile");
        assert_eq!(store.entries.len(), 2);
        let other = store
            .entries
            .iter()
            .find(|e| e.path == "E:\\Other\\b.txt")
            .expect("other retained");
        assert_eq!(other.content, "b");
        assert_eq!(other.vector, vec384(2.0));
    }

    #[test]
    fn reconcile_same_root_is_idempotent() {
        let mut store = VectorStore::new();
        let docs = || {
            vec![
                staged("D:\\Docs\\a.txt", "a", 1.0),
                staged("D:\\Docs\\sub\\b.txt", "b", 2.0),
            ]
        };
        store.reconcile(&root("D:\\Docs"), docs()).expect("first");
        let snapshot = store.serialize().expect("serialize");
        store.reconcile(&root("D:\\Docs"), docs()).expect("second");
        assert_eq!(store.entries.len(), 2);
        assert_eq!(store.serialize().expect("reserialize"), snapshot);
    }

    #[test]
    fn ambiguous_staged_aborts_without_mutation() {
        let mut store = VectorStore::new();
        store.insert("D:\\Docs\\a.txt".to_string(), "a".to_string(), vec384(1.0));
        let before = store.serialize().expect("snapshot");
        let result = store.reconcile(
            &root("D:\\Docs"),
            vec![
                staged("D:\\Docs\\a.txt", "a", 1.0),
                staged("rel\\b.txt", "b", 2.0),
            ],
        );
        assert!(result.is_err());
        assert_eq!(store.serialize().expect("after"), before);
        assert_eq!(store.entries.len(), 1);
    }

    #[test]
    fn ambiguous_root_aborts_without_mutation() {
        let mut store = VectorStore::new();
        store.insert("D:\\Docs\\a.txt".to_string(), "a".to_string(), vec384(1.0));
        let before = store.serialize().expect("snapshot");
        let result = store.reconcile(
            &root("rel\\Docs"),
            vec![staged("D:\\Docs\\a.txt", "a", 1.0)],
        );
        assert!(result.is_err());
        assert_eq!(store.serialize().expect("after"), before);
    }

    #[test]
    fn case_and_separator_variants_produce_one_entry() {
        let mut store = VectorStore::new();
        store.insert(
            "D:\\Docs\\Note.md".to_string(),
            "v1".to_string(),
            vec384(1.0),
        );
        store.insert(
            "d:/docs/note.md".to_string(),
            "v2".to_string(),
            vec384(2.0),
        );
        store.insert(
            "D:\\DOCS\\NOTE.MD".to_string(),
            "v3".to_string(),
            vec384(3.0),
        );
        assert_eq!(store.entries.len(), 1);
        assert_eq!(store.entries[0].content, "v3");
        store
            .reconcile(&root("D:\\Docs"), vec![staged("d:/docs\\NOTE.md", "v4", 4.0)])
            .expect("reconcile");
        assert_eq!(store.entries.len(), 1);
        assert_eq!(store.entries[0].content, "v4");
    }

    #[test]
    fn distinct_absolute_paths_remain_distinct() {
        let mut store = VectorStore::new();
        store.insert("D:\\Docs\\a.txt".to_string(), "a".to_string(), vec384(1.0));
        store.insert("D:\\Docs\\b.txt".to_string(), "b".to_string(), vec384(1.0));
        store.insert(
            "D:\\Docs\\sub\\a.txt".to_string(),
            "c".to_string(),
            vec384(1.0),
        );
        assert_eq!(store.entries.len(), 3);
    }

    #[test]
    fn directory_boundary_checks() {
        let docs_root = path_key("D:\\Docs").expect("root key");
        for outside in [
            "D:\\Docs-old\\a.txt",
            "D:\\Docs2\\a.txt",
            "D:\\DocsOld",
            "E:\\Docs\\a.txt",
        ] {
            let key = path_key(outside).expect("outside key");
            assert!(!is_in_root(&key, &docs_root), "{outside} must be outside");
        }
        for inside in ["D:\\Docs\\a.txt", "D:\\Docs\\sub\\b.txt"] {
            let key = path_key(inside).expect("inside key");
            assert!(is_in_root(&key, &docs_root), "{inside} must be inside");
        }
    }

    #[test]
    fn drive_root_membership() {
        let drive = path_key("D:\\").expect("drive key");
        assert_eq!(drive, "d:\\");
        let inside = path_key("D:\\Docs\\a.txt").expect("inside");
        assert!(is_in_root(&inside, &drive));
        let bare_drive = path_key("D:\\").expect("bare");
        assert!(is_in_root(&bare_drive, &drive));
    }

    #[test]
    fn unc_share_root_membership_and_boundaries() {
        let share = path_key("\\\\server\\share").expect("share key");
        assert_eq!(share, "\\\\server\\share\\");
        let with_slash = path_key("\\\\server\\share\\").expect("slash");
        assert_eq!(share, with_slash);
        let inside = path_key("\\\\server\\share\\Docs\\a.txt").expect("inside");
        assert!(is_in_root(&inside, &share));
        let docs_root = path_key("\\\\server\\share\\Docs").expect("docs");
        assert!(is_in_root(&inside, &docs_root));
        for outside in [
            "\\\\server\\share-other\\a.txt",
            "\\\\server\\other\\a.txt",
            "\\\\other\\share\\a.txt",
        ] {
            let key = path_key(outside).expect("outside");
            assert!(!is_in_root(&key, &share), "{outside} must be outside share");
        }
        let docs_old = path_key("\\\\server\\share\\Docs-old\\a.txt").expect("old");
        assert!(!is_in_root(&docs_old, &docs_root));
    }

    #[test]
    fn root_equality_is_inside() {
        assert!(is_in_root(
            &path_key("D:\\Docs").expect("a"),
            &path_key("D:\\Docs").expect("b")
        ));
        assert!(is_in_root(
            &path_key("D:\\").expect("a"),
            &path_key("D:\\").expect("b")
        ));
        assert!(is_in_root(
            &path_key("\\\\server\\share\\").expect("a"),
            &path_key("\\\\server\\share").expect("b")
        ));
    }

    #[test]
    fn extended_drive_and_unc_normalization() {
        assert_eq!(
            path_key("\\\\?\\C:\\Docs\\File.txt").expect("ext drive"),
            path_key("C:\\Docs\\File.txt").expect("plain drive")
        );
        assert_eq!(
            path_key("\\\\?\\C:\\Docs\\File.txt").expect("ext"),
            "c:\\docs\\file.txt"
        );
        assert_eq!(
            path_key("\\\\?\\UNC\\server\\share\\Docs\\File.txt").expect("ext unc"),
            path_key("\\\\server\\share\\Docs\\File.txt").expect("plain unc")
        );
        assert_eq!(
            path_key("\\\\?\\UNC\\server\\share\\Docs\\File.txt").expect("ext unc"),
            "\\\\server\\share\\docs\\file.txt"
        );
        // Forward-slash spellings converge too.
        assert_eq!(
            path_key("C:/Docs/File.txt").expect("slash"),
            "c:\\docs\\file.txt"
        );
    }

    #[test]
    fn device_and_relative_entries_are_ambiguous_and_retained() {
        for ambiguous in [
            "\\\\?\\Volume{12345678-1234-1234-1234-123456789abc}\\x.txt",
            "\\\\?\\GLOBALROOT\\Device\\HarddiskVolume1\\a.txt",
            "\\\\.\\C:\\Docs\\a.txt",
            "\\\\.\\COM1",
            "docs\\a.txt",
            "a.txt",
            "..\\shared\\b.txt",
            "C:foo.txt",
            "\\foo\\bar.txt",
            "",
        ] {
            assert!(path_key(ambiguous).is_none(), "{ambiguous} must be ambiguous");
        }
        let mut store = VectorStore::new();
        store.entries.push(Entry {
            path: "docs\\legacy.txt".to_string(),
            content: "legacy".to_string(),
            vector: vec384(9.0),
        });
        store.insert(
            "D:\\Docs\\a.txt".to_string(),
            "a".to_string(),
            vec384(1.0),
        );
        store
            .reconcile(&root("D:\\Docs"), vec![staged("D:\\Docs\\b.txt", "b", 2.0)])
            .expect("reconcile");
        // Ambiguous legacy retained even though it is not in the staged set;
        // the valid in-root entry was correctly replaced.
        assert!(store.entries.iter().any(|e| e.path == "docs\\legacy.txt"));
        assert!(!store.entries.iter().any(|e| e.path == "D:\\Docs\\a.txt"));
        assert!(store.entries.iter().any(|e| e.path == "D:\\Docs\\b.txt"));
        // An ambiguous staged path must abort instead of partially applying.
        let before = store.entries.len();
        assert!(store
            .reconcile(
                &root("D:\\Docs"),
                vec![staged("C:foo.txt", "x", 1.0)]
            )
            .is_err());
        assert_eq!(store.entries.len(), before);
    }

    #[test]
    fn root_spelling_variants_converge() {
        let base = path_key("D:\\Docs").expect("base");
        for variant in [
            "D:\\Docs",
            "D:\\Docs\\",
            "D:/Docs",
            "D:/Docs/",
            "d:\\docs",
            "D:\\Docs\\.",
            "D:\\Docs\\sub\\..",
            "\"D:\\Docs\"",
        ] {
            assert_eq!(path_key(variant).expect(variant), base, "{variant}");
        }
        assert_eq!(path_key("\\\\server\\share").expect("unc"), "\\\\server\\share\\");
        assert_eq!(
            path_key("\\\\server\\share\\").expect("unc slash"),
            "\\\\server\\share\\"
        );
    }

    #[tokio::test]
    async fn concurrent_reconciles_serialize_newest_wins() {
        use std::sync::{Arc, Mutex as StdMutex};
        let store = Arc::new(StdMutex::new(VectorStore::new()));
        let op = Arc::new(tokio::sync::Mutex::new(()));
        store.lock().unwrap().insert(
            "D:\\Docs\\old.txt".to_string(),
            "old".to_string(),
            vec384(1.0),
        );
        let acquired = Arc::new(tokio::sync::Notify::new());
        let acquired_clone = acquired.clone();
        let store_a = store.clone();
        let op_a = op.clone();
        let stale = tokio::spawn(async move {
            let _guard = op_a.lock().await;
            acquired_clone.notify_one();
            // Simulate a slow stale scan+embed while holding the op lock.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let mut guard = store_a.lock().unwrap();
            guard
                .reconcile(&root("D:\\Docs"), vec![staged("D:\\Docs\\old.txt", "old", 1.0)])
                .expect("stale reconcile");
        });
        acquired.notified().await;
        let store_b = store.clone();
        let op_b = op.clone();
        let fresh = tokio::spawn(async move {
            let _guard = op_b.lock().await;
            let mut guard = store_b.lock().unwrap();
            guard
                .reconcile(
                    &root("D:\\Docs"),
                    vec![staged("D:\\Docs\\new.txt", "new", 2.0)],
                )
                .expect("fresh reconcile");
        });
        stale.await.expect("stale task");
        fresh.await.expect("fresh task");
        let guard = store.lock().unwrap();
        assert_eq!(guard.entries.len(), 1);
        assert_eq!(guard.entries[0].path, "D:\\Docs\\new.txt");
        assert_eq!(guard.entries[0].content, "new");
    }

    #[test]
    fn registry_round_trip_persists_folders_and_entries() {
        let dir = test_dir("registry-rt");
        let dest = dir.join(INDEX_FILE_NAME);
        let mut store = VectorStore::new();
        store
            .reconcile_and_register(
                &root("D:\\Docs"),
                "D:\\Docs",
                vec![staged("D:\\Docs\\a.txt", "a", 1.0)],
            )
            .expect("register docs");
        store
            .reconcile_and_register(&root("E:\\Empty"), "E:\\Empty", vec![])
            .expect("register empty");
        store.save_to_path(&dest).expect("save");
        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        let folders = loaded.list_folders();
        assert_eq!(folders.len(), 2);
        assert_eq!(folders[0].path, "D:\\Docs");
        assert_eq!(folders[0].file_count, 1);
        assert_eq!(folders[1].path, "E:\\Empty");
        assert_eq!(folders[1].file_count, 0);
        assert_eq!(loaded.entry_count(), 1);
    }

    #[test]
    fn v1_index_loads_entries_with_empty_registry() {
        let dir = test_dir("v1compat");
        let dest = dir.join(INDEX_FILE_NAME);
        let v1 = serde_json::json!({
            "version": 1,
            "model": INDEX_MODEL_ID,
            "dim": INDEX_DIM,
            "entries": [{ "path": "D:\\Docs\\a.txt", "content": "x", "vector": vec384(1.0) }]
        });
        std::fs::write(&dest, serde_json::to_string(&v1).unwrap()).expect("seed v1");
        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(loaded.entry_count(), 1);
        assert!(loaded.list_folders().is_empty());
        assert!(loaded.registered_folders().is_empty());
    }

    #[test]
    fn folder_registration_deduplicates_equivalent_paths() {
        let mut store = VectorStore::new();
        store
            .reconcile_and_register(
                &root("D:\\Docs"),
                "D:\\Docs",
                vec![staged("D:\\Docs\\a.txt", "a", 1.0)],
            )
            .expect("first");
        store
            .reconcile_and_register(
                &root("d:/docs\\"),
                "d:/docs\\",
                vec![staged("D:\\Docs\\a.txt", "a", 1.0)],
            )
            .expect("second");
        assert_eq!(store.registered_folders().len(), 1);
        assert_eq!(store.registered_folders()[0], "d:/docs\\");
        assert!(store.is_registered(&root("D:\\DOCS")));
        assert_eq!(store.entry_count(), 1);
    }

    #[test]
    fn folder_counts_cover_empty_and_nested_roots() {
        let mut store = VectorStore::new();
        store
            .reconcile_and_register(
                &root("D:\\Docs"),
                "D:\\Docs",
                vec![
                    staged("D:\\Docs\\a.txt", "a", 1.0),
                    staged("D:\\Docs\\sub\\b.txt", "b", 2.0),
                ],
            )
            .expect("docs");
        store
            .reconcile_and_register(&root("E:\\Empty"), "E:\\Empty", vec![])
            .expect("empty");
        store
            .reconcile_and_register(
                &root("D:\\Docs\\sub"),
                "D:\\Docs\\sub",
                vec![staged("D:\\Docs\\sub\\b.txt", "b", 2.0)],
            )
            .expect("sub");
        let folders = store.list_folders();
        assert_eq!(folders.len(), 3);
        let get = |p: &str| {
            folders
                .iter()
                .find(|f| f.path == p)
                .unwrap_or_else(|| panic!("missing {p}"))
                .file_count
        };
        // Nested scopes overlap by design: the parent counts sub files too.
        assert_eq!(get("D:\\Docs"), 2);
        assert_eq!(get("D:\\Docs\\sub"), 1);
        assert_eq!(get("E:\\Empty"), 0);
    }

    #[test]
    fn remove_folder_is_root_scoped() {
        let mut store = VectorStore::new();
        store
            .reconcile_and_register(
                &root("D:\\Docs"),
                "D:\\Docs",
                vec![staged("D:\\Docs\\a.txt", "a", 1.0)],
            )
            .expect("docs");
        store
            .reconcile_and_register(
                &root("E:\\Other"),
                "E:\\Other",
                vec![staged("E:\\Other\\b.txt", "b", 2.0)],
            )
            .expect("other");
        let removed = store.remove_folder(&root("D:\\Docs")).expect("remove");
        assert_eq!(removed, 1);
        assert_eq!(store.entry_count(), 1);
        let folders = store.list_folders();
        assert_eq!(folders.len(), 1);
        assert_eq!(folders[0].path, "E:\\Other");
        assert_eq!(folders[0].file_count, 1);
    }

    #[test]
    fn remove_folder_leaves_source_files_on_disk() {
        let base = test_dir("remove-disk");
        let dir_a = base.join("folderA");
        let dir_b = base.join("folderB");
        std::fs::create_dir_all(&dir_a).expect("dirA");
        std::fs::create_dir_all(&dir_b).expect("dirB");
        let file_a = dir_a.join("note.txt");
        let file_b = dir_b.join("keep.txt");
        std::fs::write(&file_a, "alpha").expect("fileA");
        std::fs::write(&file_b, "beta").expect("fileB");
        let display_a = dir_a.to_string_lossy().to_string();
        let display_b = dir_b.to_string_lossy().to_string();
        let mut store = VectorStore::new();
        store
            .reconcile_and_register(
                &dir_a,
                &display_a,
                vec![staged(&file_a.to_string_lossy(), "alpha", 1.0)],
            )
            .expect("register A");
        store
            .reconcile_and_register(
                &dir_b,
                &display_b,
                vec![staged(&file_b.to_string_lossy(), "beta", 2.0)],
            )
            .expect("register B");
        assert_eq!(store.entry_count(), 2);
        let removed = store.remove_folder(&dir_a).expect("remove A");
        assert_eq!(removed, 1);
        assert_eq!(store.entry_count(), 1);
        // Source files are untouched: removal is index-only.
        assert!(file_a.exists());
        assert!(file_b.exists());
        assert_eq!(file_a.metadata().expect("meta").len(), 5);
    }

    #[test]
    fn ambiguous_folder_identity_rejected_without_mutation() {
        let mut store = VectorStore::new();
        store
            .reconcile_and_register(
                &root("D:\\Docs"),
                "D:\\Docs",
                vec![staged("D:\\Docs\\a.txt", "a", 1.0)],
            )
            .expect("seed");
        let before = store.serialize().expect("snapshot");
        assert!(store
            .reconcile_and_register(&root("rel\\Docs"), "rel\\Docs", vec![])
            .is_err());
        assert!(store.remove_folder(&root("C:foo")).is_err());
        assert!(store.remove_folder(&root("")).is_err());
        assert_eq!(store.serialize().expect("after"), before);
        assert_eq!(store.registered_folders().len(), 1);
    }

    #[test]
    fn failed_reconciliation_preserves_entries_and_registrations() {
        let mut store = VectorStore::new();
        store
            .reconcile_and_register(
                &root("D:\\Docs"),
                "D:\\Docs",
                vec![staged("D:\\Docs\\a.txt", "a", 1.0)],
            )
            .expect("seed");
        let before = store.serialize().expect("snapshot");
        let result = store.reconcile_and_register(
            &root("D:\\Docs"),
            "D:\\Docs",
            vec![
                staged("D:\\Docs\\b.txt", "b", 2.0),
                staged("C:foo.txt", "x", 1.0),
            ],
        );
        assert!(result.is_err());
        assert_eq!(store.serialize().expect("after"), before);
        assert_eq!(store.registered_folders(), vec!["D:\\Docs".to_string()]);
    }

    #[test]
    fn registry_and_entries_survive_persistence_reload() {
        let dir = test_dir("registry-reload");
        let dest = dir.join(INDEX_FILE_NAME);
        let mut store = VectorStore::new();
        store
            .reconcile_and_register(
                &root("D:\\Docs"),
                "D:\\Docs",
                vec![
                    staged("D:\\Docs\\a.txt", "a", 1.0),
                    staged("D:\\Docs\\sub\\b.txt", "b", 2.0),
                ],
            )
            .expect("docs");
        store
            .reconcile_and_register(&root("E:\\Empty"), "E:\\Empty", vec![])
            .expect("empty");
        let before = store.serialize().expect("snapshot");
        store.save_to_path(&dest).expect("save");
        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(loaded.serialize().expect("reserialize"), before);
        let folders = loaded.list_folders();
        assert_eq!(folders.len(), 2);
        assert_eq!(loaded.entry_count(), 2);
    }

    fn outcome_name(outcome: &LoadOutcome) -> &'static str {
        match outcome {
            LoadOutcome::Empty => "Empty",
            LoadOutcome::Loaded(_) => "Loaded",
            LoadOutcome::Quarantined { .. } => "Quarantined",
        }
    }
}
