// Minimal in-memory vector store with versioned JSON persistence.
// Fine for a few thousand documents; once this stops being fine,
// swap the backend — the `insert`/`search` interface here is the
// contract to preserve.
//
// The on-disk format is a versioned JSON envelope holding each
// entry's path, content, and embedding plus the embedding model id
// and dimension. Entries carry either a whole-file vector (legacy
// v1/v2 indexes and the current whole-file indexing pipeline) or chunk
// vectors, never both. It lives in the app-specific Local AppData
// directory as `index-v1.json` and contains full document contents.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub const INDEX_SCHEMA_VERSION: u32 = 3;
/// Stable identifier for the current local embedding model
/// (`fastembed::EmbeddingModel::AllMiniLML6V2`).
pub const INDEX_MODEL_ID: &str = "all-MiniLM-L6-v2";
/// Embedding dimension emitted by `AllMiniLML6V2`.
pub const INDEX_DIM: usize = 384;
/// Experimental minimum cosine similarity for a search match to be
/// returned. Chosen from a tiny exploratory sample (3 relevant queries
/// scoring 46–75%, 3 unrelated queries scoring 2–17%): 25% keeps the
/// relevant sample while dropping the unrelated sample. These are raw
/// similarity scores, not probabilities, and six queries prove nothing
/// general — treat this as a starting point to tune with real usage.
pub const MIN_RELEVANCE_SCORE: f32 = 0.25;
pub const INDEX_FILE_NAME: &str = "index-v1.json";
/// Maximum Unicode characters retained in a search snippet.
const SNIPPET_CHARS: usize = 200;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// One embedded span of a document. `start`/`end` are byte offsets into
/// the owning entry's exact `content`; both must fall on UTF-8 character
/// boundaries and `start < end`. Chunk ranges must cover the whole
/// content with no gaps and strictly advancing ends; neighboring ranges
/// may overlap.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct StoredChunk {
    pub start: usize,
    pub end: usize,
    pub vector: Vec<f32>,
}

/// Mutually exclusive embedding source for an entry. A whole-file vector
/// and a chunk list can never coexist, and `Empty` is valid only for
/// zero-length content.
#[derive(Clone, Debug, PartialEq)]
enum Embedding {
    Whole(Vec<f32>),
    Chunks(Vec<StoredChunk>),
    Empty,
}

#[derive(Clone, Debug)]
struct Entry {
    path: String,
    content: String,
    embedding: Embedding,
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
/// succeeds. A staged document provides either `vector` or `chunks`,
/// never both; both empty is only valid for empty content.
pub struct StagedDoc {
    pub path: String,
    pub content: String,
    pub vector: Vec<f32>,
    pub chunks: Vec<StoredChunk>,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    vector: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    chunks: Vec<StoredChunk>,
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
        let embedding = if vector.is_empty() {
            Embedding::Empty
        } else {
            Embedding::Whole(vector)
        };
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|e| paths_match(&e.path, &path))
        {
            existing.content = content;
            existing.embedding = embedding;
            existing.path = path;
        } else {
            self.entries.push(Entry { path, content, embedding });
        }
    }

    /// Atomically reconcile the selected absolute root with a completely
    /// staged set: upsert every staged doc by normalized identity, then
    /// drop in-root entries absent from the staged set. Unrelated roots
    /// and ambiguous legacy entries are always retained. Validates every
    /// staged key and embedding before mutating, so an ambiguous root,
    /// ambiguous staged path, or malformed staged embedding aborts
    /// without changing the store.
    pub fn reconcile(&mut self, root: &Path, staged: Vec<StagedDoc>) -> Result<usize, String> {
        let root_key = path_key(&root.to_string_lossy()).ok_or_else(|| {
            format!(
                "selected folder has ambiguous identity ({}); refusing to reconcile",
                root.display()
            )
        })?;
        let mut staged_keys: HashSet<String> = HashSet::with_capacity(staged.len());
        let mut staged_with_keys: Vec<(String, StagedDoc, Embedding)> =
            Vec::with_capacity(staged.len());
        for doc in staged {
            let key = path_key(&doc.path).ok_or_else(|| {
                format!(
                    "scanned file has ambiguous identity ({}); refusing to reconcile",
                    doc.path
                )
            })?;
            let embedding = staged_embedding(&doc)?;
            staged_keys.insert(key.clone());
            staged_with_keys.push((key, doc, embedding));
        }
        for (key, doc, embedding) in staged_with_keys {
            if let Some(existing) = self.entries.iter_mut().find(|e| match path_key(&e.path) {
                Some(existing_key) => existing_key == key,
                None => false,
            }) {
                existing.content = doc.content;
                existing.embedding = embedding;
                existing.path = doc.path;
            } else {
                self.entries.push(Entry {
                    path: doc.path,
                    content: doc.content,
                    embedding,
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

    /// One result per file. Whole-file entries keep their head snippet;
    /// chunk entries use the highest finite chunk score and a snippet
    /// bounded to that winning chunk. Non-finite scores are excluded,
    /// which also keeps the `partial_cmp` sort below from ever seeing
    /// NaN. Empty entries are never searchable.
    pub fn search(&self, query_vector: &[f32], limit: usize) -> Vec<SearchResult> {
        let mut scored: Vec<SearchResult> = Vec::new();
        for e in &self.entries {
            let hit: Option<(String, f32)> = match &e.embedding {
                Embedding::Empty => None,
                Embedding::Whole(vector) => {
                    Some((snippet(&e.content), cosine_similarity(query_vector, vector)))
                }
                Embedding::Chunks(chunks) => {
                    let mut best: Option<(f32, &StoredChunk)> = None;
                    for c in chunks {
                        let score = cosine_similarity(query_vector, &c.vector);
                        // Strict `>` keeps the earliest chunk on ties; chunk
                        // ranges are validated to advance deterministically.
                        if score.is_finite() && best.map_or(true, |(b, _)| score > b) {
                            best = Some((score, c));
                        }
                    }
                    best.map(|(score, c)| (chunk_snippet(&e.content, c), score))
                }
            };
            // Drop weak matches before the limit so they never consume
            // result slots.
            if let Some((snippet, score)) = hit {
                if score.is_finite() && score >= MIN_RELEVANCE_SCORE {
                    scored.push(SearchResult {
                        path: e.path.clone(),
                        snippet,
                        score,
                    });
                }
            }
        }

        // Deterministic order: highest score first, then normalized path
        // identity, with a raw-path fallback for ambiguous paths.
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap()
                .then_with(|| tie_key(&a.path).cmp(&tie_key(&b.path)))
        });
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
                .map(|e| match &e.embedding {
                    Embedding::Whole(vector) => SavedEntry {
                        path: e.path.clone(),
                        content: e.content.clone(),
                        vector: vector.clone(),
                        chunks: Vec::new(),
                    },
                    Embedding::Chunks(chunks) => SavedEntry {
                        path: e.path.clone(),
                        content: e.content.clone(),
                        vector: Vec::new(),
                        chunks: chunks.clone(),
                    },
                    Embedding::Empty => SavedEntry {
                        path: e.path.clone(),
                        content: e.content.clone(),
                        vector: Vec::new(),
                        chunks: Vec::new(),
                    },
                })
                .collect(),
            folders: self.folders.clone(),
        };
        serde_json::to_string(&saved).map_err(|e| format!("could not serialize index: {e}"))
    }

    /// Validate a parsed envelope and convert it. v1 envelopes (no
    /// folders field) load entries with an empty registry. v2 envelopes
    /// load entries strictly and sanitize duplicate or ambiguous folder
    /// paths without discarding valid entries. v3 envelopes additionally
    /// accept chunk-only and empty entries; legacy entries may not carry
    /// chunk data. Anything else quarantines.
    fn from_saved(saved: SavedIndex) -> Result<Self, String> {
        if saved.version != 1 && saved.version != 2 && saved.version != INDEX_SCHEMA_VERSION {
            return Err(format!(
                "unsupported index version {} (expected 1, 2, or {})",
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
        let legacy = saved.version == 1 || saved.version == 2;
        let mut store = VectorStore::new();
        for e in saved.entries {
            if e.path.is_empty() {
                return Err("index entry has an empty path".to_string());
            }
            let embedding = if legacy {
                // Legacy entries always carry a valid whole-file vector and
                // never carry chunk data.
                if !e.chunks.is_empty() {
                    return Err(format!(
                        "legacy entry '{}' contains chunk data",
                        e.path
                    ));
                }
                validate_whole_vector(&e.vector, &e.path)?;
                Embedding::Whole(e.vector)
            } else {
                match (e.vector.is_empty(), e.chunks.is_empty()) {
                    (false, false) => {
                        return Err(format!(
                            "entry '{}' has both a whole-file vector and chunks",
                            e.path
                        ));
                    }
                    (false, true) => {
                        validate_whole_vector(&e.vector, &e.path)?;
                        Embedding::Whole(e.vector)
                    }
                    (true, false) => {
                        validate_chunks(&e.content, &e.chunks)
                            .map_err(|reason| format!("entry '{}': {reason}", e.path))?;
                        Embedding::Chunks(e.chunks)
                    }
                    (true, true) => {
                        if e.content.is_empty() {
                            Embedding::Empty
                        } else {
                            return Err(format!(
                                "entry '{}' has no embedding but non-empty content",
                                e.path
                            ));
                        }
                    }
                }
            };
            store.entries.push(Entry {
                path: e.path,
                content: e.content,
                embedding,
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

/// Outcome of a legacy index migration attempt. Paths are explicit so
/// the helper stays independent of Tauri state and unit-testable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationOutcome {
    /// Legacy bytes were copied atomically to the new location.
    Migrated,
    /// The new location already has an index; nothing was copied.
    SkippedDestinationExists,
    /// No legacy index exists; nothing was copied.
    SkippedNoLegacy,
}

/// Copy a legacy saved index to a new location exactly once. Never
/// overwrites an existing destination and never modifies or deletes the
/// legacy source. Publication is no-replace: the copy is fully written
/// and flushed to a temporary file first, then published with a hard
/// link that fails if the destination appeared in the meantime, in
/// which case the destination is left untouched. Callers must load the
/// destination through the normal loader afterwards so validation and
/// quarantine still apply.
pub fn migrate_legacy_index(new_path: &Path, legacy_path: &Path) -> Result<MigrationOutcome, String> {
    if fs::metadata(new_path).is_ok() {
        return Ok(MigrationOutcome::SkippedDestinationExists);
    }
    let legacy_bytes = match fs::read(legacy_path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(MigrationOutcome::SkippedNoLegacy)
        }
        Err(e) => {
            return Err(format!(
                "could not read legacy index {}: {e}",
                legacy_path.display()
            ))
        }
    };
    publish_new_file_no_replace(new_path, &legacy_bytes)
}

/// Atomically publish a new file only if the destination is still
/// absent. Used only for migration; normal index saves intentionally
/// replace the current index and keep using the overwriting rename.
fn publish_new_file_no_replace(dest: &Path, bytes: &[u8]) -> Result<MigrationOutcome, String> {
    let parent = dest
        .parent()
        .ok_or_else(|| format!("index path {} has no parent directory", dest.display()))?;
    fs::create_dir_all(parent)
        .map_err(|e| format!("could not create index directory {}: {e}", parent.display()))?;
    let tmp = unique_tmp_path(dest);
    let write_result: std::io::Result<()> = (|| {
        let mut file = fs_create_tmp(&tmp)?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(e) = write_result {
        let _ = fs::remove_file(&tmp);
        return Err(format!("could not write temporary index {}: {e}", tmp.display()));
    }
    match fs::hard_link(&tmp, dest) {
        Ok(()) => {
            let _ = fs::remove_file(&tmp);
            Ok(MigrationOutcome::Migrated)
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            let _ = fs::remove_file(&tmp);
            // Another publisher won the race: confirm the destination is
            // really there and leave its bytes untouched.
            if fs::metadata(dest).is_ok() {
                return Ok(MigrationOutcome::SkippedDestinationExists);
            }
            Err(format!(
                "could not publish migrated index {}: {e}",
                dest.display()
            ))
        }
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(format!(
                "could not publish migrated index {}: {e}",
                dest.display()
            ))
        }
    }
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

/// Whole-file vectors are always exactly `INDEX_DIM` finite values.
fn validate_whole_vector(vector: &[f32], path: &str) -> Result<(), String> {
    if vector.len() != INDEX_DIM {
        return Err(format!(
            "entry '{path}' has dimension {} (expected {INDEX_DIM})",
            vector.len()
        ));
    }
    if !vector.iter().all(|v| v.is_finite()) {
        return Err(format!("entry '{path}' has a non-finite embedding value"));
    }
    Ok(())
}

/// Chunk ranges must be UTF-8-safe, cover the whole content with no gaps,
/// and have strictly advancing starts and ends. Neighboring ranges may
/// still overlap as long as `current.start <= previous.end`.
fn validate_chunks(content: &str, chunks: &[StoredChunk]) -> Result<(), String> {
    if chunks.is_empty() {
        return Err("chunk list is empty".to_string());
    }
    if chunks[0].start != 0 {
        return Err("chunks do not cover the start of the content".to_string());
    }
    if chunks[chunks.len() - 1].end != content.len() {
        return Err("chunks do not cover the end of the content".to_string());
    }
    let mut last_end: Option<usize> = None;
    let mut last_start: Option<usize> = None;
    for c in chunks {
        if c.start >= c.end {
            return Err(format!(
                "chunk range {}..{} must satisfy start < end",
                c.start, c.end
            ));
        }
        if c.end > content.len() {
            return Err(format!(
                "chunk range {}..{} exceeds content length {}",
                c.start,
                c.end,
                content.len()
            ));
        }
        if !content.is_char_boundary(c.start) || !content.is_char_boundary(c.end) {
            return Err(format!(
                "chunk range {}..{} is not on UTF-8 character boundaries",
                c.start, c.end
            ));
        }
        if c.vector.len() != INDEX_DIM {
            return Err(format!(
                "chunk at {}..{} has dimension {} (expected {INDEX_DIM})",
                c.start,
                c.end,
                c.vector.len()
            ));
        }
        if !c.vector.iter().all(|v| v.is_finite()) {
            return Err(format!(
                "chunk at {}..{} has a non-finite embedding value",
                c.start, c.end
            ));
        }
        if let Some(prev_end) = last_end {
            if c.end <= prev_end {
                return Err(format!(
                    "chunk ends must strictly advance (chunk ending at {} after {})",
                    c.end, prev_end
                ));
            }
            if c.start > prev_end {
                return Err(format!(
                    "gap between chunks (chunk starting at {} after {})",
                    c.start, prev_end
                ));
            }
            if let Some(prev_start) = last_start {
                if c.start <= prev_start {
                    return Err(format!(
                        "chunk starts must strictly increase (chunk starting at {} after {})",
                        c.start, prev_start
                    ));
                }
            }
        }
        last_end = Some(c.end);
        last_start = Some(c.start);
    }
    Ok(())
}

/// Convert a staged document into the mutually exclusive in-memory
/// embedding, rejecting a non-empty document with no embedding and a
/// document that supplies both a whole-file vector and chunks. Chunk
/// ranges are validated completely here, before any store mutation.
fn staged_embedding(doc: &StagedDoc) -> Result<Embedding, String> {
    match (doc.vector.is_empty(), doc.chunks.is_empty()) {
        (false, true) => {
            validate_whole_vector(&doc.vector, &doc.path)
                .map_err(|reason| format!("scanned file '{}': {reason}", doc.path))?;
            Ok(Embedding::Whole(doc.vector.clone()))
        }
        (true, false) => {
            validate_chunks(&doc.content, &doc.chunks)
                .map_err(|reason| format!("scanned file '{}': {reason}", doc.path))?;
            Ok(Embedding::Chunks(doc.chunks.clone()))
        }
        (true, true) => {
            if doc.content.is_empty() {
                Ok(Embedding::Empty)
            } else {
                Err(format!(
                    "scanned file '{}' has no embedding but non-empty content; refusing to reconcile",
                    doc.path
                ))
            }
        }
        (false, false) => Err(format!(
            "scanned file '{}' has both a whole-file vector and chunks; refusing to reconcile",
            doc.path
        )),
    }
}

/// Sort key for equal-scoring results: normalized path identity first,
/// with a raw-path fallback for ambiguous paths.
fn tie_key(path: &str) -> (u8, String) {
    match path_key(path) {
        Some(key) => (0, key),
        None => (1, path.to_string()),
    }
}

fn snippet(content: &str) -> String {
    content.chars().take(SNIPPET_CHARS).collect()
}

/// Snippet bounded to the winning chunk's range, then capped at
/// `SNIPPET_CHARS` Unicode characters.
fn chunk_snippet(content: &str, chunk: &StoredChunk) -> String {
    let start = chunk.start.min(content.len());
    let end = chunk.end.min(content.len());
    if start >= end || !content.is_char_boundary(start) || !content.is_char_boundary(end) {
        return snippet(content);
    }
    content[start..end].chars().take(SNIPPET_CHARS).collect()
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
        assert_eq!(whole_vector(&store.entries[0]), vec![0.0, 1.0]);
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
        // Similar enough to the all-ones query to clear the relevance
        // threshold (cosine ≈ 0.71), unlike the previous near-orthogonal
        // vector which the threshold now excludes.
        let mut other = vec384(1.0);
        for v in other.iter_mut().skip(INDEX_DIM / 2) {
            *v = 0.0;
        }
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
        assert_eq!(whole_vector(entry), vec384(2.0));
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
        assert_eq!(whole_vector(&restarted.entries[0]), vec384(5.0));
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
            chunks: Vec::new(),
        }
    }

    fn chunk(start: usize, end: usize, fill: f32) -> StoredChunk {
        StoredChunk {
            start,
            end,
            vector: vec384(fill),
        }
    }

    fn chunk_with(start: usize, end: usize, vector: Vec<f32>) -> StoredChunk {
        StoredChunk { start, end, vector }
    }

    fn chunked_staged(path: &str, content: &str, chunks: Vec<StoredChunk>) -> StagedDoc {
        StagedDoc {
            path: path.to_string(),
            content: content.to_string(),
            vector: Vec::new(),
            chunks,
        }
    }

    fn empty_staged(path: &str, content: &str) -> StagedDoc {
        StagedDoc {
            path: path.to_string(),
            content: content.to_string(),
            vector: Vec::new(),
            chunks: Vec::new(),
        }
    }

    /// Extract the whole-file vector of a test entry, panicking for any
    /// other representation.
    fn whole_vector(entry: &Entry) -> Vec<f32> {
        match &entry.embedding {
            Embedding::Whole(vector) => vector.clone(),
            other => panic!("expected whole-file embedding, got {other:?}"),
        }
    }

    fn chunks_of(entry: &Entry) -> Vec<StoredChunk> {
        match &entry.embedding {
            Embedding::Chunks(chunks) => chunks.clone(),
            other => panic!("expected chunk embedding, got {other:?}"),
        }
    }

    fn chunked_entry(path: &str, content: &str, chunks: Vec<StoredChunk>) -> Entry {
        Entry {
            path: path.to_string(),
            content: content.to_string(),
            embedding: Embedding::Chunks(chunks),
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
        assert_eq!(whole_vector(&store.entries[0]), vec384(2.0));
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
            embedding: Embedding::Whole(vec384(1.0)),
        });
        store.entries.push(Entry {
            path: "d:/docs/note.md".to_string(),
            content: "stale two".to_string(),
            embedding: Embedding::Whole(vec384(1.0)),
        });
        // An unrelated entry must survive untouched.
        store.entries.push(Entry {
            path: "E:\\Other\\keep.txt".to_string(),
            content: "keep".to_string(),
            embedding: Embedding::Whole(vec384(7.0)),
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
        assert_eq!(whole_vector(matching[0]), vec384(4.0));
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
        assert_eq!(whole_vector(other), vec384(2.0));
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
            embedding: Embedding::Whole(vec384(9.0)),
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

    #[test]
    fn v2_registry_sanitizes_invalid_entries_without_losing_documents() {
        let dir = test_dir("v2sanitize");
        let dest = dir.join(INDEX_FILE_NAME);
        let v2 = serde_json::json!({
            "version": 2,
            "model": INDEX_MODEL_ID,
            "dim": INDEX_DIM,
            "entries": [
                { "path": "D:\\Docs\\a.txt", "content": "a", "vector": vec384(1.0) },
                { "path": "E:\\Other\\b.txt", "content": "b", "vector": vec384(2.0) }
            ],
            "folders": ["D:\\Docs", "d:/docs\\", "rel\\Docs", "", "E:\\Other"]
        });
        std::fs::write(&dest, serde_json::to_string(&v2).unwrap()).expect("seed v2");
        // Must load normally: invalid registry paths must not quarantine
        // the whole index.
        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(loaded.entry_count(), 2);
        let folders = loaded.list_folders();
        assert_eq!(folders.len(), 2);
        assert_eq!(folders[0].path, "D:\\Docs");
        assert_eq!(folders[0].file_count, 1);
        assert_eq!(folders[1].path, "E:\\Other");
        assert_eq!(folders[1].file_count, 1);
        assert!(dest.exists());
        assert!(corrupt_files_in(&dir).is_empty());
    }

    #[test]
    fn v1_index_upgrades_to_v3_on_next_save() {
        let dir = test_dir("v1upgrade");
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
        loaded.save_to_path(&dest).expect("save");
        let raw = std::fs::read_to_string(&dest).expect("read back");
        let saved: serde_json::Value = serde_json::from_str(&raw).expect("parse");
        assert_eq!(saved["version"], serde_json::Value::from(3));
        assert_eq!(saved["entries"].as_array().expect("entries").len(), 1);
        assert_eq!(
            saved["entries"][0]["path"],
            serde_json::Value::from("D:\\Docs\\a.txt")
        );
        let folders = saved["folders"].as_array().expect("folders field exists");
        assert!(folders.is_empty());
    }

    #[test]
    fn unregistered_legacy_root_removal_is_index_only() {
        let base = test_dir("legacy-remove");
        let dir_a = base.join("folderA");
        let dir_b = base.join("folderB");
        std::fs::create_dir_all(&dir_a).expect("dirA");
        std::fs::create_dir_all(&dir_b).expect("dirB");
        let file_a1 = dir_a.join("one.txt");
        let file_a2 = dir_a.join("two.txt");
        let file_b = dir_b.join("keep.txt");
        std::fs::write(&file_a1, "one").expect("fileA1");
        std::fs::write(&file_a2, "two").expect("fileA2");
        std::fs::write(&file_b, "keep").expect("fileB");
        // Genuine v1 index: entries under two roots, no registry.
        let dest = base.join(INDEX_FILE_NAME);
        let v1 = serde_json::json!({
            "version": 1,
            "model": INDEX_MODEL_ID,
            "dim": INDEX_DIM,
            "entries": [
                { "path": file_a1.to_string_lossy(), "content": "one", "vector": vec384(1.0) },
                { "path": file_a2.to_string_lossy(), "content": "two", "vector": vec384(2.0) },
                { "path": file_b.to_string_lossy(), "content": "keep", "vector": vec384(3.0) }
            ]
        });
        std::fs::write(&dest, serde_json::to_string(&v1).unwrap()).expect("seed v1");
        let mut loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(loaded.entry_count(), 3);
        assert!(loaded.list_folders().is_empty());
        let removed = loaded.remove_folder(&dir_a).expect("remove");
        assert_eq!(removed, 2);
        assert_eq!(loaded.entry_count(), 1);
        // No registration is created as a side effect of removal.
        assert!(loaded.list_folders().is_empty());
        // Removal is index-only: every source file still exists.
        assert!(file_a1.exists());
        assert!(file_a2.exists());
        assert!(file_b.exists());
    }

    fn migration_paths(case: &str) -> (PathBuf, PathBuf) {
        let base = test_dir(case);
        let new_dir = base.join("new-id");
        let legacy_dir = base.join("com.example.deepsearch");
        std::fs::create_dir_all(&legacy_dir).expect("legacy dir");
        (new_dir.join(INDEX_FILE_NAME), legacy_dir.join(INDEX_FILE_NAME))
    }

    fn seed_index(dest: &Path, version: u32, folders: Option<serde_json::Value>) {
        let mut value = serde_json::json!({
            "version": version,
            "model": INDEX_MODEL_ID,
            "dim": INDEX_DIM,
            "entries": [{ "path": "D:\\Docs\\a.txt", "content": "x", "vector": vec384(1.0) }]
        });
        if let Some(f) = folders {
            value["folders"] = f;
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).expect("parent dir");
        }
        std::fs::write(dest, serde_json::to_string(&value).unwrap()).expect("seed index");
    }

    /// Seed a v3 envelope with an arbitrary `entries` array so loader
    /// validation paths can be exercised directly.
    fn seed_v3_index(dest: &Path, entries: serde_json::Value) {
        let value = serde_json::json!({
            "version": 3,
            "model": INDEX_MODEL_ID,
            "dim": INDEX_DIM,
            "entries": entries,
        });
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).expect("parent dir");
        }
        std::fs::write(dest, serde_json::to_string(&value).unwrap()).expect("seed v3 index");
    }

    fn chunk_json(start: usize, end: usize, vector: Vec<f32>) -> serde_json::Value {
        serde_json::json!({ "start": start, "end": end, "vector": vector })
    }

    fn assert_quarantined(dest: &Path, dir: &Path) {
        match VectorStore::load_from_path(dest).expect("load") {
            LoadOutcome::Quarantined { store, .. } => assert_eq!(store.entry_count(), 0),
            other => panic!("expected Quarantined, got {}", outcome_name(&other)),
        }
        assert_eq!(corrupt_files_in(dir).len(), 1);
    }

    #[test]
    fn migration_copies_legacy_and_preserves_source() {
        let (new_path, legacy_path) = migration_paths("migrate-ok");
        seed_index(&legacy_path, 2, Some(serde_json::json!(["D:\\Docs"])));
        let outcome = migrate_legacy_index(&new_path, &legacy_path).expect("migrate");
        assert_eq!(outcome, MigrationOutcome::Migrated);
        assert_eq!(
            std::fs::read(&new_path).expect("new bytes"),
            std::fs::read(&legacy_path).expect("legacy bytes")
        );
        // Loading the migrated copy retains entries and registrations.
        let loaded = match VectorStore::load_from_path(&new_path).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(loaded.entry_count(), 1);
        assert_eq!(loaded.registered_folders(), vec!["D:\\Docs".to_string()]);
    }

    #[test]
    fn migration_never_overwrites_existing_destination() {
        let (new_path, legacy_path) = migration_paths("migrate-keep");
        seed_index(&new_path, 2, Some(serde_json::json!(["E:\\Other"])));
        seed_index(&legacy_path, 1, None);
        let before_new = std::fs::read(&new_path).expect("new bytes");
        let before_legacy = std::fs::read(&legacy_path).expect("legacy bytes");
        let outcome = migrate_legacy_index(&new_path, &legacy_path).expect("migrate");
        assert_eq!(outcome, MigrationOutcome::SkippedDestinationExists);
        assert_eq!(std::fs::read(&new_path).expect("new after"), before_new);
        assert_eq!(
            std::fs::read(&legacy_path).expect("legacy after"),
            before_legacy
        );
    }

    #[test]
    fn migration_skips_cleanly_when_both_absent() {
        let (new_path, legacy_path) = migration_paths("migrate-empty");
        let missing_legacy = legacy_path.with_file_name("no-such-index.json");
        let outcome = migrate_legacy_index(&new_path, &missing_legacy).expect("migrate");
        assert_eq!(outcome, MigrationOutcome::SkippedNoLegacy);
        assert!(!new_path.exists());
    }

    #[test]
    fn migration_failure_preserves_source_and_creates_no_destination() {
        let base = test_dir("migrate-fail");
        let legacy_dir = base.join("com.example.deepsearch");
        std::fs::create_dir_all(&legacy_dir).expect("legacy dir");
        let new_path = base.join("new-id").join(INDEX_FILE_NAME);
        // A directory cannot be read as an index file: deterministic failure.
        let result = migrate_legacy_index(&new_path, &legacy_dir);
        assert!(result.is_err());
        assert!(!new_path.exists());
        // The source is preserved and still a directory.
        assert!(legacy_dir.is_dir());
        // A genuine legacy file stays recoverable after an unrelated
        // failure: point the destination somewhere uncreatable (nested
        // under a regular file) so the atomic write cannot proceed.
        let base = test_dir("migrate-recover");
        let blocker = base.join("blocker.txt");
        std::fs::write(&blocker, "not a directory").expect("blocker");
        let legacy = base.join("legacy").join(INDEX_FILE_NAME);
        seed_index(&legacy, 1, None);
        let legacy_bytes = std::fs::read(&legacy).expect("legacy bytes");
        let bad_new = blocker.join(INDEX_FILE_NAME);
        assert!(migrate_legacy_index(&bad_new, &legacy).is_err());
        assert!(!bad_new.exists());
        assert_eq!(std::fs::read(&legacy).expect("legacy after"), legacy_bytes);
        let loaded = match VectorStore::load_from_path(&legacy).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(loaded.entry_count(), 1);
    }

    #[test]
    fn migrated_v1_and_v2_indexes_load_normally() {
        for (case, version, folders) in [
            ("migrate-v1", 1, None),
            (
                "migrate-v2",
                2,
                Some(serde_json::json!(["D:\\Docs", "d:/docs\\", "rel\\Docs", ""])),
            ),
        ] {
            let (new_path, legacy_path) = migration_paths(case);
            seed_index(&legacy_path, version, folders);
            migrate_legacy_index(&new_path, &legacy_path).expect("migrate");
            let loaded = match VectorStore::load_from_path(&new_path).expect("load") {
                LoadOutcome::Loaded(s) => s,
                other => panic!("expected Loaded for {case}, got {}", outcome_name(&other)),
            };
            assert_eq!(loaded.entry_count(), 1);
            if version == 1 {
                assert!(loaded.list_folders().is_empty());
            } else {
                // Duplicates collapse and invalid paths are sanitized.
                assert_eq!(loaded.registered_folders(), vec!["D:\\Docs".to_string()]);
            }
            // The legacy source is retained for rollback in both cases.
            assert!(legacy_path.exists());
        }
    }

    #[test]
    fn no_replace_publish_writes_complete_copy() {
        let dir = test_dir("no-replace-ok");
        let dest = dir.join(INDEX_FILE_NAME);
        let outcome = publish_new_file_no_replace(&dest, b"payload").expect("publish");
        assert_eq!(outcome, MigrationOutcome::Migrated);
        assert_eq!(std::fs::read(&dest).expect("dest bytes"), b"payload");
        let strays: Vec<_> = std::fs::read_dir(&dir)
            .expect("read dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().contains(".tmp."))
            .collect();
        assert!(strays.is_empty());
    }

    #[test]
    fn no_replace_publish_never_overwrites_existing_destination() {
        // Destination already present at publication time: this is exactly
        // the state a lost check-to-publish race would leave behind, so it
        // deterministically proves the primitive cannot overwrite.
        let dir = test_dir("no-replace-race");
        let dest = dir.join(INDEX_FILE_NAME);
        std::fs::write(&dest, b"original-bytes").expect("seed dest");
        let outcome = publish_new_file_no_replace(&dest, b"staged-bytes").expect("publish");
        assert_eq!(outcome, MigrationOutcome::SkippedDestinationExists);
        assert_eq!(
            std::fs::read(&dest).expect("dest bytes"),
            b"original-bytes"
        );
        let strays: Vec<_> = std::fs::read_dir(&dir)
            .expect("read dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().contains(".tmp."))
            .collect();
        assert!(strays.is_empty());
    }

    #[test]
    fn search_returns_matches_above_threshold() {
        let mut store = VectorStore::new();
        store.insert("a.txt".to_string(), "kept".to_string(), vec![1.0, 1.0]);
        let results = store.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "a.txt");
        assert!(results[0].score >= MIN_RELEVANCE_SCORE);
    }

    #[test]
    fn search_keeps_matches_at_threshold() {
        let mut store = VectorStore::new();

        // The query's dot product is 1 and the stored vector's norm is 4,
        // so cosine similarity is exactly 0.25.
        store.insert(
            "boundary.txt".to_string(),
            "boundary".to_string(),
            vec![1.0, 3.0, 2.0, 1.0, 1.0],
        );

        let results = store.search(&[1.0, 0.0, 0.0, 0.0, 0.0], 10);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "boundary.txt");
        assert_eq!(results[0].score, MIN_RELEVANCE_SCORE);
    }

    #[test]
    fn search_excludes_below_threshold_and_negative_matches() {
        let mut store = VectorStore::new();
        store.insert("weak.txt".to_string(), "weak".to_string(), vec![0.2, 0.98]);
        store.insert("neg.txt".to_string(), "neg".to_string(), vec![-1.0, 0.0]);
        store.insert("kept.txt".to_string(), "kept".to_string(), vec![1.0, 0.0]);
        let results = store.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "kept.txt");
    }

    #[test]
    fn search_results_remain_sorted_by_score() {
        let mut store = VectorStore::new();
        store.insert("mid.txt".to_string(), "mid".to_string(), vec![1.0, 1.0]);
        store.insert("top.txt".to_string(), "top".to_string(), vec![1.0, 0.0]);
        store.insert("low.txt".to_string(), "low".to_string(), vec![0.3, 0.95]);
        let results = store.search(&[1.0, 0.0], 10);
        let paths: Vec<_> = results.iter().map(|r| r.path.as_str()).collect();
        assert_eq!(paths, vec!["top.txt", "mid.txt", "low.txt"]);
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);
    }

    #[test]
    fn search_filters_before_applying_limit() {
        let mut store = VectorStore::new();
        store.insert("weak1.txt".to_string(), "w".to_string(), vec![0.0, 1.0]);
        store.insert("weak2.txt".to_string(), "w".to_string(), vec![0.0, 1.0]);
        store.insert("a.txt".to_string(), "a".to_string(), vec![1.0, 0.2]);
        store.insert("b.txt".to_string(), "b".to_string(), vec![1.0, 0.5]);
        store.insert("c.txt".to_string(), "c".to_string(), vec![1.0, 1.0]);
        // Three qualifying matches and limit 2: the weak entries must not
        // consume slots, so both slots hold qualifying matches.
        let results = store.search(&[1.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.score >= MIN_RELEVANCE_SCORE));
        assert_eq!(results[0].path, "a.txt");
        assert_eq!(results[1].path, "b.txt");
    }

    #[test]
    fn search_excludes_non_finite_scores_without_panicking() {
        let mut store = VectorStore::new();
        store.entries.push(Entry {
            path: "nan.txt".to_string(),
            content: "nan".to_string(),
            embedding: Embedding::Whole(vec![f32::NAN, 0.0]),
        });
        store.entries.push(Entry {
            path: "inf.txt".to_string(),
            content: "inf".to_string(),
            embedding: Embedding::Whole(vec![f32::INFINITY, 0.0]),
        });
        store.insert("ok.txt".to_string(), "ok".to_string(), vec![1.0, 0.0]);
        let results = store.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "ok.txt");
    }

    #[test]
    fn search_empty_store_or_no_qualifying_results_returns_empty() {
        let empty = VectorStore::new();
        assert!(empty.search(&[1.0, 0.0], 10).is_empty());
        let mut store = VectorStore::new();
        store.insert("weak.txt".to_string(), "weak".to_string(), vec![0.0, 1.0]);
        assert!(store.search(&[1.0, 0.0], 10).is_empty());
    }

    // --- Schema v3 legacy migration and searchability --------------------

    #[test]
    fn legacy_v1_entries_remain_searchable_without_reembedding() {
        let dir = test_dir("v1search");
        let dest = dir.join(INDEX_FILE_NAME);
        seed_index(&dest, 1, None);
        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(loaded.entry_count(), 1);
        assert_eq!(whole_vector(&loaded.entries[0]), vec384(1.0));
        let results = loaded.search(&vec384(1.0), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "D:\\Docs\\a.txt");
    }

    #[test]
    fn legacy_v2_entries_remain_searchable_with_sanitized_registry() {
        let dir = test_dir("v2search");
        let dest = dir.join(INDEX_FILE_NAME);
        let v2 = serde_json::json!({
            "version": 2,
            "model": INDEX_MODEL_ID,
            "dim": INDEX_DIM,
            "entries": [{ "path": "D:\\Docs\\a.txt", "content": "x", "vector": vec384(1.0) }],
            "folders": ["D:\\Docs", "d:/docs\\", "rel\\Docs", "", "E:\\Other"]
        });
        std::fs::write(&dest, serde_json::to_string(&v2).unwrap()).expect("seed v2");
        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(loaded.entry_count(), 1);
        assert_eq!(whole_vector(&loaded.entries[0]), vec384(1.0));
        assert_eq!(loaded.search(&vec384(1.0), 10).len(), 1);
    }

    #[test]
    fn legacy_entries_with_chunk_data_quarantine() {
        let dir = test_dir("v2chunks");
        let dest = dir.join(INDEX_FILE_NAME);
        let value = serde_json::json!({
            "version": 2,
            "model": INDEX_MODEL_ID,
            "dim": INDEX_DIM,
            "entries": [{
                "path": "D:\\Docs\\a.txt",
                "content": "abcdef",
                "vector": vec384(1.0),
                "chunks": [chunk_json(0, 6, vec384(1.0))]
            }],
            "folders": []
        });
        std::fs::write(&dest, serde_json::to_string(&value).unwrap()).expect("seed v2");
        assert_quarantined(&dest, &dir);
    }

    // --- Schema v3 round trips ------------------------------------------

    #[test]
    fn v3_whole_entry_round_trips_with_vector_only() {
        let dir = test_dir("v3whole");
        let dest = dir.join(INDEX_FILE_NAME);
        let mut store = VectorStore::new();
        store
            .reconcile(
                &root("D:\\Docs"),
                vec![staged("D:\\Docs\\a.txt", "hello world", 1.0)],
            )
            .expect("reconcile");
        store.save_to_path(&dest).expect("save");

        let raw = std::fs::read_to_string(&dest).expect("read");
        let value: serde_json::Value = serde_json::from_str(&raw).expect("parse");
        assert_eq!(value["version"], serde_json::Value::from(3));
        assert!(value["entries"][0].get("chunks").is_none());

        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(whole_vector(&loaded.entries[0]), vec384(1.0));
        let results = loaded.search(&vec384(1.0), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].snippet, "hello world");
    }

    #[test]
    fn v3_chunked_entry_round_trips_with_chunks_only() {
        let dir = test_dir("v3chunks");
        let dest = dir.join(INDEX_FILE_NAME);
        let mut store = VectorStore::new();
        store
            .reconcile(
                &root("D:\\Docs"),
                vec![chunked_staged(
                    "D:\\Docs\\a.txt",
                    "abcdef",
                    vec![chunk(0, 3, 1.0), chunk(3, 6, 0.5)],
                )],
            )
            .expect("reconcile");
        assert_eq!(
            chunks_of(&store.entries[0]),
            vec![chunk(0, 3, 1.0), chunk(3, 6, 0.5)]
        );
        store.save_to_path(&dest).expect("save");

        let raw = std::fs::read_to_string(&dest).expect("read");
        let value: serde_json::Value = serde_json::from_str(&raw).expect("parse");
        assert_eq!(value["version"], serde_json::Value::from(3));
        assert!(value["entries"][0].get("vector").is_none());
        assert_eq!(
            value["entries"][0]["chunks"].as_array().map(|a| a.len()),
            Some(2)
        );

        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(
            chunks_of(&loaded.entries[0]),
            vec![chunk(0, 3, 1.0), chunk(3, 6, 0.5)]
        );
        let results = loaded.search(&vec384(1.0), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].snippet, "abc");
    }

    #[test]
    fn v3_empty_entry_round_trips_without_embedding() {
        let dir = test_dir("v3empty");
        let dest = dir.join(INDEX_FILE_NAME);
        let mut store = VectorStore::new();
        store
            .reconcile_and_register(
                &root("D:\\Docs"),
                "D:\\Docs",
                vec![empty_staged("D:\\Docs\\empty.txt", "")],
            )
            .expect("reconcile");
        assert_eq!(store.entry_count(), 1);
        store.save_to_path(&dest).expect("save");

        let raw = std::fs::read_to_string(&dest).expect("read");
        let value: serde_json::Value = serde_json::from_str(&raw).expect("parse");
        assert!(value["entries"][0].get("vector").is_none());
        assert!(value["entries"][0].get("chunks").is_none());

        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(loaded.entry_count(), 1);
        assert!(matches!(loaded.entries[0].embedding, Embedding::Empty));
        assert_eq!(loaded.list_folders()[0].file_count, 1);
        assert!(loaded.search(&vec384(1.0), 10).is_empty());
    }

    // --- Mutually exclusive representation validation -------------------

    #[test]
    fn v3_entry_with_both_representations_quarantines() {
        let dir = test_dir("v3both");
        let dest = dir.join(INDEX_FILE_NAME);
        seed_v3_index(
            &dest,
            serde_json::json!([{
                "path": "D:\\Docs\\a.txt",
                "content": "abcdef",
                "vector": vec384(1.0),
                "chunks": [chunk_json(0, 6, vec384(1.0))]
            }]),
        );
        assert_quarantined(&dest, &dir);
    }

    #[test]
    fn v3_nonempty_entry_without_embedding_quarantines() {
        let dir = test_dir("v3nembed");
        let dest = dir.join(INDEX_FILE_NAME);
        seed_v3_index(
            &dest,
            serde_json::json!([{ "path": "D:\\Docs\\a.txt", "content": "abcdef" }]),
        );
        assert_quarantined(&dest, &dir);
    }

    #[test]
    fn v3_empty_entry_without_embedding_loads_as_empty() {
        let dir = test_dir("v3emptyload");
        let dest = dir.join(INDEX_FILE_NAME);
        seed_v3_index(
            &dest,
            serde_json::json!([{ "path": "D:\\Docs\\a.txt", "content": "" }]),
        );
        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert!(matches!(loaded.entries[0].embedding, Embedding::Empty));
    }

    // --- Malformed chunk ranges and vectors ------------------------------

    #[test]
    fn v3_chunk_start_not_before_end_quarantines() {
        let dir = test_dir("v3startend");
        let dest = dir.join(INDEX_FILE_NAME);
        seed_v3_index(
            &dest,
            serde_json::json!([{
                "path": "D:\\Docs\\a.txt",
                "content": "abcdef",
                "chunks": [chunk_json(0, 6, vec384(1.0)), chunk_json(6, 6, vec384(1.0))]
            }]),
        );
        assert_quarantined(&dest, &dir);
    }

    #[test]
    fn v3_chunk_range_beyond_content_quarantines() {
        let dir = test_dir("v3beyond");
        let dest = dir.join(INDEX_FILE_NAME);
        seed_v3_index(
            &dest,
            serde_json::json!([{
                "path": "D:\\Docs\\a.txt",
                "content": "abcdef",
                "chunks": [chunk_json(0, 9, vec384(1.0))]
            }]),
        );
        assert_quarantined(&dest, &dir);
    }

    #[test]
    fn v3_chunk_gap_quarantines() {
        let dir = test_dir("v3gap");
        let dest = dir.join(INDEX_FILE_NAME);
        seed_v3_index(
            &dest,
            serde_json::json!([{
                "path": "D:\\Docs\\a.txt",
                "content": "abcdef",
                "chunks": [chunk_json(0, 2, vec384(1.0)), chunk_json(4, 6, vec384(1.0))]
            }]),
        );
        assert_quarantined(&dest, &dir);
    }

    #[test]
    fn v3_chunk_nested_or_unordered_quarantines() {
        let dir = test_dir("v3nested");
        let dest = dir.join(INDEX_FILE_NAME);
        seed_v3_index(
            &dest,
            serde_json::json!([{
                "path": "D:\\Docs\\a.txt",
                "content": "abcdef",
                "chunks": [chunk_json(0, 4, vec384(1.0)), chunk_json(1, 3, vec384(1.0))]
            }]),
        );
        assert_quarantined(&dest, &dir);
    }

    #[test]
    fn v3_chunk_missing_start_coverage_quarantines() {
        let dir = test_dir("v3nostart");
        let dest = dir.join(INDEX_FILE_NAME);
        seed_v3_index(
            &dest,
            serde_json::json!([{
                "path": "D:\\Docs\\a.txt",
                "content": "abcdef",
                "chunks": [chunk_json(1, 6, vec384(1.0))]
            }]),
        );
        assert_quarantined(&dest, &dir);
    }

    #[test]
    fn v3_chunk_missing_end_coverage_quarantines() {
        let dir = test_dir("v3noend");
        let dest = dir.join(INDEX_FILE_NAME);
        seed_v3_index(
            &dest,
            serde_json::json!([{
                "path": "D:\\Docs\\a.txt",
                "content": "abcdef",
                "chunks": [chunk_json(0, 5, vec384(1.0))]
            }]),
        );
        assert_quarantined(&dest, &dir);
    }

    #[test]
    fn v3_duplicate_chunk_starts_quarantine() {
        let dir = test_dir("v3dupstart");
        let dest = dir.join(INDEX_FILE_NAME);
        // Ends strictly advance and coverage is complete, but the second
        // chunk restarts at the same offset as the first.
        seed_v3_index(
            &dest,
            serde_json::json!([{
                "path": "D:\\Docs\\a.txt",
                "content": "abcdef",
                "chunks": [chunk_json(0, 5, vec384(1.0)), chunk_json(0, 6, vec384(1.0))]
            }]),
        );
        assert_quarantined(&dest, &dir);
    }

    #[test]
    fn v3_descending_chunk_starts_quarantine() {
        let dir = test_dir("v3descstart");
        let dest = dir.join(INDEX_FILE_NAME);
        // Ends advance (2, 4, 6) and there are no gaps, but the third
        // chunk starts before the second.
        seed_v3_index(
            &dest,
            serde_json::json!([{
                "path": "D:\\Docs\\a.txt",
                "content": "abcdef",
                "chunks": [
                    chunk_json(0, 2, vec384(1.0)),
                    chunk_json(1, 4, vec384(1.0)),
                    chunk_json(0, 6, vec384(1.0))
                ]
            }]),
        );
        assert_quarantined(&dest, &dir);
    }

    #[test]
    fn v3_chunk_non_char_boundary_quarantines() {
        let dir = test_dir("v3boundary");
        let dest = dir.join(INDEX_FILE_NAME);
        // "aéz" is 4 bytes; byte offset 2 falls inside the 2-byte 'é'.
        seed_v3_index(
            &dest,
            serde_json::json!([{
                "path": "D:\\Docs\\a.txt",
                "content": "aéz",
                "chunks": [chunk_json(0, 2, vec384(1.0)), chunk_json(2, 4, vec384(1.0))]
            }]),
        );
        assert_quarantined(&dest, &dir);
    }

    #[test]
    fn v3_chunk_dimension_mismatch_quarantines() {
        let dir = test_dir("v3dim");
        let dest = dir.join(INDEX_FILE_NAME);
        seed_v3_index(
            &dest,
            serde_json::json!([{
                "path": "D:\\Docs\\a.txt",
                "content": "abcdef",
                "chunks": [chunk_json(0, 6, vec![1.0, 0.0])]
            }]),
        );
        assert_quarantined(&dest, &dir);
    }

    #[test]
    fn v3_chunk_non_finite_vector_quarantines() {
        let dir = test_dir("v3nonfinite");
        let dest = dir.join(INDEX_FILE_NAME);
        // A finite JSON number (1e39) overflows f32 on load, yielding
        // infinity that the finite-value check must reject.
        let mut bad: Vec<f64> = vec![1.0; INDEX_DIM];
        bad[0] = 1.0e39;
        seed_v3_index(
            &dest,
            serde_json::json!([{
                "path": "D:\\Docs\\a.txt",
                "content": "abcdef",
                "chunks": [{ "start": 0, "end": 6, "vector": bad }]
            }]),
        );
        assert_quarantined(&dest, &dir);
    }

    // --- Valid chunk layouts load ---------------------------------------

    #[test]
    fn v3_overlapping_chunk_ranges_load() {
        let dir = test_dir("v3overlap");
        let dest = dir.join(INDEX_FILE_NAME);
        seed_v3_index(
            &dest,
            serde_json::json!([{
                "path": "D:\\Docs\\a.txt",
                "content": "abcdef",
                "chunks": [
                    chunk_json(0, 4, vec384(1.0)),
                    chunk_json(2, 6, vec384(0.5))
                ]
            }]),
        );
        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(chunks_of(&loaded.entries[0]).len(), 2);
    }

    #[test]
    fn v3_overlapping_chunks_with_increasing_starts_load() {
        let dir = test_dir("v3overlap3");
        let dest = dir.join(INDEX_FILE_NAME);
        // Three overlapping ranges: starts 0, 2, 4 strictly increase and
        // ends 4, 5, 6 strictly advance, so the layout is valid.
        seed_v3_index(
            &dest,
            serde_json::json!([{
                "path": "D:\\Docs\\a.txt",
                "content": "abcdef",
                "chunks": [
                    chunk_json(0, 4, vec384(1.0)),
                    chunk_json(2, 5, vec384(0.75)),
                    chunk_json(4, 6, vec384(0.5))
                ]
            }]),
        );
        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(chunks_of(&loaded.entries[0]).len(), 3);
    }

    #[test]
    fn v3_multibyte_chunk_boundaries_load() {
        let dir = test_dir("v3multi");
        let dest = dir.join(INDEX_FILE_NAME);
        // "aé" is 3 bytes: 'a' at 0, 'é' at 1..3.
        seed_v3_index(
            &dest,
            serde_json::json!([{
                "path": "D:\\Docs\\a.txt",
                "content": "aé",
                "chunks": [
                    chunk_json(0, 1, vec384(1.0)),
                    chunk_json(1, 3, vec384(0.5))
                ]
            }]),
        );
        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(chunks_of(&loaded.entries[0])[1].end, 3);
    }

    // --- Staged validation is atomic ------------------------------------

    #[test]
    fn reconcile_rejects_malformed_staged_chunks_without_mutation() {
        let mut store = VectorStore::new();
        store.insert("D:\\Docs\\keep.txt".to_string(), "keep".to_string(), vec384(1.0));
        let before = store.serialize().expect("snapshot");
        let result = store.reconcile(
            &root("D:\\Docs"),
            vec![chunked_staged(
                "D:\\Docs\\bad.txt",
                "abcdef",
                vec![chunk(0, 2, 1.0), chunk(4, 6, 1.0)],
            )],
        );
        assert!(result.is_err());
        assert_eq!(store.serialize().expect("after"), before);
    }

    #[test]
    fn reconcile_rejects_staged_nonfinite_chunk_vector_without_mutation() {
        let mut store = VectorStore::new();
        store.insert("D:\\Docs\\keep.txt".to_string(), "keep".to_string(), vec384(1.0));
        let before = store.serialize().expect("snapshot");
        let mut bad = vec384(1.0);
        bad[0] = f32::NAN;
        let result = store.reconcile(
            &root("D:\\Docs"),
            vec![chunked_staged(
                "D:\\Docs\\bad.txt",
                "abcdef",
                vec![chunk_with(0, 6, bad)],
            )],
        );
        assert!(result.is_err());
        assert_eq!(store.serialize().expect("after"), before);
    }

    #[test]
    fn reconcile_rejects_staged_wrong_dimension_whole_vector_without_mutation() {
        let mut store = VectorStore::new();
        store.insert("D:\\Docs\\keep.txt".to_string(), "keep".to_string(), vec384(1.0));
        let before = store.serialize().expect("snapshot");
        let mut bad = staged("D:\\Docs\\bad.txt", "abcdef", 1.0);
        bad.vector = vec![1.0; INDEX_DIM - 1];
        let result = store.reconcile(&root("D:\\Docs"), vec![bad]);
        assert!(result.is_err());
        assert_eq!(store.serialize().expect("after"), before);
    }

    #[test]
    fn reconcile_rejects_staged_nonfinite_whole_vector_without_mutation() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut store = VectorStore::new();
            store.insert("D:\\Docs\\keep.txt".to_string(), "keep".to_string(), vec384(1.0));
            let before = store.serialize().expect("snapshot");
            let mut bad = staged("D:\\Docs\\bad.txt", "abcdef", 1.0);
            bad.vector[0] = value;
            let result = store.reconcile(&root("D:\\Docs"), vec![bad]);
            assert!(result.is_err(), "expected rejection for {value}");
            assert_eq!(store.serialize().expect("after"), before);
        }
    }

    #[test]
    fn failed_staged_whole_vector_reconciliation_preserves_valid_entries() {
        let mut store = VectorStore::new();
        store
            .reconcile(
                &root("D:\\Docs"),
                vec![chunked_staged(
                    "D:\\Docs\\chunked.txt",
                    "abcdef",
                    vec![chunk(0, 6, 1.0)],
                )],
            )
            .expect("seed chunked entry");
        store.insert(
            "D:\\Docs\\whole.txt".to_string(),
            "whole".to_string(),
            vec384(1.0),
        );
        let before = store.serialize().expect("snapshot");
        let hits_before = store.search(&vec384(1.0), 10);

        let mut bad = staged("D:\\Docs\\bad.txt", "text", 1.0);
        bad.vector[0] = f32::NAN;
        let result = store.reconcile(&root("D:\\Docs"), vec![bad]);
        assert!(result.is_err());

        assert_eq!(store.entry_count(), 2);
        assert_eq!(store.serialize().expect("after"), before);
        let hits_after = store.search(&vec384(1.0), 10);
        assert_eq!(hits_after.len(), hits_before.len());
        for (after, before) in hits_after.iter().zip(&hits_before) {
            assert_eq!(after.path, before.path);
            assert_eq!(after.score, before.score);
        }
    }

    #[test]
    fn reconcile_rejects_staged_document_with_both_representations() {
        let mut store = VectorStore::new();
        store.insert("D:\\Docs\\keep.txt".to_string(), "keep".to_string(), vec384(1.0));
        let before = store.serialize().expect("snapshot");
        let mut bad = staged("D:\\Docs\\bad.txt", "abcdef", 1.0);
        bad.chunks = vec![chunk(0, 6, 1.0)];
        let result = store.reconcile(&root("D:\\Docs"), vec![bad]);
        assert!(result.is_err());
        assert_eq!(store.serialize().expect("after"), before);
    }

    #[test]
    fn reconcile_rejects_nonempty_document_without_embedding() {
        let mut store = VectorStore::new();
        store.insert("D:\\Docs\\keep.txt".to_string(), "keep".to_string(), vec384(1.0));
        let before = store.serialize().expect("snapshot");
        let result = store.reconcile(
            &root("D:\\Docs"),
            vec![empty_staged("D:\\Docs\\bad.txt", "text")],
        );
        assert!(result.is_err());
        assert_eq!(store.serialize().expect("after"), before);
    }

    #[test]
    fn reconcile_accepts_empty_document_as_unsearchable() {
        let mut store = VectorStore::new();
        store
            .reconcile(
                &root("D:\\Docs"),
                vec![
                    empty_staged("D:\\Docs\\empty.txt", ""),
                    staged("D:\\Docs\\a.txt", "hello", 1.0),
                ],
            )
            .expect("reconcile");
        assert_eq!(store.entry_count(), 2);
        let results = store.search(&vec384(1.0), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "D:\\Docs\\a.txt");
    }

    // --- Whole/chunk reconciliation --------------------------------------

    #[test]
    fn reconcile_replaces_whole_vector_with_chunks() {
        let mut store = VectorStore::new();
        store.insert("D:\\Docs\\a.txt".to_string(), "abcdef".to_string(), vec384(1.0));
        let n = store
            .reconcile(
                &root("D:\\Docs"),
                vec![chunked_staged(
                    "D:\\Docs\\a.txt",
                    "abcdef",
                    vec![chunk(0, 3, 2.0), chunk(3, 6, 3.0)],
                )],
            )
            .expect("reconcile");
        assert_eq!(n, 1);
        assert_eq!(store.entry_count(), 1);
        assert_eq!(chunks_of(&store.entries[0]).len(), 2);
        assert_eq!(chunks_of(&store.entries[0])[0].vector, vec384(2.0));
    }

    #[test]
    fn reconcile_replaces_chunks_with_whole_vector() {
        let mut store = VectorStore::new();
        store
            .reconcile(
                &root("D:\\Docs"),
                vec![chunked_staged(
                    "D:\\Docs\\a.txt",
                    "abcdef",
                    vec![chunk(0, 3, 2.0), chunk(3, 6, 3.0)],
                )],
            )
            .expect("seed chunks");
        let n = store
            .reconcile(
                &root("D:\\Docs"),
                vec![staged("D:\\Docs\\a.txt", "abcdef", 5.0)],
            )
            .expect("reconcile");
        assert_eq!(n, 1);
        assert_eq!(store.entry_count(), 1);
        assert_eq!(whole_vector(&store.entries[0]), vec384(5.0));
    }

    // --- Folder counts, isolation, and removal --------------------------

    #[test]
    fn folder_registration_counts_empty_files_but_isolates_search() {
        let mut store = VectorStore::new();
        store
            .reconcile_and_register(
                &root("D:\\Docs"),
                "D:\\Docs",
                vec![
                    empty_staged("D:\\Docs\\empty.txt", ""),
                    staged("D:\\Docs\\a.txt", "hello", 1.0),
                ],
            )
            .expect("docs");
        assert_eq!(store.list_folders()[0].file_count, 2);
        store
            .reconcile_and_register(
                &root("E:\\Other"),
                "E:\\Other",
                vec![staged("E:\\Other\\b.txt", "keep", 2.0)],
            )
            .expect("other");
        assert_eq!(store.list_folders().len(), 2);
        let results = store.search(&vec384(1.0), 10);
        assert_eq!(results.len(), 2);
        let paths: HashSet<&str> = results.iter().map(|r| r.path.as_str()).collect();
        assert!(paths.contains("D:\\Docs\\a.txt"));
        assert!(paths.contains("E:\\Other\\b.txt"));
    }

    #[test]
    fn remove_folder_drops_chunked_entries_and_keeps_other_roots() {
        let mut store = VectorStore::new();
        store
            .reconcile_and_register(
                &root("D:\\Docs"),
                "D:\\Docs",
                vec![chunked_staged(
                    "D:\\Docs\\a.txt",
                    "abcdef",
                    vec![chunk(0, 6, 1.0)],
                )],
            )
            .expect("docs");
        store
            .reconcile_and_register(
                &root("E:\\Other"),
                "E:\\Other",
                vec![staged("E:\\Other\\b.txt", "keep", 2.0)],
            )
            .expect("other");
        let removed = store.remove_folder(&root("D:\\Docs")).expect("remove");
        assert_eq!(removed, 1);
        assert_eq!(store.entry_count(), 1);
        assert_eq!(store.entries[0].path, "E:\\Other\\b.txt");
        assert_eq!(store.registered_folders(), vec!["E:\\Other".to_string()]);
    }

    // --- Chunk search ranking and snippets ------------------------------

    #[test]
    fn chunk_search_returns_one_result_per_file_using_best_chunk() {
        let mut store = VectorStore::new();
        store.entries.push(chunked_entry(
            "D:\\Docs\\a.txt",
            "abcdef",
            vec![
                chunk_with(0, 3, vec![1.0, 0.0]),
                chunk_with(3, 6, vec![0.0, 1.0]),
            ],
        ));
        store.entries.push(chunked_entry(
            "D:\\Docs\\b.txt",
            "ghijkl",
            vec![chunk_with(0, 6, vec![0.0, 1.0])],
        ));
        let results = store.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "D:\\Docs\\a.txt");
        assert_eq!(results[0].score, 1.0);
        assert_eq!(results[0].snippet, "abc");
    }

    #[test]
    fn chunk_search_max_chunk_wins_over_file_head() {
        let mut store = VectorStore::new();
        store.entries.push(chunked_entry(
            "D:\\Docs\\a.txt",
            "zzz match",
            vec![
                chunk_with(0, 3, vec![0.0, 1.0]),
                chunk_with(3, 9, vec![1.0, 0.0]),
            ],
        ));
        let results = store.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].score, 1.0);
        assert_eq!(results[0].snippet, " match");
    }

    #[test]
    fn chunk_search_tie_uses_earliest_chunk() {
        let mut store = VectorStore::new();
        store.entries.push(chunked_entry(
            "D:\\Docs\\a.txt",
            "abcdef",
            vec![
                chunk_with(0, 3, vec![1.0, 0.0]),
                chunk_with(3, 6, vec![1.0, 0.0]),
            ],
        ));
        let results = store.search(&[1.0, 0.0], 10);
        assert_eq!(results[0].snippet, "abc");
    }

    #[test]
    fn chunk_search_keeps_score_at_threshold() {
        let mut store = VectorStore::new();
        store.entries.push(chunked_entry(
            "D:\\Docs\\boundary.txt",
            "abcde",
            vec![chunk_with(0, 5, vec![1.0, 3.0, 2.0, 1.0, 1.0])],
        ));
        let results = store.search(&[1.0, 0.0, 0.0, 0.0, 0.0], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].score, MIN_RELEVANCE_SCORE);
        assert_eq!(results[0].snippet, "abcde");
    }

    #[test]
    fn chunk_search_drops_scores_below_threshold() {
        let mut store = VectorStore::new();
        store.entries.push(chunked_entry(
            "D:\\Docs\\a.txt",
            "abcde",
            vec![chunk_with(0, 5, vec![0.2, 0.98])],
        ));
        assert!(store.search(&[1.0, 0.0], 10).is_empty());
    }

    #[test]
    fn chunk_search_snippet_is_bounded_to_winning_chunk() {
        let content = format!("{}match{}", "x".repeat(10), "y".repeat(10));
        let mut store = VectorStore::new();
        store.entries.push(chunked_entry(
            "D:\\Docs\\a.txt",
            &content,
            vec![
                chunk_with(0, 10, vec![0.0, 1.0]),
                chunk_with(10, 15, vec![1.0, 0.0]),
            ],
        ));
        let results = store.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].snippet, "match");
    }

    #[test]
    fn chunk_search_caps_snippet_at_snippet_chars() {
        let content = "x".repeat(250);
        let mut store = VectorStore::new();
        store.entries.push(chunked_entry(
            "D:\\Docs\\a.txt",
            &content,
            vec![chunk_with(0, 250, vec![1.0, 0.0])],
        ));
        let results = store.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].snippet.chars().count(), SNIPPET_CHARS);
    }

    #[test]
    fn chunk_search_snippet_handles_multibyte_range() {
        let content = "héllo wörld";
        let mut store = VectorStore::new();
        store.entries.push(chunked_entry(
            "D:\\Docs\\a.txt",
            content,
            vec![chunk_with(0, content.len(), vec![1.0, 0.0])],
        ));
        let results = store.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].snippet, content);
    }

    // --- Deterministic ordering ------------------------------------------

    #[test]
    fn search_ties_break_by_normalized_path() {
        let mut store = VectorStore::new();
        store.insert("D:\\Docs\\b.txt".to_string(), "b".to_string(), vec![1.0, 0.0]);
        store.insert("d:/docs/A.txt".to_string(), "a".to_string(), vec![1.0, 0.0]);
        let results = store.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].path, "d:/docs/A.txt");
        assert_eq!(results[1].path, "D:\\Docs\\b.txt");
    }

    #[test]
    fn search_ties_with_ambiguous_paths_fall_back_to_raw_path() {
        let mut store = VectorStore::new();
        store.insert("b.txt".to_string(), "b".to_string(), vec![1.0, 0.0]);
        store.insert("a.txt".to_string(), "a".to_string(), vec![1.0, 0.0]);
        let results = store.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].path, "a.txt");
        assert_eq!(results[1].path, "b.txt");
    }

    // --- Whole-file compatibility ---------------------------------------

    #[test]
    fn whole_file_search_keeps_head_snippet_and_scoring() {
        let mut store = VectorStore::new();
        store.insert(
            "D:\\Docs\\a.txt".to_string(),
            "head text".to_string(),
            vec384(1.0),
        );
        let results = store.search(&vec384(1.0), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].snippet, "head text");
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn whole_and_chunk_entries_rank_together() {
        let mut store = VectorStore::new();
        store.insert(
            "D:\\Docs\\whole.txt".to_string(),
            "whole".to_string(),
            vec![1.0, 0.0],
        );
        store.entries.push(chunked_entry(
            "D:\\Docs\\chunk.txt",
            "abcdef",
            vec![chunk_with(0, 6, vec![1.0, 1.0])],
        ));
        let results = store.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].path, "D:\\Docs\\whole.txt");
        assert_eq!(results[0].score, 1.0);
        assert_eq!(results[1].path, "D:\\Docs\\chunk.txt");
        assert!(results[1].score > MIN_RELEVANCE_SCORE);
        assert!(results[1].score < results[0].score);
    }

    // --- Persistence failure --------------------------------------------

    #[test]
    fn failed_chunked_persistence_preserves_prior_disk_index() {
        let dir = test_dir("chunkfail");
        let dest = dir.join(INDEX_FILE_NAME);
        let mut original = VectorStore::new();
        original.insert("a.txt".to_string(), "original".to_string(), vec384(1.0));
        original.save_to_path(&dest).expect("seed valid index");

        let mut other = VectorStore::new();
        other
            .reconcile(
                &root("D:\\Docs"),
                vec![chunked_staged(
                    "D:\\Docs\\x.txt",
                    "abcdef",
                    vec![chunk(0, 6, 2.0)],
                )],
            )
            .expect("reconcile");
        let result = other.save_to_path_with_rename(&dest, |_, _| {
            Err(std::io::Error::other("injected rename failure"))
        });
        assert!(result.is_err());

        let loaded = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        assert_eq!(loaded.entry_count(), 1);
        assert_eq!(loaded.entries[0].path, "a.txt");
        assert_eq!(whole_vector(&loaded.entries[0]), vec384(1.0));
        assert!(corrupt_files_in(&dir).is_empty());
    }

    fn outcome_name(outcome: &LoadOutcome) -> &'static str {
        match outcome {
            LoadOutcome::Empty => "Empty",
            LoadOutcome::Loaded(_) => "Loaded",
            LoadOutcome::Quarantined { .. } => "Quarantined",
        }
    }
}
