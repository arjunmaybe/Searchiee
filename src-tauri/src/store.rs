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
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub const INDEX_SCHEMA_VERSION: u32 = 1;
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
}

#[derive(Serialize)]
pub struct SearchResult {
    pub path: String,
    pub snippet: String,
    pub score: f32,
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
        VectorStore { entries: Vec::new() }
    }

    pub fn insert(&mut self, path: String, content: String, vector: Vec<f32>) {
        if let Some(existing) = self.entries.iter_mut().find(|e| e.path == path) {
            existing.content = content;
            existing.vector = vector;
        } else {
            self.entries.push(Entry { path, content, vector });
        }
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
        };
        serde_json::to_string(&saved).map_err(|e| format!("could not serialize index: {e}"))
    }

    /// Validate a parsed envelope and convert it. Rejects the whole
    /// index on any mismatch; never returns a partially mixed store.
    fn from_saved(saved: SavedIndex) -> Result<Self, String> {
        if saved.version != INDEX_SCHEMA_VERSION {
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
        store.insert("a.txt".to_string(), "hello".to_string(), vec![1.0, 0.0]);
        store.insert("a.txt".to_string(), "hello".to_string(), vec![1.0, 0.0]);

        assert_eq!(store.entries.len(), 1);
        assert_eq!(store.search(&[1.0, 0.0], 10).len(), 1);
    }

    #[test]
    fn reinserting_same_path_updates_content_and_embedding() {
        let mut store = VectorStore::new();
        store.insert("a.txt".to_string(), "old".to_string(), vec![1.0, 0.0]);
        store.insert("a.txt".to_string(), "new".to_string(), vec![0.0, 1.0]);

        assert_eq!(store.entries.len(), 1);
        assert_eq!(store.entries[0].content, "new");
        assert_eq!(store.entries[0].vector, vec![0.0, 1.0]);
    }

    #[test]
    fn different_paths_remain_distinct_entries() {
        let mut store = VectorStore::new();
        store.insert("a.txt".to_string(), "same text".to_string(), vec![1.0, 0.0]);
        store.insert("b.txt".to_string(), "same text".to_string(), vec![1.0, 0.0]);

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
        store.insert("a.txt".to_string(), "hello".to_string(), vec384(1.0));
        store.save_to_path(&dest).expect("save");

        let mut restarted = match VectorStore::load_from_path(&dest).expect("load") {
            LoadOutcome::Loaded(s) => s,
            other => panic!("expected Loaded, got {}", outcome_name(&other)),
        };
        restarted.insert("a.txt".to_string(), "hello updated".to_string(), vec384(5.0));
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

    fn outcome_name(outcome: &LoadOutcome) -> &'static str {
        match outcome {
            LoadOutcome::Empty => "Empty",
            LoadOutcome::Loaded(_) => "Loaded",
            LoadOutcome::Quarantined { .. } => "Quarantined",
        }
    }
}
