// Entry point. Wires the file watcher, embedding pipeline, and vector
// store together, and exposes a couple of commands to the frontend.

mod chunk;
mod embed;
mod store;
mod watcher;

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use store::{LoadOutcome, VectorStore};
use tauri::{Manager, State};

struct AppState {
    store: Mutex<VectorStore>,
    save_lock: Mutex<()>,
    /// Serializes whole indexing operations (scan+embed+reconcile+save)
    /// so an older scan can never commit after a newer one.
    index_op_lock: tokio::sync::Mutex<()>,
    /// `None` means persistence is disabled for this session: search and
    /// indexing still work in memory, but nothing is written to disk so a
    /// valid-but-unreadable or unquarantinable file can never be clobbered.
    index_path: Option<PathBuf>,
    startup_notice: Mutex<Option<String>>,
}

#[tauri::command]
async fn index_folder(path: String, state: State<'_, AppState>) -> Result<usize, String> {
    run_index_pipeline(&path, false, &state).await
}

#[tauri::command]
async fn refresh_folder(path: String, state: State<'_, AppState>) -> Result<usize, String> {
    run_index_pipeline(&path, true, &state).await
}

#[tauri::command]
fn list_folders(state: State<'_, AppState>) -> Result<Vec<store::FolderInfo>, String> {
    let store = state
        .store
        .lock()
        .map_err(|e| format!("search index is unavailable: {e}"))?;
    Ok(store.list_folders())
}

#[tauri::command]
async fn remove_folder(path: String, state: State<'_, AppState>) -> Result<usize, String> {
    let _op_guard = state.index_op_lock.lock().await;
    let root = std::path::PathBuf::from(path.trim());
    // Removal needs no filesystem access: the folder may already be
    // offline. Identity is purely normalized, so any spelling works.
    let persist_path: PathBuf = match state.index_path.clone() {
        Some(p) => p,
        None => {
            let removed = state
                .store
                .lock()
                .map_err(|e| format!("search index is unavailable: {e}"))?
                .remove_folder(&root)?;
            return Err(format!(
                "Removed {removed} indexed files in memory, but the persistent index is unavailable this session, so the change cannot be saved and may reappear after restart."
            ));
        }
    };
    let _save_guard = state
        .save_lock
        .lock()
        .map_err(|e| format!("could not save search index: index is unavailable: {e}"))?;
    let (removed, snapshot) = {
        let mut store = state
            .store
            .lock()
            .map_err(|e| format!("could not save search index: index is unavailable: {e}"))?;
        let removed = store.remove_folder(&root)?;
        let snapshot = store.serialize()?;
        (removed, snapshot)
    };
    match store::write_serialized_atomic(&persist_path, &snapshot) {
        Ok(()) => Ok(removed),
        Err(e) => Err(format!(
            "Removed {removed} indexed files in memory, but saving the persistent index failed: {e}. They may reappear after restart."
        )),
    }
}

/// Shared indexing pipeline: fail-closed scan, staged embeddings, then
/// atomic reconcile-plus-register and atomic save. When
/// `require_registered` is set, unregistered roots are rejected after
/// the scan but before any embedding work mutates nothing.
async fn run_index_pipeline(
    input: &str,
    require_registered: bool,
    state: &AppState,
) -> Result<usize, String> {
    // Held across validation, scan, embedding, reconciliation, and save.
    // Nothing here is captured before this lock; overlapping calls queue
    // and the later scan always commits last.
    let _op_guard = state.index_op_lock.lock().await;

    // Stage first without touching the live store.
    let scan = watcher::scan_folder(input).map_err(|e| e.to_string())?;
    if require_registered {
        let registered = state
            .store
            .lock()
            .map_err(|e| format!("search index is unavailable: {e}"))?
            .is_registered(&scan.root);
        if !registered {
            return Err(format!(
                "Folder is not registered: {}. Add it again to index it.",
                scan.root.display()
            ));
        }
    }
    let display = watcher::storage_path(&scan.root);
    if store::path_key(&display).is_none() {
        return Err(format!(
            "Folder has ambiguous identity ({}); refusing to index",
            scan.root.display()
        ));
    }
    // Plan chunks for every document (pure; no model calls).
    let docs: Vec<(String, String)> = scan
        .docs
        .into_iter()
        .map(|doc| (doc.path, doc.content))
        .collect();
    let ranges = plan_chunk_ranges(&docs);
    let owners = chunk_owners(&ranges);

    // Embed every chunk in bounded batches. A single failing batch aborts
    // the whole pass here, before `staged` exists and before the store is
    // touched, so no folder — related or unrelated — is modified.
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(owners.len());
    for batch in owners.chunks(embed::EMBED_BATCH_SIZE) {
        let texts: Vec<String> = batch
            .iter()
            .map(|&(doc_index, range_index)| {
                chunk::chunk_text(&docs[doc_index].1, &ranges[doc_index][range_index]).to_string()
            })
            .collect();
        let mut embedded = embed::embed_texts(texts).await.map_err(|e| e.to_string())?;
        if embedded.len() != batch.len() {
            return Err(format!(
                "embedding model returned {} vectors for {} chunk texts",
                embedded.len(),
                batch.len()
            ));
        }
        vectors.append(&mut embedded);
    }

    let staged = assemble_staged(docs, ranges, vectors)?;
    let staged_len = staged.len();

    // Serialize saves through `save_lock` so concurrent index passes
    // commit in order; each snapshot is taken after acquiring the save
    // lock and therefore includes everything committed before it. The
    // store lock is held only for the in-memory snapshot, never for
    // disk I/O.
    let persist_path: PathBuf = match state.index_path.clone() {
        Some(p) => p,
        None => {
            state
                .store
                .lock()
                .map_err(|e| format!("search index is unavailable: {e}"))?
                .reconcile_and_register(&scan.root, &display, staged)?;
            return Err(format!(
                "Indexed {staged_len} files in memory, but the persistent index is unavailable this session, so changes cannot be saved and may be lost after restart. Search works for now; restart the app and re-index to retry."
            ));
        }
    };
    let _save_guard = state
        .save_lock
        .lock()
        .map_err(|e| format!("could not save search index: index is unavailable: {e}"))?;
    let snapshot = {
        let mut store = state
            .store
            .lock()
            .map_err(|e| format!("could not save search index: index is unavailable: {e}"))?;
        store.reconcile_and_register(&scan.root, &display, staged)?;
        store.serialize()?
    };
    match store::write_serialized_atomic(&persist_path, &snapshot) {
        Ok(()) => Ok(staged_len),
        Err(e) => Err(format!(
            "Indexed {staged_len} files in memory, but saving the persistent index failed: {e}. Results will work until the app restarts."
        )),
    }
}

/// One ordered list of chunk byte-ranges per document. Pure and
/// deterministic; empty content yields an empty list.
fn plan_chunk_ranges(docs: &[(String, String)]) -> Vec<Vec<chunk::Chunk>> {
    docs.iter()
        .map(|(_, content)| chunk::chunk_content(content))
        .collect()
}

/// Flattened `(document index, range index)` owners, in document order,
/// describing which chunk each embedding position belongs to.
fn chunk_owners(ranges: &[Vec<chunk::Chunk>]) -> Vec<(usize, usize)> {
    let mut owners = Vec::new();
    for (doc_index, doc_ranges) in ranges.iter().enumerate() {
        for range_index in 0..doc_ranges.len() {
            owners.push((doc_index, range_index));
        }
    }
    owners
}

/// Pair each chunk's byte range with its embedding, preserving the exact
/// UTF-8 offsets produced by the chunker. A zero-length document has no
/// ranges and no embeddings and stages as an empty, counted-but-not-
/// searchable entry. A vector-count mismatch is rejected rather than
/// misaligning offsets.
fn assemble_staged(
    docs: Vec<(String, String)>,
    ranges: Vec<Vec<chunk::Chunk>>,
    vectors: Vec<Vec<f32>>,
) -> Result<Vec<store::StagedDoc>, String> {
    let expected: usize = ranges.iter().map(|r| r.len()).sum();
    if vectors.len() != expected {
        return Err(format!(
            "internal error: {} chunk embeddings for {} chunks",
            vectors.len(),
            expected
        ));
    }
    let mut vectors = vectors.into_iter();
    let mut staged = Vec::with_capacity(docs.len());
    for ((path, content), doc_ranges) in docs.into_iter().zip(ranges) {
        let mut chunks = Vec::with_capacity(doc_ranges.len());
        for range in doc_ranges {
            let vector = vectors
                .next()
                .ok_or_else(|| format!("internal error: missing chunk embedding for {path}"))?;
            chunks.push(store::StoredChunk {
                start: range.start,
                end: range.end,
                vector,
            });
        }
        staged.push(store::StagedDoc {
            path,
            content,
            vector: Vec::new(),
            chunks,
        });
    }
    Ok(staged)
}

#[tauri::command]
async fn search(query: String, state: State<'_, AppState>) -> Result<Vec<store::SearchResult>, String> {
    let query_vector = embed::embed_text(&query).await.map_err(|e| e.to_string())?;
    let results = state
        .store
        .lock()
        .map_err(|e| format!("search index is unavailable: {e}"))?
        .search(&query_vector, 10);
    Ok(results)
}

#[tauri::command]
fn get_startup_notice(state: State<'_, AppState>) -> Option<String> {
    state.startup_notice.lock().ok()?.take()
}

fn index_file_path(app: &tauri::App) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_local_data_dir()
        .map_err(|e| format!("could not resolve local app data directory: {e}"))?;
    Ok(dir.join(store::INDEX_FILE_NAME))
}

/// Identifier used before the Searchiee rebrand. Kept only to locate a
/// legacy index for one-time migration; the legacy file is never
/// modified or deleted.
const LEGACY_INDEX_IDENTIFIER: &str = "com.example.deepsearch";
const CURRENT_INDEX_IDENTIFIER: &str = "io.github.arjunmaybe.searchiee";

/// Derive the legacy index path from the current one by swapping only
/// the final directory component when it exactly matches the current
/// identifier. Returns `None` rather than guessing otherwise.
fn legacy_index_path(new_index: &Path) -> Option<PathBuf> {
    use std::ffi::OsStr;
    let dir = new_index.parent()?;
    if dir.file_name() != Some(OsStr::new(CURRENT_INDEX_IDENTIFIER)) {
        return None;
    }
    let parent = dir.parent()?;
    Some(parent.join(LEGACY_INDEX_IDENTIFIER).join(store::INDEX_FILE_NAME))
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let (store, notice, index_path): (VectorStore, Option<String>, Option<PathBuf>) =
                match index_file_path(app) {
                    Err(e) => (
                        VectorStore::new(),
                        Some(format!(
                            "{e}; persistence is unavailable this session. Search and indexing still work in memory, but results will be lost after restart. No files were changed."
                        )),
                        None,
                    ),
                    Ok(index_path) => 'setup_index: {
                        let migration_notice = match legacy_index_path(&index_path) {
                            Some(legacy) => {
                                match store::migrate_legacy_index(&index_path, &legacy) {
                                    Ok(store::MigrationOutcome::Migrated) => Some(format!(
                                        "Migrated the saved index from {}. The original was kept as a backup.",
                                        legacy.display()
                                    )),
                                    Ok(_) => None,
                                    Err(e) => {
                                        // Persistence stays disabled so a later
                                        // indexing pass cannot create the new
                                        // file and permanently skip the legacy
                                        // index; restart to retry migration.
                                        break 'setup_index (
                                            VectorStore::new(),
                                            Some(format!(
                                                "Could not migrate the previous search index from {} ({e}); persistence is disabled this session and the old file was left untouched. Fix the problem and restart to retry migration.",
                                                legacy.display()
                                            )),
                                            None,
                                        );
                                    }
                                }
                            }
                            None => None,
                        };
                        let loaded = match VectorStore::load_from_path(&index_path) {
                            Ok(LoadOutcome::Empty) => (VectorStore::new(), None, Some(index_path)),
                            Ok(LoadOutcome::Loaded(store)) => (store, None, Some(index_path)),
                            Ok(LoadOutcome::Quarantined {
                                store,
                                notice,
                                quarantine_succeeded: true,
                            }) => (store, Some(notice), Some(index_path)),
                            Ok(LoadOutcome::Quarantined {
                                store,
                                notice,
                                quarantine_succeeded: false,
                            }) => (
                                store,
                                Some(format!(
                                    "{notice} Persistence remains disabled this session; indexing will work in memory only."
                                )),
                                None,
                            ),
                            Err(e) => {
                                // A successful migration creates the new file
                                // first, so "no files were changed" would be
                                // false in that case; only claim it when no
                                // migration occurred.
                                let suffix = match &migration_notice {
                                    Some(_) => "A migrated copy was created; the legacy original was left untouched.",
                                    None => "No files were changed.",
                                };
                                (
                                    VectorStore::new(),
                                    Some(format!(
                                        "Saved search index could not be read ({e}); persistence is disabled this session to protect the existing file. Search and indexing still work in memory. {suffix}"
                                    )),
                                    None,
                                )
                            }
                        };
                        match (loaded, migration_notice) {
                            ((store, None, path), notice) => (store, notice, path),
                            ((store, Some(first), path), Some(second)) => {
                                (store, Some(format!("{first} {second}")), path)
                            }
                            ((store, Some(only), path), None) => (store, Some(only), path),
                        }
                    }
                };
            app.manage(AppState {
                store: Mutex::new(store),
                save_lock: Mutex::new(()),
                index_op_lock: tokio::sync::Mutex::new(()),
                index_path,
                startup_notice: Mutex::new(notice),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            index_folder,
            refresh_folder,
            remove_folder,
            list_folders,
            search,
            get_startup_notice
        ])
        .run(tauri::generate_context!())
        .expect("error while running Searchiee");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_legacy_path_for_expected_identifier() {
        let new = PathBuf::from(
            "D:\\Users\\x\\AppData\\Local\\io.github.arjunmaybe.searchiee\\index-v1.json",
        );
        let expected = PathBuf::from(
            "D:\\Users\\x\\AppData\\Local\\com.example.deepsearch\\index-v1.json",
        );
        assert_eq!(legacy_index_path(&new), Some(expected));
    }

    #[test]
    fn rejects_non_matching_final_component() {
        let other = PathBuf::from("D:\\Data\\something-else\\index-v1.json");
        assert_eq!(legacy_index_path(&other), None);
    }

    #[test]
    fn rejects_paths_without_parent_structure() {
        assert_eq!(legacy_index_path(Path::new("index-v1.json")), None);
        assert_eq!(legacy_index_path(Path::new("D:\\index-v1.json")), None);
    }

    /// Deterministic stand-in for a real embedding so unit tests never
    /// load or call the model.
    fn fake_vector() -> Vec<f32> {
        vec![1.0; store::INDEX_DIM]
    }

    /// Distinct, finite `INDEX_DIM` marker for one `(document, chunk)`
    /// owner, so a test can trace each assembled chunk back to the exact
    /// owner whose text it was embedded from.
    fn marker_vector(doc_index: usize, chunk_index: usize) -> Vec<f32> {
        let mut vector = vec![0.0; store::INDEX_DIM];
        vector[0] = 1.0;
        vector[1] = (doc_index + 1) as f32;
        vector[2] = (chunk_index + 1) as f32;
        vector
    }

    #[test]
    fn empty_content_stages_no_chunks() {
        let docs = vec![(r"D:\Docs\empty.txt".to_string(), String::new())];
        let ranges = plan_chunk_ranges(&docs);
        assert!(ranges[0].is_empty());
        let staged = assemble_staged(docs, ranges, Vec::new()).expect("assemble");
        assert_eq!(staged.len(), 1);
        assert!(staged[0].vector.is_empty());
        assert!(staged[0].chunks.is_empty());
    }

    #[test]
    fn chunk_offsets_match_content_and_chunker() {
        let content = format!("{}\n\n{}", "A".repeat(500), "B".repeat(1500));
        let docs = vec![(r"D:\Docs\a.txt".to_string(), content.clone())];
        let ranges = plan_chunk_ranges(&docs);
        let owners = chunk_owners(&ranges);
        let vectors = vec![fake_vector(); owners.len()];
        let staged = assemble_staged(docs, ranges.clone(), vectors).expect("assemble");

        let doc = &staged[0];
        assert!(doc.vector.is_empty());
        assert!(doc.chunks.len() > 1);
        assert_eq!(doc.chunks.len(), ranges[0].len());
        for (chunk, range) in doc.chunks.iter().zip(&ranges[0]) {
            assert_eq!(chunk.start, range.start);
            assert_eq!(chunk.end, range.end);
            assert!(content.is_char_boundary(chunk.start));
            assert!(content.is_char_boundary(chunk.end));
            assert_eq!(
                &content[chunk.start..chunk.end],
                chunk::chunk_text(&content, range)
            );
        }
        assert_eq!(doc.chunks[0].start, 0);
        assert_eq!(doc.chunks.last().unwrap().end, content.len());
    }

    #[test]
    fn incorrect_embedding_counts_are_rejected() {
        let docs = vec![(r"D:\Docs\a.txt".to_string(), "hello world".to_string())];
        let ranges = plan_chunk_ranges(&docs);
        assert_eq!(ranges[0].len(), 1);
        let err = assemble_staged(docs, ranges, Vec::new())
            .err()
            .expect("expected a count mismatch error");
        assert!(err.contains("chunk embeddings"));
    }

    #[test]
    fn chunk_ownership_is_complete_across_batches() {
        // 40 single-chunk documents -> 40 owners, split into two batches at 32.
        let docs: Vec<(String, String)> = (0..40)
            .map(|i| (format!(r"D:\Docs\{i}.txt"), format!("doc {i}")))
            .collect();
        let ranges = plan_chunk_ranges(&docs);
        let owners = chunk_owners(&ranges);
        assert_eq!(owners.len(), 40);
        assert_eq!(owners.chunks(embed::EMBED_BATCH_SIZE).count(), 2);
        // Every position maps back to exactly one (doc, range) pair.
        for (position, &(doc_index, range_index)) in owners.iter().enumerate() {
            assert_eq!(doc_index, position);
            assert_eq!(range_index, 0);
        }
    }

    #[test]
    fn chunk_markers_survive_a_batch_boundary_across_documents() {
        // The first document is long enough to span more than one
        // embedding batch, so the batch boundary at owner position
        // EMBED_BATCH_SIZE falls inside its chunk list rather than
        // aligning with a document edge. The second document is a short
        // tail that begins mid-batch.
        let docs = vec![
            (r"D:\Docs\long.txt".to_string(), "x".repeat(25_000)),
            (r"D:\Docs\short.txt".to_string(), "second document".to_string()),
        ];
        let ranges = plan_chunk_ranges(&docs);
        assert!(ranges[0].len() > embed::EMBED_BATCH_SIZE);
        assert_eq!(ranges[1].len(), 1);

        let owners = chunk_owners(&ranges);
        assert_eq!(owners.len(), ranges[0].len() + ranges[1].len());
        // The boundary position belongs to the first document.
        assert!(embed::EMBED_BATCH_SIZE < ranges[0].len());
        assert_eq!(owners[embed::EMBED_BATCH_SIZE], (0, embed::EMBED_BATCH_SIZE));

        // Assign a distinct marker per (document, chunk) in owner order,
        // feeding them through the same bounded batching the pipeline
        // uses, without loading the model.
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(owners.len());
        for batch in owners.chunks(embed::EMBED_BATCH_SIZE) {
            for &(doc_index, chunk_index) in batch {
                vectors.push(marker_vector(doc_index, chunk_index));
            }
        }
        assert_eq!(vectors.len(), owners.len());

        let staged = assemble_staged(docs, ranges.clone(), vectors).expect("assemble");
        assert_eq!(staged.len(), 2);
        for (doc_index, doc) in staged.iter().enumerate() {
            assert_eq!(doc.chunks.len(), ranges[doc_index].len());
            for (chunk_index, chunk) in doc.chunks.iter().enumerate() {
                assert_eq!(chunk.vector, marker_vector(doc_index, chunk_index));
                assert_eq!(chunk.start, ranges[doc_index][chunk_index].start);
                assert_eq!(chunk.end, ranges[doc_index][chunk_index].end);
            }
        }
    }

    #[test]
    fn staged_chunks_reconcile_and_are_searchable_above_threshold() {
        let content = format!("{}\n\n{}", "A".repeat(500), "B".repeat(1500));
        let docs = vec![(r"D:\Docs\a.txt".to_string(), content)];
        let ranges = plan_chunk_ranges(&docs);
        let owners = chunk_owners(&ranges);
        let vectors = vec![fake_vector(); owners.len()];
        let staged = assemble_staged(docs, ranges, vectors).expect("assemble");

        let mut store = store::VectorStore::new();
        store
            .reconcile_and_register(&PathBuf::from(r"D:\Docs"), r"D:\Docs", staged)
            .expect("reconcile");
        assert_eq!(store.entry_count(), 1);
        // Unit query vs unit chunk vectors scores 1.0, above the 0.25 floor.
        let hits = store.search(&fake_vector(), 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, r"D:\Docs\a.txt");
    }

    #[test]
    fn empty_documents_remain_counted_but_not_searchable() {
        let docs = vec![(r"D:\Docs\empty.txt".to_string(), String::new())];
        let ranges = plan_chunk_ranges(&docs);
        let staged = assemble_staged(docs, ranges, Vec::new()).expect("assemble");

        let mut store = store::VectorStore::new();
        store
            .reconcile_and_register(&PathBuf::from(r"D:\Docs"), r"D:\Docs", staged)
            .expect("reconcile");
        assert_eq!(store.entry_count(), 1);
        assert!(store.search(&fake_vector(), 10).is_empty());
    }
}
