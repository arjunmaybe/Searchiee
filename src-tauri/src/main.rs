// Entry point. Wires the file watcher, embedding pipeline, and vector
// store together, and exposes a couple of commands to the frontend.

mod embed;
mod store;
mod watcher;

use std::path::PathBuf;
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
    let mut staged: Vec<store::StagedDoc> = Vec::with_capacity(scan.docs.len());
    for doc in scan.docs {
        let vector = embed::embed_text(&doc.content).await.map_err(|e| e.to_string())?;
        staged.push(store::StagedDoc {
            path: doc.path,
            content: doc.content,
            vector,
        });
    }
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
                    Ok(index_path) => match VectorStore::load_from_path(&index_path) {
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
                        Err(e) => (
                            VectorStore::new(),
                            Some(format!(
                                "Saved search index could not be read ({e}); persistence is disabled this session to protect the existing file. Search and indexing still work in memory. No files were changed."
                            )),
                            None,
                        ),
                    },
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
        .expect("error while running deep-search");
}
