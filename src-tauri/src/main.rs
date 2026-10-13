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
    /// `None` means persistence is disabled for this session: search and
    /// indexing still work in memory, but nothing is written to disk so a
    /// valid-but-unreadable or unquarantinable file can never be clobbered.
    index_path: Option<PathBuf>,
    startup_notice: Mutex<Option<String>>,
}

#[tauri::command]
async fn index_folder(path: String, state: State<'_, AppState>) -> Result<usize, String> {
    let docs = watcher::scan_folder(&path).map_err(|e| e.to_string())?;
    let mut indexed = 0;

    for doc in docs {
        let vector = embed::embed_text(&doc.content).await.map_err(|e| e.to_string())?;
        state
            .store
            .lock()
            .map_err(|e| format!("search index is unavailable: {e}"))?
            .insert(doc.path, doc.content, vector);
        indexed += 1;
    }

    // Serialize saves through `save_lock` so concurrent index passes
    // commit in order; each snapshot is taken after acquiring the save
    // lock and therefore includes everything committed before it. The
    // store lock is held only for the in-memory snapshot, never for
    // disk I/O.
    let persist_path: PathBuf = match state.index_path.clone() {
        Some(p) => p,
        None => {
            return Err(format!(
                "Indexed {indexed} files in memory, but the persistent index is unavailable this session, so changes cannot be saved and may be lost after restart. Search works for now; restart the app and re-index to retry."
            ));
        }
    };
    let _save_guard = state
        .save_lock
        .lock()
        .map_err(|e| format!("could not save search index: index is unavailable: {e}"))?;
    let snapshot = state
        .store
        .lock()
        .map_err(|e| format!("could not save search index: index is unavailable: {e}"))?
        .serialize()?;
    match store::write_serialized_atomic(&persist_path, &snapshot) {
        Ok(()) => Ok(indexed),
        Err(e) => Err(format!(
            "Indexed {indexed} files in memory, but saving the persistent index failed: {e}. Results will work until the app restarts."
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
                index_path,
                startup_notice: Mutex::new(notice),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![index_folder, search, get_startup_notice])
        .run(tauri::generate_context!())
        .expect("error while running deep-search");
}
