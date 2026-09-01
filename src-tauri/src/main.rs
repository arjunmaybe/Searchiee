// Entry point. Wires the file watcher, embedding pipeline, and vector
// store together, and exposes a couple of commands to the frontend.

mod embed;
mod store;
mod watcher;

use std::sync::Mutex;
use store::VectorStore;
use tauri::State;

struct AppState {
    store: Mutex<VectorStore>,
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
            .unwrap()
            .insert(doc.path, doc.content, vector);
        indexed += 1;
    }

    Ok(indexed)
}

#[tauri::command]
async fn search(query: String, state: State<'_, AppState>) -> Result<Vec<store::SearchResult>, String> {
    let query_vector = embed::embed_text(&query).await.map_err(|e| e.to_string())?;
    let results = state.store.lock().unwrap().search(&query_vector, 10);
    Ok(results)
}

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            store: Mutex::new(VectorStore::new()),
        })
        .invoke_handler(tauri::generate_handler![index_folder, search])
        .run(tauri::generate_context!())
        .expect("error while running deep-search");
}
