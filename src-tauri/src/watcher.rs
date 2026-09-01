// Folder scanning + (eventually) live file watching.
//
// For v0 this does a one-shot scan when the user picks a folder.
// The `notify` crate is already a dependency — wiring up live
// create/modify/delete events is the natural next step once the
// one-shot scan and embedding pipeline both work end to end.

use std::fs;
use std::path::Path;
use walkdir::WalkDir;

pub struct ScannedDoc {
    pub path: String,
    pub content: String,
}

const SUPPORTED_EXTENSIONS: &[&str] = &["md", "txt"];

pub fn scan_folder(root: &str) -> std::io::Result<Vec<ScannedDoc>> {
    let mut docs = Vec::new();

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        if !is_supported(path) {
            continue;
        }

        // Skip anything unreadable rather than failing the whole scan —
        // permission errors, binary-in-disguise files, etc. shouldn't
        // block indexing everything else.
        if let Ok(content) = fs::read_to_string(path) {
            docs.push(ScannedDoc {
                path: path.to_string_lossy().to_string(),
                content,
            });
        }
    }

    Ok(docs)
}

fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| SUPPORTED_EXTENSIONS.contains(&ext))
        .unwrap_or(false)
}

// TODO next: replace the one-shot scan with a live `notify::Watcher`
// that pushes create/modify/delete events through a channel, so the
// index stays current without the user re-triggering a scan.
