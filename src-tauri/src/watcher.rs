// Folder scanning + (eventually) live file watching.
//
// For v0 this does a one-shot scan when the user picks a folder.
// The `notify` crate is already a dependency — wiring up live
// create/modify/delete events is the natural next step once the
// one-shot scan and embedding pipeline both work end to end.

use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub struct ScannedDoc {
    pub path: String,
    pub content: String,
}

/// Validated scan result. `root` is the stable absolute folder that was
/// actually enumerated; `docs` holds every supported file read from it.
/// A successful return means complete enumeration — any traversal or
/// read failure is an `Err` and must never be treated as an empty folder.
pub struct ScanOutcome {
    pub root: PathBuf,
    pub docs: Vec<ScannedDoc>,
}

const SUPPORTED_EXTENSIONS: &[&str] = &["md", "txt"];

pub fn scan_folder(root: &str) -> std::io::Result<ScanOutcome> {
    let mut cleaned = root.trim();
    if cleaned.len() >= 2 && cleaned.starts_with('"') && cleaned.ends_with('"') {
        cleaned = cleaned[1..cleaned.len() - 1].trim();
    }
    if cleaned.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("invalid folder path: {root}"),
        ));
    }
    let requested = Path::new(cleaned);
    let candidate = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        std::env::current_dir()?.join(requested)
    };
    let meta = fs::metadata(&candidate).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("could not access folder {}: {e}", candidate.display()),
        )
    })?;
    if !meta.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("not a folder: {}", candidate.display()),
        ));
    }
    // Stable absolute root for scanning and reconciliation. Canonicalize
    // resolves symlinks and `.`/`..`; entry paths below inherit it.
    let canon = fs::canonicalize(&candidate).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("could not resolve folder {}: {e}", candidate.display()),
        )
    })?;

    let mut docs = Vec::new();
    for entry in WalkDir::new(&canon) {
        let entry = entry.map_err(|e| {
            std::io::Error::other(format!(
                "could not traverse folder {}: {e}",
                canon.display()
            ))
        })?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if !is_supported(path) {
            continue;
        }
        // Fail the whole scan on any unreadable supported file so a
        // partial enumeration can never drive stale-entry deletion.
        // Traverse and read via the canonical path, but store the
        // readable display form (extended `\\?\` drive/UNC prefixes are
        // converted back to ordinary paths for the index and UI).
        let content = fs::read_to_string(path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("could not read file {}: {e}", path.display()),
            )
        })?;
        docs.push(ScannedDoc {
            path: storage_path(path),
            content,
        });
    }

    Ok(ScanOutcome { root: canon, docs })
}

/// Readable form of a canonical path for index storage and display.
/// Converts recognized extended drive (`\\?\C:\…`) and extended UNC
/// (`\\?\UNC\server\share\…`) prefixes back to ordinary paths while
/// preserving the remainder's spelling. Device-namespace paths
/// (`\\?\Volume{…}`, `\\?\GLOBALROOT`, `\\.\…`) are preserved as-is so
/// they stay ambiguous for reconciliation instead of collapsing into an
/// unrelated path.
fn storage_path(canonical: &Path) -> String {
    let s = canonical.to_string_lossy();
    let lower = s.to_lowercase();
    if lower.starts_with("\\\\?\\unc\\") {
        return format!("\\\\{}", &s["\\\\?\\unc\\".len()..]);
    }
    if lower.starts_with("\\\\?\\") {
        let prefix_len = "\\\\?\\".len();
        let rest_lower = &lower[prefix_len..];
        let rest = &s[prefix_len..];
        if rest_lower.len() >= 3
            && rest_lower.as_bytes()[0].is_ascii_alphabetic()
            && rest_lower.as_bytes()[1] == b':'
            && (rest_lower.as_bytes()[2] == b'\\' || rest_lower.as_bytes()[2] == b'/')
        {
            return rest.to_string();
        }
        if rest_lower.len() == 2
            && rest_lower.as_bytes()[0].is_ascii_alphabetic()
            && rest_lower.as_bytes()[1] == b':'
        {
            return rest.to_string();
        }
        return s.into_owned();
    }
    s.into_owned()
}

fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            SUPPORTED_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str())
        })
        .unwrap_or(false)
}

// TODO next: replace the one-shot scan with a live `notify::Watcher`
// that pushes create/modify/delete events through a channel, so the
// index stays current without the user re-triggering a scan.

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn test_dir(name: &str) -> PathBuf {
        let count = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "searchiee-watcher-{}-{}-{name}-{count}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("test dir");
        dir
    }

    #[test]
    fn missing_root_returns_error_not_empty() {
        let dir = test_dir("missing-parent");
        let missing = dir.join("does-not-exist");
        assert!(!missing.exists());
        let result = scan_folder(&missing.to_string_lossy());
        assert!(result.is_err());
    }

    #[test]
    fn empty_path_and_file_as_root_return_error() {
        assert!(scan_folder("").is_err());
        assert!(scan_folder("   ").is_err());
        let dir = test_dir("file-root");
        let file = dir.join("note.txt");
        std::fs::write(&file, "hello").expect("seed file");
        assert!(scan_folder(&file.to_string_lossy()).is_err());
    }

    #[test]
    fn empty_folder_scans_successfully_with_no_docs() {
        let dir = test_dir("empty");
        let outcome = scan_folder(&dir.to_string_lossy()).expect("empty scan");
        assert!(outcome.docs.is_empty());
        assert!(outcome.root.is_absolute());
    }

    #[test]
    fn successful_scan_returns_absolute_supported_files() {
        let dir = test_dir("scan");
        std::fs::write(dir.join("a.txt"), "alpha").expect("a");
        std::fs::write(dir.join("b.md"), "beta").expect("b");
        std::fs::write(dir.join("c.bin"), "ignored").expect("c");
        std::fs::create_dir_all(dir.join("sub")).expect("sub");
        std::fs::write(dir.join("sub").join("d.txt"), "delta").expect("d");
        let outcome = scan_folder(&dir.to_string_lossy()).expect("scan");
        assert!(outcome.root.is_absolute());
        let mut paths: Vec<String> = outcome.docs.iter().map(|d| d.path.clone()).collect();
        paths.sort();
        assert_eq!(paths.len(), 3);
        assert!(paths.iter().all(|p| Path::new(p).is_absolute()));
        assert!(outcome.docs.iter().any(|d| d.content == "alpha"));
        assert!(outcome.docs.iter().any(|d| d.content == "beta"));
        assert!(outcome.docs.iter().any(|d| d.content == "delta"));
    }

    #[test]
    fn scan_includes_uppercase_and_mixed_case_supported_extensions() {
        let dir = test_dir("case-ext");
        std::fs::write(dir.join("lower.txt"), "lower").expect("lower");
        std::fs::write(dir.join("UPPER.TXT"), "upper").expect("upper");
        std::fs::write(dir.join("Mixed.Md"), "mixed md").expect("mixed md");
        std::fs::write(dir.join("DOC.MD"), "doc").expect("doc");
        std::fs::write(dir.join("note.TxT"), "mixed txt").expect("mixed txt");
        std::fs::write(dir.join("image.PNG"), "unsupported").expect("png");
        std::fs::write(dir.join("data.BIN"), "unsupported").expect("bin");
        std::fs::write(dir.join("noext"), "unsupported").expect("noext");
        let outcome = scan_folder(&dir.to_string_lossy()).expect("scan");
        let mut contents: Vec<String> =
            outcome.docs.iter().map(|d| d.content.clone()).collect();
        contents.sort();
        assert_eq!(
            contents,
            vec!["doc", "lower", "mixed md", "mixed txt", "upper"]
        );
    }

    #[test]
    fn storage_path_converts_extended_forms_for_display() {
        // Extended drive and UNC forms become ordinary paths while
        // preserving the remainder's spelling.
        assert_eq!(
            storage_path(Path::new("\\\\?\\C:\\Docs\\File.TXT")),
            "C:\\Docs\\File.TXT"
        );
        assert_eq!(
            storage_path(Path::new("\\\\?\\UNC\\server\\share\\Docs\\File.txt")),
            "\\\\server\\share\\Docs\\File.txt"
        );
        // Device-namespace paths are preserved, never stripped into an
        // unrelated relative path.
        assert_eq!(
            storage_path(Path::new(
                "\\\\?\\Volume{12345678-1234-1234-1234-123456789abc}\\x.txt"
            )),
            "\\\\?\\Volume{12345678-1234-1234-1234-123456789abc}\\x.txt"
        );
        assert_eq!(
            storage_path(Path::new("\\\\.\\C:\\Docs\\a.txt")),
            "\\\\.\\C:\\Docs\\a.txt"
        );
        // Ordinary paths pass through unchanged.
        assert_eq!(
            storage_path(Path::new("C:\\Docs\\a.txt")),
            "C:\\Docs\\a.txt"
        );
    }
}
