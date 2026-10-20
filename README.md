# Searchiee

Searchiee is a local-first desktop semantic search app for Windows, built
with Tauri v2, Rust, TypeScript, and Vite. You add folders containing
`.txt` and `.md` files, search their contents by meaning, refresh folders
after files change, and remove folders from the index. All indexing and
search runs on your machine; nothing is uploaded anywhere.

## What you can do

- **Add a folder** ("Index a folder"): scans the folder for supported
  files, chunks and embeds the text locally, and stores the results. The
  folder is registered so it appears in the folder list afterwards.
- **Search**: type a query to semantically search everything indexed, with
  a short snippet and match score per result.
- **Refresh a registered folder**: re-scans that folder and reconciles
  changed, renamed, and deleted files. Files in other registered folders
  are left alone.
- **Remove a registered folder**: deletes that folder's registration and
  its index entries. Original files on disk are never touched.

Only Markdown (`.md`, any letter case) and plain-text (`.txt`, any letter
case) files are indexed. Long documents are split into overlapping chunks
sized with the embedding model's own tokenizer, with a 4,096-character
secondary cap; up to 120 characters of context can overlap between
neighbouring chunks when the token budget allows. Indexing embeds up to 32
chunk texts per model call. Search returns one result per file. For chunked
entries, it ranks each file by its best-matching chunk and shows a snippet
from that chunk; legacy whole-file entries continue to use their stored
whole-file vector. Scores below the `0.25` threshold are filtered out, and
the top 10 results show snippets of up to 200 characters.

## How indexing stays safe

- A scan must complete fully to count: a missing folder, an unreadable
  folder, a traversal error, or an unreadable supported file fails the
  whole operation instead of looking like an empty folder.
- Refresh embeds everything first and only then updates the in-memory
  index, so a scan, traversal, supported-file read, or embedding failure
  aborts before indexed entries **and** the folder registration change.
- If persistence fails after a successful in-memory reconciliation or
  removal, the current session may already reflect the change even though
  the app reports it could not be saved and warns it may be lost after
  restart.
- Refreshing or removing one folder never touches entries from other
  folders. Registering nested folders (a folder and its subfolder) is
  allowed; their file counts overlap because the index is shared, and
  removing an inner root removes its matching entries from the shared
  index even when the outer root stays registered.
- Indexing operations run one at a time; search stays available except
  for the brief in-memory update.
- There is no background filesystem watcher. Indexing and refreshing only
  happen when you click.

## Folders, persistence, and privacy

- Registered folders and document entries are stored together in one
  versioned JSON index file, `index-v1.json` (currently schema v3), inside
  Tauri's application-specific local data directory. The file holds folder
  paths, document paths, document contents, and per-chunk embedding
  vectors; legacy entries may still carry a whole-file vector.
- Saves are atomic: the new index is written to a temporary file and
  renamed over the old one, so a failed save never destroys the last good
  index. If saving fails, the app says so and warns the in-memory changes
  may be lost on restart.
- A corrupt or incompatible index is moved aside next to the original
  (never deleted silently) and reported on startup; indexing then starts
  from an empty in-memory index.
- Deleting `index-v1.json` resets the saved index and requires
  re-indexing. It does not delete your documents.
- Indexes written by older schema-v1 or schema-v2 formats still load
  without re-embedding. v1 has no folder registry, so add folders again to
  refresh or remove them; v2's valid folder registrations are preserved,
  with duplicate or ambiguous paths dropped.

## Local embeddings and the model cache

- Embeddings are computed locally with FastEmbed model `AllMiniLML6V2`
  (384 dimensions). No API key, `.env` file, or cloud service is needed.
- On first use the app downloads the model (~90 MB, per the code comment),
  so the first indexing or search needs network access. Later runs reuse
  the cached model and work offline.
- The code does not set a cache directory, so FastEmbed uses its default:
  a `.fastembed_cache` directory relative to the app's working directory,
  overridable with the `FASTEMBED_CACHE_DIR` environment variable. When
  developing from `src-tauri/`, that is `src-tauri/.fastembed_cache/`
  (untracked build artifact, safe to delete — the model just downloads
  again). Offline first-run behavior is not promised.

## Prerequisites

- Windows 10/11
- Microsoft C++ Build Tools with the **Desktop development with C++**
  workload (required to compile Tauri apps)
- WebView2, required by Tauri to render the app window; it is normally
  included on Windows 10 version 1803 and later and on Windows 11
- Rust stable toolchain + Cargo: https://rustup.rs
- Node.js 20 LTS or newer
- Tauri CLI v2: `cargo install tauri-cli`

See the official requirements for details:
https://v2.tauri.app/start/prerequisites/

No API keys and no `.env` file are required (the leftover `.env.example`
is not used).

## Setup and run (PowerShell)

```powershell
git clone https://github.com/arjunmaybe/Searchiee.git
cd Searchiee
npm install
cargo tauri dev
```

## Using the app

1. Click **Index a folder**, enter a folder path, and wait for the
   "Indexed N files" confirmation. The folder appears in the folder list
   with its indexed-file count.
2. Type in the search box; results appear as you type.
3. Click **Refresh** next to a folder after its files change. If the
   folder is missing or a file cannot be read, refresh fails with an
   error and the previously saved entries and registration are kept.
4. Click **Remove** next to a folder to drop it from the index after a
   confirmation dialog. Only index entries are removed; source files stay
   on disk.

## Verification

```powershell
cargo test --manifest-path .\src-tauri\Cargo.toml
cargo check --manifest-path .\src-tauri\Cargo.toml
npm run build
```

CI reports 168 Rust application tests passed, 0 failed, and 1 ignored; the
ignored test is the optional model-dependent smoke test. The evaluator
example has a separate unit-test suite that reports 35 passed. CI compiles
and unit-tests the evaluator, but does not run the evaluator itself.

The Rust application suite covers path identity and root boundaries,
fail-closed scanning, chunk planning and token budgets, staged
reconciliation, chunked search and snippets, folder registration and
root-scoped removal, schema-v1/v2 compatibility and schema-v3 persistence,
atomic persistence, and search-relevance threshold filtering.

## Project layout

| Path | Role |
|---|---|
| `src/main.ts`, `index.html` | Search UI and registered-folder list |
| `src-tauri/src/main.rs` | Tauri commands: index, refresh, remove, list, search |
| `src-tauri/src/chunk.rs` | Deterministic token-budgeted document chunker |
| `src-tauri/src/watcher.rs` | Fail-closed folder scan (`.md`/`.txt`) |
| `src-tauri/src/store.rs` | Vector store, folder registry, JSON persistence |
| `src-tauri/src/embed.rs` | Local FastEmbed pipeline |

## Limitations

- Only `.md` and `.txt` files are indexed.
- Long documents are split into multiple chunk embeddings, so index size
  grows with the number of chunks; search still returns one result per file
  using its best-matching chunk.
- No background watching: changed files appear only after a manual refresh.
- The minimum cosine-similarity threshold of `0.25` is experimental, not calibrated against a broad labelled evaluation set, and can still admit irrelevant results scoring above the threshold.
