# deep-search (starter scaffold)

A from-scratch desktop semantic search app. This is a minimal starting
skeleton, not a finished product — it gives you a working file watcher,
a stub embedding pipeline, and a bare search UI, wired together so you
can start filling in the real logic.

## Why these choices

- **Tauri, not Electron** — much lower baseline memory usage, which matters
  on an 8GB machine. The webview uses the OS's native renderer instead of
  bundling Chromium.
- **Vanilla TS frontend, not React** — one less framework's overhead while
  the app is this small. Swap in React/Vite later if the UI grows.
- **Hosted embeddings (Voyage) to start** — your machine doesn't have the
  RAM/VRAM headroom to comfortably run a local embedding model alongside
  a Rust build and a dev server. The embedding call is isolated in one
  function (`src-tauri/src/embed.rs`) so switching to local models later
  is a small change, not a rewrite.
- **In-memory vector store for v0** — cosine similarity over a `Vec` is
  fine until you have a few thousand documents. Swap in LanceDB or Qdrant
  once that stops being true.

## Prerequisites (install these on your own machine, not here)

- Rust + Cargo: https://rustup.rs
- Node.js (you already have this)
- Tauri CLI: `cargo install tauri-cli`
- A Voyage AI API key: https://www.voyageai.com

## Setup

```bash
npm install
cp .env.example .env   # then add your VOYAGE_API_KEY
cargo tauri dev
```

## What's stubbed vs. real

| Piece | Status |
|---|---|
| File watcher (`watcher.rs`) | Real — watches a folder, detects create/modify/delete |
| Text extraction | Stub — only reads plain `.md`/`.txt` for now |
| Embedding call (`embed.rs`) | Stub — has the HTTP shape, needs your API key wired in |
| Vector store (`store.rs`) | Real — simple in-memory cosine similarity |
| Search UI (`src/main.ts`) | Real — basic input + results list |

## Next steps (see the build order from earlier)

1. Wire up your Voyage API key and confirm `embed.rs` returns real vectors
2. Point the watcher at a real folder and confirm indexing works end to end
3. Add chunking for longer documents before embedding them
4. Add a lexical/keyword pass alongside the vector search (hybrid search)
