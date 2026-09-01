// Minimal in-memory vector store. Fine for a few thousand documents;
// once this stops being fine, swap in LanceDB or Qdrant — the
// `insert`/`search` interface here is the contract to preserve.

use serde::Serialize;

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

impl VectorStore {
    pub fn new() -> Self {
        VectorStore { entries: Vec::new() }
    }

    pub fn insert(&mut self, path: String, content: String, vector: Vec<f32>) {
        self.entries.push(Entry { path, content, vector });
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
