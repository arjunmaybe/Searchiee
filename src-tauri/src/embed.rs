// Local, free embedding pipeline using fastembed-rs — a small (~90MB)
// ONNX embedding model that runs on CPU with no API key and no cost.
// Good for testing the pipeline end to end before deciding whether
// to pay for a hosted API's better embedding quality later.
//
// Swap back to a hosted API (see embed_hosted.rs.bak) once you want
// higher-quality embeddings — the function signature is identical,
// so main.rs doesn't need to change either way.

use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use std::sync::{Mutex, OnceLock};

static MODEL: OnceLock<Mutex<TextEmbedding>> = OnceLock::new();

fn get_model() -> &'static Mutex<TextEmbedding> {
    MODEL.get_or_init(|| {
        // Downloads the model once on first run (~90MB) and caches it
        // locally after that — every run after the first is offline.
        Mutex::new(
            TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
            )
            .expect("failed to load local embedding model"),
        )
    })
}

pub async fn embed_text(text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut model = get_model()
        .lock()
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    let embeddings = model.embed(vec![text], None)?;
    Ok(embeddings.into_iter().next().unwrap())
}
