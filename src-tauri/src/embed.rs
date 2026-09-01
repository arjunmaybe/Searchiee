// Embedding pipeline. Isolated here so swapping providers — hosted API
// now, local model later — only touches this file.

use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: Vec<&'a str>,
    model: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDatum>,
}

#[derive(Deserialize)]
struct EmbedDatum {
    embedding: Vec<f32>,
}

/// A stub for generating text embeddings using the Voyage AI API.
/// This function currently handles one text input at a time.
/// It expects the VOYAGE_API_KEY environment variable to be set.
pub async fn embed_text(text: &str) -> Result<Vec<f32>, reqwest::Error> {
    let api_key = std::env::var("VOYAGE_API_KEY").expect(
        "VOYAGE_API_KEY not set — copy .env.example to .env and add your key",
    );

    let client = reqwest::Client::new();
    let body = EmbedRequest {
        input: vec![text],
        model: "voyage-3-lite", // cheaper/faster model, fine for a v0
    };

    let response = client
        .post("https://api.voyageai.com/v1/embeddings")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?
        .json::<EmbedResponse>()
        .await?;

    Ok(response.data.into_iter().next().unwrap().embedding)
}

// TODO next: batch multiple documents into one request instead of
// embedding one at a time — Voyage's API accepts an array of inputs,
// and batching will matter a lot once you're indexing real folders.
