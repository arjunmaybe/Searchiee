// Local, free embedding pipeline using fastembed-rs — a small (~90MB)
// ONNX embedding model that runs on CPU with no API key and no cost.
// Good for testing the pipeline end to end before deciding whether
// to pay for a hosted API's better embedding quality later.
//
// Swap back to a hosted API (see embed_hosted.rs.bak) once you want
// higher-quality embeddings — the function signature is identical,
// so main.rs doesn't need to change either way.

use crate::chunk::TokenBudget;
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

/// Token accounting backed by the embedding model's own tokenizer.
///
/// Nothing here is guessed: the input limit is read from the tokenizer
/// FastEmbed configured for the model (`get_truncation().max_length`,
/// which includes the special tokens), and the token count is what that
/// same tokenizer produces with special tokens included. A chunk is
/// embedded in full exactly when `count_tokens(chunk) <= max_tokens()`.
///
/// The count must be the *full*, untruncated number of tokens, but
/// `Tokenizer::encode` truncates to the configured limit. tokenizers
/// 0.23.2 applies that truncation while pre-tokenizing and drops the
/// remaining splits without recording them in `Encoding::get_overflowing()`
/// (huggingface/tokenizers#2091), so summing the encoding and its overflow
/// undercounts oversized input — precisely the case this budget exists to
/// detect. Instead of reconstructing the count from the truncated encoding,
/// we keep a *clone* of the model's tokenizer with truncation switched off
/// and count with the clone; the model's real tokenizer is never modified.
/// The clone is built once, lazily, and cached here (never per call), so an
/// empty scan still touches no model.
pub struct EmbeddingTokenBudget {
    counter: OnceLock<Result<UntruncatedCounter, String>>,
}

/// A truncation-free tokenizer clone cached together with the effective
/// input limit. The tokenizer is held inside a boxed closure so it can be
/// cached without naming `tokenizers::Tokenizer`, which is not a direct
/// dependency of this crate.
struct UntruncatedCounter {
    max_tokens: usize,
    count_fn: Box<dyn Fn(&str) -> usize + Send + Sync>,
}

/// Clone the model's tokenizer, disable truncation on the clone only, and
/// resolve the effective input limit. Returns an error (never a guessed
/// limit) if the tokenizer has no truncation configured or the clone cannot
/// be reconfigured.
fn build_counter() -> Result<UntruncatedCounter, String> {
    let model = get_model()
        .lock()
        .map_err(|e| format!("embedding model is unavailable: {e}"))?;
    let max_tokens = model
        .tokenizer
        .get_truncation()
        .ok_or_else(|| {
            "embedding tokenizer has no truncation limit configured; \
             refusing to guess a token budget"
                .to_string()
        })?
        .max_length;
    let mut untruncated = model.tokenizer.clone();
    untruncated
        .with_truncation(None)
        .map_err(|e| format!("could not disable truncation on the counting tokenizer: {e}"))?;
    let count_fn = move |text: &str| {
        untruncated
            .encode(text, true)
            .expect("counting tokenizer could not encode text")
            .len()
    };
    Ok(UntruncatedCounter {
        max_tokens,
        count_fn: Box::new(count_fn),
    })
}

impl EmbeddingTokenBudget {
    /// A budget that touches the model only when it is first used, so an
    /// empty scan (no non-empty content to chunk) never triggers a download
    /// or load.
    pub fn new() -> Self {
        Self {
            counter: OnceLock::new(),
        }
    }

    /// Resolve and validate the counting tokenizer now, surfacing a missing
    /// or unusable truncation configuration as an error instead of guessing
    /// a limit. A no-op once the counter has been built. Callers that skip
    /// it (because there is no text to chunk) keep the lazy, model-free path.
    pub fn prepare(&self) -> Result<(), String> {
        self.counter
            .get_or_init(build_counter)
            .as_ref()
            .map(|_| ())
            .map_err(|e| e.clone())
    }

    fn counter(&self) -> &UntruncatedCounter {
        self.counter
            .get_or_init(build_counter)
            .as_ref()
            .unwrap_or_else(|e| panic!("{e}"))
    }
}

impl Default for EmbeddingTokenBudget {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenBudget for EmbeddingTokenBudget {
    fn count_tokens(&self, text: &str) -> usize {
        (self.counter().count_fn)(text)
    }

    fn max_tokens(&self) -> usize {
        self.counter().max_tokens
    }
}

pub async fn embed_text(text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut model = get_model()
        .lock()
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    let embeddings = model.embed(vec![text], None)?;
    Ok(embeddings.into_iter().next().unwrap())
}

/// Chunk texts per embedding call. 32 balances throughput against peak
/// CPU memory: the dominant cost is transformer activations, which grow
/// with batch × sequence length, while one failing batch aborts the whole
/// indexing pass. fastembed's own default is 256.
pub const EMBED_BATCH_SIZE: usize = 32;

/// Embed several texts in one call, preserving order exactly: output `i`
/// belongs to input `i`. Used only by the indexing pipeline; query
/// embedding keeps using `embed_text`.
pub async fn embed_texts(
    texts: Vec<String>,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut model = get_model()
        .lock()
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    let embeddings = model.embed(texts, Some(EMBED_BATCH_SIZE))?;
    Ok(embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk;

    /// Opt-in smoke test against the real FastEmbed tokenizer.
    ///
    /// Ordinary tests stay model-free; this one is `#[ignore]`d and must be
    /// run explicitly:
    ///
    /// ```text
    /// cargo test --bin searchiee embedding_token_budget_matches_the_model -- --ignored --nocapture
    /// ```
    ///
    /// It is forced offline. hf-hub 0.5.0's `ApiRepo::get` is cache-first
    /// (a cached file is returned without any request), and only a cache
    /// miss reaches the network. `HF_ENDPOINT` is pinned to a closed
    /// loopback port so a miss fails fast instead of downloading, and the
    /// cache is pinned to this crate's existing `.fastembed_cache`. If the
    /// assets are absent the test fails; it never downloads.
    #[test]
    #[ignore = "loads the cached FastEmbed model; run explicitly with --ignored"]
    fn embedding_token_budget_matches_the_model() {
        std::env::set_var("HF_ENDPOINT", "http://127.0.0.1:9");
        std::env::remove_var("HF_HOME");
        std::env::set_var(
            "FASTEMBED_CACHE_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/.fastembed_cache"),
        );

        let budget = EmbeddingTokenBudget::new();
        budget
            .prepare()
            .expect("the cached FastEmbed model assets are required");

        let limit = budget.max_tokens();
        println!("configured token limit: {limit}");

        // Deterministic, token-dense text: punctuation, accented Latin,
        // CJK, Cyrillic, Arabic and digits. The repeats make this many
        // times longer than the limit, so a truncating tokenizer cannot
        // report its full length.
        let unit =
            "The quick brown fox; 你好，世界！Привет, мир! مرحبا بالعالم café/naïve—résumé? 1234567890. ";
        let repeats = limit.clamp(64, 512);
        let text = unit.repeat(repeats);
        let full = budget.count_tokens(&text);
        println!(
            "sample: {} bytes / {} chars, {full} untruncated tokens",
            text.len(),
            text.chars().count()
        );
        assert!(
            full > limit,
            "sample must exceed the limit (count {full} <= limit {limit})"
        );

        let chunks =
            chunk::chunk_content(&text, &budget).expect("the real budget must chunk the sample");
        assert!(
            chunks.len() > 1,
            "oversized text must split into more than one chunk"
        );

        let mut covered = vec![false; text.len()];
        let mut previous_end = 0usize;
        for (index, chunk) in chunks.iter().enumerate() {
            assert!(chunk.start < chunk.end, "chunk {index} is empty");
            assert!(chunk.start <= previous_end, "chunk {index} leaves a gap");
            assert!(chunk.end > previous_end, "chunk {index} does not advance");
            assert!(
                text.is_char_boundary(chunk.start),
                "chunk {index} start is not a UTF-8 boundary"
            );
            assert!(
                text.is_char_boundary(chunk.end),
                "chunk {index} end is not a UTF-8 boundary"
            );

            let slice = &text[chunk.start..chunk.end];
            for flag in &mut covered[chunk.start..chunk.end] {
                *flag = true;
            }
            previous_end = chunk.end;

            let counted = budget.count_tokens(slice);
            assert!(
                counted <= limit,
                "chunk {index} needs {counted} tokens, above the limit {limit}"
            );

            // Within the limit the model's own tokenizer cannot truncate, so
            // its encoding length must equal the untruncated count. This is
            // the check that the clone and the real tokenizer agree exactly
            // where the model is actually used.
            let model = get_model().lock().expect("embedding model lock");
            let encoded = model
                .tokenizer
                .encode(slice, true)
                .expect("model tokenizer failed to encode a chunk")
                .len();
            assert_eq!(
                encoded, counted,
                "chunk {index}: model encoding length {encoded} != untruncated count {counted}"
            );
        }

        assert!(
            covered.iter().all(|&byte| byte),
            "every byte must be covered by some chunk"
        );
        println!(
            "ok: {} chunks, all within {limit} tokens and fully covered",
            chunks.len()
        );
    }
}
