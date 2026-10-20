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

/// Test-only access to the loaded model so the opt-in smoke test in
/// `main.rs` can compare the real tokenizer's encoding with the untruncated
/// counter. The accessor lives here because `MODEL` and `get_model` are
/// private to this module.
#[cfg(test)]
pub(crate) fn model_for_test() -> &'static Mutex<TextEmbedding> {
    get_model()
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

    /// Full, untruncated token count for `text`, special tokens included.
    /// Equivalent to `chunk::TokenBudget::count_tokens`; the inherent method
    /// keeps this module independent of the chunker so the
    /// `evaluate_relevance` example can compile it on its own.
    pub fn count_tokens(&self, text: &str) -> usize {
        (self.counter().count_fn)(text)
    }

    /// Effective input limit, special tokens included. Equivalent to
    /// `chunk::TokenBudget::max_tokens`.
    pub fn max_tokens(&self) -> usize {
        self.counter().max_tokens
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
