// Deterministic, UTF-8-safe, token-budgeted text chunker.
//
// `chunk_content` splits a document into `Chunk`s that record UTF-8-safe
// byte offsets, and `chunk_text` resolves a `Chunk` back to its slice of
// the original content.
//
// Sizing is by model tokens, not raw characters. The caller supplies a
// `TokenBudget` describing, for the embedding model actually in use, how
// many tokens a piece of text costs and how many tokens one input may
// contain. A chunk is only emitted if its exact token count (special
// tokens included) fits that budget, so the embedder never silently
// truncates part of a chunk.
//
// Guarantees:
//   * Deterministic. Output depends only on the input content and the
//     (deterministic) `TokenBudget`.
//   * Within budget. Every chunk's `TokenBudget::count_tokens` is at most
//     `TokenBudget::max_tokens`, i.e. the embedder's effective input
//     limit with special tokens included. A budget too small to hold even
//     one character of the content is reported as an error rather than
//     silently emitting an over-budget chunk.
//   * Bounded length. No chunk exceeds `MAX_CHARS` characters. This is a
//     secondary bound: the token budget stays authoritative, but the
//     character ceiling keeps token-sparse text (very long unbroken
//     words, base64, and the like) from producing very large chunks.
//   * Bounded working memory. Only the current `MAX_CHARS` window's
//     character boundaries are materialized, so peak auxiliary metadata
//     does not grow with document length.
//   * UTF-8-safe. Cuts are taken only at `char` boundaries, so every
//     chunk is a valid UTF-8 slice and no code point is split.
//   * Complete coverage. Every character appears in at least one chunk.
//   * Intentional overlap. Neighbouring chunks share up to `OVERLAP_CHARS`
//     characters. If the configured overlap cannot fit the budget it is
//     reduced to the largest affordable value (possibly zero) so that
//     chunks still overlap where possible and always advance. The overlap
//     scheduled for the next window is additionally capped at half of the
//     emitted chunk's character length (rounded down), so a short chunk
//     never makes the next window start inside its own last half.
//   * Forward progress. Each chunk's end strictly advances past the
//     previous end: a boundary already used as the previous chunk's end
//     is never reused, so a finite input yields a finite, non-repeating
//     chunk list with no redundant nested chunks. Each scheduled next
//     start also advances by at least half of the emitted chunk's length.
//     How far the next end advances is decided by the token counter,
//     budget, and boundary choice, not by the half-overlap cap: a budget
//     that cannot fit more than a few characters beyond the overlap can
//     still advance the end one character at a time. The stronger "at
//     least half a chunk of end progress" bound asserted by the tests
//     below applies only to their boundary-free `CharTokens` fixture; when
//     paragraph or sentence boundary selection is active it can be lower
//     even for a monotone counter.
//   * Strictly increasing starts. Each window opens strictly after the
//     previous window's start, so no start is repeated.
//   * LF and CRLF line endings are both understood for paragraph breaks.
//   * For longer text, paragraph and sentence boundaries are preferred
//     when they fall within the token budget; otherwise the split is
//     taken at a character boundary.
#![allow(dead_code)]

/// Target number of characters shared between adjacent chunks. Kept in
/// characters (not tokens) so the overlap stays byte-stable and easy to
/// reason about. When the budget cannot afford this much overlap it is
/// reduced (see `largest_fitting_overlap`).
///
/// The overlap scheduled after a chunk is additionally capped at half that
/// chunk's character length, so the next window starts at least half a
/// chunk later. This guarantees start progress only. End progress depends
/// on the token counter and budget, and preferred paragraph/sentence
/// boundaries can stop the next chunk early even with a monotone counter.
const OVERLAP_CHARS: usize = 120;

/// Secondary hard maximum on a chunk's length, in Unicode scalar values.
///
/// The token budget is authoritative for correctness, but some inputs are
/// token-sparse (a very long unbroken word collapses to a single `[UNK]`
/// token, base64 has no delimiters, and so on), so a token-only bound
/// could emit huge chunks. This ceiling is set well above the character
/// count that normally fills the model's token budget, so it constrains
/// only such pathological inputs.
const MAX_CHARS: usize = 4096;

/// Token accounting for the embedding model.
///
/// `count_tokens` must report the full number of tokens the model's
/// tokenizer produces for `text`, **including special tokens** and
/// **ignoring truncation**; `max_tokens` must report the tokenizer's
/// effective input limit (its truncation length), which likewise
/// includes special tokens. With those definitions,
/// `count_tokens(text) <= max_tokens()` means the model embeds `text`
/// without dropping any of it.
pub trait TokenBudget {
    /// Full, untruncated token count for `text`, special tokens included.
    fn count_tokens(&self, text: &str) -> usize;
    /// Effective input limit, special tokens included.
    fn max_tokens(&self) -> usize;
}

/// Returned by [`chunk_content`] when the token budget cannot hold even a
/// single character of the content, so no valid chunk can be produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenBudgetTooSmall {
    /// The configured maximum number of tokens per input.
    pub max_tokens: usize,
    /// Byte offset of a character that alone exceeds the budget.
    pub offset: usize,
}

impl std::fmt::Display for TokenBudgetTooSmall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "token budget of {} is too small to hold the character at byte {}",
            self.max_tokens, self.offset
        )
    }
}

impl std::error::Error for TokenBudgetTooSmall {}

/// A chunk of the source content, expressed as byte offsets into that
/// content. `start` is inclusive and `end` is exclusive. Both are UTF-8
/// character boundaries, so `content[start..end]` is always valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub start: usize,
    pub end: usize,
}

/// Split `content` into deterministic, UTF-8-safe chunks that each fit
/// `budget` and are at most `MAX_CHARS` characters long.
///
/// Empty content yields no chunks. Non-empty content whose whole token
/// count fits the budget and length yields one whole-content chunk.
/// Returns [`TokenBudgetTooSmall`] if no single character of `content`
/// can fit the budget.
pub fn chunk_content(
    content: &str,
    budget: &dyn TokenBudget,
) -> Result<Vec<Chunk>, TokenBudgetTooSmall> {
    if content.is_empty() {
        return Ok(Vec::new());
    }

    let max_tokens = budget.max_tokens();
    let mut window = Window::new(content);
    let mut chunks = Vec::new();
    // The current window and the previous chunk's end are tracked in both
    // coordinate systems: absolute byte offsets for slices and global char
    // indices for the overlap search's original `previous_end + 1` bound.
    let mut start_byte = 0usize;
    let mut start_char = 0usize;
    let mut previous_end_byte = 0usize;
    let mut previous_end_char = 0usize;

    loop {
        window.reset(start_byte);

        // The largest window from `start_byte` allowed by the token budget,
        // additionally capped by the secondary character bound.
        let hard_end = match fitting_end(&mut window, budget, max_tokens) {
            Some(end) => end,
            None => {
                return Err(TokenBudgetTooSmall {
                    max_tokens,
                    offset: start_byte,
                })
            }
        };

        // `hard_end` is within budget and the cap. If it does not pass the
        // previous end, the configured overlap does not fit this budget, so
        // shrink the overlap. `largest_fitting_overlap` normally returns a
        // start strictly after the current one; a non-monotone token counter
        // can, however, make it report no reduction even though a longer
        // window fits. Reusing the current start would then spin forever, so
        // fall back to the previous end and drop the overlap for this window.
        // Because `hard_end >= 1` and `hard_end <= previous_end_local` here,
        // `previous_end_local > 0`, so this always strictly advances.
        debug_assert!(previous_end_char >= start_char);
        let previous_end_local = previous_end_char - start_char;
        debug_assert!(previous_end_local <= OVERLAP_CHARS);
        if hard_end <= previous_end_local {
            let previous_start_byte = start_byte;
            let reduced = largest_fitting_overlap(
                content,
                previous_end_byte,
                previous_end_char,
                budget,
                max_tokens,
            );
            match reduced {
                // The overlap shrank to a start strictly after the current
                // one; keep it.
                Some((reduced_byte, width)) if reduced_byte > previous_start_byte => {
                    start_byte = reduced_byte;
                    start_char = previous_end_char + 1 - width;
                }
                // Either no overlap fits (not even a single character at
                // `previous_end`), or a non-monotone counter reported no
                // reduction. Retry from `previous_end`: if the character
                // there cannot fit, the next pass reports the explicit
                // `TokenBudgetTooSmall` error.
                _ => {
                    start_byte = previous_end_byte;
                    start_char = previous_end_char;
                }
            }
            debug_assert!(start_byte > previous_start_byte);
            continue;
        }

        // Prefer a paragraph/sentence boundary inside the window; fall back
        // to the hard end. End-of-document is a byte comparison: the window
        // may stop at `MAX_CHARS` while more input remains.
        let chosen = if window.byte(hard_end) == content.len() {
            hard_end
        } else {
            // Only boundaries strictly after the previous end and more
            // than `OVERLAP_CHARS` past this chunk's start are eligible.
            // The latter is the maximum target overlap; the overlap
            // scheduled for the next window is capped at half this chunk's
            // length, and when it does not fit the `hard_end` fallback is
            // used and the overlap is reduced.
            let first_eligible = (previous_end_local + 1).max(OVERLAP_CHARS + 1);
            choose_end(&window, hard_end, first_eligible)
        };

        // Verify the emitted chunk. `hard_end` is known to be within
        // budget, so it is the fallback if a preferred boundary is not
        // (possible only for a token counter that is not monotone in text
        // length). This guarantees the chunk never exceeds the budget.
        let end = if budget.count_tokens(&content[start_byte..window.byte(chosen)]) <= max_tokens {
            chosen
        } else {
            hard_end
        };
        let end_byte = window.byte(end);

        chunks.push(Chunk {
            start: start_byte,
            end: end_byte,
        });

        if end_byte == content.len() {
            break;
        }
        previous_end_byte = end_byte;
        previous_end_char = start_char + end;
        // Begin the next window by backing up over the overlap. The
        // scheduled overlap is capped at half this chunk's character
        // length, so the next window starts at least half a chunk later
        // (and a one-character chunk has no overlap at all). This bounds
        // the next start, not how far the next end advances: that still
        // depends on what the token counter and budget can fit. The
        // reduced-overlap path above applies when the budget cannot
        // afford even this much overlap.
        let chunk_len = end; // local window start is 0
        let overlap = OVERLAP_CHARS.min(chunk_len / 2);
        start_byte = window.byte(end - overlap);
        start_char = previous_end_char - overlap;
    }

    Ok(chunks)
}

/// Resolve `chunk` back to its slice of `content`.
pub fn chunk_text<'a>(content: &'a str, chunk: &Chunk) -> &'a str {
    &content[chunk.start..chunk.end]
}

/// A lazy, reusable view of at most `MAX_CHARS` characters starting at an
/// absolute UTF-8 byte offset. Only the boundaries needed by the current
/// window are materialized, so peak metadata is independent of document
/// length.
struct Window<'a> {
    content: &'a str,
    /// Absolute byte offset of this window's first character.
    base: usize,
    /// Byte offsets of every materialized char boundary, relative to `base`.
    /// `cuts[0]` is always zero; `cuts[local]` is the boundary before the
    /// char at local index `local`.
    cuts: Vec<usize>,
}

impl<'a> Window<'a> {
    fn new(content: &'a str) -> Self {
        Self {
            content,
            base: 0,
            cuts: Vec::with_capacity(MAX_CHARS + 1),
        }
    }

    /// Point the window at `base`, discarding all previously materialized
    /// boundaries while retaining the fixed-capacity allocation.
    fn reset(&mut self, base: usize) {
        debug_assert!(base <= self.content.len());
        debug_assert!(self.content.is_char_boundary(base));
        self.base = base;
        self.cuts.clear();
        self.cuts.push(0);
    }

    /// Number of characters materialized so far.
    fn chars(&self) -> usize {
        self.cuts.len() - 1
    }

    /// Absolute byte offset of local char boundary `local`.
    fn byte(&self, local: usize) -> usize {
        debug_assert!(local <= self.chars());
        self.base + self.cuts[local]
    }

    /// Materialize boundaries through local char index `local_end`, stopping
    /// at the end of the input. Returns the number of characters available
    /// in the window (at most `MAX_CHARS`).
    fn ensure_through(&mut self, local_end: usize) -> usize {
        debug_assert!(local_end <= MAX_CHARS);
        while self.chars() < local_end {
            let last_rel = *self.cuts.last().unwrap();
            let abs = self.base + last_rel;
            let Some(ch) = self.content[abs..].chars().next() else {
                break;
            };
            self.cuts.push(last_rel + ch.len_utf8());
        }
        debug_assert!(self.cuts.len() <= MAX_CHARS + 1);
        self.chars()
    }
}

/// Largest local char index `e`, `0 < e <= MAX_CHARS`, whose window
/// `content[base..byte(e)]` costs at most `max_tokens`, or `None` when even
/// the single character at the window start exceeds the budget.
///
/// The window is grown geometrically until it overflows, then narrowed by
/// binary search between a fitting and an overflowing length. The probe
/// sequence is unchanged from the whole-document implementation: 1, 2, 4,
/// ... up to the `MAX_CHARS` cap or the end of input, then binary search.
/// A token-sparse but very long input is never tokenized in full.
fn fitting_end(
    window: &mut Window<'_>,
    budget: &dyn TokenBudget,
    max_tokens: usize,
) -> Option<usize> {
    // The caller guarantees non-empty content.
    if window.ensure_through(1) == 0 {
        return None;
    }
    if budget.count_tokens(&window.content[window.byte(0)..window.byte(1)]) > max_tokens {
        return None;
    }

    // `fitted` is the largest known-fitting length; grow geometrically
    // until the window overflows or reaches the character cap.
    let mut fitted = 1usize;
    let mut probe = 1usize;
    loop {
        let target = probe.saturating_mul(2).min(MAX_CHARS);
        let available = window.ensure_through(target);
        if available <= probe {
            return Some(fitted);
        }
        let next = available;
        if budget.count_tokens(&window.content[window.byte(0)..window.byte(next)]) > max_tokens {
            // `fitted` fits and `next` overflows; narrow in between.
            let mut lo = fitted;
            let mut hi = next;
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if budget.count_tokens(&window.content[window.byte(0)..window.byte(mid)])
                    <= max_tokens
                {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return Some(lo);
        }
        fitted = next;
        probe = next;
    }
}

/// Reduce the overlap when the configured one does not fit the budget.
///
/// Returns the largest absolute byte start `s`, together with the candidate
/// width in characters, such that `content[s..tail_end]` fits `max_tokens`,
/// where `tail_end` is one character past `previous_end_byte`. The overlap
/// is the number of characters before `previous_end_byte`, so the largest
/// candidate has `OVERLAP_CHARS` preceding characters plus one character
/// beyond the previous end.
///
/// This search is intentionally independent of the current main window: it
/// may return a start before the main window's base. The caller's strict
/// `reduced_start > previous_start` guard rejects a result that does not
/// actually advance. Returns `None` when even the single character at
/// `previous_end` exceeds the budget.
fn largest_fitting_overlap(
    content: &str,
    previous_end_byte: usize,
    previous_end_char: usize,
    budget: &dyn TokenBudget,
    max_tokens: usize,
) -> Option<(usize, usize)> {
    debug_assert!(previous_end_char > 0);
    debug_assert!(content.is_char_boundary(previous_end_byte));

    let next_char_len = content[previous_end_byte..].chars().next()?.len_utf8();
    let tail_end = previous_end_byte + next_char_len;

    // A window of `w` chars ending at `previous_end + 1` has an overlap of
    // `w - 1` chars. `previous_end_char + 1` is the original global bound,
    // not a distance from the current window's start.
    let max_window = (OVERLAP_CHARS + 1).min(previous_end_char + 1);

    // Boundaries of the last `max_window` chars ending at `tail_end`, in
    // ascending absolute byte order. At most OVERLAP_CHARS + 2 entries.
    let mut bounds = [0usize; OVERLAP_CHARS + 2];
    let mut count = 1usize;
    bounds[0] = tail_end;
    let mut boundary = tail_end;
    while count <= max_window {
        let (previous, _) = content[..boundary].char_indices().next_back()?;
        boundary = previous;
        bounds[count] = boundary;
        count += 1;
    }
    bounds[..count].reverse();
    debug_assert_eq!(bounds[count - 1], tail_end);

    let fits =
        |w: usize| budget.count_tokens(&content[bounds[count - 1 - w]..tail_end]) <= max_tokens;

    // Preserve the original probe order exactly: one character, then the
    // maximum candidate, then binary-search midpoints.
    if !fits(1) {
        return None;
    }
    if fits(max_window) {
        return Some((bounds[count - 1 - max_window], max_window));
    }

    let mut lo = 1usize;
    let mut hi = max_window;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some((bounds[count - 1 - lo], lo))
}

/// A boundary between two chars that the splitter prefers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BreakKind {
    Paragraph,
    Sentence,
}

/// Pick the char index at which to cut the window `[first_eligible, hard_end]`.
///
/// Only boundaries at or after `first_eligible` are eligible. Callers pass
/// a value strictly past the previous end and more than `OVERLAP_CHARS`
/// after the current chunk's start, so a chunk never ends where the
/// previous chunk ended. The overlap scheduled for the next window is
/// capped at half the emitted chunk's length, so every non-final chunk is
/// longer than its overlap, which keeps the next start strictly advancing.
/// This guarantees start progress only. How far the next end advances
/// depends on the token counter and budget, and choosing an earlier
/// paragraph or sentence boundary here can make it less than half a chunk
/// even for a monotone counter. A paragraph boundary is preferred; if none
/// is present, a sentence boundary; if neither is present, `hard_end` (a
/// hard split at a character boundary, which is within budget).
fn choose_end(window: &Window<'_>, hard_end: usize, first_eligible: usize) -> usize {
    let mut sentence = None;
    // Scan backward so the first hit of each kind is its rightmost.
    for cut in (first_eligible..=hard_end).rev() {
        match break_kind_before(window.content, window.byte(cut)) {
            Some(BreakKind::Paragraph) => return cut,
            Some(BreakKind::Sentence) => {
                if sentence.is_none() {
                    sentence = Some(cut);
                }
            }
            None => {}
        }
    }
    sentence.unwrap_or(hard_end)
}

/// Classify the absolute byte boundary `cut` by the character just before it.
/// Using the whole source, rather than the current window, keeps newline runs
/// that cross a window base exactly as they were in the whole-document scan.
fn break_kind_before(content: &str, cut: usize) -> Option<BreakKind> {
    if cut == 0 || cut > content.len() {
        return None;
    }
    let prev = content[..cut].chars().next_back()?;
    if (prev == '\n' || prev == '\r') && line_break_count(content, cut) >= 2 {
        return Some(BreakKind::Paragraph);
    }
    if is_sentence_end(prev) {
        return Some(BreakKind::Sentence);
    }
    None
}

/// Number of logical line breaks in the run of `\r`/`\n` bytes ending just
/// before `end`, counting CRLF as a single break and capped at two. The
/// caller only distinguishes fewer than two from two or more, so the scan
/// stops as soon as a blank line is proved. `end` is an absolute byte
/// boundary.
fn line_break_count(content: &str, end: usize) -> usize {
    let bytes = content.as_bytes();
    let mut count = 0usize;
    let mut i = end;
    while i > 0 && count < 2 {
        match bytes[i - 1] {
            b'\n' => {
                count += 1;
                i -= 1;
            }
            b'\r' => {
                // A '\r' immediately followed by '\n' (toward `end`) is
                // the start of a CRLF pair already counted by that '\n'.
                if i < end && bytes[i] == b'\n' {
                    i -= 1;
                } else {
                    count += 1;
                    i -= 1;
                }
            }
            _ => break,
        }
    }
    count
}

/// Whether `c` commonly ends a sentence. Includes ASCII and full-width
/// CJK punctuation forms, plus the ellipsis.
fn is_sentence_end(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '。' | '！' | '？' | '…')
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use super::*;

    /// A `TokenBudget` charging one token per Unicode scalar value, so
    /// `max_tokens` behaves like the previous character cap. This is the
    /// worst case for character-counted tokens (a BERT-style token never
    /// spans less than one scalar), and it exercises the splitter without
    /// a real model.
    struct CharTokens(usize);

    impl TokenBudget for CharTokens {
        fn count_tokens(&self, text: &str) -> usize {
            text.chars().count()
        }
        fn max_tokens(&self) -> usize {
            self.0
        }
    }

    /// Like `CharTokens`, but also charges the two special tokens the
    /// sentence-embedding model adds to every input, so the budget counts
    /// specials exactly as the real model budget does.
    struct CharTokensWithSpecials(usize);

    impl TokenBudget for CharTokensWithSpecials {
        fn count_tokens(&self, text: &str) -> usize {
            text.chars().count() + 2
        }
        fn max_tokens(&self) -> usize {
            self.0
        }
    }

    /// A budget under which any text costs a single content token,
    /// mimicking a WordPiece tokenizer that maps an over-long, unbroken
    /// word to one `[UNK]`. The token budget never binds, so the secondary
    /// character bound is what limits chunk size.
    struct LongWordIsOneToken;

    impl TokenBudget for LongWordIsOneToken {
        fn count_tokens(&self, _text: &str) -> usize {
            3 // [CLS] + the word + [SEP]
        }
        fn max_tokens(&self) -> usize {
            512
        }
    }

    /// A deliberately non-monotone `TokenBudget`: a slice costs
    /// `budget + 1` when its length (in scalars) is listed in `overflowing`
    /// and one token otherwise, so longer text can be cheaper than shorter
    /// text. This is the pathological input that previously made the
    /// reduced-overlap fallback return the same start and spin forever. The
    /// internal `calls` counter turns any such non-termination into a fast
    /// assertion failure instead of a hang.
    struct NonMonotoneTokens {
        overflowing: &'static [usize],
        budget: usize,
        calls: Cell<usize>,
    }

    impl TokenBudget for NonMonotoneTokens {
        fn count_tokens(&self, text: &str) -> usize {
            let calls = self.calls.get() + 1;
            self.calls.set(calls);
            assert!(
                calls < 100_000,
                "chunker did not terminate (a start boundary was likely repeated)"
            );
            if self.overflowing.contains(&text.chars().count()) {
                self.budget + 1
            } else {
                1
            }
        }
        fn max_tokens(&self) -> usize {
            self.budget
        }
    }

    /// One token per scalar that records the character length of every
    /// probed slice. Used to pin the search order as well as the result.
    struct RecordingTokens {
        budget: usize,
        calls: RefCell<Vec<usize>>,
    }

    impl RecordingTokens {
        fn new(budget: usize) -> Self {
            Self {
                budget,
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl TokenBudget for RecordingTokens {
        fn count_tokens(&self, text: &str) -> usize {
            let len = text.chars().count();
            self.calls.borrow_mut().push(len);
            len
        }
        fn max_tokens(&self) -> usize {
            self.budget
        }
    }

    /// Budget used by the layout tests; matches the previous 800-character
    /// cap so their expectations are unchanged.
    const BUDGET: usize = 800;

    fn chunk(content: &str, budget: usize) -> Vec<Chunk> {
        chunk_content(content, &CharTokens(budget)).expect("chunking should succeed")
    }

    /// Assert every chunk is non-empty, on char boundaries, within both the
    /// character bound and the token budget, that each successive end
    /// strictly advances, that windows never open after the previous end,
    /// and that the chunks cover every character.
    fn assert_valid_and_covered(content: &str, budget: &dyn TokenBudget, chunks: &[Chunk]) {
        assert!(!chunks.is_empty(), "expected at least one chunk");
        let max_tokens = budget.max_tokens();
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(chunk.start < chunk.end, "chunk {i} is empty");
            assert!(chunk.end <= content.len(), "chunk {i} runs past the end");
            assert!(
                chunk_text(content, chunk).chars().count() <= MAX_CHARS,
                "chunk {i} exceeds the {MAX_CHARS}-character bound"
            );
            assert!(
                content.get(chunk.start..chunk.end).is_some(),
                "chunk {i} is not on char boundaries"
            );
            assert!(
                budget.count_tokens(chunk_text(content, chunk)) <= max_tokens,
                "chunk {i} exceeds the {max_tokens}-token budget"
            );
            if i > 0 {
                let previous = chunks[i - 1];
                assert!(
                    chunk.start <= previous.end,
                    "chunk {i} opens after the previous end"
                );
                assert!(
                    chunk.end > previous.end,
                    "chunk {i} end ({}) does not advance past the previous end ({})",
                    chunk.end,
                    previous.end
                );
            }
        }

        for (byte_start, ch) in content.char_indices() {
            let byte_end = byte_start + ch.len_utf8();
            assert!(
                chunks
                    .iter()
                    .any(|c| c.start <= byte_start && byte_end <= c.end),
                "char at bytes {byte_start}..{byte_end} is not covered"
            );
        }
    }

    #[test]
    fn empty_input_yields_no_chunks() {
        assert!(chunk("", BUDGET).is_empty());
    }

    #[test]
    fn short_input_returns_one_whole_content_chunk() {
        for content in ["a", "hello, world", "line one\nline two"] {
            let chunks = chunk(content, BUDGET);
            assert_eq!(chunks.len(), 1, "content: {content:?}");
            assert_eq!(
                chunks[0],
                Chunk {
                    start: 0,
                    end: content.len()
                }
            );
            assert_eq!(chunk_text(content, &chunks[0]), content);
        }
    }

    #[test]
    fn exactly_budget_chars_is_one_chunk() {
        let content = "a".repeat(BUDGET);
        let chunks = chunk(&content, BUDGET);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunk_text(&content, &chunks[0]), content);
    }

    #[test]
    fn one_char_over_budget_splits_into_two_chunks() {
        let content = "a".repeat(BUDGET + 1);
        let chunks = chunk(&content, BUDGET);
        assert_eq!(chunks.len(), 2);
        assert_valid_and_covered(&content, &CharTokens(BUDGET), &chunks);
    }

    #[test]
    fn prefers_a_paragraph_boundary() {
        // Paragraph break at char 502, inside the first 800-char window.
        let content = format!("{}\n\n{}", "A".repeat(500), "B".repeat(1000));
        let chunks = chunk(&content, BUDGET);
        let first = chunk_text(&content, &chunks[0]);
        assert_eq!(first, format!("{}\n\n", "A".repeat(500)));
        assert_valid_and_covered(&content, &CharTokens(BUDGET), &chunks);
    }

    #[test]
    fn crlf_blank_line_is_a_paragraph_boundary() {
        let content = format!("{}\r\n\r\n{}", "A".repeat(500), "B".repeat(1000));
        let chunks = chunk(&content, BUDGET);
        let first = chunk_text(&content, &chunks[0]);
        assert!(first.ends_with("\r\n\r\n"), "first chunk was {first:?}");
        assert_eq!(first, format!("{}\r\n\r\n", "A".repeat(500)));
        assert_valid_and_covered(&content, &CharTokens(BUDGET), &chunks);
    }

    #[test]
    fn prefers_a_sentence_boundary() {
        // Sentence end at char 701, inside the first 800-char window.
        let content = format!("{}. {}", "x".repeat(700), "y".repeat(1000));
        let chunks = chunk(&content, BUDGET);
        let first = chunk_text(&content, &chunks[0]);
        assert_eq!(first, format!("{}.", "x".repeat(700)));
        assert_valid_and_covered(&content, &CharTokens(BUDGET), &chunks);
    }

    #[test]
    fn early_paragraph_boundary_keeps_starts_and_ends_increasing() {
        // Paragraph break at char 42, well inside the first OVERLAP_CHARS
        // of the window, followed by a long tail. A naive overlap start
        // would compute 42 - OVERLAP_CHARS = 0 and repeat the first
        // chunk's start; the early boundary is therefore ineligible and
        // the window is split at the hard limit instead.
        let content = format!("{}\n\n{}", "A".repeat(40), "B".repeat(2000));
        let chunks = chunk(&content, BUDGET);
        assert!(chunks.len() > 1);

        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks.last().unwrap().end, content.len());

        // The early boundary cannot be used, so the first window fills to
        // the budget rather than cutting at char 42.
        assert_eq!(chunks[0].end - chunks[0].start, BUDGET);

        for pair in chunks.windows(2) {
            let (previous, current) = (pair[0], pair[1]);
            assert!(
                current.start > previous.start,
                "chunk start did not strictly increase: {previous:?} -> {current:?}"
            );
            assert!(
                current.end > previous.end,
                "chunk end did not strictly increase: {previous:?} -> {current:?}"
            );
            // Neighbouring chunks overlap by the target amount.
            assert_eq!(
                previous.end - current.start,
                OVERLAP_CHARS,
                "unexpected overlap: {previous:?} -> {current:?}"
            );
        }

        // UTF-8 safety, the token budget, and full coverage.
        assert_valid_and_covered(&content, &CharTokens(BUDGET), &chunks);
    }

    #[test]
    fn repeated_paragraph_boundary_is_not_reused_as_a_nested_chunk() {
        // The only paragraph break is at char 502, so it fills the first
        // window. The next window must not cut there again.
        let content = format!("{}\n\n{}", "A".repeat(500), "B".repeat(2000));
        let chunks = chunk(&content, BUDGET);
        assert!(chunks.len() >= 2);

        let mut previous_end = 0usize;
        for (i, chunk) in chunks.iter().enumerate() {
            if i > 0 {
                assert!(
                    chunk.start < previous_end,
                    "chunk {i} does not reach back to the previous end"
                );
                assert!(
                    chunk.end > previous_end,
                    "chunk {i} reuses the previous end {previous_end} (redundant chunk)"
                );
            }
            previous_end = chunk.end;
        }
        // The paragraph boundary is used exactly once, as the first end.
        assert_eq!(
            chunks.iter().filter(|c| c.end == 502).count(),
            1,
            "paragraph boundary reused: {chunks:?}"
        );
        assert_valid_and_covered(&content, &CharTokens(BUDGET), &chunks);
    }

    #[test]
    fn repeated_sentence_boundary_is_not_reused_as_a_nested_chunk() {
        // The only sentence end is at char 701, so it fills the first
        // window. The next window must not cut there again.
        let content = format!("{}.{}", "x".repeat(700), "y".repeat(2000));
        let chunks = chunk(&content, BUDGET);
        assert!(chunks.len() >= 2);

        let mut previous_end = 0usize;
        for (i, chunk) in chunks.iter().enumerate() {
            if i > 0 {
                assert!(
                    chunk.start < previous_end,
                    "chunk {i} does not reach back to the previous end"
                );
                assert!(
                    chunk.end > previous_end,
                    "chunk {i} reuses the previous end {previous_end} (redundant chunk)"
                );
            }
            previous_end = chunk.end;
        }
        // The sentence boundary is used exactly once, as the first end.
        assert_eq!(
            chunks.iter().filter(|c| c.end == 701).count(),
            1,
            "sentence boundary reused: {chunks:?}"
        );
        assert_valid_and_covered(&content, &CharTokens(BUDGET), &chunks);
    }

    #[test]
    fn long_line_is_hard_split_at_char_boundaries() {
        let content = "z".repeat(2500);
        let chunks = chunk(&content, BUDGET);
        assert!(chunks.len() >= 4);
        assert_valid_and_covered(&content, &CharTokens(BUDGET), &chunks);
    }

    #[test]
    fn overlap_is_targeted_between_neighbours() {
        // A long, unbroken line is hard-split, so each split shares
        // exactly OVERLAP_CHARS of context with the previous chunk.
        let content = "z".repeat(2500);
        let chunks = chunk(&content, BUDGET);
        for pair in chunks.windows(2) {
            assert_eq!(
                pair[0].end - pair[1].start,
                OVERLAP_CHARS,
                "unexpected overlap between {:?} and {:?}",
                pair[0],
                pair[1]
            );
        }
        // The overlapped region is byte-identical when resolved through
        // either neighbouring chunk.
        for pair in chunks.windows(2) {
            let previous = chunk_text(&content, &pair[0]);
            let current = chunk_text(&content, &pair[1]);
            assert_eq!(
                &previous[previous.len() - OVERLAP_CHARS..],
                &current[..OVERLAP_CHARS]
            );
        }
    }

    #[test]
    fn short_tail_is_kept_as_its_own_chunk_within_max() {
        // The budget fills the first chunk; the 30-char tail must not be
        // merged back past the budget.
        let content = "w".repeat(BUDGET + 30);
        let chunks = chunk(&content, BUDGET);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].end - chunks[0].start, BUDGET);
        assert_eq!(chunks[1].end, content.len());
        assert!(chunks[1].end - chunks[1].start <= BUDGET);
        assert_valid_and_covered(&content, &CharTokens(BUDGET), &chunks);
    }

    #[test]
    fn unicode_offsets_are_char_boundaries_and_covered() {
        let content = "é😀漢字🌍ß ".repeat(400); // far beyond the budget
        let chunks = chunk(&content, BUDGET);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(
                content.get(chunk.start..chunk.end).is_some(),
                "offset pair not on char boundaries: {chunk:?}"
            );
        }
        assert_valid_and_covered(&content, &CharTokens(BUDGET), &chunks);
    }

    #[test]
    fn output_is_deterministic() {
        let content = format!("{}\n\n{}", "sentence. ".repeat(200), "tail".repeat(50));
        let first = chunk(&content, BUDGET);
        for _ in 0..5 {
            assert_eq!(chunk(&content, BUDGET), first);
        }
    }

    #[test]
    fn chunk_text_resolves_offset_pairs() {
        let content = "abcdefghij".repeat(200);
        let chunks = chunk(&content, BUDGET);
        let mut rebuilt = String::new();
        for chunk in &chunks {
            rebuilt.push_str(chunk_text(&content, chunk));
        }
        assert_valid_and_covered(&content, &CharTokens(BUDGET), &chunks);
        // First chunk starts at the content start and the last reaches
        // the content end, so coverage spans the whole document.
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks.last().unwrap().end, content.len());
        assert!(!rebuilt.is_empty());
    }

    #[test]
    fn token_budget_bounds_every_chunk() {
        // One token per char: a chunk may hold at most `budget` chars,
        // which is far below the previous 800-character cap.
        let budget = 300usize;
        let content = "a".repeat(2000);
        let chunks = chunk(&content, budget);
        assert!(chunks.len() > 1);
        assert_valid_and_covered(&content, &CharTokens(budget), &chunks);
        for c in &chunks {
            assert!(chunk_text(&content, c).chars().count() <= budget);
        }
    }

    #[test]
    fn special_tokens_count_against_the_budget() {
        // The model charges two special tokens, so content capacity is two
        // tokens below `max_tokens`. The counter must account for them.
        let budget = 302usize;
        let counter = CharTokensWithSpecials(budget);
        let content = "a".repeat(2000);
        let chunks = chunk_content(&content, &counter).expect("chunking should succeed");
        assert!(chunks.len() > 1);
        assert_valid_and_covered(&content, &counter, &chunks);
        for c in &chunks {
            assert!(chunk_text(&content, c).chars().count() + 2 <= budget);
        }
    }

    #[test]
    fn token_dense_punctuation_stays_within_budget() {
        // BERT-style tokenizers isolate punctuation, so each symbol here
        // costs about one token: token-dense text the old char cap could
        // not bound.
        let text = "\"{\",:\"[]\";(){}<>=+*/%!?@#$^&|~`".repeat(60);
        let budget = 256usize;
        let counter = CharTokensWithSpecials(budget);
        let chunks = chunk_content(&text, &counter).expect("chunking should succeed");
        assert!(chunks.len() > 1);
        assert_valid_and_covered(&text, &counter, &chunks);
    }

    #[test]
    fn code_like_text_stays_within_budget() {
        let text = "fn main(){let x=1+2;if x>0{println!(\"{}\",x);}}".repeat(40);
        let budget = 200usize;
        let counter = CharTokensWithSpecials(budget);
        let chunks = chunk_content(&text, &counter).expect("chunking should succeed");
        assert!(!chunks.is_empty());
        assert_valid_and_covered(&text, &counter, &chunks);
    }

    #[test]
    fn multilingual_text_stays_within_budget() {
        // Mixed scripts, including CJK that the model's normalizer spaces
        // out so each ideograph tends to cost a token.
        let text = "漢字かなカナ한글é😀ß ".repeat(120);
        let budget = 256usize;
        let counter = CharTokensWithSpecials(budget);
        let chunks = chunk_content(&text, &counter).expect("chunking should succeed");
        assert!(chunks.len() > 1);
        assert_valid_and_covered(&text, &counter, &chunks);
    }

    #[test]
    fn budget_boundary_exactly_and_one_over() {
        let budget = 137usize;
        let exact = "b".repeat(budget);
        let exact_chunks = chunk(&exact, budget);
        assert_eq!(exact_chunks.len(), 1);
        assert_eq!(chunk_text(&exact, &exact_chunks[0]), exact);

        let over = "b".repeat(budget + 1);
        let over_chunks = chunk(&over, budget);
        assert!(over_chunks.len() >= 2);
        assert_valid_and_covered(&over, &CharTokens(budget), &over_chunks);
    }

    #[test]
    fn budget_larger_than_overlap_covers_everything_with_overlap() {
        let budget = 200usize; // > OVERLAP_CHARS, so the normal path runs
        let content = "word ".repeat(500);
        let chunks = chunk(&content, budget);
        assert_valid_and_covered(&content, &CharTokens(budget), &chunks);
        // Neighbouring chunks still overlap.
        for pair in chunks.windows(2) {
            assert!(
                pair[1].start < pair[0].end,
                "no overlap between {:?} and {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn every_chunk_fits_the_token_budget() {
        // Cross-check the core invariant directly on a mix of dense,
        // multilingual and structured inputs.
        let cases = [
            "a".repeat(3000),
            "!@#$%^&*()_+-=[]{}|;:',.<>/?".repeat(100),
            "漢字😀éß한글".repeat(300),
            "line one\n\nline two. more text here!\n".repeat(200),
        ];
        let budget = 256usize;
        let counter = CharTokensWithSpecials(budget);
        for content in cases {
            let chunks = chunk_content(&content, &counter).expect("chunking should succeed");
            assert_valid_and_covered(&content, &counter, &chunks);
        }
    }

    #[test]
    fn token_sparse_long_word_is_bounded_by_character_cap() {
        // A single unbroken "word" many times longer than `MAX_CHARS`. The
        // token budget never binds, so only the secondary character bound
        // can stop the whole document becoming one giant chunk.
        let content = "a".repeat(MAX_CHARS * 3 + 7);
        let counter = LongWordIsOneToken;
        let chunks = chunk_content(&content, &counter).expect("chunking should succeed");
        assert!(
            chunks.len() >= 3,
            "character cap did not split the long word"
        );
        for chunk in &chunks {
            assert!(chunk_text(&content, chunk).chars().count() <= MAX_CHARS);
        }
        assert_valid_and_covered(&content, &counter, &chunks);
    }

    #[test]
    fn token_sparse_multibyte_run_is_bounded_and_char_aligned() {
        // Same idea with multi-byte scalars, to exercise UTF-8 safety on
        // the cap-driven path.
        let content = "é😀漢".repeat(MAX_CHARS);
        let counter = LongWordIsOneToken;
        let chunks = chunk_content(&content, &counter).expect("chunking should succeed");
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(chunk_text(&content, chunk).chars().count() <= MAX_CHARS);
            assert!(
                content.get(chunk.start..chunk.end).is_some(),
                "offset pair not on char boundaries: {chunk:?}"
            );
        }
        assert_valid_and_covered(&content, &counter, &chunks);
    }

    #[test]
    fn budget_smaller_than_overlap_adapts_overlap() {
        // The configured overlap (OVERLAP_CHARS) cannot fit a budget this
        // small, so the chunker must reduce it while still advancing.
        let budget = 50usize; // < OVERLAP_CHARS
        let content = "q".repeat(200);
        let chunks = chunk(&content, budget);
        assert!(chunks.len() > 1);
        assert_valid_and_covered(&content, &CharTokens(budget), &chunks);
        for pair in chunks.windows(2) {
            let (previous, current) = (pair[0], pair[1]);
            // Overlap is reduced but not discarded, and chunks advance.
            assert!(current.start < previous.end, "lost all overlap");
            assert!(current.end > previous.end, "no forward progress");
            assert!(
                previous.end - current.start <= OVERLAP_CHARS,
                "overlap exceeded the configured target"
            );
        }
    }

    #[test]
    fn small_budget_multibyte_covers_and_progresses() {
        // A budget below the configured overlap combined with multi-byte
        // input: sizes must adapt while boundaries stay UTF-8-safe.
        let budget = 40usize;
        let content = "漢é😀ß ".repeat(80);
        let chunks = chunk(&content, budget);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(content.get(chunk.start..chunk.end).is_some());
        }
        assert_valid_and_covered(&content, &CharTokens(budget), &chunks);
        for pair in chunks.windows(2) {
            assert!(pair[1].start < pair[0].end);
            assert!(pair[1].end > pair[0].end);
        }
    }

    #[test]
    fn small_budgets_bound_overlap_and_keep_coverage() {
        // For every budget on either side of OVERLAP_CHARS, the overlap
        // into the next chunk must not exceed either the configured target
        // or half of the chunk just emitted. This is what stops a small
        // budget from re-reading all but a few characters of every chunk.
        let content = "q".repeat(4000);
        let budgets = [1usize, 2, 49, 50, 119, 120, 121, 122, 200, 800];

        for budget in budgets {
            let counter = CharTokens(budget);
            let chunks = chunk_content(&content, &counter).expect("chunking should succeed");
            assert_valid_and_covered(&content, &counter, &chunks);

            for pair in chunks.windows(2) {
                let (previous, current) = (pair[0], pair[1]);
                let previous_len = content[previous.start..previous.end].chars().count();
                let overlap = content[current.start..previous.end].chars().count();

                assert!(
                    current.start > previous.start,
                    "budget {budget}: start repeated"
                );
                assert!(
                    current.end > previous.end,
                    "budget {budget}: end did not advance"
                );
                assert!(
                    overlap <= OVERLAP_CHARS,
                    "budget {budget}: overlap {overlap} exceeds OVERLAP_CHARS"
                );
                assert!(
                    overlap <= previous_len / 2,
                    "budget {budget}: overlap {overlap} exceeds half of {previous_len}"
                );
            }
        }
    }

    #[test]
    fn small_budgets_keep_chunk_count_and_progress_bounded() {
        // Test-scoped bound: only for this boundary-free fixture under the
        // monotone one-token-per-character counter does the half-overlap cap
        // make each non-final end advance by at least half of `capacity`.
        // Paragraph/sentence boundary selection can make end progress
        // smaller even with a monotone counter, so this is not a general
        // chunker guarantee. That bound makes the chunk count roughly twice
        // the ideal `n / capacity`, instead of one chunk per character when
        // the budget is at or just above OVERLAP_CHARS.
        let content = "q".repeat(4000);
        let char_count = content.chars().count();
        let budgets = [1usize, 2, 49, 50, 119, 120, 121, 122, 200, 800];

        for budget in budgets {
            let chunks = chunk(&content, budget);
            let capacity = budget.min(MAX_CHARS).max(1);
            let min_progress = (capacity + 1) / 2;
            let bound = (char_count + min_progress - 1) / min_progress + 2;

            assert!(
                chunks.len() <= bound,
                "budget {budget}: {} chunks exceeds the bound {bound}",
                chunks.len()
            );
            // The transition into the final chunk may be shorter than a
            // whole step because the tail is whatever remains.
            for pair in chunks.windows(2).take(chunks.len().saturating_sub(2)) {
                let progress = pair[1].end - pair[0].end;
                assert!(
                    progress >= min_progress,
                    "budget {budget}: end advanced by {progress}, expected at least {min_progress}"
                );
            }
        }
    }

    #[test]
    fn small_budget_multibyte_progress_is_bounded() {
        // The same test-scoped progress check on boundary-free mixed-width
        // UTF-8 under the monotone CharTokens counter: overlap and progress
        // are measured in characters while offsets stay in bytes.
        let budget = 40usize;
        let content = "漢é😀ß ".repeat(800);
        let char_count = content.chars().count();
        let chunks = chunk(&content, budget);
        let capacity = budget.min(MAX_CHARS).max(1);
        let min_progress = (capacity + 1) / 2;
        let bound = (char_count + min_progress - 1) / min_progress + 2;

        assert!(chunks.len() > 1);
        assert!(
            chunks.len() <= bound,
            "{} chunks exceeds the bound {bound}",
            chunks.len()
        );
        assert_valid_and_covered(&content, &CharTokens(budget), &chunks);

        for pair in chunks.windows(2) {
            let (previous, current) = (pair[0], pair[1]);
            let previous_len = content[previous.start..previous.end].chars().count();
            let overlap = content[current.start..previous.end].chars().count();

            assert!(current.start > previous.start, "start repeated");
            assert!(current.end > previous.end, "end did not advance");
            assert!(
                overlap <= OVERLAP_CHARS,
                "overlap {overlap} exceeds OVERLAP_CHARS"
            );
            assert!(
                overlap <= previous_len / 2,
                "overlap {overlap} exceeds half of {previous_len}"
            );
        }
        for pair in chunks.windows(2).take(chunks.len().saturating_sub(2)) {
            let progress = content[pair[0].end..pair[1].end].chars().count();
            assert!(
                progress >= min_progress,
                "end advanced by {progress}, expected at least {min_progress}"
            );
        }
    }

    #[test]
    fn non_monotone_budget_terminates_and_covers_everything() {
        // The token counter is not monotone in text length, so
        // `largest_fitting_overlap` can be asked whether a longer window
        // fits and answer yes even though `fitting_end` stopped earlier.
        // Before the fallback guard this returned the current start and
        // looped forever; now it must drop the overlap and still cover every
        // character, in budget, with strictly advancing boundaries.
        const OVERFLOWING: [usize; 5] = [4, 6, 8, 9, 14];
        let content = "x".repeat(14);
        let budget = NonMonotoneTokens {
            overflowing: &OVERFLOWING,
            budget: 2,
            calls: Cell::new(0),
        };
        let chunks = chunk_content(&content, &budget).expect("chunking should terminate");
        assert!(chunks.len() > 1);
        assert_valid_and_covered(&content, &budget, &chunks);
        for pair in chunks.windows(2) {
            assert!(pair[1].end > pair[0].end, "no forward progress");
            assert!(pair[1].start > pair[0].start, "a start boundary repeated");
        }
    }

    #[test]
    fn golden_chunk_layout_matches_the_pre_refactor_chunker() {
        // Captured from the whole-document implementation with the
        // half-overlap cap in place, before boundary metadata was windowed.
        let mixed = format!(
            "{}\n\n{}. {}\r\n\r\n{}",
            "A".repeat(500),
            "x".repeat(700),
            "sentence tail",
            "漢é😀ß ".repeat(300),
        );
        assert_eq!(
            chunk(&mixed, 800),
            vec![
                Chunk { start: 0, end: 502 },
                Chunk {
                    start: 382,
                    end: 1182
                },
                Chunk {
                    start: 1062,
                    end: 1221
                },
                Chunk {
                    start: 1142,
                    end: 2952
                },
                Chunk {
                    start: 2664,
                    end: 4584
                },
                Chunk {
                    start: 4296,
                    end: 4821
                },
            ]
        );
        assert_valid_and_covered(&mixed, &CharTokens(800), &chunk(&mixed, 800));

        let small = format!(
            "{}\n\n{}. {}\r\n\r\n{}",
            "word ".repeat(40),
            "y".repeat(150),
            "end",
            "漢é😀ß ".repeat(80),
        );
        assert_eq!(
            chunk(&small, 137),
            vec![
                Chunk { start: 0, end: 137 },
                Chunk {
                    start: 69,
                    end: 202
                },
                Chunk {
                    start: 136,
                    end: 273
                },
                Chunk {
                    start: 205,
                    end: 342
                },
                Chunk {
                    start: 274,
                    end: 481
                },
                Chunk {
                    start: 343,
                    end: 648
                },
                Chunk {
                    start: 484,
                    end: 814
                },
                Chunk {
                    start: 649,
                    end: 978
                },
                Chunk {
                    start: 816,
                    end: 1144
                },
                Chunk {
                    start: 982,
                    end: 1309
                },
                Chunk {
                    start: 1146,
                    end: 1321
                },
            ]
        );
        assert_valid_and_covered(&small, &CharTokens(137), &chunk(&small, 137));
    }

    #[test]
    fn max_chars_boundary_is_exact_and_one_over_splits() {
        let exact = "a".repeat(MAX_CHARS);
        let counter = LongWordIsOneToken;
        assert_eq!(
            chunk_content(&exact, &counter).expect("chunking should succeed"),
            vec![Chunk {
                start: 0,
                end: MAX_CHARS
            }]
        );

        let over = "a".repeat(MAX_CHARS + 1);
        let chunks = chunk_content(&over, &counter).expect("chunking should succeed");
        assert_eq!(
            chunks,
            vec![
                Chunk {
                    start: 0,
                    end: MAX_CHARS
                },
                Chunk {
                    start: MAX_CHARS - OVERLAP_CHARS,
                    end: MAX_CHARS + 1
                },
            ]
        );
        assert_valid_and_covered(&over, &counter, &chunks);
    }

    #[test]
    fn max_chars_multibyte_boundary_is_char_aligned() {
        let exact = "漢".repeat(MAX_CHARS);
        let counter = LongWordIsOneToken;
        let chunks = chunk_content(&exact, &counter).expect("chunking should succeed");
        assert_eq!(
            chunks,
            vec![Chunk {
                start: 0,
                end: exact.len()
            }]
        );

        let over = "漢".repeat(MAX_CHARS + 1);
        let chunks = chunk_content(&over, &counter).expect("chunking should succeed");
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[0].end, MAX_CHARS * "漢".len());
        assert_eq!(chunks[1].start, (MAX_CHARS - OVERLAP_CHARS) * "漢".len());
        assert_eq!(chunks[1].end, over.len());
        assert_valid_and_covered(&over, &counter, &chunks);
    }

    #[test]
    fn paragraph_run_crossing_a_window_start_is_classified_absolutely() {
        // The second window opens inside a long run of newlines. The run's
        // blank-line boundary at char 800 is still eligible and must win.
        let content = format!(
            "{}{}{}",
            "A".repeat(500),
            "\n".repeat(300),
            "B".repeat(1000)
        );
        let chunks = chunk(&content, 670);
        assert_eq!(chunks[0], Chunk { start: 0, end: 670 });
        assert_eq!(chunks[1].start, 550);
        assert_eq!(chunks[1].end, 800);
        assert_valid_and_covered(&content, &CharTokens(670), &chunks);
    }

    #[test]
    fn crlf_run_crossing_a_window_start_is_classified_absolutely() {
        // Same layout with CRLF pairs: the run crossing the second window's
        // start must count as a single logical paragraph break per pair.
        let content = format!(
            "{}{}{}",
            "A".repeat(500),
            "\r\n".repeat(150),
            "B".repeat(1000)
        );
        let chunks = chunk(&content, 670);
        assert_eq!(chunks[0], Chunk { start: 0, end: 670 });
        assert_eq!(chunks[1].start, 550);
        assert_eq!(chunks[1].end, 800);
        assert_valid_and_covered(&content, &CharTokens(670), &chunks);
    }

    #[test]
    fn long_line_runs_are_classified_with_a_bounded_scan() {
        // Beyond the second break the helper must saturate at two. The old
        // implementation returned the full prefix length at every offset.
        let lf = "\n".repeat(1_000_000);
        for end in [2usize, 3, 1_000, 500_000, lf.len()] {
            assert_eq!(line_break_count(&lf, end), 2, "LF run ending at {end}");
        }

        let crlf = "\r\n".repeat(500_000);
        for end in [4usize, 6, 1_000, 500_000, crlf.len()] {
            assert_eq!(line_break_count(&crlf, end), 2, "CRLF run ending at {end}");
        }

        // One logical break is still not a paragraph; CRLF counts once.
        assert_eq!(line_break_count("\n", 1), 1);
        assert_eq!(line_break_count("\r\n", 2), 1);
        assert_eq!(line_break_count("\r\n\r\n", 4), 2);
        assert_eq!(line_break_count("\r\n", 1), 1); // cut between CR and LF

        assert_eq!(
            break_kind_before(&lf, lf.len()),
            Some(BreakKind::Paragraph)
        );
        assert_eq!(
            break_kind_before(&crlf, crlf.len()),
            Some(BreakKind::Paragraph)
        );
        assert_eq!(break_kind_before("\n", 1), None);
        assert_eq!(break_kind_before("\r\n", 2), None);
    }

    #[test]
    fn non_monotone_counter_near_the_max_chars_edge_terminates_and_covers() {
        // The first window stops one character before `MAX_CHARS` because
        // that exact probe overflows. Later probes force the
        // reduced-overlap path near the window edge; the non-monotone
        // counter must not cause a repeated start or a hang.
        const OVERFLOWING: [usize; 3] = [121, 122, MAX_CHARS];
        let content = "x".repeat(MAX_CHARS + 1);
        let budget = NonMonotoneTokens {
            overflowing: &OVERFLOWING,
            budget: 2,
            calls: Cell::new(0),
        };
        let chunks = chunk_content(&content, &budget).expect("chunking should terminate");
        assert!(chunks.len() > 2);
        assert_valid_and_covered(&content, &budget, &chunks);
        for pair in chunks.windows(2) {
            assert!(pair[1].start > pair[0].start, "a start boundary repeated");
            assert!(pair[1].end > pair[0].end, "no forward progress");
        }
    }

    #[test]
    fn overlap_search_keeps_the_global_candidate_range_and_probe_order() {
        // `previous_end_char` is 60, so the original search may look back 60
        // characters even if the current main window starts much later.
        let content = "a".repeat(200);
        let budget = RecordingTokens::new(30);
        let reduced = largest_fitting_overlap(&content, 60, 60, &budget, 30);
        assert_eq!(reduced, Some((31, 30)));
        assert_eq!(
            budget.calls.borrow().as_slice(),
            &[1, 61, 31, 16, 23, 27, 29, 30]
        );
    }

    #[test]
    fn overlap_search_caps_the_candidate_width_at_overlap_chars_plus_one() {
        let content = "a".repeat(600);
        let budget = RecordingTokens::new(10);
        let reduced = largest_fitting_overlap(&content, 500, 500, &budget, 10);
        let (start, width) = reduced.expect("single characters fit");
        assert!(width <= OVERLAP_CHARS + 1);
        assert_eq!(&budget.calls.borrow()[..2], &[1, OVERLAP_CHARS + 1]);
        assert_eq!(start, 500 + 1 - width);
    }

    #[test]
    fn fitting_end_keeps_the_geometric_and_binary_probe_order() {
        let content = "a".repeat(200);
        let budget = RecordingTokens::new(30);
        let mut window = Window::new(&content);
        window.reset(0);
        assert_eq!(fitting_end(&mut window, &budget, 30), Some(30));
        assert_eq!(
            budget.calls.borrow().as_slice(),
            &[1, 2, 4, 8, 16, 32, 24, 28, 30, 31]
        );
    }

    #[test]
    fn budget_too_small_to_hold_one_character_is_an_error() {
        /// A budget that cannot hold even a single character.
        struct ZeroBudget;

        impl TokenBudget for ZeroBudget {
            fn count_tokens(&self, text: &str) -> usize {
                text.chars().count()
            }
            fn max_tokens(&self) -> usize {
                0
            }
        }

        let err = chunk_content("abc", &ZeroBudget).expect_err("expected an error");
        assert_eq!(err.max_tokens, 0);
        assert_eq!(err.offset, 0);
    }

    #[test]
    fn token_budget_too_small_reports_the_absolute_byte_offset() {
        /// One token per `a`; any `X` costs more than the whole budget.
        struct ExpensiveX;

        impl TokenBudget for ExpensiveX {
            fn count_tokens(&self, text: &str) -> usize {
                if text.chars().all(|ch| ch == 'a') {
                    text.chars().count()
                } else {
                    100
                }
            }
            fn max_tokens(&self) -> usize {
                2
            }
        }

        let content = "aaaX";
        let err = chunk_content(content, &ExpensiveX).expect_err("expected an error");
        assert_eq!(err.max_tokens, 2);
        assert_eq!(err.offset, 3);
        assert!(content.is_char_boundary(err.offset));
    }
}
