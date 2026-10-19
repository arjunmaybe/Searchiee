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
//   * UTF-8-safe. Cuts are taken only at `char` boundaries, so every
//     chunk is a valid UTF-8 slice and no code point is split.
//   * Complete coverage. Every character appears in at least one chunk.
//   * Intentional overlap. Neighbouring chunks share up to `OVERLAP_CHARS`
//     characters. If the configured overlap cannot fit the budget it is
//     reduced to the largest affordable value (possibly zero) so that
//     chunks still overlap where possible and always advance.
//   * Forward progress. Each chunk's end strictly advances past the
//     previous end: a boundary already used as the previous chunk's end
//     is never reused, so a finite input yields a finite, non-repeating
//     chunk list with no redundant nested chunks.
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
    // Byte offset of every char boundary, plus the end offset.
    let mut cuts: Vec<usize> = content.char_indices().map(|(i, _)| i).collect();
    cuts.push(content.len());
    let char_count = cuts.len() - 1;
    if char_count == 0 {
        return Ok(Vec::new());
    }

    let chars: Vec<char> = content.chars().collect();
    let max_tokens = budget.max_tokens();
    let mut chunks = Vec::new();
    let mut start = 0usize; // char index where the next window begins
    let mut previous_end = 0usize; // previous chunk's end, in chars

    loop {
        // The largest window from `start` allowed by the token budget,
        // additionally capped by the secondary character bound.
        let cap = (start + MAX_CHARS).min(char_count);
        let hard_end = match fitting_end(content, &cuts, start, cap, budget, max_tokens) {
            Some(end) => end,
            None => {
                return Err(TokenBudgetTooSmall {
                    max_tokens,
                    offset: cuts[start],
                })
            }
        };

        // `hard_end` is within budget and the cap. If it does not pass the
        // previous end, the configured overlap does not fit this budget, so
        // shrink the overlap. `largest_fitting_overlap` normally returns a
        // start strictly after the current one; a non-monotone token counter
        // can, however, make it report no reduction even though a longer
        // window fits. Reusing `start` would then spin forever, so fall back
        // to `previous_end` and drop the overlap for this window. Because
        // `hard_end >= start + 1` and `hard_end <= previous_end` here,
        // `previous_end > start`, so this always strictly advances `start`.
        if hard_end <= previous_end {
            let previous_start = start;
            let reduced =
                largest_fitting_overlap(content, &cuts, start, previous_end, budget, max_tokens);
            start = match reduced {
                // The overlap shrank to a start strictly after the current
                // one; keep it.
                Some(reduced_start) if reduced_start > previous_start => reduced_start,
                // Either no overlap fits (not even a single character at
                // `previous_end`), or a non-monotone counter reported no
                // reduction. Retry from `previous_end`: if the character
                // there cannot fit, the next pass reports the explicit
                // `TokenBudgetTooSmall` error.
                _ => previous_end,
            };
            debug_assert!(start > previous_start);
            continue;
        }

        // Prefer a paragraph/sentence boundary inside the window; fall back
        // to the hard end.
        let chosen = if hard_end == char_count {
            char_count
        } else {
            // Only boundaries strictly after the previous end and more
            // than `OVERLAP_CHARS` past this chunk's start are eligible.
            // The latter is the target overlap; when it does not fit, the
            // `hard_end` fallback is used and the overlap is reduced.
            let first_eligible = (previous_end + 1).max(start + OVERLAP_CHARS + 1);
            choose_end(&chars, hard_end, first_eligible)
        };

        // Verify the emitted chunk. `hard_end` is known to be within
        // budget, so it is the fallback if a preferred boundary is not
        // (possible only for a token counter that is not monotone in text
        // length). This guarantees the chunk never exceeds the budget.
        let end = if budget.count_tokens(&content[cuts[start]..cuts[chosen]]) <= max_tokens {
            chosen
        } else {
            hard_end
        };

        chunks.push(Chunk {
            start: cuts[start],
            end: cuts[end],
        });

        if end == char_count {
            break;
        }
        previous_end = end;
        // Begin the next window `OVERLAP_CHARS` before this chunk's end.
        // The reduced-overlap path above adjusts `start` further forward
        // when the budget cannot afford this much overlap.
        start = end.saturating_sub(OVERLAP_CHARS);
    }

    Ok(chunks)
}

/// Resolve `chunk` back to its slice of `content`.
pub fn chunk_text<'a>(content: &'a str, chunk: &Chunk) -> &'a str {
    &content[chunk.start..chunk.end]
}

/// Largest char index `e` with `start < e <= cap` whose window
/// `content[cuts[start]..cuts[e]]` costs at most `max_tokens`, or `None`
/// when even the single character at `start` exceeds the budget.
///
/// The window is grown geometrically until it overflows, then narrowed by
/// binary search between a fitting and an overflowing length, so the work
/// per chunk is proportional to the chunk size rather than to the
/// remaining document. `cap` bounds the search (the token budget and the
/// secondary character bound combined), so a token-sparse but very long
/// input is never tokenized in full.
fn fitting_end(
    content: &str,
    cuts: &[usize],
    start: usize,
    cap: usize,
    budget: &dyn TokenBudget,
    max_tokens: usize,
) -> Option<usize> {
    // The caller guarantees `cap > start`, so there is at least one char.
    let count_upto = |end: usize| budget.count_tokens(&content[cuts[start]..cuts[end]]);

    if count_upto(start + 1) > max_tokens {
        return None;
    }

    // `fitted` is the largest known-fitting length; grow geometrically
    // until the window overflows or reaches `cap`.
    let mut fitted = start + 1;
    let mut probe = start + 1;
    while probe < cap {
        let next = (start + (probe - start).saturating_mul(2)).min(cap);
        if count_upto(next) > max_tokens {
            // `fitted` fits and `next` overflows; narrow in between.
            let mut lo = fitted;
            let mut hi = next;
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if count_upto(mid) <= max_tokens {
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
    Some(fitted)
}

/// Reduce the overlap when the configured one does not fit the budget.
///
/// Returns the largest start `s` with `start < s <= previous_end` such that
/// the window `content[s..previous_end + 1]` fits `max_tokens`, i.e. the
/// largest affordable overlap (`previous_end - s`), capped at
/// `OVERLAP_CHARS`. The returned start therefore always lets the next
/// chunk advance past `previous_end` within budget. Returns `None` when
/// even the single character at `previous_end` exceeds the budget, so no
/// in-budget chunk can advance.
fn largest_fitting_overlap(
    content: &str,
    cuts: &[usize],
    start: usize,
    previous_end: usize,
    budget: &dyn TokenBudget,
    max_tokens: usize,
) -> Option<usize> {
    debug_assert!(start < previous_end);

    // A window of `w` chars ending at `previous_end + 1` has an overlap of
    // `w - 1` chars; grow `w` up to the configured overlap plus one.
    let max_window = (OVERLAP_CHARS + 1).min(previous_end + 1);
    let fits = |w: usize| {
        budget.count_tokens(&content[cuts[previous_end + 1 - w]..cuts[previous_end + 1]])
            <= max_tokens
    };

    if !fits(1) {
        return None;
    }
    if fits(max_window) {
        return Some(previous_end + 1 - max_window);
    }

    // `fits(1)` holds and `fits(max_window)` does not; narrow between them.
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
    Some(previous_end + 1 - lo)
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
/// previous chunk ended and, when the overlap is affordable, every
/// non-final chunk is longer than the overlap (which keeps the next start
/// strictly advancing). A paragraph boundary is preferred; if none is
/// present, a sentence boundary; if neither is present, `hard_end` (a hard
/// split at a character boundary, which is within budget).
fn choose_end(chars: &[char], hard_end: usize, first_eligible: usize) -> usize {
    let mut sentence = None;
    // Scan backward so the first hit of each kind is its rightmost.
    for cut in (first_eligible..=hard_end).rev() {
        match break_kind_before(chars, cut) {
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

/// Classify the boundary at char index `cut` by the char just before it.
fn break_kind_before(chars: &[char], cut: usize) -> Option<BreakKind> {
    if cut == 0 || cut > chars.len() {
        return None;
    }
    let prev = chars[cut - 1];
    if (prev == '\n' || prev == '\r') && line_break_count(chars, cut) >= 2 {
        return Some(BreakKind::Paragraph);
    }
    if is_sentence_end(prev) {
        return Some(BreakKind::Sentence);
    }
    None
}

/// Number of logical line breaks in the run of `\r`/`\n` chars ending
/// just before `end`, counting CRLF as a single break. Two or more means
/// a blank line, i.e. a paragraph break.
fn line_break_count(chars: &[char], end: usize) -> usize {
    let mut count = 0usize;
    let mut i = end;
    while i > 0 {
        match chars[i - 1] {
            '\n' => {
                count += 1;
                i -= 1;
            }
            '\r' => {
                // A '\r' immediately followed by '\n' (toward `end`) is
                // the start of a CRLF pair already counted by that '\n'.
                if i < end && chars[i] == '\n' {
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
    use std::cell::Cell;

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
            assert_eq!(chunks[0], Chunk { start: 0, end: content.len() });
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
        assert!(chunks.len() >= 3, "character cap did not split the long word");
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
}
