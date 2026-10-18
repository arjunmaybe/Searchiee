// Deterministic, UTF-8-safe text chunker (Phase 1).
//
// `chunk_content` splits a document into `Chunk`s that record UTF-8-safe
// byte offsets, and `chunk_text` resolves a `Chunk` back to its slice of
// the original content.
//
// Sizing is by Unicode scalar values ("chars"), not model tokens. The
// 800-character cap is a hard per-chunk maximum and is NOT a strict
// token limit: a downstream embedder tokenizes independently, and the
// token count for 800 characters varies with language and content, so
// this cap only bounds character length.
//
// Guarantees:
//   * Deterministic. Output depends only on the input content.
//   * UTF-8-safe. Cuts are taken only at `char` boundaries, so every
//     chunk is a valid UTF-8 slice and no code point is split.
//   * Complete coverage. Every character appears in at least one chunk.
//   * Intentional overlap. Adjacent chunks share up to `OVERLAP_CHARS`
//     characters (at least one), since each next chunk starts before the
//     previous chunk ended.
//   * Forward progress. Each chunk's end strictly advances past the
//     previous end: a boundary already used as the previous chunk's end
//     is never reused, so a finite input yields a finite, non-repeating
//     chunk list with no redundant nested chunks.
//   * Strictly increasing starts. Every non-final chunk is strictly
//     longer than `OVERLAP_CHARS`, so each window opens strictly after
//     the previous start and no start is ever repeated.
//   * Hard maximum. No chunk exceeds `MAX_CHARS` characters; a short
//     tail is emitted as its own chunk rather than merged past the cap.
//   * LF and CRLF line endings are both understood for paragraph breaks.
//   * For longer text, paragraph and sentence boundaries are preferred;
//     otherwise the split is taken at a character boundary.
#![allow(dead_code)]

/// Hard maximum chunk size, in Unicode characters.
const MAX_CHARS: usize = 800;
/// Target number of characters shared between adjacent chunks.
const OVERLAP_CHARS: usize = 120;

/// A chunk of the source content, expressed as byte offsets into that
/// content. `start` is inclusive and `end` is exclusive. Both are UTF-8
/// character boundaries, so `content[start..end]` is always valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub start: usize,
    pub end: usize,
}

/// Split `content` into deterministic, UTF-8-safe chunks.
///
/// Empty content yields no chunks. Non-empty content of at most
/// `MAX_CHARS` characters yields one whole-content chunk.
pub fn chunk_content(content: &str) -> Vec<Chunk> {
    // Byte offset of every char boundary, plus the end offset.
    let mut cuts: Vec<usize> = content.char_indices().map(|(i, _)| i).collect();
    cuts.push(content.len());
    let char_count = cuts.len() - 1;
    if char_count == 0 {
        return Vec::new();
    }

    let chars: Vec<char> = content.chars().collect();
    let mut chunks = Vec::new();
    let mut start = 0usize; // char index where the next window begins
    let mut previous_end = 0usize; // previous chunk's end, in chars

    loop {
        let hard_end = (start + MAX_CHARS).min(char_count);
        let end = if hard_end == char_count {
            char_count
        } else {
            // Only boundaries strictly after the previous end and more
            // than `OVERLAP_CHARS` past this chunk's start are eligible.
            // The latter guarantees every non-final chunk is longer than
            // the overlap, so the next chunk's start strictly advances
            // instead of saturating back to this start.
            let first_eligible = (previous_end + 1).max(start + OVERLAP_CHARS + 1);
            choose_end(&chars, hard_end, first_eligible)
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
        // This is always strictly less than `end`, so neighbours overlap
        // while each chunk's end strictly advances.
        start = end.saturating_sub(OVERLAP_CHARS);
    }

    chunks
}

/// Resolve `chunk` back to its slice of `content`.
pub fn chunk_text<'a>(content: &'a str, chunk: &Chunk) -> &'a str {
    &content[chunk.start..chunk.end]
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
/// previous chunk ended and every non-final chunk is longer than the
/// overlap (which keeps the next start strictly advancing). A paragraph
/// boundary is preferred; if none is present, a sentence boundary; if
/// neither is present, `hard_end` (a hard split at a character boundary,
/// which never exceeds the cap).
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
    use super::*;

    /// Assert every chunk is non-empty, on char boundaries, within the
    /// hard cap, that each successive end strictly advances (with the
    /// next start before the previous end), and that the chunks cover
    /// every character.
    fn assert_valid_and_covered(content: &str, chunks: &[Chunk]) {
        assert!(!chunks.is_empty(), "expected at least one chunk");
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(chunk.start < chunk.end, "chunk {i} is empty");
            assert!(chunk.end <= content.len(), "chunk {i} runs past the end");
            assert!(
                content.get(chunk.start..chunk.end).is_some(),
                "chunk {i} is not on char boundaries"
            );
            assert!(
                chunk_text(content, chunk).chars().count() <= MAX_CHARS,
                "chunk {i} exceeds the {MAX_CHARS}-char cap"
            );
            if i > 0 {
                let previous = chunks[i - 1];
                assert!(
                    chunk.start < previous.end,
                    "chunk {i} does not start before the previous end"
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
        assert!(chunk_content("").is_empty());
    }

    #[test]
    fn short_input_returns_one_whole_content_chunk() {
        for content in ["a", "hello, world", "line one\nline two"] {
            let chunks = chunk_content(content);
            assert_eq!(chunks.len(), 1, "content: {content:?}");
            assert_eq!(chunks[0], Chunk { start: 0, end: content.len() });
            assert_eq!(chunk_text(content, &chunks[0]), content);
        }
    }

    #[test]
    fn exactly_max_chars_is_one_chunk() {
        let content = "a".repeat(MAX_CHARS);
        let chunks = chunk_content(&content);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunk_text(&content, &chunks[0]), content);
    }

    #[test]
    fn one_char_over_max_splits_into_two_chunks() {
        let content = "a".repeat(MAX_CHARS + 1);
        let chunks = chunk_content(&content);
        assert_eq!(chunks.len(), 2);
        assert_valid_and_covered(&content, &chunks);
    }

    #[test]
    fn prefers_a_paragraph_boundary() {
        // Paragraph break at char 502, inside the first 800-char window.
        let content = format!("{}\n\n{}", "A".repeat(500), "B".repeat(1000));
        let chunks = chunk_content(&content);
        let first = chunk_text(&content, &chunks[0]);
        assert_eq!(first, format!("{}\n\n", "A".repeat(500)));
        assert_valid_and_covered(&content, &chunks);
    }

    #[test]
    fn crlf_blank_line_is_a_paragraph_boundary() {
        let content = format!("{}\r\n\r\n{}", "A".repeat(500), "B".repeat(1000));
        let chunks = chunk_content(&content);
        let first = chunk_text(&content, &chunks[0]);
        assert!(first.ends_with("\r\n\r\n"), "first chunk was {first:?}");
        assert_eq!(first, format!("{}\r\n\r\n", "A".repeat(500)));
        assert_valid_and_covered(&content, &chunks);
    }

    #[test]
    fn prefers_a_sentence_boundary() {
        // Sentence end at char 701, inside the first 800-char window.
        let content = format!("{}. {}", "x".repeat(700), "y".repeat(1000));
        let chunks = chunk_content(&content);
        let first = chunk_text(&content, &chunks[0]);
        assert_eq!(first, format!("{}.", "x".repeat(700)));
        assert_valid_and_covered(&content, &chunks);
    }

    #[test]
    fn early_paragraph_boundary_keeps_starts_and_ends_increasing() {
        // Paragraph break at char 42, well inside the first OVERLAP_CHARS
        // of the window, followed by a long tail. A naive overlap start
        // would compute 42 - OVERLAP_CHARS = 0 and repeat the first
        // chunk's start; the early boundary is therefore ineligible and
        // the window is split at the hard limit instead.
        let content = format!("{}\n\n{}", "A".repeat(40), "B".repeat(2000));
        let chunks = chunk_content(&content);
        assert!(chunks.len() > 1);

        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks.last().unwrap().end, content.len());

        // The early boundary cannot be used, so the first window fills to
        // the 800-character cap rather than cutting at char 42.
        assert_eq!(chunks[0].end - chunks[0].start, MAX_CHARS);

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

        // UTF-8 safety, the 800-char cap, and full coverage.
        assert_valid_and_covered(&content, &chunks);
    }

    #[test]
    fn repeated_paragraph_boundary_is_not_reused_as_a_nested_chunk() {
        // The only paragraph break is at char 502, so it fills the first
        // window. The next window must not cut there again.
        let content = format!("{}\n\n{}", "A".repeat(500), "B".repeat(2000));
        let chunks = chunk_content(&content);
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
        assert_valid_and_covered(&content, &chunks);
    }

    #[test]
    fn repeated_sentence_boundary_is_not_reused_as_a_nested_chunk() {
        // The only sentence end is at char 701, so it fills the first
        // window. The next window must not cut there again.
        let content = format!("{}.{}", "x".repeat(700), "y".repeat(2000));
        let chunks = chunk_content(&content);
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
        assert_valid_and_covered(&content, &chunks);
    }

    #[test]
    fn long_line_is_hard_split_at_char_boundaries() {
        let content = "z".repeat(2500);
        let chunks = chunk_content(&content);
        assert!(chunks.len() >= 4);
        assert_valid_and_covered(&content, &chunks);
    }

    #[test]
    fn overlap_is_targeted_between_neighbours() {
        // A long, unbroken line is hard-split, so each split shares
        // exactly OVERLAP_CHARS of context with the previous chunk.
        let content = "z".repeat(2500);
        let chunks = chunk_content(&content);
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
        // 800 chars fill the first chunk; the 30-char tail must not be
        // merged back past the hard maximum.
        let content = "w".repeat(MAX_CHARS + 30);
        let chunks = chunk_content(&content);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].end - chunks[0].start, MAX_CHARS);
        assert_eq!(chunks[1].end, content.len());
        assert!(chunks[1].end - chunks[1].start <= MAX_CHARS);
        assert_valid_and_covered(&content, &chunks);
    }

    #[test]
    fn unicode_offsets_are_char_boundaries_and_covered() {
        let content = "é😀漢字🌍ß ".repeat(400); // far beyond MAX_CHARS
        let chunks = chunk_content(&content);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(
                content.get(chunk.start..chunk.end).is_some(),
                "offset pair not on char boundaries: {chunk:?}"
            );
        }
        assert_valid_and_covered(&content, &chunks);
    }

    #[test]
    fn output_is_deterministic() {
        let content = format!("{}\n\n{}", "sentence. ".repeat(200), "tail".repeat(50));
        let first = chunk_content(&content);
        for _ in 0..5 {
            assert_eq!(chunk_content(&content), first);
        }
    }

    #[test]
    fn chunk_text_resolves_offset_pairs() {
        let content = "abcdefghij".repeat(200);
        let chunks = chunk_content(&content);
        let mut rebuilt = String::new();
        for chunk in &chunks {
            rebuilt.push_str(chunk_text(&content, chunk));
        }
        assert_valid_and_covered(&content, &chunks);
        // First chunk starts at the content start and the last reaches
        // the content end, so coverage spans the whole document.
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks.last().unwrap().end, content.len());
        assert!(!rebuilt.is_empty());
    }
}
