#[path = "../src/embed.rs"]
mod embed;

use serde::Deserialize;
use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

const THRESHOLDS: [f32; 9] = [0.00, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50];
const BASELINE_THRESHOLD: f32 = 0.25;
const RESULT_LIMIT: usize = 10;
const EXPLORATORY_CORPUS: &str = "evaluation/relevance-corpus.json";
const BASELINE_ONLY_FLAG: &str = "--baseline-only";
const SUMMARY_ONLY_FLAG: &str = "--summary-only";

#[derive(Debug, Deserialize)]
struct Corpus {
    documents: Vec<Document>,
    queries: Vec<Query>,
}

#[derive(Debug, Deserialize)]
struct Document {
    id: String,
    title: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct Query {
    id: String,
    text: String,
    relevant_document_ids: Vec<String>,
}

struct ScoredDocument {
    id: String,
    title: String,
    score: f32,
}

struct EvaluatedQuery {
    id: String,
    text: String,
    relevant_document_ids: Vec<String>,
    results: Vec<ScoredDocument>,
}

#[derive(Default, Debug, PartialEq)]
struct Metrics {
    positive_queries: usize,
    positive_queries_hit: usize,
    positive_results: usize,
    positive_false_positive_docs: usize,
    true_positive_docs: usize,
    total_labels: usize,
    reciprocal_rank_sum: f64,
    negative_queries: usize,
    negative_queries_hit: usize,
    negative_results: usize,
    total_results: usize,
}

impl Metrics {
    /// Share of labelled query-document pairs that were returned, as a
    /// percentage. Each label counts separately, so multi-label queries
    /// contribute more than once and the denominator never shrinks when a
    /// label is missed.
    fn label_recall(&self) -> f64 {
        if self.total_labels == 0 {
            0.0
        } else {
            100.0 * self.true_positive_docs as f64 / self.total_labels as f64
        }
    }

    /// Mean reciprocal rank of the first relevant result across positive
    /// queries, computed within the returned top-10 list. A positive query
    /// with no relevant result contributes zero but stays in the denominator.
    fn mrr_at_10(&self) -> f64 {
        if self.positive_queries == 0 {
            0.0
        } else {
            self.reciprocal_rank_sum / self.positive_queries as f64
        }
    }

    /// Share of returned documents on positive queries that are labelled
    /// relevant, as a percentage.
    fn positive_precision(&self) -> f64 {
        if self.positive_results == 0 {
            0.0
        } else {
            100.0 * self.true_positive_docs as f64 / self.positive_results as f64
        }
    }

    /// Share of all returned documents that are labelled relevant, as a
    /// percentage. Negative controls have no labels, so every document they
    /// return counts as a false positive.
    fn overall_precision(&self) -> f64 {
        if self.total_results == 0 {
            0.0
        } else {
            100.0 * self.true_positive_docs as f64 / self.total_results as f64
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Options {
    corpus_path: PathBuf,
    baseline_only: bool,
    summary_only: bool,
}

impl Options {
    /// Argument defaults: sweep the exploratory corpus across every candidate
    /// threshold. Baseline-only mode is opt-in via `--baseline-only <corpus>`,
    /// which measures a single supplied corpus at the fixed production
    /// threshold. `--summary-only` suppresses the per-query detail block and is
    /// only meaningful next to `--baseline-only`, so it is rejected elsewhere.
    fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self, String> {
        let mut baseline_only = false;
        let mut summary_only = false;
        let mut corpus_path = None;

        for arg in args {
            // Both flags must be matched before the generic '-' branch, or the
            // summary-only flag would be reported as an unknown option.
            if arg == BASELINE_ONLY_FLAG {
                baseline_only = true;
            } else if arg == SUMMARY_ONLY_FLAG {
                summary_only = true;
            } else if arg.to_string_lossy().starts_with('-') {
                return Err(format!("unknown option: {}", arg.to_string_lossy()));
            } else if corpus_path.is_some() {
                return Err(format!(
                    "expected at most one corpus path, got an extra argument: {}",
                    arg.to_string_lossy()
                ));
            } else {
                corpus_path = Some(PathBuf::from(arg));
            }
        }

        match (baseline_only, corpus_path, summary_only) {
            (true, Some(path), summary_only) => Ok(Self {
                corpus_path: path,
                baseline_only: true,
                summary_only,
            }),
            (true, None, _) => Err(format!("{BASELINE_ONLY_FLAG} requires a corpus path")),
            (false, None, true) => Err(format!(
                "{SUMMARY_ONLY_FLAG} requires {BASELINE_ONLY_FLAG} with a corpus path"
            )),
            (false, None, false) => Ok(Self {
                corpus_path: PathBuf::from(EXPLORATORY_CORPUS),
                baseline_only: false,
                summary_only: false,
            }),
            (false, Some(path), true) => Err(format!(
                "{SUMMARY_ONLY_FLAG} requires {BASELINE_ONLY_FLAG}, which was not given (got {})",
                path.display()
            )),
            (false, Some(path), false) => Err(format!(
                "a corpus path must be paired with {BASELINE_ONLY_FLAG} (got {})",
                path.display()
            )),
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("Evaluation failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn Error>> {
    // Baseline-only mode is opt-in via `--baseline-only <corpus>`; with no
    // arguments the exploratory corpus is swept across every candidate
    // threshold. Deciding the mode from the mere presence of a path argument
    // would misclassify the documented flag invocation.
    let options = Options::parse(env::args_os().skip(1))
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;

    let raw = fs::read_to_string(&options.corpus_path)?;
    let corpus: Corpus = serde_json::from_str(&raw)?;

    validate_corpus(&corpus)
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))?;

    println!(
        "Corpus: {} documents, {} queries",
        corpus.documents.len(),
        corpus.queries.len()
    );
    println!("Embedding documents with Searchiee's local model...");

    let mut dimension = None;
    let mut document_vectors = Vec::with_capacity(corpus.documents.len());

    for (index, document) in corpus.documents.iter().enumerate() {
        print!("\rDocuments: {}/{}", index + 1, corpus.documents.len());
        io::stdout().flush()?;

        let vector = embed::embed_text(&document.content).await?;
        validate_embedding(
            &vector,
            &format!("document {}", document.id),
            &mut dimension,
        )?;

        document_vectors.push((document.id.clone(), document.title.clone(), vector));
    }
    println!();

    let mut evaluated = Vec::with_capacity(corpus.queries.len());

    for (index, query) in corpus.queries.iter().enumerate() {
        print!("\rQueries: {}/{}", index + 1, corpus.queries.len());
        io::stdout().flush()?;

        let query_vector = embed::embed_text(&query.text).await?;
        validate_embedding(
            &query_vector,
            &format!("query {}", query.id),
            &mut dimension,
        )?;

        let mut results = Vec::with_capacity(document_vectors.len());

        for (id, title, vector) in &document_vectors {
            let score = cosine_similarity(&query_vector, vector);
            results.push(ScoredDocument {
                id: id.clone(),
                title: title.clone(),
                score,
            });
        }

        // Match production's descending score order, with a stable
        // ID tie-breaker so evaluation output is reproducible.
        results.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));

        evaluated.push(EvaluatedQuery {
            id: query.id.clone(),
            text: query.text.clone(),
            relevant_document_ids: query.relevant_document_ids.clone(),
            results,
        });
    }
    println!();

    let mut stdout = io::stdout();
    write_report(&mut stdout, &evaluated, &options)?;

    println!(
        "\nEvaluation note: this is a small synthetic corpus with \
         author-assigned relevance labels. Treat the results as \
         diagnostic evidence, not a general relevance benchmark."
    );

    Ok(())
}

/// Renders everything a run prints after the progress lines: the mode banner,
/// the aggregate metrics table, and — only when details are enabled — the
/// per-query block. Writing through an `io::Write` sink keeps the real output
/// path testable in memory instead of only observable by running the example.
fn write_report<W: Write>(
    out: &mut W,
    evaluated: &[EvaluatedQuery],
    options: &Options,
) -> io::Result<()> {
    if options.baseline_only {
        writeln!(
            out,
            "\nSupplied corpus: evaluated only at the fixed production threshold \
             ({BASELINE_THRESHOLD:.2}); the threshold sweep is reserved for the \
             exploratory corpus."
        )?;
    } else {
        writeln!(
            out,
            "\nThreshold comparison (maximum {RESULT_LIMIT} results per query)"
        )?;
    }

    writeln!(
        out,
        "{:>9} {:>9} {:>10} {:>8} {:>8} {:>11} {:>13} {:>9} {:>12} {:>10}",
        "Threshold",
        "Pos hit",
        "Recall@10",
        "MRR@10",
        "Pos FP",
        "Pos prec",
        "Overall prec",
        "Neg hit",
        "Neg results",
        "Avg/query"
    )?;

    for (threshold, metrics) in metric_rows(evaluated, options) {
        let positive_hits = format!(
            "{}/{}",
            metrics.positive_queries_hit, metrics.positive_queries
        );
        let negative_hits = format!(
            "{}/{}",
            metrics.negative_queries_hit, metrics.negative_queries
        );

        let average_results = metrics.total_results as f64 / evaluated.len() as f64;

        writeln!(
            out,
            "{:>9.2} {:>9} {:>9.1}% {:>8.3} {:>8} {:>10.1}% {:>12.1}% {:>9} {:>12} {:>10.2}",
            threshold,
            positive_hits,
            metrics.label_recall(),
            metrics.mrr_at_10(),
            metrics.positive_false_positive_docs,
            metrics.positive_precision(),
            metrics.overall_precision(),
            negative_hits,
            metrics.negative_results,
            average_results
        )?;
    }

    // `write_baseline_details` is the only place query text, per-query
    // scores, ranks, and document titles reach stdout, so suppressing it is
    // what keeps a sealed run aggregate-only. The aggregate table is emitted
    // from `metric_rows` regardless of this switch.
    if emits_query_details(options) {
        write_baseline_details(out, evaluated, BASELINE_THRESHOLD)?;
    }

    Ok(())
}

fn validate_corpus(corpus: &Corpus) -> Result<(), String> {
    if corpus.documents.is_empty() || corpus.queries.is_empty() {
        return Err("documents and queries must both be non-empty".into());
    }

    let mut document_ids = HashSet::new();

    for document in &corpus.documents {
        if document.id.trim().is_empty()
            || document.title.trim().is_empty()
            || document.content.trim().is_empty()
        {
            return Err("each document needs an ID, title, and content".into());
        }

        if !document_ids.insert(document.id.as_str()) {
            return Err(format!("duplicate document ID: {}", document.id));
        }
    }

    let mut query_ids = HashSet::new();

    for query in &corpus.queries {
        if query.id.trim().is_empty() || query.text.trim().is_empty() {
            return Err("each query needs an ID and query text".into());
        }

        if !query_ids.insert(query.id.as_str()) {
            return Err(format!("duplicate query ID: {}", query.id));
        }

        let mut relevant_ids = HashSet::new();

        for id in &query.relevant_document_ids {
            if !document_ids.contains(id.as_str()) {
                return Err(format!(
                    "query {} refers to unknown document {}",
                    query.id, id
                ));
            }

            if !relevant_ids.insert(id.as_str()) {
                return Err(format!(
                    "query {} repeats relevant document {}",
                    query.id, id
                ));
            }
        }
    }

    let positives = corpus
        .queries
        .iter()
        .filter(|query| !query.relevant_document_ids.is_empty())
        .count();

    let negatives = corpus.queries.len() - positives;

    if positives == 0 || negatives == 0 {
        return Err("the corpus needs both positive queries and negative controls".into());
    }

    Ok(())
}

fn validate_embedding(
    vector: &[f32],
    label: &str,
    expected_dimension: &mut Option<usize>,
) -> io::Result<()> {
    if vector.is_empty() || !vector.iter().all(|value| value.is_finite()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{label} produced an empty or non-finite embedding"),
        ));
    }

    match *expected_dimension {
        Some(expected) if vector.len() != expected => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{label} has {} dimensions; expected {expected}",
                vector.len()
            ),
        )),
        Some(_) => Ok(()),
        None => {
            *expected_dimension = Some(vector.len());
            Ok(())
        }
    }
}

// Mirrors the cosine calculation used by the production vector store.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// The exploratory corpus is the only place threshold selection happens.
/// Baseline-only mode measures a supplied (held-out) corpus solely at the
/// fixed production threshold so its results cannot inform the choice of
/// threshold.
fn thresholds_for(baseline_only: bool) -> Vec<f32> {
    if baseline_only {
        vec![BASELINE_THRESHOLD]
    } else {
        THRESHOLDS.to_vec()
    }
}

/// The aggregate metrics reported by a run, one row per threshold. These are
/// produced independently of `--summary-only`, so suppressing the per-query
/// detail block never changes a measurement, only what is printed.
fn metric_rows(queries: &[EvaluatedQuery], options: &Options) -> Vec<(f32, Metrics)> {
    thresholds_for(options.baseline_only)
        .into_iter()
        .map(|threshold| (threshold, evaluate_at(queries, threshold)))
        .collect()
}

/// Whether the per-query detail block is printed for these options.
fn emits_query_details(options: &Options) -> bool {
    !options.summary_only
}

fn results_at(query: &EvaluatedQuery, threshold: f32) -> Vec<&ScoredDocument> {
    query
        .results
        .iter()
        .filter(|result| result.score.is_finite() && result.score >= threshold)
        .take(RESULT_LIMIT)
        .collect()
}

fn evaluate_at(queries: &[EvaluatedQuery], threshold: f32) -> Metrics {
    let mut metrics = Metrics::default();

    for query in queries {
        let returned = results_at(query, threshold);
        metrics.total_results += returned.len();

        if query.relevant_document_ids.is_empty() {
            metrics.negative_queries += 1;

            if !returned.is_empty() {
                metrics.negative_queries_hit += 1;
            }

            // Negative controls have no labelled relevant documents, so every
            // returned document is a false positive.
            metrics.negative_results += returned.len();
            continue;
        }

        metrics.positive_queries += 1;
        metrics.total_labels += query.relevant_document_ids.len();
        metrics.positive_results += returned.len();

        let hits = returned
            .iter()
            .filter(|result| query.relevant_document_ids.contains(&result.id))
            .count();

        if hits > 0 {
            metrics.positive_queries_hit += 1;
        }

        metrics.true_positive_docs += hits;
        metrics.positive_false_positive_docs += returned.len() - hits;

        // Reciprocal rank of the first relevant result in the returned list.
        if let Some(position) = returned
            .iter()
            .position(|result| query.relevant_document_ids.contains(&result.id))
        {
            metrics.reciprocal_rank_sum += 1.0 / (position + 1) as f64;
        }
    }

    metrics
}

fn write_baseline_details<W: Write>(
    out: &mut W,
    queries: &[EvaluatedQuery],
    threshold: f32,
) -> io::Result<()> {
    writeln!(
        out,
        "\nPer-query details at the current threshold ({threshold:.2})"
    )?;

    for query in queries {
        let returned = results_at(query, threshold);

        let target_summary = if query.relevant_document_ids.is_empty() {
            "none (negative control)".to_string()
        } else {
            query
                .relevant_document_ids
                .iter()
                .map(|id| {
                    let score = query
                        .results
                        .iter()
                        .find(|result| &result.id == id)
                        .map(|result| format!("{:.1}%", result.score * 100.0))
                        .unwrap_or_else(|| "missing".into());

                    let rank = returned
                        .iter()
                        .position(|result| &result.id == id)
                        .map(|index| (index + 1).to_string())
                        .unwrap_or_else(|| "miss".into());

                    format!("{id} score={score} rank={rank}")
                })
                .collect::<Vec<_>>()
                .join(", ")
        };

        let top_results = returned
            .iter()
            .take(3)
            .map(|result| format!("{} ({:.1}%)", result.title, result.score * 100.0))
            .collect::<Vec<_>>()
            .join("; ");

        let label = if query.relevant_document_ids.is_empty() {
            "negative"
        } else if returned
            .iter()
            .any(|result| query.relevant_document_ids.contains(&result.id))
        {
            "target-hit"
        } else {
            "target-missed"
        };

        writeln!(
            out,
            "{} | {} | {} | query={:?} | returned={} | top={}",
            query.id,
            label,
            target_summary,
            query.text,
            returned.len(),
            if top_results.is_empty() {
                "(none)"
            } else {
                &top_results
            }
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    fn document(id: &str) -> Document {
        Document {
            id: id.to_string(),
            title: format!("{id} title"),
            content: format!("{id} content"),
        }
    }

    fn query(id: &str, relevant: &[&str]) -> Query {
        Query {
            id: id.to_string(),
            text: format!("{id} text"),
            relevant_document_ids: relevant.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn valid_corpus() -> Corpus {
        Corpus {
            documents: vec![document("d1"), document("d2")],
            queries: vec![query("q1", &["d1"]), query("q2", &[])],
        }
    }

    fn scored(id: &str, score: f32) -> ScoredDocument {
        ScoredDocument {
            id: id.to_string(),
            title: format!("{id} title"),
            score,
        }
    }

    fn evaluated_query(id: &str, relevant: &[&str], scores: &[(&str, f32)]) -> EvaluatedQuery {
        EvaluatedQuery {
            id: id.to_string(),
            text: format!("{id} text"),
            relevant_document_ids: relevant.iter().map(|s| s.to_string()).collect(),
            results: scores
                .iter()
                .map(|(rid, score)| scored(rid, *score))
                .collect(),
        }
    }

    #[test]
    fn valid_corpus_passes_validation() {
        assert!(validate_corpus(&valid_corpus()).is_ok());
    }

    #[test]
    fn duplicate_document_ids_are_rejected() {
        let mut corpus = valid_corpus();
        corpus.documents.push(document("d1"));
        let error = validate_corpus(&corpus).expect_err("duplicate doc ID");
        assert!(error.contains("duplicate document ID"));
    }

    #[test]
    fn duplicate_query_ids_are_rejected() {
        let mut corpus = valid_corpus();
        corpus.queries.push(query("q1", &[]));
        let error = validate_corpus(&corpus).expect_err("duplicate query ID");
        assert!(error.contains("duplicate query ID"));
    }

    #[test]
    fn unknown_target_ids_are_rejected() {
        let mut corpus = valid_corpus();
        corpus.queries[0].relevant_document_ids = vec!["missing".to_string()];
        let error = validate_corpus(&corpus).expect_err("unknown target");
        assert!(error.contains("unknown document"));
    }

    #[test]
    fn duplicate_target_ids_are_rejected() {
        let mut corpus = valid_corpus();
        corpus.queries[0].relevant_document_ids = vec!["d1".to_string(), "d1".to_string()];
        let error = validate_corpus(&corpus).expect_err("duplicate target");
        assert!(error.contains("repeats relevant document"));
    }

    #[test]
    fn corpus_needs_both_positive_and_negative_queries() {
        let mut no_positive = valid_corpus();
        no_positive.queries[0].relevant_document_ids.clear();
        assert!(validate_corpus(&no_positive).is_err());

        let mut no_negative = valid_corpus();
        no_negative.queries[1].relevant_document_ids = vec!["d2".to_string()];
        assert!(validate_corpus(&no_negative).is_err());

        let empty = Corpus {
            documents: vec![],
            queries: vec![],
        };
        assert!(validate_corpus(&empty).is_err());
    }

    #[test]
    fn embeddings_must_be_non_empty_and_finite() {
        let mut dimension = None;
        assert!(validate_embedding(&[0.1, 0.2, 0.3], "first", &mut dimension).is_ok());
        assert_eq!(dimension, Some(3));

        assert!(validate_embedding(&[], "empty", &mut dimension).is_err());
        assert!(validate_embedding(&[f32::NAN, 0.0, 0.0], "nan", &mut dimension).is_err());
        assert!(validate_embedding(&[f32::INFINITY, 0.0, 0.0], "inf", &mut dimension).is_err());
        // Failed validations must not clobber the tracked dimension.
        assert_eq!(dimension, Some(3));
    }

    #[test]
    fn embeddings_must_have_consistent_dimensions() {
        let mut dimension = None;
        assert!(validate_embedding(&[1.0, 2.0], "first", &mut dimension).is_ok());
        assert!(validate_embedding(&[3.0, 4.0], "same", &mut dimension).is_ok());
        let error = validate_embedding(&[1.0, 2.0, 3.0], "wide", &mut dimension)
            .expect_err("mismatched dimensions");
        assert!(error.to_string().contains("dimensions"));
    }

    #[test]
    fn cosine_similarity_reference_cases() {
        assert!((cosine_similarity(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn threshold_selection_covers_sweep_and_baseline_modes() {
        assert_eq!(thresholds_for(false), THRESHOLDS.to_vec());
        assert_eq!(thresholds_for(true), vec![BASELINE_THRESHOLD]);
    }

    #[test]
    fn default_arguments_sweep_the_exploratory_corpus() {
        let options = Options::parse(Vec::new()).expect("default options");

        assert_eq!(options.corpus_path, PathBuf::from(EXPLORATORY_CORPUS));
        assert!(!options.baseline_only);
        assert_eq!(thresholds_for(options.baseline_only), THRESHOLDS.to_vec());
    }

    #[test]
    fn baseline_only_flag_selects_the_supplied_corpus_and_baseline() {
        let options = Options::parse(args(&[
            "--baseline-only",
            "evaluation/validation-corpus.json",
        ]))
        .expect("baseline-only options");

        assert_eq!(
            options.corpus_path,
            PathBuf::from("evaluation/validation-corpus.json")
        );
        assert!(options.baseline_only);
        assert_eq!(
            thresholds_for(options.baseline_only),
            vec![BASELINE_THRESHOLD]
        );
    }

    #[test]
    fn baseline_only_flag_is_not_parsed_as_a_corpus_path() {
        let options = Options::parse(args(&[
            "--baseline-only",
            "evaluation/validation-corpus.json",
        ]))
        .expect("baseline-only options");

        // The flag must be consumed as an option; otherwise the leading
        // argument would be mistaken for the corpus path.
        assert_ne!(options.corpus_path, PathBuf::from("--baseline-only"));
    }

    #[test]
    fn baseline_only_without_a_corpus_is_rejected() {
        let error = Options::parse(args(&["--baseline-only"])).expect_err("missing corpus");
        assert!(error.contains("--baseline-only"));
    }

    #[test]
    fn corpus_path_without_baseline_only_is_rejected() {
        let error = Options::parse(args(&["evaluation/validation-corpus.json"]))
            .expect_err("path needs the flag");
        assert!(error.contains("--baseline-only"));
    }

    #[test]
    fn unknown_options_are_rejected() {
        let error = Options::parse(args(&["--sweep"])).expect_err("unknown option");
        assert!(error.contains("unknown option"));
    }

    #[test]
    fn default_arguments_keep_per_query_details_enabled() {
        let options = Options::parse(Vec::new()).expect("default options");

        // The exploratory sweep must keep printing per-query details.
        assert!(!options.summary_only);
    }

    #[test]
    fn baseline_only_without_summary_only_keeps_details() {
        let options = Options::parse(args(&[
            "--baseline-only",
            "evaluation/relevance-corpus.json",
        ]))
        .expect("baseline-only options");

        // Regression debugging relies on per-query output staying on by default.
        assert!(options.baseline_only);
        assert!(!options.summary_only);
    }

    #[test]
    fn summary_only_combines_with_baseline_only_in_either_order() {
        let flags_first = Options::parse(args(&[
            "--summary-only",
            "--baseline-only",
            "evaluation/relevance-corpus.json",
        ]))
        .expect("summary-only after baseline-only");

        let path_first = Options::parse(args(&[
            "--baseline-only",
            "evaluation/relevance-corpus.json",
            "--summary-only",
        ]))
        .expect("summary-only before path");

        for options in [flags_first, path_first] {
            assert_eq!(
                options.corpus_path,
                PathBuf::from("evaluation/relevance-corpus.json")
            );
            assert!(options.baseline_only);
            assert!(options.summary_only);
            // The sweep guard must not be weakened by the new flag.
            assert_eq!(
                thresholds_for(options.baseline_only),
                vec![BASELINE_THRESHOLD]
            );
        }
    }

    #[test]
    fn summary_only_without_baseline_only_is_rejected() {
        let error = Options::parse(args(&["--summary-only"])).expect_err("summary-only alone");
        assert!(error.contains("--summary-only"));
    }

    #[test]
    fn summary_only_with_a_corpus_path_is_rejected() {
        let error = Options::parse(args(&[
            "--summary-only",
            "evaluation/relevance-corpus.json",
        ]))
        .expect_err("summary-only needs the baseline flag");

        // A path still has to be paired with --baseline-only, with or without
        // the summary-only flag.
        assert!(error.contains("--baseline-only"));
    }

    #[test]
    fn summary_only_renders_aggregate_table_without_per_query_details() {
        let queries = vec![
            evaluated_query("hit", &["target"], &[("target", 0.8), ("other", 0.6)]),
            evaluated_query("neg", &[], &[("noise", 0.5)]),
        ];

        let sealed = Options::parse(args(&[
            "--baseline-only",
            "evaluation/sealed-corpus.json",
            "--summary-only",
        ]))
        .expect("sealed options");

        let debugging = Options::parse(args(&["--baseline-only", "evaluation/sealed-corpus.json"]))
            .expect("debug options");

        let render = |options: &Options| {
            let mut buffer = Vec::new();
            write_report(&mut buffer, &queries, options).expect("render report");
            String::from_utf8(buffer).expect("report is valid utf-8")
        };

        let sealed_output = render(&sealed);
        let debug_output = render(&debugging);

        // The aggregate metrics table is emitted in both modes, with the same
        // values: a label recall of 100.0% and an MRR of 1.000.
        for output in [&sealed_output, &debug_output] {
            assert!(output.contains("Threshold"));
            assert!(output.contains("Recall@10"));
            assert!(output.contains("Overall prec"));
            assert!(output.contains("100.0%"));
            assert!(output.contains("1.000"));
        }

        let table_rows = |output: &str| {
            output
                .lines()
                .filter(|line| line.trim_start().starts_with("0.25"))
                .map(str::to_string)
                .collect::<Vec<_>>()
        };
        assert_eq!(table_rows(&sealed_output).len(), 1);
        assert_eq!(table_rows(&sealed_output), table_rows(&debug_output));

        // Query text, per-query scores, ranks, and document titles appear
        // only when summary-only is disabled.
        assert!(debug_output.contains("Per-query details"));
        assert!(debug_output.contains("query="));
        assert!(debug_output.contains("rank=1"));
        assert!(debug_output.contains("target title"));

        assert!(!sealed_output.contains("Per-query details"));
        assert!(!sealed_output.contains("query="));
        assert!(!sealed_output.contains("rank="));
        assert!(!sealed_output.contains("score="));
        assert!(!sealed_output.contains("target title"));
    }

    #[test]
    fn results_at_keeps_threshold_and_applies_limit() {
        let query = evaluated_query(
            "q",
            &["keep"],
            &[
                ("keep", 0.90),
                ("boundary", 0.25),
                ("weak", 0.24),
                ("bad", f32::NAN),
            ],
        );
        let ids: Vec<_> = results_at(&query, 0.25)
            .iter()
            .map(|r| r.id.as_str())
            .collect();
        assert_eq!(ids, vec!["keep", "boundary"]);
    }

    #[test]
    fn results_at_respects_the_ten_result_limit() {
        let scores: Vec<(String, f32)> = (0..12)
            .map(|i| (format!("d{i:02}"), 0.99 - i as f32 * 0.01))
            .collect();
        let query = EvaluatedQuery {
            id: "q".to_string(),
            text: "q text".to_string(),
            relevant_document_ids: vec![],
            results: scores
                .iter()
                .map(|(id, score)| scored(id, *score))
                .collect(),
        };
        let returned = results_at(&query, 0.25);
        assert_eq!(returned.len(), 10);
        assert_eq!(returned[0].id, "d00");
        assert_eq!(returned[9].id, "d09");
    }

    #[test]
    fn evaluate_at_counts_hits_and_false_positives() {
        let queries = vec![
            evaluated_query("hit", &["target"], &[("target", 0.8), ("other", 0.6)]),
            evaluated_query("miss", &["absent"], &[("other", 0.6)]),
            evaluated_query("neg-hit", &[], &[("noise", 0.5)]),
            evaluated_query("neg-clean", &[], &[("quiet", 0.1)]),
        ];
        let metrics = evaluate_at(&queries, 0.25);

        assert_eq!(metrics.positive_queries, 2);
        assert_eq!(metrics.positive_queries_hit, 1);
        assert_eq!(metrics.true_positive_docs, 1);
        assert_eq!(metrics.total_labels, 2);
        assert_eq!(metrics.positive_results, 3);
        // "other" beside the hit plus the lone miss result.
        assert_eq!(metrics.positive_false_positive_docs, 2);

        assert_eq!(metrics.negative_queries, 2);
        assert_eq!(metrics.negative_queries_hit, 1);
        assert_eq!(metrics.negative_results, 1);

        assert_eq!(metrics.total_results, 4);
        assert_eq!(
            metrics.positive_false_positive_docs + metrics.negative_results,
            3
        );

        // The hit query's only relevant document is the first result.
        assert_eq!(metrics.reciprocal_rank_sum, 1.0);
        assert_eq!(metrics.label_recall(), 50.0);
        assert_eq!(metrics.mrr_at_10(), 0.5);
        assert!((metrics.positive_precision() - 100.0 / 3.0).abs() < 1e-9);
        assert_eq!(metrics.overall_precision(), 25.0);
    }

    #[test]
    fn hit_rate_is_query_level_while_recall_is_label_level() {
        // A four-label query with one retrieved label is a single hit, but
        // only a quarter of its labels count toward recall.
        let queries = vec![evaluated_query(
            "multi",
            &["a", "b", "c", "d"],
            &[("a", 0.9), ("x", 0.8)],
        )];
        let metrics = evaluate_at(&queries, 0.25);

        assert_eq!(metrics.positive_queries, 1);
        assert_eq!(metrics.positive_queries_hit, 1);
        assert_eq!(metrics.true_positive_docs, 1);
        assert_eq!(metrics.total_labels, 4);
        assert_eq!(metrics.label_recall(), 25.0);
    }

    #[test]
    fn multi_label_recall_counts_each_labelled_document() {
        let queries = vec![
            evaluated_query(
                "many",
                &["a", "b", "c", "d"],
                &[("a", 0.9), ("b", 0.8), ("c", 0.7), ("x", 0.6)],
            ),
            evaluated_query("one", &["e"], &[("e", 0.9)]),
        ];
        let metrics = evaluate_at(&queries, 0.25);

        assert_eq!(metrics.true_positive_docs, 4);
        assert_eq!(metrics.total_labels, 5);
        assert_eq!(metrics.positive_queries_hit, 2);
        assert_eq!(metrics.label_recall(), 80.0);
    }

    #[test]
    fn mrr_uses_only_the_first_relevant_rank() {
        // The best-ranked relevant document is second, so the relevant
        // document ranked third must not be added in.
        let queries = vec![evaluated_query(
            "two-labels",
            &["a", "b"],
            &[("x", 0.9), ("a", 0.8), ("b", 0.7)],
        )];
        let metrics = evaluate_at(&queries, 0.25);

        assert_eq!(metrics.reciprocal_rank_sum, 0.5);
        assert_eq!(metrics.mrr_at_10(), 0.5);
    }

    #[test]
    fn mrr_is_one_when_the_first_result_is_relevant() {
        // A later relevant document must not change the rank-one result.
        let queries = vec![evaluated_query(
            "first",
            &["a", "b"],
            &[("a", 0.9), ("x", 0.8), ("b", 0.7)],
        )];
        let metrics = evaluate_at(&queries, 0.25);

        assert_eq!(metrics.mrr_at_10(), 1.0);
    }

    #[test]
    fn mrr_is_zero_when_no_relevant_result_is_returned() {
        let queries = vec![
            evaluated_query("found", &["a"], &[("a", 0.9)]),
            evaluated_query("lost", &["b"], &[("x", 0.9)]),
        ];
        let metrics = evaluate_at(&queries, 0.25);

        // The missed query still counts toward the positive-query denominator.
        assert_eq!(metrics.positive_queries, 2);
        assert_eq!(metrics.positive_queries_hit, 1);
        assert_eq!(metrics.mrr_at_10(), 0.5);
    }

    #[test]
    fn relevant_document_at_rank_ten_counts() {
        let queries = vec![evaluated_query(
            "rank-ten",
            &["a"],
            &[
                ("x0", 0.90),
                ("x1", 0.89),
                ("x2", 0.88),
                ("x3", 0.87),
                ("x4", 0.86),
                ("x5", 0.85),
                ("x6", 0.84),
                ("x7", 0.83),
                ("x8", 0.82),
                ("a", 0.81),
            ],
        )];
        let metrics = evaluate_at(&queries, 0.25);

        assert_eq!(metrics.positive_queries_hit, 1);
        assert_eq!(metrics.true_positive_docs, 1);
        assert_eq!(metrics.label_recall(), 100.0);
        assert_eq!(metrics.reciprocal_rank_sum, 0.1);
        assert_eq!(metrics.mrr_at_10(), 0.1);
    }

    #[test]
    fn relevant_document_beyond_the_top_ten_does_not_count() {
        let queries = vec![evaluated_query(
            "rank-eleven",
            &["a"],
            &[
                ("x0", 0.91),
                ("x1", 0.90),
                ("x2", 0.89),
                ("x3", 0.88),
                ("x4", 0.87),
                ("x5", 0.86),
                ("x6", 0.85),
                ("x7", 0.84),
                ("x8", 0.83),
                ("x9", 0.82),
                ("a", 0.81),
            ],
        )];
        let metrics = evaluate_at(&queries, 0.25);

        // The labelled document passes the threshold but sits outside the
        // top-10 cap, so it counts neither as a hit nor as retrieved.
        assert_eq!(metrics.positive_queries_hit, 0);
        assert_eq!(metrics.true_positive_docs, 0);
        assert_eq!(metrics.label_recall(), 0.0);
        assert_eq!(metrics.mrr_at_10(), 0.0);
        assert_eq!(metrics.positive_results, 10);
        assert_eq!(metrics.positive_false_positive_docs, 10);
    }

    #[test]
    fn negative_hit_count_and_result_count_are_distinct() {
        let queries = vec![
            evaluated_query(
                "neg-many",
                &[],
                &[
                    ("n0", 0.90),
                    ("n1", 0.89),
                    ("n2", 0.88),
                    ("n3", 0.87),
                    ("n4", 0.86),
                    ("n5", 0.85),
                    ("n6", 0.84),
                    ("n7", 0.83),
                    ("n8", 0.82),
                    ("n9", 0.81),
                    ("n10", 0.80),
                    ("n11", 0.79),
                ],
            ),
            evaluated_query("neg-one", &[], &[("only", 0.5)]),
            evaluated_query("neg-none", &[], &[("weak", 0.1)]),
        ];
        let metrics = evaluate_at(&queries, 0.25);

        assert_eq!(metrics.negative_queries, 3);
        // Two of the three controls returned something ...
        assert_eq!(metrics.negative_queries_hit, 2);
        // ... but the returned-document count is capped at ten per query.
        assert_eq!(metrics.negative_results, 11);
        assert_eq!(metrics.total_results, 11);

        // With no positive queries every rate falls back to zero, not NaN.
        assert_eq!(metrics.positive_queries, 0);
        assert_eq!(metrics.label_recall(), 0.0);
        assert_eq!(metrics.mrr_at_10(), 0.0);
        assert_eq!(metrics.positive_precision(), 0.0);
        assert_eq!(metrics.overall_precision(), 0.0);
    }

    #[test]
    fn precision_denominators_are_explicit() {
        let queries = vec![
            evaluated_query("hit", &["a"], &[("a", 0.9), ("x", 0.8)]),
            evaluated_query("neg", &[], &[("n", 0.5)]),
        ];
        let metrics = evaluate_at(&queries, 0.25);

        assert_eq!(metrics.true_positive_docs, 1);
        assert_eq!(metrics.positive_results, 2);
        assert_eq!(metrics.total_results, 3);
        // Positive precision ignores the control result ...
        assert!((metrics.positive_precision() - 50.0).abs() < 1e-9);
        // ... while overall precision counts it as a false positive.
        assert!((metrics.overall_precision() - 100.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn precision_is_zero_when_nothing_is_returned() {
        let queries = vec![
            evaluated_query("high", &["a"], &[("a", 0.1)]),
            evaluated_query("higher", &["b", "c"], &[("b", 0.1)]),
            evaluated_query("neg", &[], &[("n", 0.05)]),
        ];
        let metrics = evaluate_at(&queries, 0.25);

        assert_eq!(metrics.positive_queries, 2);
        assert_eq!(metrics.positive_queries_hit, 0);
        assert_eq!(metrics.positive_results, 0);
        assert_eq!(metrics.total_results, 0);
        assert_eq!(metrics.total_labels, 3);

        // Zero denominators report 0.0 rather than NaN.
        assert_eq!(metrics.positive_precision(), 0.0);
        assert_eq!(metrics.overall_precision(), 0.0);
        assert_eq!(metrics.label_recall(), 0.0);
        assert_eq!(metrics.mrr_at_10(), 0.0);
    }
}
