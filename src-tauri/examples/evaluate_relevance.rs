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

#[derive(Default)]
struct Metrics {
    positive_queries: usize,
    positive_queries_hit: usize,
    positive_false_positive_docs: usize,
    negative_queries: usize,
    negative_queries_hit: usize,
    negative_false_positive_docs: usize,
    true_positive_docs: usize,
    total_results: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct Options {
    corpus_path: PathBuf,
    baseline_only: bool,
}

impl Options {
    /// Argument defaults: sweep the exploratory corpus across every candidate
    /// threshold. Baseline-only mode is opt-in via `--baseline-only <corpus>`,
    /// which measures a single supplied corpus at the fixed production
    /// threshold.
    fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self, String> {
        let mut baseline_only = false;
        let mut corpus_path = None;

        for arg in args {
            if arg == BASELINE_ONLY_FLAG {
                baseline_only = true;
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

        match (baseline_only, corpus_path) {
            (true, Some(path)) => Ok(Self {
                corpus_path: path,
                baseline_only: true,
            }),
            (true, None) => Err(format!("{BASELINE_ONLY_FLAG} requires a corpus path")),
            (false, None) => Ok(Self {
                corpus_path: PathBuf::from(EXPLORATORY_CORPUS),
                baseline_only: false,
            }),
            (false, Some(path)) => Err(format!(
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

    if options.baseline_only {
        println!(
            "\nHeld-out corpus: measured only at the fixed production threshold \
             ({BASELINE_THRESHOLD:.2}); the threshold sweep is reserved for the \
             exploratory corpus."
        );
    } else {
        println!("\nThreshold comparison (maximum {RESULT_LIMIT} results per query)");
    }
    println!(
        "{:>9} {:>12} {:>13} {:>13} {:>12} {:>11} {:>11}",
        "Threshold",
        "Target hits",
        "Positive FP",
        "Negative hit",
        "Negative FP",
        "Precision",
        "Avg/query"
    );

    for threshold in thresholds_for(options.baseline_only) {
        let metrics = evaluate_at(&evaluated, threshold);

        let target_hits = format!(
            "{}/{}",
            metrics.positive_queries_hit, metrics.positive_queries
        );
        let negative_hits = format!(
            "{}/{}",
            metrics.negative_queries_hit, metrics.negative_queries
        );

        let precision = if metrics.total_results == 0 {
            0.0
        } else {
            100.0 * metrics.true_positive_docs as f64 / metrics.total_results as f64
        };

        let average_results = metrics.total_results as f64 / evaluated.len() as f64;

        println!(
            "{:>9.2} {:>12} {:>13} {:>13} {:>12} {:>10.1}% {:>11.2}",
            threshold,
            target_hits,
            metrics.positive_false_positive_docs,
            negative_hits,
            metrics.negative_false_positive_docs,
            precision,
            average_results
        );
    }

    print_baseline_details(&evaluated, BASELINE_THRESHOLD);

    println!(
        "\nEvaluation note: this is a small synthetic corpus with \
         author-assigned relevance labels. Treat the results as \
         diagnostic evidence, not a general relevance benchmark."
    );

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

            // Negative controls have no labelled relevant documents.
            metrics.negative_false_positive_docs += returned.len();
            continue;
        }

        metrics.positive_queries += 1;

        let hits = returned
            .iter()
            .filter(|result| query.relevant_document_ids.contains(&result.id))
            .count();

        if hits > 0 {
            metrics.positive_queries_hit += 1;
        }

        metrics.true_positive_docs += hits;
        metrics.positive_false_positive_docs += returned.len() - hits;
    }

    metrics
}

fn print_baseline_details(queries: &[EvaluatedQuery], threshold: f32) {
    println!("\nPer-query details at the current threshold ({threshold:.2})");

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

        println!(
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
        );
    }
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
        // "other" beside the hit plus the lone miss result.
        assert_eq!(metrics.positive_false_positive_docs, 2);

        assert_eq!(metrics.negative_queries, 2);
        assert_eq!(metrics.negative_queries_hit, 1);
        assert_eq!(metrics.negative_false_positive_docs, 1);

        assert_eq!(metrics.total_results, 4);
        assert_eq!(
            metrics.positive_false_positive_docs + metrics.negative_false_positive_docs,
            3
        );
    }
}
