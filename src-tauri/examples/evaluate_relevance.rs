#[path = "../src/embed.rs"]
mod embed;

use serde::Deserialize;
use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

const THRESHOLDS: [f32; 9] = [
    0.00, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
];
const BASELINE_THRESHOLD: f32 = 0.25;
const RESULT_LIMIT: usize = 10;

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

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("Evaluation failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn Error>> {
    let corpus_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("evaluation/relevance-corpus.json"));

    let raw = fs::read_to_string(&corpus_path)?;
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
        print!(
            "\rDocuments: {}/{}",
            index + 1,
            corpus.documents.len()
        );
        io::stdout().flush()?;

        let vector = embed::embed_text(&document.content).await?;
        validate_embedding(
            &vector,
            &format!("document {}", document.id),
            &mut dimension,
        )?;

        document_vectors.push((
            document.id.clone(),
            document.title.clone(),
            vector,
        ));
    }
    println!();

    let mut evaluated = Vec::with_capacity(corpus.queries.len());

    for (index, query) in corpus.queries.iter().enumerate() {
        print!(
            "\rQueries: {}/{}",
            index + 1,
            corpus.queries.len()
        );
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
        results.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.id.cmp(&b.id))
        });

        evaluated.push(EvaluatedQuery {
            id: query.id.clone(),
            text: query.text.clone(),
            relevant_document_ids: query.relevant_document_ids.clone(),
            results,
        });
    }
    println!();

    println!("\nThreshold comparison (maximum 10 results per query)");
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

    for threshold in THRESHOLDS {
        let metrics = evaluate_at(&evaluated, threshold);

        let target_hits = format!(
            "{}/{}",
            metrics.positive_queries_hit,
            metrics.positive_queries
        );
        let negative_hits = format!(
            "{}/{}",
            metrics.negative_queries_hit,
            metrics.negative_queries
        );

        let precision = if metrics.total_results == 0 {
            0.0
        } else {
            100.0 * metrics.true_positive_docs as f64
                / metrics.total_results as f64
        };

        let average_results =
            metrics.total_results as f64 / evaluated.len() as f64;

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
        "\nEvaluation note: this is a small synthetic corpus with one \
         labelled target per positive query. Treat the results as \
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
        return Err(
            "the corpus needs both positive queries and negative controls".into()
        );
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
        Some(expected) if vector.len() != expected => {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{label} has {} dimensions; expected {expected}",
                    vector.len()
                ),
            ))
        }
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

fn results_at(
    query: &EvaluatedQuery,
    threshold: f32,
) -> Vec<&ScoredDocument> {
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
    println!(
        "\nPer-query details at the current threshold ({threshold:.2})"
    );

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
            .map(|result| format!(
                "{} ({:.1}%)",
                result.title,
                result.score * 100.0
            ))
            .collect::<Vec<_>>()
            .join("; ");

        let label = if query.relevant_document_ids.is_empty() {
            "negative"
        } else if returned.iter().any(|result| {
            query.relevant_document_ids.contains(&result.id)
        }) {
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