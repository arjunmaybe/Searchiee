# Search relevance evaluation

This is a synthetic diagnostic corpus for evaluating Searchiee's minimum cosine-similarity threshold. It does not change production search behavior.

## Corpus

- 24 synthetic documents. Several topics carry related but distinguishable documents so within-topic discrimination is exercised: water (wastewater, rainwater harvesting, septic systems, private well testing), security (password managers, home router, TLS, two-factor authentication, phishing), bread and sourdough (yeast bread, sourdough starter, no-knead, whole wheat), and earth science (volcanic ash, plate tectonics, earthquake measurement, tsunamis).
- 55 queries: 28 positive queries and 27 negative controls.
- 36 individual relevance labels across the positive queries.
- 5 positive queries carry multiple relevance labels (up to four documents each); the remaining positives have a single labelled target.
- Positive queries deliberately mix keyword-style strings, colloquial questions, paraphrases with little wording overlap, and a small number of realistic typos.
- Negative controls are either clearly out of the corpus or nearby near-misses whose answers are absent; 15 of the 27 negatives were added as hard/adjacent cases.
- Other returned documents on positive queries are counted as non-target results under this strict labelling.

### Label policy

A document is labelled relevant to a query only when its content meaningfully answers or supports that query; sharing a broad topic or a few keywords is not enough. When several documents each genuinely address a query, all of them are labelled. Documents that are merely related are deliberately left unlabelled, so they count as non-target results rather than being added on a guess. Ambiguous cases are avoided rather than resolved arbitrarily. This keeps the labels defensible and the metrics meaningful.

The corpus is a synthetic diagnostic fixture, not a broad or independently validated benchmark.

## Run

From the repository root in PowerShell:

```powershell
$env:FASTEMBED_CACHE_DIR = "D:\Searchiee-eval-fastembed-cache"
cargo run --manifest-path .\src-tauri\Cargo.toml --example evaluate_relevance
```

This evaluates the default exploratory corpus
(`evaluation/relevance-corpus.json`) and compares all nine candidate
thresholds. To evaluate the regression-baseline validation corpus at the
fixed production threshold only, pass `--baseline-only` followed by the
corpus path:

```powershell
$env:FASTEMBED_CACHE_DIR = "D:\Searchiee-eval-fastembed-cache"
cargo run --manifest-path .\src-tauri\Cargo.toml --example evaluate_relevance -- --baseline-only evaluation\validation-corpus.json
```

The first run downloads the local embedding model into the separate cache directory. Later runs reuse it. The evaluator uses Searchiee's existing local embedding function. Running with no arguments sweeps every candidate threshold on the exploratory corpus. Passing `--baseline-only <corpus>` instead measures that corpus solely at the fixed production threshold (0.25), so the sweep cannot be used to tune against a supplied corpus. A corpus path is only accepted together with `--baseline-only`; any other option is rejected. Production threshold behavior is never changed.

## Exploratory results

Observed on the current 24-document, 55-query synthetic corpus (28 positive queries, 27 negative controls):

| Threshold | Positive queries hit | Negative queries with results | Non-target results on positive queries | Results on negative queries | Precision | Results per query |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 28/28 | 27/27 | 239 | 267 | 6.6% | 9.85 |
| 0.15 | 28/28 | 26/27 | 102 | 80 | 16.5% | 3.96 |
| 0.20 | 28/28 | 20/27 | 62 | 48 | 24.7% | 2.65 |
| 0.25 | 28/28 | 16/27 | 39 | 39 | 30.4% | 2.04 |
| 0.30 | 28/28 | 12/27 | 30 | 23 | 39.1% | 1.58 |
| 0.35 | 27/28 | 10/27 | 17 | 17 | 49.3% | 1.22 |
| 0.40 | 27/28 | 5/27 | 9 | 7 | 66.7% | 0.87 |
| 0.45 | 26/28 | 1/27 | 6 | 3 | 76.3% | 0.69 |
| 0.50 | 24/28 | 1/27 | 1 | 3 | 87.1% | 0.56 |

These are the raw figures printed by the evaluator. Precision is its document-level figure: labelled hits divided by all returned documents, including those returned for negative controls, so it is a strict lower bound. The results are diagnostic, not a production-quality claim. The same corpus was used to compare thresholds, so the apparent best result must not be treated as independent validation.

At the production threshold of 0.25 every positive query still returned at least one labelled document, 16 of 27 negative controls returned something, and 39 non-target documents were returned on positive queries. Raising the threshold trades positive coverage for fewer negative results: at 0.45 two of the 28 positive queries no longer retrieve any labelled target, and at 0.50 four do. No threshold was selected or changed on the basis of these observations; the production threshold stays at 0.25.

## Validation corpus (already-observed regression baseline)

`validation-corpus.json` is a separate synthetic diagnostic dataset covering topics absent from the exploratory corpus (backyard beekeeping, bicycle repair, pottery, knitting, home coffee brewing, aquarium fishkeeping, amateur astronomy, and chess improvement). It was inspected once to record the baseline below, so it is no longer an untouched holdout; treat it as a regression baseline that detects changes, not as unbiased evidence about the threshold.

- 24 documents across 8 topic clusters, 3 related but distinguishable documents per cluster.
- 36 queries: 24 positive queries and 12 negative controls.
- 6 clearly out-of-corpus negatives (scuba diving, tax deductions, ballet, lighthouse keepers, Tokyo subway navigation, wedding planning).
- 6 difficult near-miss negatives whose answers are not covered by any document (bee-sting treatment, bike-frame welding, porcelain clay formulation, coffee roasting, clownfish breeding, solar-eclipse predictions).
- Positive queries use varied wording and specificity; 5 carry multiple relevance labels.

Like the exploratory corpus, this is a synthetic diagnostic dataset, not an independently validated real-world benchmark.

## Validation regression baseline at the production threshold (0.25)

Observed by running the evaluator with `--baseline-only` against this corpus, which reports only the fixed production threshold:

- 24/24 positive queries returned at least one labelled relevant document.
- 29/31 individual relevance labels were retrieved; the broad bicycle-maintenance query missed two labelled targets (8.6% and 0.6%).
- 22/24 positive queries had a labelled relevant document at rank 1; the beekeeping mite-monitoring and aquarium-water-testing targets ranked second.
- 37 non-target results were returned across positive queries.
- 6/12 negative queries returned results: 1/6 clearly out-of-corpus queries (ballet) and 5/6 difficult near-misses (only frame welding returned nothing).
- The evaluator reported 12 results on negative queries in total.

The 24/24 query-hit rate does not mean every relevant document was retrieved (2 of 31 labels were missed), nor that the top-ranked result was always relevant (2 queries ranked a labelled target second). Separately, 37 non-target results were returned across positive queries. No threshold was selected or changed on the basis of these results; the exploratory sweep above stays separate from this baseline.

## Limitations

Both corpora are small synthetic diagnostic datasets, not independently validated benchmarks. The exploratory corpus now mixes single- and multi-label positives, but it still uses author-assigned labels and was itself used to compare thresholds, so its results do not establish general retrieval quality. The validation corpus (`validation-corpus.json`) has already been inspected once and is therefore a regression baseline rather than an untouched holdout; like the exploratory set it uses author-assigned synthetic labels, so its baseline is diagnostic rather than a general estimate of search quality. No production threshold decision should be made from either set alone; that would require representative real documents and queries with independent labelling.
