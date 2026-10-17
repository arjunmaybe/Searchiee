# Search relevance evaluation

This is a synthetic diagnostic corpus for evaluating Searchiee's minimum cosine-similarity threshold. It does not change production search behavior.

## Corpus

- 16 synthetic documents.
- 28 queries: 16 positive queries and 12 negative controls.
- Each positive query has one intended relevant document.
- Negative controls have no relevant document in this corpus.
- Other returned documents on positive queries are counted as non-target results under this fixture's single-target labels.

These labels are deliberately simplified. The corpus is not a broad or independently validated benchmark.

## Run

From the repository root in PowerShell:

```powershell
$env:FASTEMBED_CACHE_DIR = "D:\Searchiee-eval-fastembed-cache"
cargo run --manifest-path .\src-tauri\Cargo.toml --example evaluate_relevance
```

This evaluates the default exploratory corpus
(`evaluation/relevance-corpus.json`) and compares all nine candidate
thresholds. To evaluate the held-out validation corpus at the fixed
production threshold only, pass `--baseline-only` followed by the corpus
path:

```powershell
$env:FASTEMBED_CACHE_DIR = "D:\Searchiee-eval-fastembed-cache"
cargo run --manifest-path .\src-tauri\Cargo.toml --example evaluate_relevance -- --baseline-only evaluation\validation-corpus.json
```

The first run downloads the local embedding model into the separate cache directory. Later runs reuse it. The evaluator uses Searchiee's existing local embedding function. Running with no arguments sweeps every candidate threshold on the exploratory corpus. Passing `--baseline-only <corpus>` instead measures that corpus solely at the fixed production threshold (0.25), so the sweep cannot be used to tune against held-out data. A corpus path is only accepted together with `--baseline-only`; any other option is rejected. Production threshold behavior is never changed.

## Exploratory results

Observed on the current 16-document, 28-query synthetic corpus:

| Threshold | Positive targets found | Negative queries with results | Non-target results on positive queries | Results on negative queries |
|---:|---:|---:|---:|---:|
| 0.25 | 16/16 | 8/12 | 10 | 13 |
| 0.35 | 16/16 | 4/12 | 3 | 5 |
| 0.40 | 16/16 | 1/12 | 1 | 1 |
| 0.45 | 16/16 | 0/12 | 1 | 0 |
| 0.50 | 15/16 | 0/12 | 0 | 0 |

These figures are diagnostic, not a production-quality claim. The same corpus was used to compare thresholds, so the apparent best result must not be treated as independent validation.

At 0.45, all 16 intended targets were retained and none of the 12 negative controls returned results, but one non-target result remained on a positive query. At 0.50, the TLS-certificate target was missed. Do not change the production threshold solely on these observations.

## Held-out validation corpus

`validation-corpus.json` is a separate synthetic diagnostic dataset covering topics absent from the exploratory corpus (backyard beekeeping, bicycle repair, pottery, knitting, home coffee brewing, aquarium fishkeeping, amateur astronomy, and chess improvement):

- 24 documents across 8 topic clusters, 3 related but distinguishable documents per cluster.
- 36 queries: 24 positive queries and 12 negative controls.
- 6 clearly out-of-corpus negatives (scuba diving, tax deductions, ballet, lighthouse keepers, Tokyo subway navigation, wedding planning).
- 6 difficult near-miss negatives whose answers are not covered by any document (bee-sting treatment, bike-frame welding, porcelain clay formulation, coffee roasting, clownfish breeding, solar-eclipse predictions).
- Positive queries use varied wording and specificity; 5 carry multiple relevance labels.

Like the exploratory corpus, this is a synthetic diagnostic dataset, not an independently validated real-world benchmark.

## Validation baseline at the production threshold (0.25)

Observed by running the evaluator against the held-out corpus, which reports only the fixed production threshold:

- 24/24 positive queries returned at least one labelled relevant document.
- 29/31 individual relevance labels were retrieved; the broad bicycle-maintenance query missed two labelled targets (8.6% and 0.6%).
- 22/24 positive queries had a labelled relevant document at rank 1; the beekeeping mite-monitoring and aquarium-water-testing targets ranked second.
- 37 non-target results were returned across positive queries.
- 6/12 negative queries returned results: 1/6 clearly out-of-corpus queries (ballet) and 5/6 difficult near-misses (only frame welding returned nothing).
- The evaluator reported 12 results on negative queries in total.

The 24/24 query-hit rate does not mean every relevant document was retrieved (2 of 31 labels were missed), nor that the top-ranked result was always relevant (2 queries ranked a labelled target second). Separately, 37 non-target results were returned across positive queries. No threshold was selected or changed on the basis of these results; the exploratory sweep above stays separate from this baseline.

## Limitations

The exploratory corpus (`relevance-corpus.json`) is a small synthetic diagnostic dataset, not an independently validated benchmark. It uses simplified single-target labels, and the same data was used to compare thresholds. These results do not establish general retrieval quality. The 0.45 result is a candidate for further evaluation, not sufficient evidence to change the production threshold. The held-out `validation-corpus.json` is likewise synthetic and uses author-assigned labels, so its baseline is diagnostic rather than a general estimate of search quality.
