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

The first run downloads the local embedding model into the separate cache directory. Later runs reuse it. The evaluator uses Searchiee's existing local embedding function and compares candidate thresholds without changing the production threshold.

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

## Limitations

This is a small synthetic diagnostic dataset, not an independently validated benchmark. The corpus uses simplified single-target labels, and the same data was used to compare thresholds. These results do not establish general retrieval quality. The 0.45 result is a candidate for further evaluation, not sufficient evidence to change the production threshold.
