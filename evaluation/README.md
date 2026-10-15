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