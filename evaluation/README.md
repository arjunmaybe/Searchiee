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

To suppress the per-query detail block, add `--summary-only`. It is only accepted next to `--baseline-only <corpus>`, and without it `--baseline-only <corpus>` continues to print per-query details for debugging. The flag removes query text, per-query scores, ranks, and document titles from the output while leaving the aggregate metrics table unchanged:

```powershell
$env:FASTEMBED_CACHE_DIR = "D:\Searchiee-eval-fastembed-cache"
cargo run --manifest-path .\src-tauri\Cargo.toml --example evaluate_relevance -- --baseline-only evaluation\validation-corpus.json --summary-only
```

The first run downloads the local embedding model into the separate cache directory. Later runs reuse it. The evaluator uses Searchiee's existing local embedding function. Running with no arguments sweeps every candidate threshold on the exploratory corpus. Passing `--baseline-only <corpus>` instead measures that corpus solely at the fixed production threshold (0.25), so the sweep cannot be used to tune against a supplied corpus. A corpus path is only accepted together with `--baseline-only`; any other option is rejected. Production threshold behavior is never changed.

## Metric definitions

Every result-based metric below is computed from the evaluator's returned list for a query: documents whose finite cosine score meets the candidate threshold, sorted by descending score (with a document-ID tie-break for reproducibility), then capped at the first 10. Filtering before the cap matches production, which filters, sorts, and truncates to the limit, so a below-threshold document never occupies a result slot. `RESULT_LIMIT = 10` in the evaluator mirrors the literal `10` passed to production search (`.search(&query_vector, 10)` in `src-tauri/src/main.rs`); the two must be kept in sync if production's limit changes.

- **Positive queries hit** (query-level): positive queries whose returned top-10 list contains at least one labelled relevant document, over all positive queries. A query that returns nothing, or nothing relevant, stays in the denominator and is not a hit.
- **Label recall@10** (label-level): labelled relevant documents found in the returned top-10 lists, summed over positive queries, divided by the total number of relevance labels. Each labelled query-document pair counts separately, so a multi-label query can contribute several hits, and the denominator does not shrink when a label is missed.
- **MRR@10** (query-level): for each positive query, the reciprocal of the 1-based rank of its first labelled relevant document in the returned top-10 list (0 if none appears), summed over positive queries and divided by the number of positive queries. Only the best-ranked relevant document counts for a query; a positive query with no relevant result contributes 0 but stays in the denominator. Negative controls are excluded.
- **Non-target results on positive queries** (document-level count): documents returned for positive queries that are not labelled relevant.
- **Positive precision** (document-level): labelled relevant documents returned for positive queries divided by all documents returned for positive queries. Non-target documents are the false positives.
- **Overall precision** (document-level): labelled relevant documents returned for all queries divided by all documents returned for all queries. Negative controls have no labels, so every document returned for a control is a false positive; this is the stricter figure.
- **Negative queries with results** (query-level): negative controls whose returned top-10 list is non-empty, over all negative controls.
- **Results on negative queries** (document-level count): the total number of documents returned for negative controls (each control contributes between 0 and 10).
- **Results per query** (document-level): the total number of documents returned for all queries divided by the number of queries.

A precision of `0.0` means either that nothing was returned (the denominator is zero) or that the returned documents were all non-targets; it is reported as `0.0` rather than undefined so the table stays numeric, and the neighbouring result-count columns distinguish the two cases.

Because every metric is computed from the same threshold-filtered, top-10-capped list, the numbers move together with the threshold. Raising the threshold returns fewer documents: it can push labelled documents out of the list (reducing the hit rate, label recall@10, and — for affected queries — MRR@10) while removing non-target documents (raising precision and reducing negative-query results). Ranks are assigned within the filtered list, so dropping a non-target document ahead of a relevant one can also raise that query's reciprocal rank.

## Exploratory results

Observed on the current 24-document, 55-query synthetic corpus (28 positive queries, 27 negative controls):

| Threshold | Positive queries hit | Label recall@10 | MRR@10 | Non-target results on positive queries | Positive precision | Overall precision | Negative queries with results | Results on negative queries | Results per query |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 28/28 | 100.0% | 0.982 | 239 | 13.1% | 6.6% | 27/27 | 267 | 9.85 |
| 0.15 | 28/28 | 100.0% | 0.982 | 102 | 26.1% | 16.5% | 26/27 | 80 | 3.96 |
| 0.20 | 28/28 | 100.0% | 0.982 | 62 | 36.7% | 24.7% | 20/27 | 48 | 2.65 |
| 0.25 | 28/28 | 94.4% | 0.982 | 39 | 46.6% | 30.4% | 16/27 | 39 | 2.04 |
| 0.30 | 28/28 | 94.4% | 0.982 | 30 | 53.1% | 39.1% | 12/27 | 23 | 1.58 |
| 0.35 | 27/28 | 91.7% | 0.946 | 17 | 66.0% | 49.3% | 10/27 | 17 | 1.22 |
| 0.40 | 27/28 | 88.9% | 0.946 | 9 | 78.0% | 66.7% | 5/27 | 7 | 0.87 |
| 0.45 | 26/28 | 80.6% | 0.929 | 6 | 82.9% | 76.3% | 1/27 | 3 | 0.69 |
| 0.50 | 24/28 | 75.0% | 0.857 | 1 | 96.4% | 87.1% | 1/27 | 3 | 0.56 |

These figures are the raw output of the run above. Precision appears twice: positive precision counts only documents returned for positive queries, while overall precision also counts the documents returned for negative controls as false positives, making it the stricter figure. The results are diagnostic, not a production-quality claim. The same corpus was used to compare thresholds, so the apparent best result must not be treated as independent validation.

At the production threshold of 0.25 every positive query still returned at least one labelled document (28/28), label recall@10 was 94.4% (34 of the 36 relevance labels), and MRR@10 was 0.982. Negative controls produced 39 returned documents and positive queries produced 39 non-target results, leaving 46.6% positive precision and 30.4% overall precision. Raising the threshold trades positive coverage for fewer negative results: the first positive query loses its labelled target at 0.35, label recall@10 falls to 75.0% by 0.50, and overall precision rises to 87.1%. No threshold was selected or changed on the basis of these observations; the production threshold stays at 0.25.

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

## Sealed test corpus (not yet evaluated)

`evaluation/sealed-corpus.json` is a third synthetic diagnostic corpus written after the development and validation corpora already existed. Its purpose is to characterize Searchiee at the fixed production threshold on topics that do not overlap either earlier corpus. It is intended as a disposable regression and smoke benchmark, not as a general quality estimate.

- 24 synthetic documents across 8 topic clusters, 3 distinguishable documents per cluster.
- 48 queries: 32 positive queries and 16 negative controls.
- 40 individual relevance labels.
- 24 single-label positives plus 8 positives that each carry exactly two relevance labels.
- 8 hard/adjacent negative controls, one per cluster, plus 8 clear out-of-domain negative controls.

The topics are archery, model railroading, rock climbing, soap making, leatherworking, amateur radio, bookbinding, and origami. Positive queries mix keyword fragments, conversational wording, low-overlap paraphrases, and a small number of realistic typos. Some clusters are deliberately adjacent in vocabulary (leatherworking and bookbinding overlap on stitching and sewing vocabulary, sharing words such as stitch, thread, sewing, and waxed; bookbinding and origami share paper vocabulary; model railroading and amateur radio reuse terms such as station and ground in different senses; archery and rock climbing share terms such as anchor, cam, energy, fall, shoulder, stance, and stable), which is treated as useful within-corpus retrieval stress while labels stay strict and content-based: a document is labelled relevant only when its own content answers or supports the query, never because the wording happens to overlap.

### Disjointness from the earlier corpora

Disjointness from the development corpus was machine-checked against `evaluation/relevance-corpus.json`: no sealed document title, document content, or query text duplicates a development entry. Duplicate comparison used exact string equality and, separately, a normalized form obtained by lowercasing, collapsing whitespace, and stripping every non-alphanumeric character. No sealed topic appears as a development document or query topic, including topics that occur only in the development negative controls (music practice, influenza, Roman aqueducts, compound interest, descaling, utility billing, pool chemistry, router quality of service, password reset, container shipping, personal loans, TLS cipher suites, sink drains, desalination, frozen pipes, operating-system password resets, firewalls, VPNs, gluten-free baking, croissant lamination, bread storage, hurricanes, index fossils, aurora, raised garden beds, broker-versus-advisor comparisons, and home-insurance appeals).

`evaluation/validation-corpus.json` was **not** opened or machine-compared for this work. Disjointness from it rests only on its published topic summary in this README (beekeeping, bicycle repair, pottery, knitting, home coffee brewing, aquarium fishkeeping, amateur astronomy, and chess improvement, together with their listed negative-control topics). No machine comparison against that file was performed, and none is being claimed.

### Label policy and bias

The same label policy as the earlier corpora applies: a document is labelled relevant to a query only when its content meaningfully answers or supports that query, topic similarity or shared vocabulary alone is not enough, and ambiguous cases are avoided rather than guessed. Every document is the target of at least one positive query, and every label references an existing document ID.

Topic separation reduces overlap with the earlier corpora but does not remove author bias. The labels are author-assigned, the author has already seen results from the earlier corpora, and each query was written with knowledge of the sealed documents. This corpus remains a synthetic diagnostic fixture, not an independently labelled benchmark.

## Sealed test protocol (pre-registered before any run)

This section is the acceptance protocol for the sealed corpus. The protocol, the corpus, and the evaluator safeguards are committed **before** any evaluation run so that the criteria cannot be chosen after seeing results.

### Purpose and fixed threshold

Characterize Searchiee at the fixed production threshold of **0.25** on a topic-disjoint corpus. This is a diagnostic and regression measurement only. It is not threshold selection, and no change to `0.25` is proposed or implied.

### Metrics recorded

The Milestone 2 evaluator reports, at the fixed threshold only: positive queries hit, label recall@10, MRR@10, non-target results on positive queries, positive precision, overall precision, negative queries with results, results on negative queries, and results per query. The definitions in the metric-definitions section apply unchanged.

### Acceptance criteria

There are **no pre-agreed search-quality pass/fail cutoffs**. No product requirement is invented here, and no threshold change is recommended. The run is a descriptive characterization: the aggregate metrics are recorded and interpreted afterwards, not scored against a gate. The only mandatory pass conditions are the objective corpus-integrity checks below.

### Corpus-integrity checks (objective, verified before freezing)

- The file parses as JSON against the evaluator's schema.
- 24 documents, 48 queries, and 40 relevance labels.
- 32 positive queries and 16 negative controls.
- Exactly 8 positive queries carry exactly 2 labels; the remaining 24 positives carry exactly 1.
- 3 documents and 4 positive queries per topic cluster, that is, 3 single-label positives plus 1 two-label positive.
- 8 hard/adjacent negative controls, one per cluster, plus 8 clear out-of-domain controls.
- Document IDs, query IDs, and normalized query texts are unique.
- Every label references an existing document ID.
- Every document is the target of at least one positive query.
- No development-corpus document title, document content, or query text is reused.

### Resolution of the negative controls

Only 16 negative controls contribute to the negative-control metrics, so those figures are coarse and directional. Binomial uncertainty across 16 observations is large, and the numbers must not be read as precise general-quality estimates. The positive side, 32 queries carrying 40 labels, is likewise small enough to be a regression signal rather than a general benchmark.

### One-look procedure

- **The look versus execution attempts.** The corpus is authorized for exactly one look. An *execution attempt* is any run of the evaluator against `evaluation/sealed-corpus.json`. An attempt that fails **before** the aggregate metrics table begins printing reveals no results and may be retried without consuming the look. An execution attempt that reaches the aggregate metrics table consumes the look, whether it completes successfully or fails afterwards.
- **Recorded after the fact.** The number of execution attempts, each attempt's outcome, and whether the look was consumed are recorded in the results section below only once they have actually happened. Nothing is pre-filled here: no attempt count, outcome, or date is asserted in advance.
- **Consumption policy.** If the results motivate any change to labels, queries, corpus contents, or the production threshold, this corpus is consumed. A new independent corpus is required before another decision-grade evaluation, and the sealed corpus is not re-run to confirm a revision.
- **Freeze metadata.** The SHA-256 digest of the frozen corpus file and the frozen commit reference are captured from actual pre-run values: the digest is computed from the committed `evaluation/sealed-corpus.json` once the freeze commit exists, and the commit reference is that freeze commit. Both are now recorded from actual values in the results section below. The execution-attempt count, attempt outcomes, and the evaluation date stay unset until they actually occur, so that no post-run value can be mistaken for a recorded one.
- **Commit sequence.** The freeze commit contains the sealed corpus, the `--summary-only` evaluator change together with its tests, and this protocol section. After the freeze commit, and before the evaluation, the actual corpus SHA-256 digest and the frozen commit reference are recorded in a documentation-only change. After the authorized evaluation, the aggregate results together with the evaluation date, the execution-attempt count, and each attempt's outcome are recorded in a subsequent documentation-only change. Neither documentation-only change may modify `evaluation/sealed-corpus.json`, so the frozen digest remains valid.

## Sealed test results (not yet run)

**Status: NOT YET RUN.** No run has been performed against `evaluation/sealed-corpus.json`. The pre-run metadata is recorded below; no aggregate metrics, evaluation date, execution-attempt count, or attempt outcome exists yet.

Pre-run metadata, captured from actual values after the freeze commit and before any attempt:

- **Frozen corpus and evaluator safeguards commit:** `76b56952dc18c6516a7e88e494a5934826cb9736`. This reference identifies the frozen `evaluation/sealed-corpus.json` together with the `--summary-only` evaluator safeguards and their tests, and this pre-registered protocol.
- **Sealed corpus SHA-256:** `9279B792761A0D0F37E23EACC129964D55BC5FC75D2D207B735E7098502845C2`, computed from the committed `evaluation/sealed-corpus.json` at that commit.

These are pre-run values: they fix which corpus and code the authorized run will use. They are deliberately separate from the post-run records, so the evaluation date, the number of execution attempts, and each attempt's outcome remain unset until the authorized attempt actually happens.

The run will be performed once, after the frozen commit is explicitly approved, with:

```powershell
$env:FASTEMBED_CACHE_DIR = "D:\Searchiee-eval-fastembed-cache"
cargo run --manifest-path .\src-tauri\Cargo.toml --example evaluate_relevance -- --baseline-only evaluation\sealed-corpus.json --summary-only
```

The command above runs with `--summary-only`, so only the aggregate metrics table is emitted (see the Run section). After the authorized attempt, this section will record the aggregate output together with the evaluation date, the number of execution attempts, and each attempt's outcome. No post-run metadata is pre-filled.

## Limitations

All three corpora are small synthetic diagnostic datasets, not independently validated benchmarks. The exploratory corpus now mixes single- and multi-label positives, but it still uses author-assigned labels and was itself used to compare thresholds, so its results do not establish general retrieval quality. The validation corpus (`validation-corpus.json`) has already been inspected once and is therefore a regression baseline rather than an untouched holdout; like the exploratory set it uses author-assigned synthetic labels, so its baseline is diagnostic rather than a general estimate of search quality. The sealed corpus (`sealed-corpus.json`) has not yet been evaluated and is topic-disjoint from the other two, but its labels are also author-assigned, so a future run would be a regression signal rather than a general estimate of search quality. No production threshold decision should be made from any of them alone; that would require representative real documents and queries with independent labelling.
