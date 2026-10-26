# Reproducing the reported outputs

All commands below run from the artifact root. They read the released records,
use Python 3.10+ and its standard library, require no network, and leave input
files unchanged.

| Command | Output and expected status |
|---|---|
| `python3 scripts/verify_artifact.py` | JSON summary, `status: ok`, exit 0; otherwise an explicit error and exit 1 |
| `python3 scripts/verify_artifact.py --result-index` | JSON membership index with supporting report/cause/attempt IDs; all integrity checks run first |
| `python3 -m unittest discover -s tests -v` | Offline unit and package tests, exit 0 |

`make reproduce-main` and `make test` invoke the first and third commands.
See [VERIFICATION.json](VERIFICATION.json) for the recorded run and fingerprints.

## Result index

Display names are used here because table numbers can change during layout.

| Paper output | Released input and check |
|---|---|
| Collection table and 7,509-record frame | `collection-frame.json`, `selection.csv`, CVE source inventory: 6,816 distinct GitHub IDs and 693 distinct CVE IDs/paths |
| Human exclusion audit and accuracy/recall formulas | `statistics/phase1-human-review.json`: recompute 600/4/592 totals, exact hypergeometric bounds, 44.86 missed failures, 85.48% estimated initial recall |
| Review-flow and appendix review tables | Same aggregate file: each stage balances, outputs join, 544 proposals minus 28 reversals gives 516 removals; individual stage memberships unavailable |
| Analysis sets | `data/final/cases.csv`: 264 reports partition into 162 repair-linked and 102 report-only; 162 links map to 149 causes |
| RQ1 relation distribution | `rq-cause-analysis.csv`: one primary relation per cause, by ecosystem |
| RQ2 provenance table | `rq2-provenance-analysis.csv`: origin and consumer endpoints; 71 different categories, 101 compiler and 48 runtime consumers |
| RQ3 relation–transition matrix | `rq-cause-analysis.csv`: one contribution per cause; matrix margins agree |
| RQ4 obligation table | Same cause table: 15 primary families, 149 contributions |
| RQ4 secondary results | `statistics/archived-taxonomy-assignments.jsonl`: mask final primary family separately, compare all 2,235 pairs, retain intersections, check against final labels |
| RQ5 consequence table and flags | `rq5-report-impacts.csv`: 264 primary consequences; flag union 60, with counts 23/14/24 |
| RQ5 failure–repair and obligation–repair matrices | Cause table: 149 contributions; 104 leading-family causes and 94 corresponding repairs |
| RQ5 sensitivity | `historical-alignment.csv`, cause and report tables: 136 test-or-diagnosis causes; removing PyTorch repository retains Triton and yields 193 reports/112 causes |
| RQ6 diagnostics | `rq6-diagnostic-attempts.csv`: four attempts, three causes; uncovered paths leave the other dimensions not evaluated |
| Taxonomy definitions | `taxonomy-codebook.json`: 45 families and positive-example links |
| Repair evidence | `repair-patch-index.jsonl`: 167 patches, file paths, hashes, and cause coverage |

Paths without a prefix above are under `results/`. The result-index command
exposes matrix-cell memberships and sensitivity subset IDs. It reports null
membership for the human-review aggregates instead of inventing sample IDs.

## Statistical conventions

The exclusion audit is treated as the author-reported simple random sample
without replacement from 6,729 initial exclusions. Exact integer combinations
evaluate hypergeometric tails. Equal-tailed 95% bounds retain population counts
whose two tail probabilities are at least 0.025; the one-sided bound uses 0.05.
The 95%-correct exclusion test rejects populations with at least 337 missed
failures when at most 21 positives are sampled. Power at 97% correct uses
202 missed failures, the nearest integer to 3% of 6,729.

The script computes with unrounded values. Displayed percentages are rounded
to two decimals, secondary agreement to one decimal, and power to two decimals.
The estimated number missed is 44.86 (45 when reported as a whole number).
The recall transformation uses 264 / (264 + missed).

The two original taxonomy assignment sets remain unchanged. A separate view
removes the final primary family from each secondary set. Overall agreement
still includes its 149 automatic joint omissions. The resulting counts are
72 joint assignments, 2,063 joint omissions, and 100 disagreements; positive
agreement is 59.0%. Human confirmation of the retained labels is author-reported.

## Inspecting a source record

For the cached-buffer example, start with `data/final/iree/12161/report.md`
and its `fixes/` patch, then join its cause ID to
`historical-alignment.csv`, `rq-cause-analysis.csv`, and
`rq6-diagnostic-attempts.csv`. The pre-fix publication differs from the
repaired write, success check, and publication. This source comparison explains
the example figure; the figure is not inferred by the numerical script.

Three corrected JAX anchors are documented in
`results/source-location-corrections.json`. Repair-side helpers remain in
required/repaired propagation paths, while pre-fix operations occupy the
causal-location fields. These checks preserve the taxonomy and distributions;
they are revision-time evidence checks, not new human localization measurements.

## Diagnostics, figures, and manuscript

The RQ6 table's `evidence_path` identifies each archived command/output record.
An attempt can group multiple commands or allocator configurations: the
run-trace comparison and the four run-module allocator configurations each
count as one attempt. Diagnostic observations remain separate from later
source-based explanations. This verification run inspects the records; it
does not build old compiler revisions or rerun the attempts.

Paper sources are maintained in the separate TensorBound-paper repository.
Its README documents LaTeX compilation, a blue revision copy, a clean reading
copy, and ReportLab figure generation. Supplied PDF figures support compilation
without regenerating them. The matrix generator accepts this release's
`rq-analysis-summary.json` explicitly. TeX fonts, ReportLab, compiler builds,
device hardware, and model services are outside the standard-library numerical
verification environment.
