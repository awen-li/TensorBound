# Reproducing the results

Run `make reproduce-main`. The verifier reads the published records, checks
their SHA-256 hashes, and prints a JSON summary with `"status": "ok"`.
It makes no changes to the data and requires no network access.

| Result | Input |
|---|---|
| Declared 7,509-record frame | `results/collection-frame.json` |
| 264 retained and 7,245 excluded records | `results/selection.csv` |
| 162 repair-linked reports and 149 causes | `data/final/cases.csv` |
| RQ1, RQ3, RQ4, RQ5 distributions and cross-tabs | `results/rq-cause-analysis.csv` |
| RQ2 origin and consumer layers; 71 differing endpoint pairs | `results/rq2-provenance-analysis.csv` |
| RQ4 secondary obligations: 68 causes, 72 labels | `results/rq-cause-analysis.csv` |
| RQ5 consequences for 264 reports | `results/rq5-report-impacts.csv` |
| A/B and reconciled failure-family marginals | `results/statistics/taxonomy-agreement-summary.json` |
| RQ6 four diagnostic attempts | `results/rq6-diagnostic-attempts.csv` |
| 142 final-codebook rechecks and 19 observed omissions | `results/statistics/exclusion-audit.jsonl`, `omitted-positive-sensitivity.csv` |
| Full versus 136-cause stronger-evidence distributions | `results/historical-alignment.csv`, `results/evidence-sensitivity.json` |
| 167 indexed repair patches | `results/repair-patch-index.jsonl` and the referenced patches |
| 45 taxonomy definitions and examples | `results/taxonomy-codebook.json` |
| 149 selected repair-supported decisions and zero unresolved final fields | `results/causal-selection-audit.csv`, `results/causal-selection-audit-summary.json` |
| Complete two-human review coverage for final cause records | `results/human-validation-coverage.json` |
| Three paired scope boundaries and one evidence-limited exclusion | `results/scope-boundary-examples.csv` and its archived evidence paths |

The manifest verifies the packaged data inventory and file contents. The
collection totals are checked against the declared frame; this is not a new
enumeration of GitHub's changing index or verification of historical search
completeness. Source paths and case records preserve the available archived
evidence.

The causal-selection audit joins the cohort, cause analysis, and historical
alignment tables. It verifies the final repair-linked denominator and publishes
the obligation-selection basis for each cause. Pre-adjudication candidate
lists are absent from the archive, so counts of initially competing operations
or obligations remain unavailable.

The agreement summaries in `results/statistics/` are final reported outputs.
They are hash-checked rather than regenerated from intermediate reviewer
decisions. The released audit contains the 142 records that received a
final-codebook recheck. The remaining 2,374 examined exclusions retain earlier
negative decisions. Because the rechecked subset was targeted rather than
probability sampled, the artifact reports the 19 observed omissions and leaves
population recall unknown.

Run `make test` for offline tests of collection, record validation, CLI safety,
and package consistency. These tests use synthetic responses and perform no
live collection. Historical compiler failures are not rerun.
