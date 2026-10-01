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
| RQ2 origin and consumer layers; 71 cross-layer paths | `results/rq2-provenance-analysis.csv` |
| RQ4 secondary obligations: 68 causes, 72 labels | `results/rq-cause-analysis.csv` |
| RQ5 consequences for 264 reports | `results/rq5-report-impacts.csv` |
| RQ6 four diagnostic attempts | `results/rq6-diagnostic-attempts.csv` |
| 19 omissions and report-level sensitivity | `results/statistics/exclusion-audit.jsonl`, `omitted-positive-sensitivity.csv` |
| Full versus 136-cause stronger-evidence distributions | `results/historical-alignment.csv`, `results/evidence-sensitivity.json` |
| 167 indexed repair patches | `results/repair-patch-index.jsonl` and the referenced patches |
| 45 taxonomy definitions and examples | `results/taxonomy-codebook.json` |

The manifest verifies the packaged data inventory and file contents. The
collection totals are checked against the declared frame; this is not a new
enumeration of GitHub's changing index or verification of historical search
completeness. Source paths and case records preserve the available archived
evidence.

The agreement summaries in `results/statistics/` are final reported outputs.
They are hash-checked, not regenerated from intermediate reviewer decisions.
The final exclusion decisions and sampling weights support omission
sensitivity without the review execution scripts. The interval estimates
remain published outputs; the verifier recomputes the weighted point estimates.

Run `make test` for offline tests of collection, record validation, CLI safety,
and package consistency. These tests use synthetic responses and perform no
live collection. Historical compiler failures are not rerun.
