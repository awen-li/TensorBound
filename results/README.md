# Final study outputs

The primary analysis contains 264 reports and 149 distinct repaired causes.
`rq-cause-analysis.csv` stores final normalized labels; `rq-cause-paths.jsonl`
stores the source-linked paths. The RQ-specific CSV and JSON files contain the
published records and aggregate results.

`selection.csv` contains the final membership of all 7,509 collected records.
`selection-summary.json` preserves the published cohort-flow totals. Intermediate
per-pass selection columns are omitted.

`statistics/` contains final agreement summaries, final exclusion-audit
decisions and weights, the 19 observed omissions, and report-level sensitivity
results. The artifact omits intermediate A/B coding files, review scripts,
prompts, and blank audit templates. The original aggregate statistics are
preserved without treating them as new independent measurements.

Two human reviewers confirmed the final causal reconstructions after iterative
review and correction. Archived A/B agreement is descriptive: original actor
identities and a complete pre-comparison definition freeze remain unverified.
Pre-adjudication agreement on exact source operations remains unmeasured.
The taxonomy codebook is a dated documentation snapshot.

`historical-alignment.csv` records revisions, source locations, and available
causal evidence. `evidence-sensitivity.json` compares all 45 families with the
136-cause stronger-evidence subset. The largest share change is 2.44 percentage
points, and the leading repair family changes. The 19 audit omissions support
report-level sensitivity and remain outside the repaired-cause distributions.

`artifact-manifest.json` records the size and hash of every released data and
result file, excluding the manifest itself. The validator checks these hashes
and recomputes the main distributions. Source data and final labels are
unchanged; selection and audit exports omit intermediate-review fields.
