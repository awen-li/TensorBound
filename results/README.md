# Final study outputs

The primary analysis contains 264 reports and 149 distinct repaired causes.
`rq-cause-analysis.csv` stores final normalized labels; `rq-cause-paths.jsonl`
stores the source-linked paths. The RQ-specific CSV and JSON files contain the
published records and aggregate results.

`selection.csv` contains the final membership of all 7,509 collected records.
`selection-summary.json` preserves the published cohort-flow totals. Intermediate
per-pass selection columns are omitted.

`scope-boundary-examples.csv` links three included/excluded pairs to archived
source evidence and states the decisive construct distinction for each pair.
Its seventh row illustrates a separate selection limit: the record establishes
a symptom, while the proximal mechanism remains unresolved. The final
membership field is checked against `selection.csv`.

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

`causal-selection-audit.csv` contains one row for each repaired cause. It joins
the final first-invalid-decision site and primary obligation to patch, source-
contrast, regression, diagnosis, reproduction, and case-path evidence. Each
row also reproduces the case report's concrete causal basis. The 149 row-level
report counts sum to 162 repair-linked reports. The summary records
149 unique final decisions, 149 unique final primary obligations, and zero
unresolved final fields. It represents unavailable pre-adjudication candidate
counts as JSON `null` with status `not_recorded`.

The 13 causes outside the 136-cause stronger-evidence subset retain a landed
patch and pre-fix/repaired source contrast. Seven have one coded obligation;
six have secondary obligations and a row-level explanation of how the semantic
repair selects the primary condition. Six of the 13 also have public
reproduction evidence.

For the 102 report-only cases, `causal-selection-audit-summary.json` records 70
open and 32 closed reports. Every case lacks a patch and public fix URL for the
source contrast required by the cause-level analysis. One hundred record a
proposal or workaround.

`human-validation-coverage.json` records the completed human review. Both
reviewers inspected all 149 final cause records against the archived reports,
source revisions, operations, commits, patches, and available corroborating
evidence. Their review covered primary and secondary coding, provenance
endpoints, and the 13 duplicate links across 162 repair-linked reports. The
record documents final consensus after evidence-based correction; independent
pre-adjudication localization agreement remains a separate unmeasured value.

`artifact-manifest.json` records the size and hash of every released data and
result file, excluding the manifest itself. The validator checks these hashes
and recomputes the main distributions. Source data and final labels are
unchanged; selection and audit exports omit intermediate-review fields.
