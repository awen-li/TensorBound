# Final study outputs

The primary analysis contains 264 reports and 149 distinct repaired causes.
`rq-cause-analysis.csv` stores final normalized labels; `rq-cause-paths.jsonl`
stores the source-linked paths. The RQ-specific CSV and JSON files contain the
published records and aggregate results.

`rq6-diagnostic-attempts.csv` uses the manuscript's five diagnostic fields:
failing-path coverage, effect detection, effect localization, cause
localization, and cause explanation.

`selection.csv` contains the frozen primary-cohort membership of all 7,509
collected records. `selection-summary.json` preserves the published cohort-flow
totals. Intermediate per-pass selection columns are omitted. The 19 positives
found by the later recheck remain outside this frozen cohort and appear in the
separate sensitivity files.

`source-baselines.json` records the full 40-character commit identifier for
each of the five common implementation snapshots used during source tracing.
Report-specific historical revisions remain in the case and alignment records.

`scope-boundary-examples.csv` links three included/excluded pairs to archived
source evidence and states the decisive construct distinction for each pair.
Its seventh row illustrates a separate selection limit: the record establishes
a symptom, while the proximal mechanism remains unresolved. The final
cohort-membership field is checked against `selection.csv`.

Validity-obligation labels are assigned only after a record passes this scope
test. In particular, `transformation equivalence` and `value semantics
preservation` do not admit a transformation-local or numerical defect without
an established value--operation, storage, representation, ownership, or
execution relation.

`statistics/` contains final agreement summaries, the 142 records that received
a final-codebook recheck, the 19 observed omissions, and exact report-level
sensitivity results. Another 2,374 examined exclusions retain earlier negative
decisions. The targeted recheck does not estimate population recall. The
artifact omits intermediate A/B coding files, review scripts, prompts, and
blank audit templates.

Every row in `statistics/omitted-positive-sensitivity.csv` includes an
`evidence_path` to an archived report snapshot. The three positives drawn from
screening exclusions are archived alongside the 16 positives whose reports
were already retained for detailed review. The release records final decisions
for all 142 rechecks; it lacks raw snapshots for 63 screen-excluded negatives.
The three added snapshots record their 2026-10-15 retrieval time and GitHub
response metadata. They support evidence inspection; they do not recreate the
historical screening run.

`statistics/construct-boundary-sensitivity.json` removes all 23 causes in the
value--operation/identity relation family from a secondary calculation while
leaving their final scope labels unchanged. The remaining 126 causes preserve
the three leading failure families, with a different order. The repository-level
consequence counts in `rq5-impact-summary.json` keep the five query frames
visible instead of interpreting the pooled count as a common population rate.

The authors report that two human reviewers confirmed all final repair-centered
accounts after iterative review and correction. Item-level reviewer logs are
unavailable. Archived A/B agreement is descriptive: original actor identities
and a complete pre-comparison definition freeze remain unverified.
Pre-adjudication agreement on exact source operations remains unmeasured. The
taxonomy agreement summary retains the A, B, and reconciled failure-family
marginals used for the ranking-sensitivity result. The taxonomy codebook is a
dated documentation snapshot.

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
149 selected final decisions, 149 selected final primary obligations, and zero
unresolved final fields. These are repair-supported coding outcomes, not an
independent causal-localization result. It represents unavailable pre-adjudication candidate
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

`human-validation-coverage.json` records the authors' account of the completed
human review. Both reviewers inspected all 149 final cause records against the
archived reports, source revisions, operations, commits, patches, and available
corroborating evidence. Their review covered primary and secondary coding,
provenance endpoints, and the 13 duplicate links across 162 repair-linked
reports. Independent pre-adjudication localization agreement remains
unmeasured.

`artifact-manifest.json` records the size and hash of every released data and
result file, excluding the manifest itself. The validator checks these hashes
and recomputes the main distributions. Source data and final labels are
final; selection and audit exports omit intermediate-review fields.
