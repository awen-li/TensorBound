# Final analysis records

The report set contains 264 records. Its 162 repair-linked reports map to
149 distinct repaired causes; 13 is the number of extra report links, not
the number of cause groups with duplicates. The other 102 reports contribute
to consequence analysis only.

## Membership and human-review totals

`selection.csv` records final membership of all 7,509 collection identifiers:
264 retained and 7,245 outside the final set. This final exclusion total differs
from the 6,729 initial-filtering exclusions. `selection-summary.json` records
the reported selection stages.

`statistics/phase1-human-review.json` contains the authors' current review
totals and the calculation's expected values. The three 200-record samples
have 0/3/1 binding failures and 198/195/199 pre-discussion agreements.
Per-record sample lists, human labels, restoration links, reviewer IDs, and
review dates are unavailable in the released material. The 28 restorations
count reversed removal proposals; distinct restored-record membership cannot
be checked here. The final 264 already includes restored reports. The four
audit positives are separate from the analysis.

The old 142-record targeted audit and 19-omission sensitivity remain in the
previous artifact commit `e70da3b`. They are not inputs to the current paper's
Phase 1 estimates.

## Cause records and source locations

`rq-cause-analysis.csv` stores final taxonomy labels.
`historical-alignment.csv` separates affected, pre-fix, and fixed revisions,
selected operation, repair site, patch site, and evidence flags.
`causal-selection-audit.csv` joins these fields and the final cause explanation.
The fields describe reconciled selections, not independent reviewer choices.

`rq-cause-paths.jsonl` records implementation-specific relations and propagation
steps. Some steps describe required or repaired behavior. They are not a list
of exclusively pre-fix executed operations. The three corrected JAX records
explicitly name this path kind and their separate pre-fix decision.
`source-location-corrections.json` preserves the old/new anchors, alternatives,
evidence, and label checks made during the 2026-10-25 revision.

`source-baselines.json` identifies the five lookup snapshots. Case-specific
historical revisions, rather than those common snapshots, support each repair
comparison. `scope-boundary-examples.csv` gives three included/excluded pairs
and one evidence-limited exclusion.

## Taxonomy and agreement

`taxonomy-codebook.json` supplies 45 definitions and example links.
`statistics/archived-taxonomy-assignments.jsonl` preserves the A/B label
values used in the published comparison, including their original secondary
arrays. Original actor identities and a complete pre-comparison codebook
snapshot are unavailable. The numerical script derives the primary-family mask
without editing those arrays.

Final secondary labels are their masked intersection, with human confirmation
reported in `human-validation-coverage.json`. A label assigned by one set only
is omitted under this policy, not adjudicated false. The comparison yields
2,235 binary pairs, including 149 automatic primary-family omissions, and
72 retained labels in 68 causes. Positive agreement is 59.0%.

The archived failure label `premature or unordered transition` is displayed
as **operation before required event** in the current paper. This presentation
alias preserves its eight-family partition and counts; raw A/B values retain
their original strings. All other full family names correspond directly.

`human-validation-coverage.json` records the authors' account that both humans
checked all 149 causes and the report-to-cause mapping. Individual review logs
and independent pre-discussion source-operation choices are unavailable.
The October 25 source-location corrections are a separate evidence-check record.

## RQ outputs and verification

RQ CSV/JSON files contain final per-record classifications and derived
distributions. `evidence-sensitivity.json` compares the full cause set with
the 136 cases having a regression test or attributed diagnosis.
`statistics/construct-boundary-sensitivity.json` excludes 23 operation-identity
causes from a separate calculation, leaving 126.

`rq6-diagnostic-attempts.csv` separates archived diagnostic observations from
later explanations. Four attempts concern three causes; more than one command
can belong to an attempt.

`artifact-manifest.json` hashes released data and result files, excluding
itself. `release-info.json` identifies the analysis revision and codebook.
The verifier recomputes distributions, memberships, archived label comparison,
and arithmetic from supplied review totals. The result index exposes supporting
IDs. See [PROVENANCE.md](../PROVENANCE.md) for missing evidence and
[REPRODUCE.md](../REPRODUCE.md) for commands.
