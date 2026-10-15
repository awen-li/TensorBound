# Repair-grounded evidence and interpretation

A landed change supports a repair-centered account; it does not prove a unique
root cause. The study is retrospective: the patch guides source inspection,
and tracing also follows the pre-fix path upstream of the edited lines.

## Preserved fields

| Evidence | Released location |
| --- | --- |
| Affected, pre-fix, and fixed revisions | `historical-alignment.csv` |
| First invalid decision, semantic repair site, edited patch site | Separate columns in `historical-alignment.csv` |
| Native relation, obligation, physical effect, semantic repair action | `rq-cause-analysis.csv` and each case's `report.md` |
| Source-path evidence | `rq-cause-paths.jsonl` and case bundles |
| Full patches, including supporting and test edits | `data/final/*/*/fixes/` |
| All changed file paths and patch hashes | `repair-patch-index.jsonl` |
| Patch, contrast, regression, developer diagnosis, reproducer support | Five separate columns in `historical-alignment.csv` |
| Attributed explanation and reproduction evidence | Case review notes and archived public evidence |
| Final normalized labels | `rq-cause-analysis.csv` |

The first invalid decision is the earliest obligation violation; the repair
site is the semantic decision changed by the fix; the patch site is an edited
source location. These locations need not coincide. Existing site fields vary
in granularity and are not substitutes for an independent exact-operation audit.

The final denominator audit combines these locations with the cause and report
tables. All 162 repair-linked reports map to 149 causes and 13 duplicate
reports. Every cause has one final first-invalid-decision site and one primary
obligation; the final cause table contains zero unresolved records. The archive
does not retain pre-adjudication candidate lists, so initial competing-
candidate counts remain unavailable. `causal-selection-audit.csv` records each
selected repair-supported decision and its source evidence.

For a multi-hunk repair, the semantic action that restores the violated
obligation determines the repair family. Build changes, tests, and other
supporting edits remain in the complete patch. The patch index lists all changed files; it does
not assign an independently reviewed semantic role to every hunk. A separate
per-hunk role annotation is not claimed.

Developer diagnosis remains attributed evidence. If it conflicts with source
reconstruction, retain both accounts, record their source references, compare
the competing paths, and document the adjudicated choice before changing the
coding. The artifact contains the final reconstructions, without a separate
conflict census.

## Stronger-evidence sensitivity

The 136-cause subset requires a landed patch, pre-fix/fixed contrast, and either
a regression test or an attributed developer diagnosis. All five distributions
are checked from row-level data by `scripts/verify_artifact.py`.
`evidence-sensitivity.json` publishes all 45 family counts, percentages,
and percentage-point changes. The leading relation, transition, obligation,
and failure families remain unchanged. The leading repair changes from mapping
correction to stronger eligibility checks. The largest absolute family-share
change is 2.44 percentage points.

The other 13 causes still have a landed patch and pre-fix/fixed source
contrast. Seven have one coded obligation. Six have a primary and at least one
secondary obligation; their rows in `causal-selection-audit.csv` state how the
semantic repair distinguishes the primary condition. Six of the 13 also have
public reproduction evidence.
