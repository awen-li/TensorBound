# Repair-grounded evidence and interpretation

A landed change is evidence for a causal reconstruction, not proof of a unique
root cause. The study is retrospective: the patch guides source inspection,
so tracing must also follow the pre-fix path upstream of the edited lines.

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

For a multi-hunk repair, classify the semantic action that restores the violated
obligation, not the largest edit. Keep build changes, tests, and other supporting
edits in the complete patch. The patch index lists all changed files; it does
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
