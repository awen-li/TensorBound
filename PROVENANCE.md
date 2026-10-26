# Evidence and decision provenance

This inventory describes the released files. Missing history is left missing;
a checksum or successful numerical check is not evidence of independent judgment.

| Material | Released location and status | Supported check / remaining limit |
|---|---|---|
| Collection identifiers | `results/selection.csv`: 7,509 distinct IDs | Final membership; initial per-record filtering decisions are outside this release |
| Query specification | `results/collection-frame.json` | Required labels, creation cutoff, partitions and reported yields; full historical query-response logs are outside this release |
| CVE evidence | `data/raw/cve-list/*/{source,record}.json`: 693 IDs and distinct fixed-revision paths | Unique retrieved records and available capture timestamps; not completeness of the historical live search index |
| GitHub evidence | `data/raw/github/*/*/report.json`: 783 snapshots | The 780 advanced reports and three older audit-positive snapshots; not all 6,816 issue bodies |
| Historical source | `results/source-baselines.json`, `historical-alignment.csv`, indexed patches and case evidence | Common lookup snapshots are separate from report-specific pre-fix/fixed revisions; complete source trees are not bundled |
| Final cause selection | `results/causal-selection-audit.csv`, `data/final/cases.csv` | Reconciled operation/obligation and 162-to-149 mapping; original independent selections and candidate counts not recorded |
| Historical human review | `results/human-validation-coverage.json` | Author-reported coverage of all 149 causes and linked evidence; individual reviewer logs unavailable |
| Phase 1 human review | `results/statistics/phase1-human-review.json` | Author-reported audit and review-step totals; sample membership, individual labels and restoration links unavailable |
| Taxonomy assignments | `results/statistics/archived-taxonomy-assignments.jsonl` | Original A/B label values and available evidence references; actors and pre-comparison codebook snapshot unknown |
| Derived secondary view | `verify_archived_assignments()` in `scripts/verify_artifact.py` | Removes each final primary family from both secondary sets without editing the source arrays |
| Model-review history | Procedure/configuration summarized in the manuscript | Model inputs, outputs, prompts and full run manifests are outside this clean release; exact model replay unsupported |
| Diagnostic attempts | `results/rq6-diagnostic-attempts.csv` and each `evidence_path` | Preserved commands/output and separate later diagnosis; current reruns untested |
| Source-location corrections | `results/source-location-corrections.json` | Assistant evidence checks dated 2026-10-25; separate from historical human confirmation |
| Derived results | RQ summaries, sensitivity outputs and `--result-index` | Counts, matrix cells and member IDs from final labels; human-review tables remain aggregate-only |
| Manuscript and figures | Separate TensorBound-paper repository | LaTeX/ReportLab dependencies and build instructions are documented there |

## Meaning of unavailable fields

- `not_recorded` means the available history lacks that metadata.
- `unavailable_in_release` means this package cannot supply it; the field is
  not assigned a zero count or a negative decision.
- `not independently measured` applies to pre-reconciliation localization
  agreement. Final human confirmation is a different reported procedure.
- `not evaluated` in RQ6 applies when an attempt misses the failing path.

Stable A/B labels identify archived assignment sets, not identified human
reviewers. Their actor type remains `unknown`. Their original labels, the final
taxonomy assignments, and the derived secondary-family mask occupy separate
records. One-pass-only secondary labels remain available in the input arrays;
the intersection policy does not establish that those labels are false.

## Dates and versions

The GitHub issue-creation cutoff is 2026-09-24. Released GitHub evidence was
captured between 2026-09-27 and 2026-10-15; those timestamps describe snapshots,
not eligibility or original label-application times. CVE snapshots have capture
times from 2026-09-25T02:45:33Z to 2026-09-25T03:15:45Z and identify the fixed
CVE repository revision. Capture time, reported affected version, common source
baseline, pre-fix revision, and repaired revision are separate concepts.

The manuscript reports historical verification on Ubuntu 24.04.3 LTS with
Python 3.12.3 and model calls through Codex CLI 0.155.0 using the alias
`gpt-daybreak-blue-latest` with high reasoning effort. An immutable model
revision, temperature, top-p, and seed were not recorded. These historical
claims are separate from the environment actually recorded in VERIFICATION.json.

## Redactions and release integrity

Notification-service URLs containing capabilities are replaced with redaction
markers. The manifest hashes the released bytes, which can differ from the
original reviewed bytes. No comparison with a missing original packet hash is
claimed. Public report IDs, technical source locations and patch links remain
available for evidence inspection.

The numerical verification route uses no model service or live retrieval.
Optional collection writes to a separate user-selected workspace and does not
replace the frozen analysis records. The earlier 142-record audit is preserved
in release `e70da3b`, not mixed with the current 600-record audit.
