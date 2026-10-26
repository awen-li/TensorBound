# TensorBound artifact

Data, final analysis records, results, and collection code for **TensorBound:
Characterizing Tensor Binding Failures in AI Compiler and Runtime Stacks**.

The analysis contains 264 reports. Of these, 162 repair-linked reports map
to 149 distinct repaired causes. The current Phase 1 validation reports
600 sampled initial exclusions, four binding failures, and 28 reversals of
removal proposals. Restored reports are already included in the 264.

## Release and supported tasks

[release-info.json](results/release-info.json) identifies this revision, its
baseline manuscript, and data/code fingerprints. Use the commit linked by the
paper, then record `git rev-parse HEAD`.

The offline route supports:

- Recomputing RQ distributions, cross-tabulations, and sensitivity subsets
  from final report and cause records.
- Recomputing archived taxonomy-label agreement from its two assignment sets.
- Recomputing Phase 1 estimates and table arithmetic from **author-supplied
  human-review totals**. Individual human labels and sample membership are
  unavailable in this release.
- Inspecting archived reports, patches, source anchors, and diagnostic outputs.

The checks do not repeat human judgments, establish independent localization
agreement, rerun historical diagnostics, or replay historical model calls.

## Quick start

Run from this artifact checkout with Python 3.10 or later:

```sh
git rev-parse HEAD
make reproduce-main
make test
python3 scripts/verify_artifact.py --result-index
```

Without Make, run `python3 scripts/verify_artifact.py` and
`python3 -m unittest discover -s tests -v`. Analysis uses only the standard
library, without network access, GPU hardware, or API credentials. The verifier
prints JSON with `"status": "ok"` and exits zero when checks pass. A mismatch
exits nonzero; it never rewrites the supplied records to match expected counts.
The result-index command prints the identifiers supporting each table cell,
matrix cell, and sensitivity subset. Human-review tables are marked aggregate-only.

## Contents and provenance

| Path | Contents |
|---|---|
| `data/raw/` | Available GitHub snapshots and 693 unique CVE records |
| `data/final/` | 264 case reports, report-to-cause membership, and repair patches |
| `results/rq-*.csv`, `results/rq*.json` | Final RQ records and distributions |
| `results/historical-alignment.csv` | Pre-fix and fixed revisions, operation and repair sites, evidence flags |
| `results/source-location-corrections.json` | Three revision-time source-anchor corrections and their evidence |
| `results/statistics/phase1-human-review.json` | Supplied sample and review-step totals; explicit missing individual records |
| `results/statistics/archived-taxonomy-assignments.jsonl` | Original A/B assignment values used in the published comparison; actor identities unknown |
| `results/human-validation-coverage.json` | Authors' account of the completed two-human cause review |
| `results/artifact-manifest.json` | Released data/result file sizes and hashes |
| `tensorbound/` | Optional GitHub/CVE collection code |
| `scripts/verify_artifact.py` | Offline numerical and integrity checks |
| `tests/` | Offline collector and artifact tests |

The artifact omits review-execution scripts, prompts, intermediate coding
outputs, blank review forms, development caches, and paper sources. The two
archived taxonomy assignment sets are retained as inputs to the reported
agreement analysis, separately from final labels and the derived secondary mask.

## Documentation

- [REPRODUCE.md](REPRODUCE.md): tested entry points and result-to-input index.
- [PROVENANCE.md](PROVENANCE.md): available evidence, missing history, and replay boundaries.
- [COLLECTION.md](COLLECTION.md): optional live collection in a separate workspace.
- [INSTALL.md](INSTALL.md): local and container requirements.
- [results/README.md](results/README.md): record definitions and analysis conventions.
- [VERIFICATION.json](VERIFICATION.json): environment, commands, outcomes, and limitations of the verification run.

The previous release `e70da3bfe4dc69e167015692d5510e28c18a9b00` preserves
the older targeted 142-record audit and 19-omission sensitivity. Those outputs
are superseded, not relabeled as the current 600-record human audit. Their
historical evidence remains recoverable from Git.

## License and data use

TensorBound-authored code and documentation use [Apache License 2.0](LICENSE).
Archived upstream evidence retains its original terms; see
[THIRD_PARTY.md](THIRD_PARTY.md) and [ETHICS.md](ETHICS.md).
Credential-bearing notification links are redacted.
