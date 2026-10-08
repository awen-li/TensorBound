# TensorBound Artifact

Data, final results, and collection code for **TensorBound: Binding Validity
in AI Compiler and Runtime Stacks**.

The artifact contains a primary cohort of 264 public reports, including 162
repair-linked reports grouped into 149 distinct repaired causes. It also
preserves final selection and exclusion-audit results for the 7,509-record
collection frame.

## Quick start

Python 3.10 or newer is sufficient; the runtime has no third-party dependencies.
No network access, GPU, compiler installation, or API token is needed to check
the packaged results.

```sh
git clone https://github.com/awen-li/TensorBound.git
cd TensorBound
make reproduce-main
make test
```

Without Make, run `python3 scripts/verify_artifact.py` and
`python3 -m unittest discover -s tests -v`.

## Contents

| Path | Contents |
|---|---|
| `data/raw/` | Archived GitHub evidence and CVE List records |
| `data/final/` | 264 case reports, their index, and archived repair patches |
| `results/` | Final RQ tables, source paths, codebooks, selection, and sensitivity results |
| `results/causal-selection-audit.csv` | Final decision, primary-obligation status, selection basis, and evidence links for 149 repaired causes |
| `results/human-validation-coverage.json` | Scope, evidence, coded fields, and outcome of the completed two-human review |
| `results/statistics/` | Published agreement summaries and final exclusion-audit data |
| `results/artifact-manifest.json` | Data inventory with sizes, hashes, and source revision |
| `tensorbound/` | GitHub/CVE collection code and supporting storage utilities |
| `scripts/verify_artifact.py` | Read-only result and integrity checks |
| `tests/` | Offline collector and artifact tests |

The artifact excludes intermediate coding outputs, review execution scripts,
review prompts, blank audit forms, manuscript revision assessments, paper
sources, and development caches. Raw upstream evidence is retained unchanged.
The development repository is maintained separately.

## Reproduction and collection

- [REPRODUCE.md](REPRODUCE.md) maps the results to their inputs and checks.
- [COLLECTION.md](COLLECTION.md) describes optional collection in a separate workspace.
- [INSTALL.md](INSTALL.md) gives local and container setup.
- [results/README.md](results/README.md) explains the final outputs and their scope.

The offline checks recompute distributions and sensitivity results from final
records. They also verify that the 162 repair-linked reports map to 149 final
cause selections and 13 duplicates, with zero unresolved final decisions or
primary obligations. The published agreement summaries are preserved statistics; this
artifact does not rerun the historical review process or establish independent
causal-localization reliability.

The 102 report-only cases establish an in-scope failure and observed effect,
but every record lacks an archived landed patch and public fix URL for the
pre-fix/repaired causal contrast. Seventy remain open, 32 are closed, and 100
contain a proposal or workaround. These reports remain in report-level
analyses.

Both human reviewers inspected all 149 final cause records and their underlying
source and repair evidence. Their review covered primary labels, secondary
obligations, provenance endpoints, and duplicate links. Identified issues were
corrected against the evidence, and both reviewers agreed on the final records.
This iterative confirmation is documented separately from unmeasured blinded
pre-adjudication localization agreement.

## License and data use

TensorBound-authored code and documentation use the [Apache License 2.0](LICENSE).
Archived upstream records and patches retain their original terms; see
[THIRD_PARTY.md](THIRD_PARTY.md) and [ETHICS.md](ETHICS.md).
