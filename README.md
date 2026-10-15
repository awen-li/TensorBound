# TensorBound Artifact

Data, final results, and collection code for **TensorBound: A Repair-Centered
Study of Tensor Binding Failures in AI Compiler and Runtime Stacks**.

The artifact contains an analysis cohort of 264 public reports, including 162
repair-linked reports grouped into 149 distinct repaired causes. A targeted
recheck found 19 additional in-scope reports. The released selection table
records frozen primary-cohort membership for the 7,509-record collection
frame; population recall remains unknown.

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
| `data/raw/` | Archived evidence for released GitHub and CVE List records |
| `data/final/` | 264 case reports, their index, and archived repair patches |
| `results/` | Final RQ tables, source paths, codebooks, selection, and sensitivity results |
| `results/causal-selection-audit.csv` | Final decision, primary-obligation status, selection basis, and evidence links for 149 repaired causes |
| `results/human-validation-coverage.json` | Scope, evidence, coded fields, and outcome of the completed two-human review |
| `results/scope-boundary-examples.csv` | Three paired construct-boundary examples and one mechanism-documentation example, each linked to archived evidence |
| `results/statistics/` | Agreement summaries, 142 final-codebook rechecks, and the 19 observed omissions |
| `results/artifact-manifest.json` | Data inventory with sizes, hashes, and source revision |
| `tensorbound/` | GitHub/CVE collection code and supporting storage utilities |
| `scripts/verify_artifact.py` | Read-only result and integrity checks |
| `tests/` | Offline collector and artifact tests |

The artifact excludes intermediate coding outputs, review execution scripts,
review prompts, blank audit forms, manuscript revision assessments, paper
sources, and development caches. Archived evidence preserves technical content;
credential-bearing notification links are replaced with a redaction marker.
The development repository is maintained separately.

## Reproduction and collection

- [REPRODUCE.md](REPRODUCE.md) maps the results to their inputs and checks.
- [COLLECTION.md](COLLECTION.md) describes optional collection in a separate workspace.
- [INSTALL.md](INSTALL.md) gives local and container setup.
- [results/README.md](results/README.md) explains the final outputs and their scope.

The offline checks recompute distributions and observed sensitivity results
from final records. They also verify that the 162 repair-linked reports map to
149 final cause selections and 13 duplicates, with zero unresolved final
decisions or primary obligations. The checks verify packaged results; they do
not rerun the historical review process or measure independent localization
agreement.

The 102 report-only cases establish an in-scope failure and observed effect,
but every record lacks an archived landed patch and public fix URL for the
pre-fix/repaired causal contrast. Seventy remain open, 32 are closed, and 100
contain a proposal or workaround. These reports remain in report-level
analyses.

The authors report that two human reviewers inspected all 149 final cause
records and their source and repair evidence. Their review covered primary
labels, secondary obligations, provenance endpoints, and duplicate links.
Identified issues were corrected against the evidence, and both reviewers
confirmed the final records. Item-level reviewer logs and blinded
pre-adjudication localization choices are unavailable.

The paired scope examples contrast operation semantics with storage-offset
preservation, numerical semantics with external representation, and compiler
metadata with runtime ownership. A separate evidence-limited example records a
reproducible symptom whose causal operation remains unidentified. These records
distinguish construct exclusion from insufficient mechanism documentation.

## License and data use

TensorBound-authored code and documentation use the [Apache License 2.0](LICENSE).
Archived upstream records and patches retain their original terms; see
[THIRD_PARTY.md](THIRD_PARTY.md) and [ETHICS.md](ETHICS.md).
