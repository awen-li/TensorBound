# [torch.compile] as_strided produces incorrect outputs for out-of-bounds strided tensor

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/197431](https://github.com/pytorch/pytorch/issues/197431)
- Opened: 2026-09-17T17:28:25Z
- Closed: Open at collection
- Labels: `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `topic: fuzzer`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug

Here is the code to reproduce and it is found by our fuzzer.

```python
import os
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
import torch
def fn(x):
    full = x.repeat((3, 2))
    return torch.as_strided(full[:, ::2], (3, 2), full.stride())
torch.manual_seed(0)
x = torch.randn(1, 1)
out1 = fn(x)
# print(out1)
out2 = torch.compile(fn)(x)
# print(out2)
torch.testing.assert_close(out1, out2, equal_nan=True)

```

The difference between eager and torch.compile is nondeterministic, but the assertion error always trigger like this:

```
AssertionError: Tensor-likes are not close!

Mismatched elements: 3 / 6 (50.0%)
Greatest absolute difference: 1.5409960746765137 at index (1, 1) (up to 1e-05 allowed)
Greatest relative difference: inf at index (1, 1) (up to 1.3e-06 allowed)
```

Analysis:

1. `repeat` is necessary to be placed in the compiled function.
2. when directly print `out2`, a runtime error is raised: `RuntimeError: setStorage: sizes [2], strides [1], storage offset 2, and itemsize 4 requiring a storage size of 16 are out of bounds for storage of size 12`

### Error logs

_No response_

### Versions

Versions of relevant libraries:
[pip3] No relevant packages
[conda] numpy                     2.1.3                    pypi_0    pypi
[conda] torch                     2.14.0                   pypi_0    pypi

cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises torchInductor materialization of a lazy view used as the base of storage-relative as_strided.

## Observed implementation decision

Stride enforcement copies only the view's logical elements into a smaller buffer before applying a storage-relative offset.

## Physical effect

The derived view addresses storage outside the materialized allocation instead of the original base.

## Observed consequence

Compiled output disagrees with eager as_strided behavior.

## Reported proposal or workaround

PR 197712 proposes realizing the base rather than copying the view.

## Repair trace

PR 197712 was not merged, so the report has no landed base-preservation repair and regression.

## Archived evidence

- Raw report: `data/raw/github/pytorch/197431/report.json`
- SHA-256: `4690f96189c51d90bf5304a29f2f7d3bd1c17875a4674d2cad62ced067583fb5`
