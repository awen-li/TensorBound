# [inductor][CUDA][bf16] Fusion skips rounding before equality, producing zero tie counts

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/195214](https://github.com/pytorch/pytorch/issues/195214)
- Opened: 2026-08-28T21:10:52Z
- Closed: 2026-09-02T16:16:01Z
- Labels: `triaged`, `module: bfloat16`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `pt2: numerics-discrepancy`, `bot-triaged`

## Reported behavior

Human context: this was isolated from a real transformer training failure where
compiled execution produced non-finite gradients. Correct compiled execution is
important here; disabling compilation is not an acceptable long-term solution.

Codex was used to minimize and verify the reproducer and to draft the technical
report below.

<details>
<summary>Codex-assisted technical report</summary>

## Describe the bug

On CUDA, Inductor fuses a BF16 division into a following exact equality without
preserving the BF16 rounding boundary. Eager execution materializes `scaled` as
BF16. Inductor appears to compare a higher-precision division result against the
separately materialized BF16 `maximum`, so every equality is false.

This is correctness-critical in an argmax-backward-style tie count: eager finds
one or more maxima per row, while Inductor returns zero for every row. A
subsequent division by the tie count produces infinities. The original model
failure surfaced as non-finite compiled transformer gradients, but the minimized
reproducer contains no attention code.

### Minimal reproducer

```python
import math
import os

import torch


if os.environ.get("EMULATE_PRECISION_CASTS") == "1":
    torch._inductor.config.emulate_precision_casts = True

torch.manual_seed(0)
values = torch.randn((6, 4, 1, 33), device="cuda", dtype=torch.bfloat16)
scaled = values / math.sqrt(32.0)
maximum = scaled.amax(dim=-1, keepdim=True)
numerator = torch.randn((6, 4, 1, 1), device="cuda", dtype=torch.bfloat16)


def fn(maximum, values, numerator):
    scaled = values / math.sqrt(32.0)
    tie_count = (maximum == scaled).sum(dim=-1, keepdim=True)
    return numerator / tie_count, tie_count


expected, expected_count = fn(maximum, values, numerator)
actual, actual_count = torch.compile(fn, fullgraph=True)(
    maximum, values, numerator
)

print(torch.__version__, torch.version.git_version)
print("eager:", expected_count.flatten().tolist(), bool(torch.isfinite(expected).all()))
print("compiled:", actual_count.flatten().tolist(), bool(torch.isfinite(actual).all()))
torch.testing.assert_close(actual_count, expected_count)
torch.testing.assert_close(actual, expected)
```

### Actual result

With both PyTorch `2.13.0+cu126` and the 2026-08-28 CUDA 12.6 nightly:

```text
eager:    [1, 1, ..., 2, ..., 1] True
compiled: [0, 0, ..., 0, ..., 0] False

AssertionError: Tensor-likes are not equal!
Mismatched elements: 24 / 24 (100.0%)
```

### Ablations and workaround

- `backend="aot_eager"` matches eager exactly.
- Inductor with
  `torch._inductor.config.emulate_precision_casts = True` matches eager exactly.
- The failure remains when mix-order reductions are disabled.
- The failure reproduces on today's nightly, so it is not fixed on current
  mainline wheels.

This looks related to #179561 / #180575, but that fix preserves explicit
`fp32 -> bf16 -> fp32` casts. This graph has no explicit cast round trip: it
needs the implicit BF16 output rounding of the pointwise division to be
preserved before the exact equality. The current automatic low-precision
pointwise-barrier logic appears not to cover this consumer pattern.

The exact equality is intentional rather than a numerical-comparison test: both
operands represent the same BF16-scaled values, and the tie count in the maximum
backward must be positive.

## Versions

Reproduced on both:

- PyTorch `2.13.0+cu126`, git
  `cf30153c4c131c8164ee7798e5022d810682e2cb`
- PyTorch `2.15.0.dev20260828+cu126`, git
  `2695a9a323d60c3be552e880c02d62ee10de6327`

Environment:

```text
OS: Ubuntu 24.04.3 LTS under WSL2
Python: 3.12.3
GPU: NVIDIA GeForce RTX 4070 Laptop GPU (compute capability 8.9)
CUDA used to build PyTorch: 12.6
Triton: 3.7.1 (stable environment)
```

</details>


cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

A BF16 division feeds equality with a separately materialized BF16 maximum inside a fused kernel.

## Observed implementation decision

Default lowering compares a higher-precision intermediate without the eager BF16 materialization boundary.

## Physical effect

Values that are equal after BF16 rounding compare unequal before rounding, producing zero tie counts.

## Observed consequence

The compiled program produces non-finite outputs while eager execution remains finite.

## Reported proposal or workaround

The existing emulate_precision_casts compilation option preserves the rounding boundary; PR 195223 proposed automatic preservation for comparison consumers.

## Repair trace

The report was closed after identifying an existing opt-in configuration, while PR 195223 was closed without merge. Configuration-based resolution does not supply a landed default-path repair or a fixed regression for the reported compilation mode.

## Archived evidence

- Raw report: `data/raw/github/pytorch/195214/report.json`
- SHA-256: `11976c710ed1e6d1339b30313dcbe45eed5f1b8846ca39303774a23b3029a541`
