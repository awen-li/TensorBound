# [inductor] Fused reduction + argmax/argmin mis-indexes `channels_last` inputs (logical index not preserved), CPU and CUDA

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/193751](https://github.com/pytorch/pytorch/issues/193751)
- Opened: 2026-08-17T00:56:54Z
- Closed: 2026-08-28T13:21:33Z
- Labels: `triage review`, `module: memory format`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

`torch.compile` returns a wrong `argmax`/`argmin` index when a trailing-dim reduction (`mean`, `sum`, `amax`, `var`, `logsumexp`, `prod`) is fused with the index consumer and the input uses the `channels_last` memory format (4D, non-contiguous). Eager is correct; the compiled index points at a value that is not the extreme of the reduction output. The generated kernel computes the reduction values correctly but reports the physical flat index of a non-row-major intermediate buffer instead of the logical row-major index.

### To reproduce

```python
import torch
torch.manual_seed(1234)
x = torch.randn(4, 4, 8, 8).to(memory_format=torch.channels_last)  # strides (256, 1, 32, 4)

def fn(t):
    return torch.argmax(torch.mean(t, dim=-1))

print("eager   =", fn(x).item())                # 14 (correct, value 0.751697 == max)
print("compiled=", torch.compile(fn)(x).item()) # 25 (wrong, value -0.410133)
```

All six ops fail the same way (2.13 CPU, seed 1234; nightly 2.15.0.dev20260816+cu130 reproduces with identical indices on CPU and CUDA):

| op | eager idx -> value | compiled idx -> value | extreme |
|---|---|---|---|
| mean + argmax | 14 -> 0.751697 | 25 -> -0.410133 | 0.751697 |
| sum + argmax | 14 -> 6.013576 | 25 -> -3.281064 | 6.013576 |
| amax + argmax | 28 -> 3.029174 | 19 -> 0.945478 | 3.029174 |
| var + argmax | 43 -> 2.989961 | 45 -> 0.611232 | 2.989961 |
| logsumexp + argmax | 54 -> 3.537204 | 58 -> 2.333352 | 3.537204 |
| prod + argmin | 34 -> -2.680438 | 40 -> 5.749767 | -2.680438 |

`channels_last_3d` (5D, NHWDC) reproduces identically, e.g. mean+argmax: eager 232 -> 1.272395, compiled 163 -> 0.385751 (2.13 CPU, seed 1234; reproduces on nightly and CUDA too).

Controls pass: contiguous input, `[..., ::2]`, and `expand`/stride-0 views; only flat `argmax`/`argmin` consumers are affected (`max`/`min` dim and `topk` consumers pass).

### Root cause (kernel evidence, 2.13 CPU)

The fused kernel `cpp_fused_argmax_mean_0` reads the input with correct channels_last strides and computes the mean correctly, but the argmax stage reports the flat loop position as the index while the intermediate mean buffer has strides `(32, 1, 4)` (non-row-major): physical position 25 corresponds to logical row-major position 14. The candidate value is correct; only the reported index is wrong.

### Why this is still open

The root cause is that the fused arg-reduction reports the physical flat index of a non-row-major intermediate buffer instead of the logical row-major index. We observed a producer-type split: variants whose arg-reduction input is an unrealized pointwise result (mean/var/logsumexp) can be made correct by enabling the logical-index path, while variants whose input is a materialized, contiguously-allocated reduction buffer (sum/amax/prod) remain wrong under the same change — the existing layout check does not fire for a contiguous allocation. Latest nightly (2.15.0.dev20260816+cu130) still reproduces all six variants on CPU and CUDA, so nothing in the current tree fixes this.

### Expected behavior

`torch.compile` should return the same logical row-major index as eager for channels_last inputs across all fused reduction ops.

### Versions

```text
PyTorch version: 2.13.0+cu130 and nightly 2.15.0.dev20260816+cu130 (both reproduce, CPU and CUDA)
OS: Linux (AutoDL container)
```

cc @jamesr66a @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Duplicate mapping

- Primary report: `github-pytorch-pytorch-193661`
- Shared cause: `cause-pytorch-193661-argreduce-index-bound-to-preorder-loop-counter`

A maintainer closes report 193751 as a duplicate of 193661. Both cases preserve the extreme value but return the final physical reduction-loop position rather than the logical row-major index after layout-driven loop reordering, and commit d607816 explicitly fixes 193661 while covering 193751.

The sections below describe the shared cause reconstructed from the primary report. The reported behavior above remains the observation specific to this report.

## Trigger

A trailing-dimension mean is fused into flat argmax on an input with noncontiguous leading dimensions and a stride-two reduction window.

## Relevant entities

logical row-major index; arg-reduction candidate value; explicit index expression; reduction loop variables; loop reordering; Triton reduction cache

## Implemented relation

An arg-reduction returns the logical row-major position of its winning value regardless of the physical order used to traverse candidates.

## Pre-state

Lowering sees an unrealized fused producer before the scheduler has chosen its final loop order.

## Binding transition

The old lowering decides that a separate logical index is unnecessary and binds the result index to the implicit native reduction counter.

## Post-state

The scheduler reorders loops to match physical strides; loads still access the correct values, but the implicit counter now enumerates the reordered physical traversal.

## Validity obligation

The logical index remains an explicit expression through all loop transformations unless a late proof shows that it equals the final native reduction order.

## Invalid decision

Index-elision was decided from an early layout observation that subsequent layout changes and loop reordering could invalidate.

## Physical effect

The winning value is paired with its physical loop position rather than its logical row-major position.

## Observed consequence

For the reported mean-plus-argmax input, eager returns logical index 12 at the true maximum, while compiled code returns index 24 whose logical value is not maximal; channels-last variants exhibit the same mismatch.

## Repair site

torch/_inductor/lowering.py::_make_reduction_inner, optimize_indexing.py::remove_redundant_argreduce_indices, scheduler.py, and codegen/triton.py reduction CSE

## Repair

Commit d607816 always carries logical indices for arg reductions through lowering, runs a late pass that removes an index only when it matches the final reduction-variable flattening, and includes retained indices in Triton reduction cache keys. It adds CPU/GPU regressions for fused noncontiguous reductions and cache sharing.

## Why the repair works

Loop rewrites transform the explicit logical-index expression with the value computation, while the late proof permits the native counter only after the physical traversal order is fixed; distinct index mappings also cannot collide in Triton's reduction cache.

## Causal basis

The landed commit states that it fixes 193661 and covers 193751. Its description matches the issue's time-of-check mechanism, its code moves index elision after loop transforms, and its regression recreates the report's noncontiguous mean-plus-argmin/argmax inputs.

## Analysis implication

Arg-reduction analysis needs candidate values, logical index expressions, producer layouts, loop-variable permutations, final flattening order, and reduction-cache identity together.

## Fix patches

- [`fixes/pytorch-193751-commit-d607816.patch`](fixes/pytorch-193751-commit-d607816.patch) — https://github.com/pytorch/pytorch/commit/d60781628a81a6733e9939ebd70430da049b7746
