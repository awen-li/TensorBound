# [inductor] Wrong argmax after fused trailing-dim reduction on non-contiguous inputs (incl. non-unit-stride reduction window) — generic reduction fusion ignores input strides, CPU and CUDA

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/193661](https://github.com/pytorch/pytorch/issues/193661)
- Opened: 2026-08-15T10:55:26Z
- Closed: 2026-08-31T17:40:46Z
- Labels: `high priority`, `triaged`, `module: correctness (silent)`, `module: reductions`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

**Relationship to #193492**: this is the same *mechanism* (the fused kernel assumes a contiguous interpretation of the input and ignores the real strides) but a **different lowering** — the generic reduction fusion (`mean`/`var`/`sum`/`logsumexp`/`amax` + `argmax`), not the `_adaptive_avg_pool2d` path. The contiguous guard proposed in #193492 is added at the `_adaptive_avg_pool2d` lowering and does not apply to these graphs; the reduction path needs its own fix/regression coverage. I asked in the #193492 thread whether this path should be tracked separately ([[comment](https://github.com/pytorch/pytorch/issues/193492#issuecomment-5299760647)](https://github.com/pytorch/pytorch/issues/193492#issuecomment-5299760647)); opening this issue here so it doesn't get lost in the pool discussion. Happy to close this and fold everything back into #193492 if maintainers prefer a single tracking issue.

`torch.compile` with the Inductor backend returns a wrong `argmax` index when a trailing-dim reduction (`mean`, `var`, `sum`, `logsumexp`, also `amax`) is fused with `argmax` and the reduction input is a **non-contiguous view**. Eager returns the correct index; the compiled index points at a value that is **not** the extremum of the reduction output. The unfused reduction matches eager exactly (sanity `maxdiff ≈ 0`), so the mis-indexing is introduced by the fused kernel, whose generated input indexing assumes a contiguous layout and ignores the real strides. The trigger is deterministic (fixed seed, no randomness).

The previously-reported family instances used non-contiguous leading dims with a **stride-1** reduction window (e.g. `transpose`). The new instance below additionally has a **non-unit-stride reduction window** (`[..., ::2]`, stride 2 on the reduced dim) combined with non-contiguous leading dims. Neither axis triggers on its own: the same `[..., ::2]` slice on a contiguous input passes, and a fully contiguous input passes. As a scope hint for the regression test, `expand`/stride-0 views are **not** affected (all pass).

### To reproduce

```python
import torch

def fn(x):
    return torch.argmax(torch.mean(x, dim=-1))

torch.manual_seed(123)
x = torch.rand([2, 4, 4, 8]).transpose(0, 1).contiguous().transpose(0, 1)[..., ::2]
# shape (2, 4, 4, 4), strides (32, 64, 8, 2)
# non-contiguous leading dims AND a stride-2 reduction window

print("eager   =", fn(x).item())                # 12 (correct)
print("compiled=", torch.compile(fn)(x).item()) # 24 (wrong)
```

Truth check: value at the returned index vs. the true maximum of `mean(x, dim=-1)` (eager index points at the maximum; compiled index does not):

```text
eager    index 12 -> value 0.828343 (== max)
compiled index 24 -> value 0.596202 (!= max)
```

Scope / additional observations (all fixed seed 123, caches disabled, unfused reductions matching eager exactly):

- The same input mismatches for `var`, `sum`, `logsumexp` (dim=-1) and for `mean`, `sum`, `var` (dim=[-1, -2]): **7/7 reduction variants**, on CPU and CUDA, torch 2.13.0 and 2.5.1 (CPU indices identical across versions, e.g. mean: eager 12 / compiled 24 / true 12; CUDA also reproduces).
- Controls: the same `[..., ::2]` slice on a contiguous input passes, a fully contiguous input passes, and `expand`/stride-0 views pass — the trigger is the combination of non-contiguous leading dims and a non-unit-stride reduction window.
- The earlier family instances (non-contiguous leading dims, stride-1 reduction window) are also affected; see the #193492 thread for the family matrix.

Kernel evidence (CPU, torch 2.13.0): the aten graph contains only `mean.dim -> argmax` (the transpose/slice views are metadata and never enter the graph), while the fused kernel `cpp_fused_argmax_mean_0` reads the reduction window with a flat, contiguous interpretation:

```cpp
for (int64_t x0 = 0; x0 < 32; x0++) {   // x0 = flattened 2*4*4 leading positions
    auto tmp0 = in_ptr0[8 * x0];        // mean over in_ptr0[8*x0 + {0,2,4,6}]
    auto tmp1 = in_ptr0[2 + 8 * x0];
    auto tmp3 = in_ptr0[4 + 8 * x0];
    auto tmp5 = in_ptr0[6 + 8 * x0];
    ...
    tmp_acc0 = argmax_combine(tmp_acc0, mean, x0);
}
```

The true offset of element (i, j, k, k′) with strides `(32, 64, 8, 2)` is `32i + 64j + 8k + 2k′`, which the fused flat reads do not match. The wrong reads stay in bounds (max flat offset 254 < storage size 256), so the result is silently wrong — no OOB, no exception.

### Error logs

No exception — silent wrong result (wrong index returned).

### Versions

```text
PyTorch version: 2.13.0+cu130 and 2.5.1+cu121 (both reproduce, CPU and CUDA)
OS: Linux (AutoDL container)
```

cc @ezyang @gchanan @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @muchulee8 @amjames @aakhundov @coconutruben @jataylo

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

- [`fixes/pytorch-193661-commit-d607816.patch`](fixes/pytorch-193661-commit-d607816.patch) — https://github.com/pytorch/pytorch/commit/d60781628a81a6733e9939ebd70430da049b7746
