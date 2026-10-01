# [inductor] Unrolled sum reduction produces wrong results for non-contiguous (transposed) inputs after `unsqueeze+expand`

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/189401](https://github.com/pytorch/pytorch/issues/189401)
- Opened: 2026-07-09T14:24:41Z
- Closed: 2026-07-28T19:38:08Z
- Labels: `high priority`, `triaged`, `has workaround`, `module: correctness (silent)`, `module: reductions`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

When `make_pointwise(ops.mul)` + `make_reduction("sum")` operates on non-contiguous (transposed) inputs after `unsqueeze` + broadcast `expand`, the generated Triton kernel uses **wrong stride multipliers** — it assumes contiguous strides for the logical shape instead of preserving the actual transposed strides.

The `ops.dot` + `reduction("dot")` path (native_matmul) is NOT affected because it bypasses the standard tiling heuristic and forces `{y: M, x: N, r0_: K}` tiling unconditionally.

### Root cause

Traced through 4 independent investigations to:

1. **Reduction unrolling** (`ir.py:1770-1787`): When `reduction_numel < unroll_reductions_threshold` (default 8) AND `reduction_type != "dot"`, the reduction is unrolled into a `Pointwise` node via `_unroll_reduction_fn`. The `"dot"` reduction is explicitly exempt.

2. **View.create path difference**: Without `clone`, the input is a `ReinterpretView` with `is_storage_and_layout=True`, so `View.create` uses the `_compute_stride` path. With `clone`, the input becomes a `Pointwise` with `is_storage_and_layout=False`, routing through the more robust `dynamic_reshape_indexer` (symbolic reindexing).

3. **Generated code evidence**: For `mm(64×3, (4×3).T)`, the buggy kernel loads `in_ptr1 + (4*k + x0)` (contiguous strides `(4,1)` for shape `(3,4)`) when the actual transposed strides are `(1,3)`. The correct load pattern is `in_ptr1 + (k + 3*x0)`.

Related to #186850 (closed as non-reproducible, different reproducer).

### Reproducer

```python
import torch

a = torch.randn(64, 3, device='cuda')
b = torch.randn(4, 3, device='cuda')

@torch.compile
def fn(x, y):
    return torch.mm(x, y.T)

ref = torch.mm(a, b.T)
out = fn(a, b)
print(f"max diff: {(ref - out).abs().max().item()}")
# Expected: ~1e-7 (float32 tolerance)
# Actual: ~8.0 (completely wrong)
```

**Note**: This only reproduces when mm is decomposed to `unsqueeze + mul + sum` at the lowering level (e.g., via the small-mm pointwise optimization in PR #189149). Standard cuBLAS mm is not affected.

To reproduce without PR #189149, insert this in `tuned_mm` after `mm_args`:
```python
mat1 = lowerings[aten.unsqueeze](mat1, -1)
mat2 = lowerings[aten.unsqueeze](mat2, 0)
args, _ = transform_args(args=[mat1, mat2], kwargs={}, broadcast=True,
                          type_promotion_kind=None, convert_input_to_bool=False)
return make_reduction("sum")(make_pointwise(ops.mul)(*args), axis=1)
```

### Workaround

`clone` before `unsqueeze` forces the input through the `dynamic_reshape_indexer` path:
```python
mat1 = L.clone(mat1)
mat2 = L.clone(mat2)
```
The clone has zero runtime cost because the scheduler fuses the clone's Pointwise node into the downstream mul kernel.

### Versions

PyTorch 2.14.0a0+git30e161d (viable/strict), CUDA 12.8, H200

cc @ezyang @gchanan @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @muchulee8 @amjames @aakhundov @coconutruben @jataylo @jansel @eellison @yanboliang @ngimel

## Trigger

A small mm decomposes a transposed right operand into unsqueeze, broadcast multiplication, and a sum whose reduction length is unrolled.

## Relevant entities

transposed operand; ReinterpretView; base storage; view lowering; unsqueeze view; physical strides; unrolled reduction load

## Implemented relation

Each new view composes its logical index transformation with the existing operand view before reaching base storage.

## Pre-state

The right operand is a raw ReinterpretView whose shape is 3 by 4 and whose physical strides are 1 and 3 over base storage.

## Binding transition

The old view lowering reads x.data unconditionally; for a raw ReinterpretView this returns its base storage and drops the transpose before View.create builds the unsqueezed view.

## Post-state

The multiplication and unrolled reduction treat the operand as a contiguous 3 by 4 tensor with strides 4 and 1.

## Validity obligation

Lowering unwraps a TensorBox container but preserves any raw IR view node and its logical-to-physical index map.

## Invalid decision

The view lowering conflated container unwrapping with removal of an existing ReinterpretView.

## Physical effect

The generated kernel loads right-operand elements at 4 times k plus x instead of k plus 3 times x.

## Observed consequence

Compiled mm on a 64 by 3 matrix and a transposed 4 by 3 matrix has a maximum error around 8 instead of float32-level agreement.

## Repair site

torch/_inductor/lowering.py::view for aten.view, aten.reshape, and aten._unsafe_view

## Repair

Commit f2a2c90 widens view lowering to raw IRNode inputs, unwraps .data only when the input is a TensorBox, removes the clone workaround from small-mm lowering, and strengthens transposed left, right, and both-operand decomposition tests.

## Why the repair works

View.create now receives the ReinterpretView itself, so its transpose index map is composed with unsqueeze and broadcast indexing rather than replaced by the base storage layout.

## Causal basis

The landed commit explicitly fixes report 189401, names the .data view loss, changes the exact lowering site, removes the masking clone workaround, and verifies that pointwise mm decomposition runs correctly for transposed operands.

## Analysis implication

View-chain analysis needs wrapper type, existing view identity, base storage, each index map, stride composition, and the final load expression together.

## Fix patches

- [`fixes/pytorch-189401-commit-f2a2c90.patch`](fixes/pytorch-189401-commit-f2a2c90.patch) — https://github.com/pytorch/pytorch/commit/f2a2c90de1455c8b3d6fb90f880523a9a301b0c0
