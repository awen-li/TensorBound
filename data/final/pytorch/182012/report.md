# `torch.compile` silently produces all-zero `x.grad` for `diagonal_scatter(...).sum().backward()`

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/182012](https://github.com/pytorch/pytorch/issues/182012)
- Opened: 2026-04-30T11:03:55Z
- Closed: 2026-06-24T08:29:37Z
- Labels: `triaged`, `has workaround`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`, `repro:agent:h100:repro_success:reproduced`

## Reported behavior

### 🐛 Describe the bug


## 🐛 Describe the bug

`torch.compile(backend='inductor')` silently produces `x.grad = 0` for
`torch.diagonal_scatter(x, src, offset).sum().backward()`, while eager mode
correctly produces `x.grad = 1` at all non-diagonal positions (and `0` on the
diagonal, since those positions were overwritten by `src`).

The forward output is correct; only the backward gradient for `x` is wrong.
`src.grad` is computed correctly. `aot_eager` also produces the correct gradient,
confirming the bug is in Inductor's code generation, not in AOTAutograd or the
decomposition.

## Minimal reproducer

```python
import torch

x = torch.randn(8, 8, device='cuda', requires_grad=True)
src = torch.randn(8, device='cuda', requires_grad=True)

def fn(x, src):
    return torch.diagonal_scatter(x, src, 0).sum()

# Eager: correct
x1 = x.clone().detach().requires_grad_(True)
s1 = src.clone().detach().requires_grad_(True)
fn(x1, s1).backward()
print(f"eager x.grad:\n{x1.grad}")
# tensor([[0., 1., 1., 1., 1., 1., 1., 1.],
#         [1., 0., 1., 1., 1., 1., 1., 1.],
#         ...])  # 1 everywhere except diagonal

# Compiled: wrong
torch._dynamo.reset()
x2 = x.clone().detach().requires_grad_(True)
s2 = src.clone().detach().requires_grad_(True)
torch.compile(fn)(x2, s2).backward()
print(f"compiled x.grad:\n{x2.grad}")
# tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0.],
#         ...])  # ALL ZEROS — incorrect
```

## Root cause analysis

The backward FX graph for `diagonal_scatter(x, src).sum()` is:

```
expand(tangent_scalar, [N, N])  →  clone  →  diagonal_scatter(clone, zeros)  →  return
```

Inductor lowers `clone(expand(scalar, [N, N]))` into a Triton kernel that
writes a single element to `buf0`, then returns
`reinterpret_tensor(buf0, (N, N), (0, 0), 0)` — a stride-(0, 0) view where
all N² positions alias one memory cell. The subsequent `diagonal_scatter`
kernel writes `0.0` to that single cell (to zero the diagonal), so the entire
gradient reads back as 0.

Correct behavior: `clone` must materialize the expanded tensor into a
contiguous buffer so that `diagonal_scatter` can selectively zero only the
diagonal positions.

This is **not** the same root cause as #180164 / #180771 (slice_scatter
backward), which was in AOTAutograd's `meta_slice_scatter` preserving zero
strides. Here `aot_eager` is correct; only Inductor's codegen is wrong.

## Affected configurations

- **Shapes**: all tested sizes (4×4 through 256×256), square and non-square, all diagonal offsets
- **Dtypes**: float32 and float64
- **Devices**: CUDA (tested on Tesla T4)
- **Trigger pattern**: `diagonal_scatter(x, src, offset)` followed directly by `.sum()` (any dim variant)
- Adding a pointwise op between `diagonal_scatter` and `sum` (e.g. `.relu()`, `.mul(2)`) avoids the bug by preventing the problematic fusion

## Characterization

| Pattern | x.grad |
|---------|--------|
| `diagonal_scatter(x, src, 0).sum().backward()` | **all zeros (BUG)** |
| `diagonal_scatter(x, src, 0).relu().sum().backward()` | correct |
| `diagonal_scatter(x, src, 0).mul(2).sum().backward()` | correct |
| `diagonal_scatter(x, src, 0).add(1).sum().backward()` | **all zeros (BUG)** |
| `diagonal_scatter(x, src, 0).clone().sum().backward()` | **all zeros (BUG)** |


### Versions

## Versions

- **PyTorch**: 2.13.0.dev20260425+cu126
- **Triton**: 3.2.0+git4b3bb1e8
- **CUDA**: 12.6
- **GPU**: Tesla T4 (sm_75)
- **Python**: 3.11


cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Trigger

The backward of diagonal_scatter(x, src).sum() passes an expanded gradient with size 8 by 8 and strides zero by zero into the scatter decomposition.

## Relevant entities

expanded gradient; zero-stride dimensions; clone_preserve_strides; diagonal view; scatter mutation; x gradient

## Implemented relation

Before a subset mutation, distinct logical destination positions are bound to distinct physical storage unless the operation explicitly defines overlapping writes.

## Pre-state

The expanded gradient logically contains one value at every matrix position but all positions alias one physical element through zero strides.

## Binding transition

The decomposition clones while preserving the zero-stride layout and then writes through the diagonal view of that clone.

## Post-state

Diagonal and non-diagonal logical positions in the mutable output remain bound to the same physical element.

## Validity obligation

A scatter decomposition materializes an internally overlapping input before mutating only the selected diagonal positions.

## Invalid decision

clone_preserve_strides was used even when a dimension of size greater than one had stride zero.

## Physical effect

Writing the diagonal value changes the shared storage element that represents every non-diagonal position as well.

## Observed consequence

Compiled backward produces an all-zero gradient for x, while eager execution produces ones outside the diagonal and zeros on the diagonal.

## Repair site

torch/_refs/__init__.py::diagonal_scatter

## Repair

Commit c004e2d detects size-greater-than-one dimensions with stride zero and uses input.clone to create nonoverlapping contiguous storage before the diagonal mutation; otherwise it retains clone_preserve_strides. It adds an eager-versus-compiled gradient regression.

## Why the repair works

The diagonal view now selects independent storage locations, so zeroing or replacing diagonal elements leaves the non-diagonal gradient values unchanged.

## Causal basis

The landed commit explicitly fixes report 182012, describes the zero-stride alias in the decomposition, changes the clone choice at that site, and adds the report's backward comparison as a regression.

## Analysis implication

Mutation analysis needs logical size, strides, internal-overlap status, cloned-storage layout, mutation subset, and gradient consumer together.

## Fix patches

- [`fixes/pytorch-182012-commit-c004e2d.patch`](fixes/pytorch-182012-commit-c004e2d.patch) — https://github.com/pytorch/pytorch/commit/c004e2dc8082e016df1893e3f42a831ea3f12c75
