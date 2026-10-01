# `torch.compile` silently produces wrong results for `index_add`/`index_copy`/`index_fill`/`index_put` on expanded (stride-0) tensors

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/183986](https://github.com/pytorch/pytorch/issues/183986)
- Opened: 2026-05-16T01:36:25Z
- Closed: 2026-05-21T23:34:14Z
- Labels: `triaged`, `module: advanced indexing`, `has workaround`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`, `repro:agent:h100:repro_success:reproduced`

## Reported behavior

### 🐛 Describe the bug


### 🐛 Describe the bug

When a tensor is created via `.expand()` (introducing stride-0 dimensions), `index_add`, `index_copy`, `index_fill`, and `index_put` under `torch.compile(backend='inductor')` produce **silently wrong forward results**. The compiled code operates directly on the aliased storage instead of materializing the expansion, so writes intended for independent rows collide through the shared underlying memory.

Eager mode and `aot_eager` are correct. Only the Inductor backend is affected.

### Reproducer

```python
import torch

def fn(src, idx):
    x = torch.zeros(1, 8, device="cuda").expand(4, -1)  # stride-0 in dim 0
    return x.index_add(0, idx, src)

src = torch.ones(3, 8, device="cuda")
idx = torch.tensor([0, 1, 0], device="cuda")

eager = fn(src, idx)
compiled = torch.compile(fn, backend="inductor")(src, idx)

print("Eager row 0:", eager[0].tolist())      # [2, 2, 2, 2, 2, 2, 2, 2]
print("Compiled row 0:", compiled[0].tolist()) # [3, 3, 3, 3, 3, 3, 3, 3]  ← WRONG
print("Eager row 2:", eager[2].tolist())       # [0, 0, 0, 0, 0, 0, 0, 0]
print("Compiled row 2:", compiled[2].tolist()) # [3, 3, 3, 3, 3, 3, 3, 3]  ← WRONG
```

**Expected:** Row 0 = `[2, ...]` (idx 0 hit twice), Row 1 = `[1, ...]`, Rows 2–3 = `[0, ...]`
**Actual (compiled):** All rows = `[3, ...]` — the total sum leaks to every row via aliased storage.

### All affected ops

```python
import torch

def test_op(name, op_fn):
    torch._dynamo.reset()
    eager = op_fn()
    compiled = torch.compile(op_fn, backend="inductor")()
    diff = (eager - compiled).abs().max().item()
    print(f"  {name}: max_diff={diff:.4f}  {'BUG' if diff > 0.01 else 'OK'}")

idx = torch.tensor([0, 1, 0], device="cuda")
src = torch.ones(3, 8, device="cuda")

test_op("index_add", lambda: torch.zeros(1, 8, device="cuda").expand(4, -1).index_add(0, idx, src))
test_op("index_copy", lambda: torch.zeros(1, 8, device="cuda").expand(4, -1).index_copy(0, idx[:2], src[:2]))
test_op("index_fill", lambda: torch.zeros(1, 8, device="cuda").expand(4, -1).index_fill(0, idx[:2], 1.0))
test_op("index_put(acc=True)", lambda: torch.zeros(1, 8, device="cuda").expand(4, -1).index_put((idx,), src, accumulate=True))
test_op("index_put(acc=False)", lambda: torch.zeros(1, 8, device="cuda").expand(4, -1).index_put((idx[:2],), src[:2], accumulate=False))
# All print BUG
```

`scatter`, `scatter_add`, `scatter_reduce`, `select_scatter`, `slice_scatter` are NOT affected — they go through a different codegen path that correctly handles stride-0 inputs.

### Controls

| Condition | Result |
|-----------|--------|
| Non-expanded tensor (`torch.zeros(4, 8)`) | Correct (diff=0) |
| `.clone()` after `.expand()` | Correct (diff=0) |
| `aot_eager` backend | Correct (diff=0) |
| All shapes tested (4×8 to 128×256) | All wrong |
| All dtypes (float16, float32, float64) | All wrong |
| Expand on dim 0 or dim 1 | Both wrong |
| Backward (src.grad) | Correct — only forward is affected |

### Root cause

Inductor's lowering for the `index_*` family does not detect or materialize expanded (stride-0) tensors before performing indexed writes. Because all expanded rows alias the same storage, writes to different logical rows collide.

### Workaround

Insert `.clone()` or `.contiguous()` after `.expand()`:
```python
x = torch.zeros(1, 8, device="cuda").expand(4, -1).clone()  # materialize
```

### Relationship to existing issues

This is the same bug class as #180164 / #182012 (stride-0 tensor mishandling in Inductor), but:
- #180164 (`slice_scatter`/`select_scatter` backward): **fixed**
- #182012 (`diagonal_scatter` backward): **open, fix PR ready**
- This bug: `index_add`/`index_copy`/`index_fill`/`index_put` **forward** — NOT fixed

The `index_*` ops are a distinct code path from the `*_scatter` ops, so existing fixes do not cover them.



### Versions

### Versions

- PyTorch: 2.13.0.dev20260513+cu126
- GPU: Tesla T4 (SM 7.5), also reproducible on CPU
- OS: Linux

cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Duplicate mapping

- Primary report: `github-pytorch-pytorch-188084`
- Shared cause: `cause-pytorch-188084-expanded-view-reinplacement`

Reports 183986 and 188084 identify the same omitted internal-overlap guard in Inductor reinplacement, the same collapse of logically distinct indexed writes through stride-zero storage, and the same repair in PR 184488. They differ in forward versus gradient manifestations, which are consequences rather than distinct invalid decisions.

The sections below describe the shared cause reconstructed from the primary report. The reported behavior above remains the observation specific to this report.

## Trigger

A functional indexed write targets a stride-zero expanded tensor during Inductor reinplacement.

## Relevant entities

functional index operation; expanded view; stride-zero storage; logically distinct rows

## Implemented relation

Expansion maps multiple logical elements or rows to the same physical storage locations.

## Pre-state

The functional operation requires an independent result whose logical writes do not collapse through internal overlap.

## Binding transition

The reinplacement pass rewrites the functional indexed operation to its in-place form on the expanded view.

## Post-state

Distinct logical writes mutate aliased locations in one overlapping storage region.

## Validity obligation

Reinplacement is allowed only when the mutated destination has no definite internal overlap that changes functional semantics.

## Invalid decision

can_inplace checked inter-tensor aliasing but omitted the destination's definite internal-overlap check.

## Physical effect

Writes from different expanded rows fold through the same storage locations.

## Observed consequence

The compiled gradient is wrong by the batch-size factor on CPU and CUDA.

## Repair site

torch/_inductor/fx_passes/reinplace.py

## Repair

can_inplace rejects a mutated argument when _debug_has_internal_overlap returns definite overlap; regression tests cover indexed operations.

## Why the repair works

The pass retains the functional materialization path for expanded tensors, so logically distinct writes receive nonoverlapping storage.

## Causal basis

The added guard is the exact missing condition identified by the report and is exercised by the new expanded-index tests.

## Analysis implication

A reinplacement check needs internal-overlap information, not only alias relations between different tensor values.

## Fix patches

- [`fixes/pytorch-188084-fix.patch`](fixes/pytorch-188084-fix.patch) — https://github.com/pytorch/pytorch/pull/184488
