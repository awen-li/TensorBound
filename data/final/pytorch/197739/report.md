# [torch.compile] `torch.compile` returns incorrect argmin index for non-contiguous (transposed) tensor

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/197739](https://github.com/pytorch/pytorch/issues/197739)
- Opened: 2026-09-19T18:01:31Z
- Closed: open at collection time
- Labels: `high priority`, `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug

When torch.compile compiles a model that calls `.t()` followed by `argmin()` on a 2D tensor, the compiled model returns a wrong index.

Here is the code to reproduce:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        xt = x.t()
        out = xt + 0.5
        return out.argmin()

torch.manual_seed(0)
x = torch.randn((2, 2))
print(x)
# tensor([[ 1.5410, -0.2934],
#         [-2.1788,  0.5684]])

model = Model()
cmodel = torch.compile(model)

out1 = model(x)
print(out1)   # tensor(1)

out2 = cmodel(x)
print(out2)   # tensor(2)   <-- wrong

torch.testing.assert_close(out1, out2)  # fails
```
Error:

```
AssertionError: Scalars are not equal!

Expected 2 but got 1.
Absolute difference: 1
Relative difference: 0.5
```

The bug is sensitive to the operation performed between the transpose and argmin:

`out = xt + 0.5` → bug is triggered
`out = xt.contiguous()` → bug is triggered
`out = xt` → bug is not triggered

### Error logs

_No response_

### Versions

Versions of relevant libraries:
[pip3] numpy==2.1.3
[pip3] onnxruntime==1.28.0
[pip3] torch==2.14.0

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

- [`fixes/pytorch-197739-commit-d607816.patch`](fixes/pytorch-197739-commit-d607816.patch) — https://github.com/pytorch/pytorch/commit/d60781628a81a6733e9939ebd70430da049b7746
