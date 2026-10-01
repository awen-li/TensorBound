# [Inductor] Incorrect results in `index_put_` when `randperm` views are not injective

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/175903](https://github.com/pytorch/pytorch/issues/175903)
- Opened: 2026-02-26T22:45:04Z
- Closed: 2026-04-03T19:55:57Z
- Labels: `high priority`, `triaged`, `module: advanced indexing`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `PT2-Bug-Bash`

## Reported behavior

### 🐛 Describe the bug


```python
import torch
from torch.nn import Module

class Model(Module):
    def forward(self, x):
        p = torch.randperm(5)
        # Create a non-injective (overlapping) view of randperm 
        idx = torch.as_strided(p, (5,), (0,))
        y = x[idx] + 1.0
        x.index_put_((idx,), y, accumulate=False)
        return x

model = Model().eval()
compiled_model = torch.compile(model, fullgraph=True)

torch.manual_seed(42)
input_uncompiled = torch.zeros(5)
output_uncompiled = model(input_uncompiled)

torch.manual_seed(42)
input_compiled = torch.zeros(5)
output_compiled = compiled_model(input_compiled)

print("Uncompiled model output:\n", output_uncompiled)
print("Compiled model output:\n", output_compiled)
torch.testing.assert_close(output_uncompiled, output_compiled, equal_nan=True)
```


Output:
```
Uncompiled model output:
 tensor([0., 0., 1., 0., 0.])
Compiled model output:
 tensor([0., 0., 2., 0., 0.])

AssertionError: Tensor-likes are not close!

Mismatched elements: 1 / 5 (20.0%)
Greatest absolute difference: 1.0 at index (2,) (up to 1e-05 allowed)
Greatest relative difference: 0.5 at index (2,) (up to 1.3e-06 allowed)
```

Root cause: The compiler incorrectly assumes any view of `randperm` is injective and skips the `.realize()` read-after-write safeguard.



### Versions

[pip3] numpy==2.4.2
[pip3] optree==0.19.0
[pip3] torch==2.12.0a0+git93b243e
[conda] numpy                      2.4.2                py311h4bb6f22_0
[conda] numpy-base                 2.4.2                py311h23175f9_0
[conda] optree                     0.19.0               pypi_0           pypi
[conda] torch                      2.12.0a0+git93b243e  pypi_0           pypi

cc @ezyang @gchanan @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Trigger

A stride-zero view of randperm supplies five identical indices to index_put_ on x, while the update values read x through those same indices.

## Relevant entities

randperm storage; stride-zero index view; source read; index_put_ scatter; write range variables; weak dependency

## Implemented relation

Every source read used to form a scatter value observes the input state before any of that scatter's writes when multiple iterations target one location.

## Pre-state

The stride-zero index view maps all five logical iterations to one physical destination, and y reads that destination before index_put_ begins.

## Binding transition

The scheduler considers fusing the source reader with the scatter writer under a weak dependency.

## Post-state

Fused iterations can read the shared destination after another iteration has already written it.

## Validity obligation

A scatter writer whose index omits any iteration variable is treated as non-injective and is not fused with a reader of the mutated buffer.

## Invalid decision

The weak-dependency fusion check did not test whether every write-range variable appeared in the scatter's physical index expression.

## Physical effect

A later fused iteration reads the value written by an earlier iteration and increments the same destination twice.

## Observed consequence

Eager execution writes 1 at the selected position, while compiled execution writes 2.

## Repair site

torch/_inductor/scheduler.py::fusable_weak_dep

## Repair

Commit 67e982d rejects weak-dependency fusion when the write index does not contain every write-range variable. Its regressions cover duplicate-index index_put_ with and without accumulation.

## Why the repair works

The non-injective scatter executes separately from its source reader, so every update value is derived from the pre-scatter tensor state.

## Causal basis

The report identifies a non-injective index view and the missing read-before-write safeguard. Its closing verification names commit 67e982d as the intervening change that blocks this unsafe reader/writer fusion; the landed patch adds the general missing-range-variable guard and duplicate-index regressions.

## Analysis implication

Scatter analysis needs index-view strides, write-range variables, free symbols in the physical write index, source-buffer identity, and kernel fusion boundaries.

## Fix patches

- [`fixes/pytorch-175903-commit-67e982d.patch`](fixes/pytorch-175903-commit-67e982d.patch) — https://github.com/pytorch/pytorch/commit/67e982d5cb5e65208168ddc6dd6ad24df3e5db9e
