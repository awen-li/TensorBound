# [inductor] Autograd's "modified by an inplace operation" check is lost when a view of a graph input is saved for backward: `x @ w`, `bmm`, `einsum`, 3-D `F.linear` silently return wrong gradients after `x` is modified in place

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/197887](https://github.com/pytorch/pytorch/issues/197887)
- Opened: 2026-09-21T13:55:25Z
- Closed: Open at collection
- Labels: `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug

**Summary.** When a tensor that backward needs is modified in place between forward and backward, eager and `aot_eager` raise "one of the variables needed for gradient computation has been modified by an inplace operation", while `backend="inductor"` raises nothing for `mm`, `bmm`, `einsum`, `addmm`, 3-D `F.linear` and slicing, and `backward()` returns a gradient computed from the modified input (torch 2.14.0 and nightly, CPU and CUDA). A training loop that reuses its input buffer before `backward()` (`x.copy_(next_batch)`) is stopped by eager, but keeps running under `torch.compile` with gradients of no executed forward pass. Because the saved tensor is a view that Inductor rebuilds with `reinterpret_tensor` rather than the input itself, it no longer shares the input's version counter, and the check disappears for every graph whose backward saves a view of an input.

```python
import torch

def f(x, w):
    return x @ w

x = torch.arange(6.).reshape(2, 3) / 7
w = (torch.arange(9.).reshape(3, 3) / 5).requires_grad_()
true_grad = x.clone().t() @ torch.ones(2, 3)          # dL/dw of the forward that actually runs

for backend in (None, "aot_eager", "inductor"):
    torch._dynamo.reset()
    xb, wb = x.clone(), w.detach().clone().requires_grad_()
    g = f if backend is None else torch.compile(f, backend=backend)
    out = g(xb, wb)
    xb.add_(1)                                        # input modified between forward and backward
    try:
        out.sum().backward()
        print(backend, "-> no error, grad correct:", torch.allclose(wb.grad, true_grad), wb.grad[0].tolist())
    except RuntimeError as e:
        print(backend, "-> raises:", str(e)[:70])
```

```
None      -> raises: one of the variables needed for gradient computation has been modified
aot_eager -> raises: one of the variables needed for gradient computation has been modified
inductor  -> no error, grad correct: False [2.4285714626312256, 2.4285714626312256, 2.4285714626312256]
```

The correct gradient row is `[0.4286, 0.4286, 0.4286]`; Inductor returns the gradient of `(x + 1) @ w`, a forward pass that never ran.

Each (program, backend) below ran in a fresh interpreter (`saved_view_variants.py`); "raises" is the eager error above:

| forward (then `x.add_(1)`, then `backward()`) | eager | aot_eager | inductor |
|---|---|---|---|
| `x @ p` | raises | raises | **silent, wrong grad (max err 2)** |
| `torch.bmm(x, p)` | raises | raises | **silent, wrong grad** |
| `torch.einsum('ij,jk->ik', x, p)` | raises | raises | **silent, wrong grad** |
| `F.linear(x, p)` with 3-D `x` | raises | raises | **silent, wrong grad (max err 4)** |
| `torch.matmul(x, p)` with 3-D `x` | raises | raises | **silent, wrong grad** |
| `torch.addmm(p.sum(0), x, p)` | raises | raises | **silent, wrong grad** |
| `x[0] * p` | raises | raises | **silent, wrong grad** |
| `softmax(x @ p, -1) @ x.t()[:3].t()` | raises | raises | **silent, wrong grad** |
| `x.t() @ p`, `x.view(3, 2) * p`, `x * p`, `x.expand(4, 2, 3) * p`, 2-D `F.linear`, `conv2d`, a small MLP | raises | raises | raises |

**Root cause.** For `x @ w` the AOT forward graph saves a *view of the input* for backward:

```
def forward(primals_1, primals_2):
    mm = aten.mm.default(primals_1, primals_2)
    permute = aten.permute.default(primals_1, [1, 0])
    return (mm, permute)            # permute: SavedForBackwardsAOTOutput(idx=0)
```

With `aot_eager` the saved `permute` is produced by a real `aten.permute` call, so it shares `primals_1`'s version counter and the saved-variable check fires. Inductor's wrapper returns the view as a new tensor on the same storage (`reinterpret_tensor`) with its **own version counter**, so modifying `x` afterwards no longer bumps the version of what was saved. This is directly observable:

```python
out = torch.compile(lambda x, w: x @ w, backend=backend)(x, w)
saved = out.grad_fn.saved_tensors[0]
x.add_(1)
print(saved.untyped_storage().data_ptr() == x.untyped_storage().data_ptr(), x._version, saved._version)
# aot_eager: True 1 1      inductor: True 1 0
``` The min-cut partitioner is not involved (`aot_eager_decomp_partition` raises). Programs whose backward saves the input itself (2-D `linear`, `mul`, `conv2d`) keep the check.

**Possible fix.** Either (a) have AOTAutograd's runtime wrapper regenerate saved-for-backward tensors that alias a graph input from that input, the way user-visible outputs that alias inputs are already regenerated with `gen_alias_from_base`, or (b) make Inductor's `reinterpret_tensor` wrapper share the base tensor's version counter when the result escapes the graph (`torch._C._autograd._unsafe_set_version_counter`), or (c) record the input's `_version` at forward time in `CompiledFunction` and compare it in `backward`.

Expected: the same `RuntimeError` as eager and `aot_eager` (or, at the very least, a gradient that belongs to the forward pass that ran).

### Error logs

_No response_

### Versions

No error is raised under `backend="inductor"`; that is the defect. Eager / `aot_eager` raise:

```
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [3, 2]] is at version 1; expected version 0 instead.
```

### Versions

```
PyTorch version: 2.14.0+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Microsoft Windows 11 Pro for Workstations (10.0.26200 64-bit)
MSVC: Microsoft (R) C/C++ Optimizing Compiler Version 19.44.35228 for x64 (used by Inductor's C++ backend)

Python version: 3.14.7 (tags/v3.14.7:823f032, Aug  5 2026, 10:51:32) [MSC v.1944 64 bit (AMD64)] (64-bit runtime)
Python platform: Windows-11-10.0.26200-SP0
Is CUDA available: False

CPU:
Name: 11th Gen Intel(R) Core(TM) i9-11900T @ 1.50GHz

Versions of relevant libraries:
[pip3] numpy==2.5.2
[pip3] torch==2.14.0+cpu
```

The table above is identical on nightly 2.15.0.dev20260911+cpu (Windows), on Linux CPU with 2.14.0 and nightly 2.15.0.dev20260919, and on CUDA (Tesla T4, 2.14.0+cu130, Triton path), so it is neither platform- nor device-specific.


cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

[inductor] Autograd's "modified by an inplace operation" check is lost when a view of a graph input is saved for backward: `x @ w`, `bmm`, `einsum`, 3-D `F.linear` silently return wrong gradients after `x` is modified in place

## Observed implementation decision

Graph-escaping saved views use _reinterpret_tensor rather than alias regeneration from the graph input, severing the version-counter relation that represents mutation history.

## Physical effect

Inductor reconstructs a view saved for backward as a new TensorImpl on the input's storage but with an independent version counter.

## Observed consequence

In-place modification between forward and backward no longer raises and backward silently computes gradients from mutated data belonging to no executed forward pass.

## Reported proposal or workaround

Backend/operator matrix, saved-tensor pointer and version probes, and AOT graph inspection isolate the missing shared counter; the proposed fix reuses input-alias regeneration for saved views.

## Repair trace

The report proposes regenerating saved aliases from the graph input, but no landed repair or fixed regression demonstrates restored version-counter sharing.

## Archived evidence

- Raw report: `data/raw/github/pytorch/197887/report.json`
- SHA-256: `37d39778068b4137895f705350ffc255065d2ea507b7d973dc5ce2896ba0a52c`
