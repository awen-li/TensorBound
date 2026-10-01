# `from_dlpack` on transposed torch arrays gives wrong indexing

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/7657](https://github.com/jax-ml/jax/issues/7657)
- Opened: 2021-08-18T09:33:40Z
- Closed: 2025-10-09T13:25:55Z
- Labels: `bug`

## Reported behavior

When using the jax.dlpack interface with strided&transposed arrays I'm getting wrong values when indexing, whereas printing gives the correct array structure.

I think this is similar to #4196 and as explained by @zhangqiaorjc  https://github.com/google/jax/issues/4196#issuecomment-689680107, but here no explicit error is thrown. Also doing a deepcopy with `jax2torch(a.clone())` or `jax2torch(copy.deepcopy(a))` does not seem to help in this case.

In my use-case `a` is expected to come from a longer torch pipeline I don't have control over, so I can't defer transpositions/slices to the JAX side. Are there other workarounds I could try?

mwe:
```python
import torch
import jax
import jax.dlpack
import torch.utils.dlpack
from jax.config import config
config.update("jax_enable_x64", True)

def jax2torch(x):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))

def torch2jax(x):
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))

a = torch.arange(6).reshape(2,3).moveaxis(1,0)
b = torch2jax(a)

print("stride", a.stride())
# stride (1, 3)

print(a)
# tensor([[0, 3],
#         [1, 4],
#         [2, 5]])

print(b) # these do match
# [[0 3]
#  [1 4]
#  [2 5]]

print(a[1,0], b[1,0]) # these do not match
# tensor(1) 2

print(torch.__version__) # '1.9.0+cpu'
print(jax.__version__) # '0.2.19'
```

## Duplicate mapping

- Primary report: `github-jax-ml-jax-17784`
- Shared cause: `cause-jax-17784-dlpack-layout-lost-at-ifrt-boundary`

Reports 7657 and 17784 share the DLPack stride-derived layout, the loss of usable layout identity at the IFRT/JAX boundary, incorrect coordinate indexing, and the same JAX/XLA repair pair. NumPy versus Torch is a producer difference, not a different invalid decision.

The sections below describe the shared cause reconstructed from the primary report. The reported behavior above remains the observation specific to this report.

## Trigger

from_dlpack imports a transposed NumPy or Torch array whose minor-to-major layout differs from the JAX device default.

## Relevant entities

DLPack strides; PjRtBuffer layout; IFRT Array; JAX compiled consumer; default layout

## Implemented relation

The physical layout derived from DLPack strides remains attached to the array or is rejected before a consumer assumes default layout.

## Pre-state

DLPack and the created PjRtBuffer correctly describe the transposed physical layout.

## Binding transition

The buffer is wrapped as an IFRT/JAX array whose consumers assume the default layout.

## Post-state

Compiled slicing or indexing interprets the nondefault bytes under default minor-to-major order.

## Validity obligation

A layout is preserved across the interoperability boundary or incompatibility is reported before execution.

## Invalid decision

from_dlpack admitted a nondefault PjRtBuffer although the then-current IFRT Array interface could not expose that layout to JAX consumers.

## Physical effect

Logical indices address default-layout coordinates rather than the producer's stride-derived coordinates.

## Observed consequence

Printing may show the transposed array, but indexing returns values from the wrong coordinate.

## Repair site

xla/python/dlpack.cc::DLPackManagedTensorToBuffer

## Repair

XLA PR 7357 compares the imported layout with the client default and rejects nondefault layouts; JAX PR 18721 adds the transposed-array regression. Later JAX/IFRT layout support permits compact nondefault layouts, but the documented historical repair is the fail-fast boundary.

## Why the repair works

The unsupported representation cannot enter a compiled consumer under an unstated default-layout assumption.

## Causal basis

Both patches list issues 7657 and 17784 and enforce the exact layout capability boundary identified by maintainers.

## Analysis implication

An interoperability check needs DLPack strides, derived layout, consumer layout capability, executable parameter layout, and copy/reject behavior.

## Fix patches

- [`fixes/jax-7657-17784-pr18721.patch`](fixes/jax-7657-17784-pr18721.patch) — https://github.com/jax-ml/jax/pull/18721
- [`fixes/xla-7657-17784-pr7357.patch`](fixes/xla-7657-17784-pr7357.patch) — https://github.com/openxla/xla/pull/7357
