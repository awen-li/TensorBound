# `from_dlpack` silently gives wrong arrays for non-standard strides

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/17784](https://github.com/jax-ml/jax/issues/17784)
- Opened: 2023-09-26T11:32:48Z
- Closed: 2025-02-24T17:03:50Z
- Labels: `bug`

## Reported behavior

### Description

When using `from_dlpack` on an array with non-standard strides, JAX returns the wrong arrays (silently).

Think it's similar to issues mentioned before #4196, #7657, #8082. I understand it might not be possible to do no copy here, but I think it would be best if it raised an error, rather than silently creating the wrong array.

eg:
```
import numpy as np
from jax import dlpack as jdp

a = np.arange(4).reshape(2,2)
b = a.T

print(b)
# [[0 2]
#  [1 3]]

print(b[0,1])
# 2

b_jax = jdp.from_dlpack(b.__dlpack__())

print(b_jax)
# [[0 1]
#  [2 3]]

print(b_jax[0,1])
# 1
```

### What jax/jaxlib version are you using?

_No response_

### Which accelerator(s) are you using?

_No response_

### Additional system info

_No response_

### NVIDIA GPU info

_No response_

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
