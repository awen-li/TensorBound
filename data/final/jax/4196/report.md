# `from_dlpack` ignores stride information

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/4196](https://github.com/jax-ml/jax/issues/4196)
- Opened: 2020-09-02T18:48:08Z
- Closed: 2021-02-24T03:47:19Z
- Labels: `bug`

## Reported behavior

Trying to pass a DLPack tensor to JAX I noticed that the strides information gets lost. Is this a bug or a limitation of the library?

E.g., if I have a `dltensor` that corresponds to a slice from say CuPy and I export it to DLPack and import it to JAX, I see the following:

```python
import cupy
import jax

from jax.dlpack import from_dlpack
from jax.config import config

config.update('jax_enable_x64', True)

a = cupy.arange(6).reshape((2, 3))
# array([[0, 1, 2],
#        [3, 4, 5]])
b = a[:, 0:2]
# array([[0, 1],
#        [3, 4]])
c = from_dlpack(b.toDlpack())
# DeviceArray([[0, 1],
#              [2, 3]], dtype=int64)
```

Where I was expecting:
```python
# DeviceArray([[0, 1],
#              [3, 4]], dtype=int64)
```

## Trigger

A 2-by-2 view selects the first two columns of a 2-by-3 CuPy array and exports shape and stride metadata through DLPack.

## Relevant entities

DLPack shape; DLPack strides; underlying allocation; XLA minor-to-major layout; logical slice; imported buffer

## Implemented relation

A DLPack tensor's logical indices map through its strides, or import rejects a stride pattern that XLA's compact layout cannot represent.

## Pre-state

The exported view has logical rows [0,1] and [3,4] over the allocation [0,1,2,3,4,5].

## Binding transition

The old stride-handling condition uses an iterator as a Boolean instead of comparing the search result with dimensions.end().

## Post-state

Stride conversion is skipped and the buffer is treated as compact row-major storage for shape [2,2].

## Validity obligation

Interop either preserves the producer's logical-to-physical index mapping or fails before exposing a differently indexed array.

## Invalid decision

DLPackManagedTensorToBuffer failed to execute stride-to-layout validation because its zero-dimension search result was tested incorrectly.

## Physical effect

The imported XLA buffer indexes consecutive elements [0,1,2,3] rather than the sliced coordinates [0,1,3,4].

## Observed consequence

JAX returns [[0,1],[2,3]] instead of [[0,1],[3,4]].

## Repair site

tensorflow/compiler/xla/python/dlpack.cc::DLPackManagedTensorToBuffer

## Repair

TensorFlow commit 28a2d0a compares absl::c_find(dimensions, 0) with dimensions.end(), allowing stride conversion and rejection of layouts that cannot represent the slice.

## Why the repair works

Nonempty strided tensors now pass through the layout-conversion check, so an unrepresentable slice cannot silently inherit a compact layout with different indexing.

## Causal basis

The report isolates the wrong buffer at the XLA importer; the maintainer identifies the invalid stride condition and closes the report after the correcting commit.

## Analysis implication

Interop analysis needs producer shape, element strides, byte offset, consumer layout expressiveness, and the fail-or-copy policy at import.

## Fix patches

- [`fixes/jax-4196-commit28a2d0a.patch`](fixes/jax-4196-commit28a2d0a.patch) — https://github.com/tensorflow/tensorflow/commit/28a2d0a1a83b3dfa3c903ce645aa9d319a4fd5ef
