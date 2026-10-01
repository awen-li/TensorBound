# `jax.device_put` silently aliases writeable numpy buffers

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/37830](https://github.com/jax-ml/jax/issues/37830)
- Opened: 2026-05-19T18:46:42Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

When `jax.device_put` is called with a writeable numpy array and a CPU device, the returned `jax.Array` shares the underlying buffer with the input numpy array (for sufficiently large arrays). Subsequent in-place mutation of the numpy array silently changes the contents of the JAX array.

This violates JAX's value semantics: JAX arrays are documented as immutable, but here their contents can change under the user's feet.

### Minimal reproducer

```python
import numpy as np
import jax

x = np.arange(100, dtype=np.float32)
out = jax.device_put(x, jax.devices('cpu')[0])

assert x.ctypes.data == out.unsafe_buffer_pointer()  # shared buffer
assert x.flags.writeable                              # input still mutable

assert out[0] == 0.0
x[0] = -1.0                                           # mutate the numpy source
assert out[0] == -1.0                                 # JAX array changed
```

Below some threshold (~tens of elements) JAX copies, so the bug only shows for non-trivial arrays.

### Expected behavior

A jax function should never alias a writeable buffer unless the user turns on a glaring knob like `device_put(x, INVOKE_ALIASHURBANIPAL_KING_OF_BUGS_DONT_SET=True)`. This remains true even if the function already provides the setting `may_alias=False`. On the other hand, `device_put` currently serves the important function of converting numpy arrays to jax arrays zero-copy.

Potential fixes:
- Hand off numpy=>jax zero-copy to another API
- `device_put(writeable numpy array, cpu device)` raises an exception, the array must be read-only, hopefully numpy provides a way to check the buffer does not have any writeable aliases
- `device_put(writeable numpy array, cpu device)` sets `x.flags.writeable = False` on the input, weirdly invasive and potentially hard to handle all numpy buffer aliasing edge cases

### Note

Written by Opus 4.7, edited by a human.


### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.10.0
jaxlib: 0.10.0
numpy:  2.4.6
python: 3.14.0 (main, Oct 14 2025, 21:10:22) [Clang 20.1.4 ]
device info: cpu-1, 1 local devices"
process_count: 1
platform: uname_result(system='Darwin', release='25.4.0', version='Darwin Kernel Version 25.4.0: Thu Mar 19 19:30:44 PDT 2026; root:xnu-12377.101.15~1/RELEASE_ARM64_T6000', machine='arm64')
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises jax.device_put copy semantics for writeable NumPy buffers.

## Observed implementation decision

The NumPy sharding path drops may_alias=False before batched_device_put.

## Physical effect

The device array remains bound to mutable host storage despite an explicit copy request.

## Observed consequence

Later host writes silently change the JAX value.

## Reported proposal or workaround

PR 38086 proposes threading copy semantics into the NumPy path.

## Repair trace

PR 38086 was not merged and no alternative landed repair is linked.

## Archived evidence

- Raw report: `data/raw/github/jax/37830/report.json`
- SHA-256: `2b61d629d5bf4b3eb34bbc0a955c3a84d98666aa56179b99ca72ee1bc2e24d4f`
