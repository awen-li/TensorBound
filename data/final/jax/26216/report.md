# Control the lifetime of host values of the jax.Array.

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/26216](https://github.com/jax-ml/jax/issues/26216)
- Opened: 2025-01-30T20:00:35Z
- Closed: 2025-02-13T17:37:02Z
- Labels: `bug`

## Reported behavior

### Description

Sample repro of this problem. Please read the code and probably it is better than my description.

```
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import jax
import jax.numpy as jnp

from jax.sharding import SingleDeviceSharding
from jax.sharding import PartitionSpec as P
from jax.experimental import mesh_utils

import numpy as np
from ml_dtypes import bfloat16


def main():
    s = SingleDeviceSharding(jax.local_devices()[0], memory_kind="pinned_host")
    s_dev = s.with_memory_kind("device")

    # Allocates pinned memory by numpy array.
    jax.profiler.start_trace("/root/tensorboard/pinned_host")
    np_inp = np.ones((8192, 32768), dtype=bfloat16)
    weight_on_pinned_host = jax.device_put(np_inp, s)

    weight_on_device = jax.random.normal(jax.random.PRNGKey(0), (8192, 32768), dtype=jnp.bfloat16)
    weight = jax.device_put(weight_on_device, s_dev)

    def copy_weight_to_pinned_host(w, w_on_host_donate):
        w_on_host = jax.device_put(w, s)
        return w, w_on_host

    copy_jitted = jax.jit(
        copy_weight_to_pinned_host,
        in_shardings=(s_dev, s),
        out_shardings=(s_dev, s),
        donate_argnums=(0, 1),
        keep_unused=True,
    )

    def save_host_array(arr):
        a = np.asarray(arr)
        np.save('test.npy', a)

    for i in range(1000):
        print(f"iteration i: {i}")
        weight, weight_on_pinned_host = copy_jitted(weight, weight_on_pinned_host)
        save_host_array(weight_on_pinned_host)
        # If you don't use this code, it will crash.
        # weight_on_pinned_host._npy_value = np.array(0)
        # np.testing.assert_allclose(weight, weight_on_pinned_host)
    weight_on_pinned_host.block_until_ready()
    jax.profiler.stop_trace()
```

The issue here is that the `_npy_value` holds the lifetime of the external reference of the device buffer even the `np.asarray` thing goes out of the scope. The lifetime of each `external reference hold` should be bound with the np.asarray it creates but not with the jax.Array.

### System info (python version, jaxlib version, accelerator, etc.)

HEAD

## Trigger

A pinned-host JAX array is converted to NumPy without copying and later participates in repeated donation or reuse.

## Relevant entities

JAX ArrayImpl; NumPy host view; PJRT external reference; device buffer; _npy_value cache

## Implemented relation

A zero-copy NumPy array holds an external reference that keeps the backing device buffer alive for the view's lifetime.

## Pre-state

The external reference is needed while the returned NumPy view exists.

## Binding transition

ArrayImpl caches the zero-copy NumPy value in _npy_value.

## Post-state

The JAX array retains the NumPy view and its external reference beyond the caller-visible view lifetime.

## Validity obligation

A zero-copy external reference is retained only for the lifetime of the view that requires it; a persistent cache owns only independently copied host data.

## Invalid decision

The conversion path cached every materialized NumPy value without distinguishing copied from zero-copy results.

## Physical effect

The external reference holds the device buffer across donation/reuse boundaries longer than intended.

## Observed consequence

Repeated donation and conversion crashes unless _npy_value is manually cleared or replaced.

## Repair site

jax/_src/array.py and xla/python/py_array.cc

## Repair

The XLA binding returns whether host conversion copied, and JAX stores _npy_value only when did_copy is true; pinned-host regression tests require no cache.

## Why the repair works

Zero-copy views retain their external reference through the returned NumPy object but are no longer retained transitively by the JAX array cache.

## Causal basis

The paired XLA/JAX API change exposes did_copy and uses it as the cache condition, directly implementing the lifetime distinction from the report.

## Analysis implication

A host-conversion lifetime check needs the copy-versus-view outcome and the owner of the external buffer reference.

## Fix patches

- [`fixes/jax-26216-fix.patch`](fixes/jax-26216-fix.patch) — https://github.com/jax-ml/jax/pull/26460
- [`fixes/xla-22586-fix.patch`](fixes/xla-22586-fix.patch) — https://github.com/openxla/xla/pull/22586
