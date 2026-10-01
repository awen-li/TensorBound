# Creating a mutable array does not preserve its sharding

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/26338](https://github.com/jax-ml/jax/issues/26338)
- Opened: 2025-02-05T18:05:02Z
- Closed: 2025-02-07T19:05:03Z
- Labels: `bug`

## Reported behavior

### Description

Repro:

```python
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=4'

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.core import mutable_array
from jax._src.state.primitives import ref_swap

devices = mesh_utils.create_device_mesh((2, 2))
mesh = Mesh(devices, axis_names=('i', 'j'))
sharding = NamedSharding(mesh, P('i', 'j'))

a = jnp.zeros_like(mesh.device_ids, dtype=jnp.int32)
a = jax.make_array_from_callback(a.shape, sharding, lambda idx: a[idx])
print(a[...].sharding)

a_ref = mutable_array(a)
print(a_ref[...].sharding)
```

Expected output: The two lines should both be `NamedSharding` and should be the same.

Actual output:

```
NamedSharding(mesh=Mesh('i': 2, 'j': 2), spec=PartitionSpec('i', 'j'), memory_kind=unpinned_host)
SingleDeviceSharding(device=CpuDevice(id=0), memory_kind=unpinned_host)
```

### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.5.0
jaxlib: 0.5.0
numpy:  2.2.2
python: 3.13.0rc3 (main, Oct  2 2024, 17:18:08) [Clang 18.1.8 ]
device info: cpu-1, 1 local devices"
process_count: 1
platform: uname_result(system='Linux', node='ayx1', release='6.10.11-1rodete2-amd64', version='#1 SMP PREEMPT_DYNAMIC Debian 6.10.11-1rodete2 (2024-10-16)', machine='x86_64')
```

## Trigger

mutable_array receives an array distributed over a two-by-two device mesh.

## Relevant entities

input JAX array; NamedSharding; defensive copy; mutable-array buffer; device mesh

## Implemented relation

A mutable array's defensive storage copy preserves the input array's sharding.

## Pre-state

The input buffer uses NamedSharding over four CPU devices.

## Binding transition

The old mutable-array implementation invokes the input object's generic copy method.

## Post-state

The mutable buffer is placed with SingleDeviceSharding on device zero.

## Validity obligation

Creating independent mutable storage copies content without changing its device-to-index mapping.

## Invalid decision

The defensive-copy path did not preserve the initialized array's explicit sharding.

## Physical effect

The same logical array becomes bound to one device instead of the supplied mesh.

## Observed consequence

Reading the new mutable array reports SingleDeviceSharding instead of the original NamedSharding.

## Repair site

jax/_src/core.py::_mutable_array_impl

## Repair

PR 26377 uses JAX's sharding-preserving _array_copy for the defensive copy and exposes the mutable buffer's sharding property.

## Why the repair works

The independent buffer is created through a JAX array copy path that carries the established sharding.

## Causal basis

The report shows sharding loss immediately after mutable-array creation, and the patch replaces that exact copy path and asserts preservation before any ref execution.

## Analysis implication

Mutable-storage creation analysis needs both content-copy semantics and the input array's device-to-index mapping.

## Fix patches

- [`fixes/jax-26338-pr26377.patch`](fixes/jax-26338-pr26377.patch) — https://github.com/jax-ml/jax/pull/26377
