# `ref_get` mutates the sharding of the underlying buffer of a mutable array

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/26350](https://github.com/jax-ml/jax/issues/26350)
- Opened: 2025-02-06T13:55:16Z
- Closed: 2025-02-07T19:05:04Z
- Labels: `bug`

## Reported behavior

### Description

```python
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=4'

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.core import mutable_array
from jax._src.state.primitives import ref_get

devices = mesh_utils.create_device_mesh((2, 2))
mesh = Mesh(devices, axis_names=('i', 'j'))
sharding = NamedSharding(mesh, P('i', 'j'))

a = jnp.zeros_like(mesh.device_ids, dtype=jnp.int32)
a = jax.make_array_from_callback(a.shape, sharding, lambda idx: a[idx])

a_ref = mutable_array(a)
# ref_get(a_ref, ...)
print(a_ref._buf.sharding)
```

This prints:

```
NamedSharding(mesh=Mesh('i': 2, 'j': 2), spec=PartitionSpec('i', 'j'), memory_kind=unpinned_host)
```

But if we uncomment the line `ref_get(a_ref, ...)`, it prints:

```
SingleDeviceSharding(device=CpuDevice(id=0), memory_kind=unpinned_host)
```

Note that `ref_get` is purely a read operation that should not have any side effects.

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

ref_get reads a mutable array whose backing buffer has NamedSharding over four devices.

## Relevant entities

mutable reference; backing buffer; input sharding; discharged jaxpr; XLA execution

## Implemented relation

Discharging a read-only ref operation carries the mutable input buffer's existing sharding into XLA execution.

## Pre-state

The mutable buffer retains its NamedSharding before ref_get executes.

## Binding transition

_discharge_refs_jaxpr appends the mutable input with UNSPECIFIED sharding.

## Post-state

XLA execution may place the aliased mutable buffer on one device, changing the buffer observed after the read.

## Validity obligation

Every discharged mutable input is compiled with the sharding of the buffer whose identity and state it aliases.

## Invalid decision

The old ref-discharge path discarded mutable-input shardings when extending the compiled input list.

## Physical effect

A logically read-only operation changes the backing buffer's device mapping.

## Observed consequence

After ref_get, the mutable buffer reports SingleDeviceSharding instead of its original NamedSharding.

## Repair site

jax/_src/interpreters/pxla.py::_discharge_refs_jaxpr

## Repair

PR 26377 supplies c.sharding for every discharged mutable input instead of UNSPECIFIED and tests sharding before and after a jitted read.

## Why the repair works

The compiled ref operation consumes and returns the aliased buffer under its established device mapping.

## Causal basis

The report isolates sharding change to ref_get, and the patch replaces UNSPECIFIED precisely for discharged mutable inputs and checks the post-read buffer sharding.

## Analysis implication

Effect discharge needs alias identity and placement metadata together; marking a value mutable cannot erase its sharding contract.

## Fix patches

- [`fixes/jax-26350-pr26377.patch`](fixes/jax-26350-pr26377.patch) — https://github.com/jax-ml/jax/pull/26377
