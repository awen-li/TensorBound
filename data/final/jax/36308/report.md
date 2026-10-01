# Data loss on device 1 during multi-GPU sharded device_put

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/36308](https://github.com/jax-ml/jax/issues/36308)
- Opened: 2026-03-28T14:07:49Z
- Closed: Open at collection
- Labels: `bug`, `NVIDIA GPU`

## Reported behavior

### Description

#### Summary
When sharding data across 2 GPUs and then converting back to host, data corresponding to the second GPU is replaced with zeros. This occurs on a multi-GPU NVIDIA A30 setup using JAX 0.9.1. Using 1 GPU and forcing ```--xla_force_host_platform_device_count=2``` results in correct data on both simulated devices.

#### Script to reproduce
```
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

devices = jax.local_devices()
num_gpus = len(devices)

mesh = jax.make_mesh((num_gpus,), ('devices'))
sharded_sharding = NamedSharding(mesh, P("devices"))

def put_sharded(x):
    """Shard a list of num_gpus pytrees across local devices."""
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *x)
    return jax.device_put(stacked, sharded_sharding)

dim = 16
arrays = [(i + 1) * jnp.ones(shape=(dim,), dtype=jnp.float32) for i in range(num_gpus)]

sharded = put_sharded(arrays)
jax.block_until_ready(sharded)
gathered = jax.device_get(sharded)

for i in range(num_gpus):
    expected = jnp.sum(arrays[i]).item()
    observed = jnp.sum(gathered[i]).item()
    print(f"Expected {expected} | observed {observed}")
```
#### Observed Output
```
Expected 16.0 | observed 16.0
Expected 32.0 | observed 0.0
```
### System info (python version, jaxlib version, accelerator, etc.)

#### Environment info
```
jax:    0.9.1
jaxlib: 0.9.1
numpy:  2.4.3
python: 3.12.3 (main, Aug 14 2025, 17:47:21) [GCC 13.3.0]
device info: NVIDIA A30-2, 2 local devices"
process_count: 1
platform: uname_result(system='Linux', node='clapper.doc.ic.ac.uk', release='6.14.0-29-generic', version='#29~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Aug 14 16:52:50 UTC 2', machine='x86_64')
```
#### Extra details
```
CUDA Version: 13.0
Driver Version: 580.65.06
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises multi-GPU sharded device_put transfer to secondary devices.

## Observed implementation decision

The transfer path associates data with device zero or omits secondary-device payloads when building the sharded array.

## Physical effect

Secondary shards contain missing or zero data.

## Observed consequence

Values are lost on device 1.

## Reported proposal or workaround

PR 36524 proposes preserving each destination shard.

## Repair trace

PR 36524 closed without merge, leaving no landed repair or fixed regression.

## Archived evidence

- Raw report: `data/raw/github/jax/36308/report.json`
- SHA-256: `2da9fc2a650e73a3b9a8c41d0499b3558619f8d1b57259f6443daddf673dfc5d`
