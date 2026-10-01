# jnp.cumsum is incorrect when sharding over summing axis on GPUs

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/21403](https://github.com/jax-ml/jax/issues/21403)
- Opened: 2024-05-23T18:36:29Z
- Closed: 2024-05-24T14:34:39Z
- Labels: `bug`

## Reported behavior

### Description

I have this simple script below that tests `jnp.cumsum` when sharding along the same axis it is summing over. Tested on a machine with 8 40GB A100s.

```python
# test.py
import argparse
import jax
import jax.numpy as jnp
import jax.lax
from jax.sharding import PartitionSpec as PS, Mesh
from jax.experimental.pjit import pjit
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--sp', type=int, default=2)
args = parser.parse_args()

def f(mask):
    mask = jax.lax.with_sharding_constraint(mask, PS('dp', 'sp'))
    idxs = jnp.cumsum(mask, axis=-1)
    return idxs
f = pjit(f, in_shardings=PS(), out_shardings=PS())

mesh_shape = (jax.device_count() // args.sp, args.sp)
print(f"Mesh shape: {mesh_shape}")
mesh = Mesh(np.array(jax.devices()).reshape(mesh_shape), ('dp', 'sp'))

B, L = 8, 1024
with mesh:
    mask = np.ones((B, L), dtype=np.int32)
    out = jax.device_get(f(mask))
    expected = np.arange(L, dtype=np.int32)[None].repeat(B, axis=0) + 1
    print('Output:', out)
    print('Expected:', expected)
    assert np.allclose(out, expected)
    print('Passed')
```
**On TPUs, this code works completely fine for different Mesh shardings**

**On GPUs, the code produces incorrect output when args.sp > 1 (sharding over the summing axis)**

Works for no sharding (sp = 1)
```
> python test.py --sp 1
Mesh shape: (8, 1)
2024-05-23 18:27:20.884869: W external/xla/xla/service/gpu/nvptx_compiler.cc:760] The NVIDIA driver's CUDA version is 12.4 which is older than the ptxas CUDA version (12.5.40). Because the driver is older than the ptxas version, XLA is disabling parallel compilation, which may slow
down compilation. You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.
Output: [[   1    2    3 ... 1022 1023 1024]
 [   1    2    3 ... 1022 1023 1024]
 [   1    2    3 ... 1022 1023 1024]
 ...
 [   1    2    3 ... 1022 1023 1024]
 [   1    2    3 ... 1022 1023 1024]
 [   1    2    3 ... 1022 1023 1024]]
Expected: [[   1    2    3 ... 1022 1023 1024]
 [   1    2    3 ... 1022 1023 1024]
 [   1    2    3 ... 1022 1023 1024]
 ...
 [   1    2    3 ... 1022 1023 1024]
 [   1    2    3 ... 1022 1023 1024]
 [   1    2    3 ... 1022 1023 1024]]
Passed
```

Breaks for sharding (sp > 1 = 2, 4, 8 - up to 8 GPUs)
```
> python test.py --sp 2
Mesh shape: (4, 2)
Output: [[   1    2    4 ... 1278 1279 1024]
 [   1    2    4 ... 1278 1279 1024]
 [   1    2    4 ... 1278 1279 1024]
 ...
 [   1    2    4 ... 1278 1279 1024]
 [   1    2    4 ... 1278 1279 1024]
 [   1    2    4 ... 1278 1279 1024]]
Expected: [[   1    2    3 ... 1022 1023 1024]
 [   1    2    3 ... 1022 1023 1024]
 [   1    2    3 ... 1022 1023 1024]
 ...
 [   1    2    3 ... 1022 1023 1024]
 [   1    2    3 ... 1022 1023 1024]
 [   1    2    3 ... 1022 1023 1024]]
Traceback (most recent call last):
  File "/home/wilsonyan/test.py", line 31, in <module>
    assert np.allclose(out, expected)
AssertionError
```

One other bizarre thing is that if you insert `mask = jnp.ones_like(mask)` like so:
```
...
def f(mask):
    mask = jax.lax.with_sharding_constraint(mask, PS('dp', 'sp'))
    mask = jnp.ones_like(mask) # <<< new inserted line
    idxs = jnp.cumsum(mask, axis=-1)
    return idxs
...
```
The code works again on GPUs.

Any clue what might be going on?


### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.4.28
jaxlib: 0.4.28
numpy:  1.26.4
python: 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0]
jax.devices (8 total, 8 local): [cuda(id=0) cuda(id=1) ... cuda(id=6) cuda(id=7)]
process_count: 1
platform: uname_result(system='Linux', release='6.8.0-1007-gcp', version='#7-Ubuntu SMP Sat Apr 20 00:58:31 UTC 2024', machine='x86_64')

$ nvidia-smi
Thu May 23 18:28:53 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.67                 Driver Version: 550.67         CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-40GB          Off |   00000000:00:04.0 Off |                    0 |
| N/A   34C    P0             59W /  400W |     425MiB /  40960MiB |      3%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A100-SXM4-40GB          Off |   00000000:00:05.0 Off |                    0 |
| N/A   33C    P0             69W /  400W |     425MiB /  40960MiB |      2%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA A100-SXM4-40GB          Off |   00000000:00:06.0 Off |                    0 |
| N/A   31C    P0             61W /  400W |     425MiB /  40960MiB |      2%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA A100-SXM4-40GB          Off |   00000000:00:07.0 Off |                    0 |
| N/A   32C    P0             59W /  400W |     425MiB /  40960MiB |      2%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA A100-SXM4-40GB          Off |   00000000:80:00.0 Off |                    0 |
| N/A   32C    P0             61W /  400W |     425MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA A100-SXM4-40GB          Off |   00000000:80:01.0 Off |                    0 |
| N/A   33C    P0             59W /  400W |     425MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA A100-SXM4-40GB          Off |   00000000:80:02.0 Off |                    0 |
| N/A   32C    P0             60W /  400W |     425MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA A100-SXM4-40GB          Off |   00000000:80:03.0 Off |                    0 |
| N/A   34C    P0             66W /  400W |     425MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A    345969      C   python                                        416MiB |
|    1   N/A  N/A    345969      C   python                                        416MiB |
|    2   N/A  N/A    345969      C   python                                        416MiB |
|    3   N/A  N/A    345969      C   python                                        416MiB |
|    4   N/A  N/A    345969      C   python                                        416MiB |
|    5   N/A  N/A    345969      C   python                                        416MiB |
|    6   N/A  N/A    345969      C   python                                        416MiB |
|    7   N/A  N/A    345969      C   python                                        416MiB |
+-----------------------------------------------------------------------------------------+
```

## Trigger

jnp.cumsum scans an axis whose partitions require an internal concatenate operand to be padded.

## Relevant entities

sharded scan axis; SPMD partition; concatenate operand; padding elements; all-reduce; cumulative result

## Implemented relation

Elements introduced only to align an SPMD partition are the additive identity when the partitioner's internal concatenate feeds cumulative aggregation.

## Pre-state

A logical operand ends before the padded physical shard extent.

## Binding transition

The old concatenate handler reshards the operand and uses its padded physical extent without replacing out-of-domain elements by zero.

## Post-state

Internal concatenate and collective computation include arbitrary padded elements as if they belonged to the logical scan input.

## Validity obligation

Padding created for SPMD shape alignment carries the operation's neutral value before it enters cross-partition aggregation.

## Invalid decision

HandleConcatenate consumed a resharded internal operand without applying PadWithZero.

## Physical effect

Padding contributes to partial sums propagated across GPU partitions.

## Observed consequence

jnp.cumsum returns incorrect values when the summation axis is split across two, four, or eight GPUs.

## Repair site

xla/service/spmd/spmd_partitioner.cc::HandleConcatenate

## Repair

XLA PR 12476 applies PadWithZero to each resharded concatenate operand and updates partitioner tests to require masking of padded elements.

## Why the repair works

Out-of-domain physical elements become zero before concatenate and all-reduce processing, preserving the logical cumulative sum.

## Causal basis

A JAX maintainer maps the report to XLA PR 12476, whose change zeroes padded concatenate operands; the reporter then verifies the corrected nightly build.

## Analysis implication

SPMD analysis needs logical bounds, padded shard extents, neutral values, concatenate offsets, collective aggregation, and final slice bounds.

## Fix patches

- [`fixes/jax-21403-xla-pr12476.patch`](fixes/jax-21403-xla-pr12476.patch) — https://github.com/openxla/xla/pull/12476
