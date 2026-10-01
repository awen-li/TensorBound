# bool to int32 conversion broken / inconsistent

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/34751](https://github.com/jax-ml/jax/issues/34751)
- Opened: 2026-02-01T16:33:33Z
- Closed: Open at collection
- Labels: `bug`, `XLA`

## Reported behavior

### Description

This small snippet illustrates the issue:

it seems when creating a jax.numpy array with bools from a numpy array with bools it just does a raw copy of the underlying memory and then when converting that array to int32 it interprets the bytes as int8 s?



```python
import jax.numpy as jnp
import numpy as np

numpy_bool_array=np.frombuffer(b'\x00\x01\xff', dtype=bool)
print(numpy_bool_array)
jax_bool_array=jnp.array(numpy_bool_array)
print(jax_bool_array)

jax_int_array_from_numpy_array=jnp.array(numpy_bool_array, dtype=jnp.int32)
print(jax_int_array_from_numpy_array)

jax_int_array_from_jax_bool_array=jnp.array(jax_bool_array, dtype=jnp.int32)
print(jax_int_array_from_jax_bool_array)
```

I get this output:
```
[False  True  True]
W0201 16:29:57.722925  574629 cuda_executor.cc:1801] GPU interconnect information not available: INTERNAL: NVML doesn't support extracting fabric info or NVLink is not used by the device.
W0201 16:29:57.725764  574495 cuda_executor.cc:1801] GPU interconnect information not available: INTERNAL: NVML doesn't support extracting fabric info or NVLink is not used by the device.
[False  True  True]
[0 1 1]
[ 0  1 -1]
```

### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.8.2.dev20260201
jaxlib: 0.8.2.dev20260201
numpy:  2.2.6
python: 3.12.3 (main, Nov  6 2025, 13:44:16) [GCC 13.3.0]
device info: NVIDIA GeForce RTX 2060-1, 1 local devices"
process_count: 1
platform: uname_result(system='Linux', node='geli-3950', release='6.8.0-90-generic', version='#91-Ubuntu SMP PREEMPT_DYNAMIC Tue Nov 18 14:14:30 UTC 2025', machine='x86_64')
JAX_TOOLBOX_REF=34cbc5fd5adf3911e23ed67995778cf5edb68db3
XLA_FLAGS= --xla_gpu_enable_latency_hiding_scheduler=true

$ nvidia-smi
Sun Feb  1 16:30:49 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2060        On  |   00000000:25:00.0  On |                  N/A |
|  0%   43C    P2             35W /  184W |    2724MiB /  12288MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

W0201 16:30:49.959043  575262 cuda_executor.cc:1801] GPU interconnect information not available: INTERNAL: NVML doesn't support extracting fabric info or NVLink is not used by the device.
W0201 16:30:49.961548  575109 cuda_executor.cc:1801] GPU interconnect information not available: INTERNAL: NVML doesn't support extracting fabric info or NVLink is not used by the device.
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises stableHLO boolean-to-integer conversion for noncanonical boolean byte representations.

## Observed implementation decision

XLA converts the underlying byte value instead of logical false/true semantics.

## Physical effect

A true byte such as 0xff becomes integer 255 rather than 1.

## Observed consequence

JAX conversion returns values inconsistent with StableHLO semantics.

## Reported proposal or workaround

The issue is forwarded to open XLA issue 37159.

## Repair trace

The upstream issue remains open and JAX PR 34763 is unmerged; no landed repair exists.

## Archived evidence

- Raw report: `data/raw/github/jax/34751/report.json`
- SHA-256: `bf4453fe31ecfc63cfbfbc778c8477c7fc79d9a0f464a31ff51b2999c3a73396`
