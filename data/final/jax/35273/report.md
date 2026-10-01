# Inconsistent value dependent device for boolean masked array

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/35273](https://github.com/jax-ml/jax/issues/35273)
- Opened: 2026-02-20T16:12:24Z
- Closed: 2026-02-21T21:05:39Z
- Labels: `bug`

## Reported behavior

### Description

On a colab notebook with a CUDA device I observe the following:

```python
import numpy as np
import jax.numpy as jnp
import jax

cpu_device = jax.devices("cpu")[0]
cpu_array = jnp.asarray([1], device=cpu_device)
masked = cpu_array[jnp.asarray([True], device=cpu_device)]
masked.shape, masked.device
```
```
((1,), CpuDevice(id=0))
```

Which is what I would expect.

However, when the masking result in a 0-sized dimension, I get a result on the CUDA device instead of the CPU device I would have expected given the explicit device selection the above code:

```python
masked = cpu_array[jnp.asarray([False], device=cpu_device)]
masked.shape, masked.device
```
```
((0,), CudaDevice(id=0))
```

### System info (python version, jaxlib version, accelerator, etc.)

```


jax:    0.7.2
jaxlib: 0.7.2
numpy:  2.0.2
python: 3.12.12 (main, Oct 10 2025, 08:52:57) [GCC 11.4.0]
device info: Tesla T4-1, 1 local devices"
process_count: 1
platform: uname_result(system='Linux', node='966a76ee8f69', release='6.6.105+', version='#1 SMP Thu Oct  2 10:42:05 UTC 2025', machine='x86_64')

$ nvidia-smi
Fri Feb 20 16:12:03 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   71C    P0             32W /   70W |   11305MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2061      C   /usr/bin/python3                      11302MiB |
+-----------------------------------------------------------------------------------------+
```

## Trigger

Boolean masking produces an empty output shape from an array committed to a nondefault CPU device.

## Relevant entities

source array; gather indexer; slice sharding; empty full_like result; selected device

## Implemented relation

The gather indexer derives the output slice sharding from the indexed array, including its concrete single-device placement.

## Pre-state

The input array and mask are committed to a selected CPU device.

## Binding transition

The empty-shape fast path bypasses gather and constructs zeros through lax.full_like.

## Post-state

The fast path either omits slice_sharding or carries an abstract, noncanonical sharding, so full_like selects the default device.

## Validity obligation

An empty-result fast path preserves the concrete placement implied by the indexed source when no data-dependent transfer occurs.

## Invalid decision

_gather constructed the empty result without a usable concrete slice sharding.

## Physical effect

The zero-element result is allocated on CUDA or the default CPU instead of the selected CPU device.

## Observed consequence

A false boolean mask returns a zero-sized array on a different device from the source.

## Repair site

jax/_src/numpy/indexing.py and jax/_src/lax/lax.py::full_like

## Repair

PR 35281 passes slice_sharding and permits shape changes for fully replicated inputs; PR 35294 canonicalizes indexer.slice_sharding before the empty fast path consumes it.

## Why the repair works

The empty result is built with the source-derived concrete replicated sharding, so device selection remains bound to the source placement.

## Causal basis

The issue discussion shows that the first change did not complete the repair; the second canonicalizes the exact slice_sharding consumed by the new fast path and adds the original empty-mask regression.

## Analysis implication

A shape-fast-path check needs source commitment, derived sharding concreteness, replication, output shape, and final device selection.

## Fix patches

- [`fixes/jax-35273-pr35281.patch`](fixes/jax-35273-pr35281.patch) — https://github.com/jax-ml/jax/pull/35281
- [`fixes/jax-35273-pr35294.patch`](fixes/jax-35273-pr35294.patch) — https://github.com/jax-ml/jax/pull/35294
