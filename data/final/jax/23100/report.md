# Passing `DeviceLocalLayout` to `jax.jit` causes incorrect outputs on GPU

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/23100](https://github.com/jax-ml/jax/issues/23100)
- Opened: 2024-08-17T01:50:15Z
- Closed: 2024-08-19T22:10:36Z
- Labels: `bug`

## Reported behavior

### Description

When I pass a Layout with a `DeviceLocalLayout` to the `in_shardings` parameter of `jax.jit`, I get an incorrect output on a GPU. This example indicates to `jax.jit` that the input should be in column-major format, but JAX appears to still pass the matrix to XLA in row-major format, resulting in an incorrectly transposed output.

```python
import jax.numpy as jnp
import jax
from jax.experimental import layout

x = jnp.array([[10., 20.],
               [30., 40.]])
major_last_layout = layout.DeviceLocalLayout(major_to_minor=(1, 0))
sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
# Uncommenting this line causes it to work
# x = jax.device_put(x, layout.Layout(major_last_layout, sharding))

jitted_fn = jax.jit(
    lambda x: x + 1.,
    in_shardings=[
        layout.Layout(major_last_layout, sharding),
    ],
)

print(jitted_fn(x))
```

This should output `[[11 21] [31 41]]` but instead outputs `[[11 31] [21 41]]`. The `DeviceLocalLayout` value is correctly passed to XLA, but JAX still transfers the matrix in row-major format.

Uncommenting the `x = jax.device_put` line causes the example to output the correct results. During that line, I see XLA transposing the matrix on the GPU.

CC @yashk2810


### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.4.31
jaxlib: 0.4.31
numpy:  1.26.3
python: 3.11.9 (main, Apr 10 2024, 13:16:36) [GCC 13.2.0]
jax.devices (1 total, 1 local): [CudaDevice(id=0)]
process_count: 1
platform: uname_result(system='Linux', node='reedwm.svl.corp.google.com', release='6.7.12-1rodete1-amd64', version='#1 SMP PREEMPT_DYNAMIC Debian 6.7.12-1rodete1 (2024-06-12)', machine='x86_64')


$ nvidia-smi
Fri Aug 16 18:41:43 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4060 Ti     Off | 00000000:41:00.0  On |                  N/A |
|  0%   42C    P2              29W / 165W |   1044MiB / 16380MiB |      2%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      8171      G   /usr/lib/xorg/Xorg                          568MiB |
|    0   N/A  N/A      8785      G   /usr/bin/gnome-shell                         59MiB |
|    0   N/A  N/A      9772      G   ...seed-version=20240812-050059.636000      205MiB |
|    0   N/A  N/A     80631      G   ...sion,SpareRendererForSitePerProcess       53MiB |
|    0   N/A  N/A   1323876      G   /usr/bin/nautilus                            20MiB |
|    0   N/A  N/A   1352607      C   python                                      120MiB |
+---------------------------------------------------------------------------------------+
```

## Trigger

jax.jit receives an uncommitted row-major array while in_shardings requests a different device-local layout.

## Relevant entities

uncommitted input array; requested DeviceLocalLayout; sharding fast path; device buffer; compiled parameter layout

## Implemented relation

The physical layout of the input buffer passed to an executable matches the layout declared for that executable parameter.

## Pre-state

The host/JAX input uses its default row-major layout.

## Binding transition

The argument fast path compares devices and shard indices but does not materialize the requested layout.

## Post-state

The executable interprets row-major bytes using the requested column-major parameter layout.

## Validity obligation

Argument placement checks both sharding and device-local layout before reusing an existing buffer.

## Invalid decision

The Python and C++ shard-argument fast paths treated matching indices or devices as sufficient and omitted layout equality.

## Physical effect

The physical buffer is not transposed to the layout encoded in the executable contract.

## Observed consequence

A 2x2 input is interpreted as its transpose and produces [[11,31],[21,41]] instead of [[11,21],[31,41]].

## Repair site

jax/_src/array.py::_array_shard_arg and xla/python/pjit.cc::ShardArg

## Repair

JAX PR 23101 threads layouts through shard handlers and device_put; XLA PR 16187 checks the fast-path IFRT array layout and falls back when it differs. The JAX patch adds the reported layout regression.

## Why the repair works

A mismatched uncommitted input is copied or reshaped into the requested physical layout before execution.

## Causal basis

Both linked patches name issue 23100 and add layout equality to the exact argument-reuse paths described by the report.

## Analysis implication

An input-binding check needs commitment, shard indices, device identity, requested layout, current IFRT layout, and fallback/copy outcome.

## Fix patches

- [`fixes/jax-23100-pr23101.patch`](fixes/jax-23100-pr23101.patch) — https://github.com/jax-ml/jax/pull/23101
- [`fixes/xla-23100-pr16187.patch`](fixes/xla-23100-pr16187.patch) — https://github.com/openxla/xla/pull/16187
