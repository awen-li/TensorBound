# transfer_guard and jax.device_put(..., device=cpu_device)

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/40285](https://github.com/jax-ml/jax/issues/40285)
- Opened: 2026-08-28T09:12:25Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

When activating `jax.transfer_guard("disallow")` uncached (cold) GPU->CPU transfers with `jax.device_put(..., device=cpu_device)` trigger
```
jax.errors.JaxRuntimeError: INVALID_ARGUMENT: Disallowed device-to-host transfer: shape=(10), dtype=F32, device=cuda:0
```
If the array has been moved to CPU outside of the transfer guard, and then is probably cached (warm), calling `jax.device_put(..., device=cpu_device)` under the same (disallow) `transfer_guard` succeeds.

The documentation (https://docs.jax.dev/en/latest/transfer_guard.html) states
`"disallow": Disallow implicit transfers. Silently allow explicit transfers.`

So, that ** should ** probably work (it is an explicit transfer) ?

Simple test example: [test_transfer_guard.py](https://github.com/user-attachments/files/31548254/test_transfer_guard.py)

```
JAX 0.11.1
GPU: cuda:0  CPU: cpu:0

=== Warm start ===
Warm start succeeded, result on: cpu:0

=== Cold start ===
Cold start raised JaxRuntimeError: INVALID_ARGUMENT: Disallowed device-to-host transfer: shape=(10), dtype=F32, device=cuda:0
Traceback (most recent call last):
  File "<ipython-input-3-b823e162c3ad>", line 41, in <module>
    out = jax.device_put(arr, device=cpu)
  File "/tmp/venv/lib/python3.13/site-packages/jax/_src/api.py", line 2390, in device_put
    out_flat = dispatch._batched_device_put_impl(
        *x_flat, devices=device_flat, srcs=src_flat,
        copy_semantics=copy_semantics, dst_avals=dst_avals)
  File "/tmp/venv/lib/python3.13/site-packages/jax/_src/dispatch.py", line 619, in _batched_device_put_impl
    y = _device_put_impl(x, device=device, src=src, copy=cp, aval=aval)
  File "/tmp/venv/lib/python3.13/site-packages/jax/_src/dispatch.py", line 603, in _device_put_impl
    return _device_put_sharding_impl(x, aval, device, copy)
  File "/tmp/venv/lib/python3.13/site-packages/jax/_src/dispatch.py", line 557, in _device_put_sharding_impl
    return pxla.batched_device_put(aval, sharding, shards, devices)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/venv/lib/python3.13/site-packages/jax/_src/interpreters/pxla.py", line 258, in batched_device_put
    return xc.batched_device_put(aval, sharding, xs, list(devices), committed,
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                 enable_x64=enable_x64)
                                 ^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/venv/lib/python3.13/site-packages/jax/_src/profiler.py", line 420, in wrapper
    return func(*args, **kwargs)
  File "/tmp/venv/lib/python3.13/site-packages/jax/_src/array.py", line 642, in _value
    npy_value, did_copy = self._single_device_array_to_np_array_did_copy()
                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
jax.errors.JaxRuntimeError: INVALID_ARGUMENT: Disallowed device-to-host transfer: shape=(10), dtype=F32, device=cuda:0

=== Summary ===
Warm start : OK
Cold start : FAIL
```

Testing array vs dataclass pytree, different directions:  [more_transfer_guard_tests.py](https://github.com/user-attachments/files/31548343/more_transfer_guard_tests.py)
```
Summary (collapsed over default_device):
kind       src   dest  fn   | cold statuses      | warm statuses
----------------------------------------------------------------
array      cpu   cpu   put  | OK                 | OK
array      cpu   gpu   put  | OK                 | OK
array      cpu   host  get  | OK                 | OK
array      gpu   cpu   put  | FAIL               | OK
array      gpu   gpu   put  | OK                 | OK
array      gpu   host  get  | OK                 | OK
array      host  cpu   put  | OK                 | OK
array      host  gpu   put  | OK                 | OK
dataclass  cpu   cpu   put  | OK                 | OK
dataclass  cpu   gpu   put  | OK                 | OK
dataclass  cpu   host  get  | OK                 | OK
dataclass  gpu   cpu   put  | FAIL               | OK
dataclass  gpu   gpu   put  | OK                 | OK
dataclass  gpu   host  get  | OK                 | OK
dataclass  host  cpu   put  | OK                 | OK
dataclass  host  gpu   put  | OK                 | OK
```


### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.11.1
jaxlib: 0.11.1
numpy:  2.5.2
python: 3.13.6 (main, Aug  6 2025, 22:57:45) [Clang 20.1.4 ]
device info: NVIDIA GeForce RTX 3090-1, 1 local devices"
process_count: 1
platform: uname_result(system='Linux', node='c703i-gpu9', release='6.8.0-90-generic', version='#91~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Nov 20 15:20:45 UTC 2', machine='x86_64')

$ nvidia-smi
Fri Aug 28 11:09:53 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.42.06              Driver Version: 555.42.06      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:01:00.0 Off |                  N/A |
|  0%   51C    P8             21W /  350W |   18396MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A    993740      C   /tmp/venv/bin/python3                       18386MiB |
+-----------------------------------------------------------------------------------------+

```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises transfer_guard classification of an explicit cross-client device_put to a CPU device.

## Observed implementation decision

The device-to-host leg is treated as implicit even though the user explicitly requested the cross-client transfer.

## Physical effect

Transfer guard applies the wrong policy to the host leg.

## Observed consequence

An explicit device_put is rejected or reported unexpectedly.

## Reported proposal or workaround

PR 40303 proposes classifying that leg as explicit.

## Repair trace

PR 40303 was not merged and no alternative fixed regression is linked.

## Archived evidence

- Raw report: `data/raw/github/jax/40285/report.json`
- SHA-256: `b874b4a3bbd0ca0cac5ed5b7a93314dd2644c7f4fcc9a5ce3d03c9be62fb86a4`
