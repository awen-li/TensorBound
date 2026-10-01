# `host_callback.call(..., call_with_device=True)` specifies the wrong device

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/10082](https://github.com/jax-ml/jax/issues/10082)
- Opened: 2022-03-30T12:43:11Z
- Closed: 2023-05-14T02:58:12Z
- Labels: `bug`, `P3 (no schedule)`

## Reported behavior

I *think* the issue is `host_callback.call`, but I suppose it might be something else I haven't clocked.

On a machine with 4x GPUs, this script:
```python
import jax
import jax.numpy as jnp
import jax.experimental.host_callback as hcb

def _print_device(_, device):
    print(device)

@jax.jit
def print_device(x):
    hcb.call(_print_device, x, call_with_device=True)

cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]
a = jax.device_put(jnp.array(1), cpu)
b = jax.device_put(jnp.array(1), gpu)
print_device(a)
print_device(b)
```
prints:
```
gpu:0
gpu:0
```

Expected behaviour:
```
TFRT_CPU_0
gpu:0
```

This seems to happen with or without the `jax.jit`.

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

Two jitted callback calls execute on different platforms in one process and request their originating device.

## Observed implementation decision

The callback's device reconstruction reports gpu:0 for the CPU-originating invocation even though the returned value remains on CPU.

## Physical effect

The callback receives a device identity that is inconsistent with the computation and result placement.

## Observed consequence

Both callback invocations print gpu:0 instead of distinct CPU and GPU devices.

## Reported proposal or workaround

Commit 30cf057 adds device_index selection for SPMD callbacks and documents limitations of the callback device mechanism; host_callback was later deprecated.

## Repair trace

The inspected commit changes explicit SPMD device-index selection but retains the single-device reconstruction path implicated by this report. The issue closes because host_callback is deprecated, without a linked repair or fixed regression for CPU-versus-GPU call_with_device identity.

## Archived evidence

- Raw report: `data/raw/github/jax/10082/report.json`
- SHA-256: `f11bb2ff436e949bf16e573cc7d573da9c20700f0db46e033420663b3f0c9782`
