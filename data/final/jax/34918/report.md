# jax parallelism and tensorflow

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/34918](https://github.com/jax-ml/jax/issues/34918)
- Opened: 2026-02-08T21:55:59Z
- Closed: 2026-07-26T21:20:48Z
- Labels: `bug`

## Reported behavior

### Description

This issue is similar to https://github.com/google/flax/issues/4942. However, the proposed fix does not work.

```python
if True:
    import os
    os.environ["KERAS_BACKEND"] = "jax"
    import jax
    _ = jax.devices()

    import tensorflow


import jax
import jax.numpy as jnp


def f(p, x):
    m = len(jax.devices())
    return jnp.mean(jax.shard_map(
        lambda x: p * x,
        mesh=jax.make_mesh(axis_shapes=[m], axis_names=["x"], axis_types=(jax.sharding.AxisType.Auto,)),
        in_specs=(jax.P("x"),),
        out_specs=jax.P("x")
    )(x))  # type: ignore


p = jnp.zeros(())
x = jnp.zeros(100)
print(jax.jit(jax.grad(f))(p, x))
```

```
jax.errors.JaxRuntimeError: INTERNAL: NCCL operation ncclAllReduce( send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype, ToNcclReduction(reduction_kind), comm_, AsCudaStream(stream)) failed: invalid argument (run with NCCL_DEBUG=WARN for details). Last NCCL warning(error) log entry (may be unrelated) ''.: while running replica 0 and partition 0 of a replicated computation (other replicas may have failed as well).
```

I use `tensorflow==2.20.0`

### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.9.0.1
jaxlib: 0.9.0.1
numpy:  2.3.5
python: 3.13.0 | packaged by conda-forge | (main, Nov 27 2024, 19:18:50) [GCC 13.3.0]
device info: NVIDIA A100-SXM4-40GB-4, 4 local devices"
process_count: 1
platform: uname_result(system='Linux', node='sgc01', release='5.15.0-126-generic', version='#136-Ubuntu SMP Wed Nov 6 10:38:22 UTC 2024', machine='x86_64')

$ nvidia-smi
Sun Feb  8 22:55:20 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  |   00000000:2F:00.0 Off |                    0 |
| N/A   29C    P0             59W /  400W |     509MiB /  40960MiB |      1%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A100-SXM4-40GB          On  |   00000000:30:00.0 Off |                    0 |
| N/A   28C    P0             57W /  400W |     544MiB /  40960MiB |      3%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA A100-SXM4-40GB          On  |   00000000:AF:00.0 Off |                    0 |
| N/A   27C    P0             60W /  400W |     544MiB /  40960MiB |      1%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA A100-SXM4-40GB          On  |   00000000:B0:00.0 Off |                    0 |
| N/A   30C    P0             57W /  400W |     544MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            6642      G   /usr/lib/xorg/Xorg                       65MiB |
|    0   N/A  N/A         1659239      C   python                                  424MiB |
|    1   N/A  N/A            6642      G   /usr/lib/xorg/Xorg                      106MiB |
|    1   N/A  N/A         1659239      C   python                                  418MiB |
|    2   N/A  N/A            6642      G   /usr/lib/xorg/Xorg                      106MiB |
|    2   N/A  N/A         1659239      C   python                                  418MiB |
|    3   N/A  N/A            6642      G   /usr/lib/xorg/Xorg                      106MiB |
|    3   N/A  N/A         1659239      C   python                                  418MiB |
+-----------------------------------------------------------------------------------------+
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises jAX multi-GPU execution in a process that also imports TensorFlow.

## Observed implementation decision

The two frameworks' runtime/library initialization creates an incompatible collective or device binding in the reported environment.

## Physical effect

JAX collective execution fails after TensorFlow import.

## Observed consequence

Parallel execution does not complete correctly.

## Reported proposal or workaround

The reporter observes success after changing TensorFlow/JAX package versions.

## Repair trace

No responsible source change, causal runtime relation, or fixed regression is identified.

## Archived evidence

- Raw report: `data/raw/github/jax/34918/report.json`
- SHA-256: `a56dcb7efe24c0eab74d2ee4c4d44672e49d0efe284e310cd629b064202b16fe`
