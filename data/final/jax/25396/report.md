# jax.full allocates memory on the wrong device

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/25396](https://github.com/jax-ml/jax/issues/25396)
- Opened: 2024-12-11T08:34:09Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

When I use jax.numpy.full, it allocates memory on TPU, even when instructed to allocate it on the CPU

if the memory requested is smaller then the free memory on the TPU, the memory is deallocated and reallocated on the CPU instead

however if the memory requested is more than the free memory, the device crashes with OOM

in my snippet, it's an extremely edge case where I am trying to allocate 40GB, which normally never happens, however, during training and execution, allocating jax buffers on CPU may be the difference for some training pipelines fitting withing specific memory constraints or not.

snippet:
```
import jax
import jax.numpy as jnp
cpu = jax.devices('cpu')[0]
a = jnp.full(shape=(40*(2**30)), fill_value=1.0, dtype=jnp.uint8, device=cpu)
```


### System info (python version, jaxlib version, accelerator, etc.)

jax:    0.4.37
jaxlib: 0.4.36
numpy:  1.26.4
python: 3.10.12 (main, Nov  6 2024, 20:22:13) [GCC 11.4.0]
device info: TPU v6 lite-4, 4 local devices"
process_count: 1
platform: uname_result(system='Linux', node='t1v-n-7691bba4-w-0', release='6.8.0-1015-gcp', version='#17~22.04.1-Ubuntu SMP Tue Sep  3 16:11:52 UTC 2024', machine='x86_64')

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises jnp.full placement when the requested large array is intended for a specific/default host device.

## Observed implementation decision

Internal operations select an accelerator/default placement rather than preserving the caller's desired device context.

## Physical effect

The large allocation is created on the wrong device.

## Observed consequence

The program exhausts or uses unintended device memory.

## Reported proposal or workaround

Wrapping construction in jax.default_device is a workaround.

## Repair trace

No landed factory-placement repair or fixed regression is linked.

## Archived evidence

- Raw report: `data/raw/github/jax/25396/report.json`
- SHA-256: `e8c07b517c9fc6f3e213b2175fbe91cdb815d514c8e1b67fdadca781a1b3905c`
