# [Pallas TPU] Wrong value when using `input_output_aliases` with multiple arrays

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/24023](https://github.com/jax-ml/jax/issues/24023)
- Opened: 2024-09-30T17:56:16Z
- Closed: Open at collection
- Labels: `bug`, `pallas`

## Reported behavior

## Repro

```python
import functools
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

@functools.partial(
    pl.pallas_call,
    out_shape=(
        jax.ShapeDtypeStruct((2,), jnp.float32),
        jax.ShapeDtypeStruct((2,), jnp.float32),
    ),
    grid=1,
    input_output_aliases={0: 0, 1: 1},
)
def kernel(_, _2, x_ref, y_ref):
    pass

def main():
    x = jnp.array([1, 1], dtype=jnp.float32)
    y = jnp.array([2, 2], dtype=jnp.float32)

    x_out, y_out = kernel(x, y)

    print(x_out)
    print(y_out)

if __name__ == '__main__':
    main()
```

## Expected behaviour

Prints out

```
[1. 1.]
[2. 2.]
```

because this is the normal behaviour (which can be confirmed in interpret mode).

The kernel should essentially do nothing but passing the inputs directly to the outputs.

## Actual behaviour

Prints out

```
[0. 0.]
[0. 0.]
```

Note that this issue does not happens when there is only 1 array.

The repro is originally from a test https://github.com/jax-ml/jax/blob/ff1c2ac152b6fa5e07724417b83de6b711ab5104/tests/pallas/ops_test.py#L1344. It was identified while working on https://github.com/jax-ml/jax/pull/23967.

## System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.4.34.dev20240924+85a466d73
jaxlib: 0.4.33
numpy:  2.1.0
python: 3.12.4 (main, Jun  8 2024, 18:29:57) [GCC 11.4.0]
jax.devices (8 total, 8 local): [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0) TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0) ... TpuDevice(id=6, process_index=0, coords=(2,1,0), core_on_chip=0) TpuDevice(id=7, process_index=0, coords=(3,1,0), core_on_chip=0)]
process_count: 1
platform: uname_result(system='Linux', node='t1v-n-ab2ce832-w-0', release='5.19.0-1027-gcp', version='#29~22.04.1-Ubuntu SMP Thu Jun 22 05:13:17 UTC 2023', machine='x86_64')
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises pallas TPU input_output_aliases with more than one array.

## Observed implementation decision

Multiple logical alias declarations are mapped to an incorrect physical input/output correspondence.

## Physical effect

A result reads or overwrites storage belonging to another aliased array.

## Observed consequence

Hardware execution returns wrong values while interpret mode succeeds.

## Reported proposal or workaround

Interpret mode is an observed comparison, not a repair.

## Repair trace

The issue remains open without a source-localized alias repair or fixed hardware regression.

## Archived evidence

- Raw report: `data/raw/github/jax/24023/report.json`
- SHA-256: `63130166517055578e25919fae34472cc49406813760c7a59635b61a5760abd9`
