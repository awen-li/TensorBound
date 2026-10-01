# [Pallas TPU] HBM↔VMEM DMA round trip returns zero for 4-byte payloads

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/39505](https://github.com/jax-ml/jax/issues/39505)
- Opened: 2026-07-28T16:39:33Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

SHAPE = (1,)
# SHAPE = (1, 1)
# SHAPE = (1, 2)
# SHAPE = (2,)
# SHAPE = (8, 128)

def main():
    single_core_mesh = pltpu.TensorCoreMesh(axis_name="tc_core", num_cores=1)

    @jax.jit(
        compiler_options={
            "xla_msa_enable": "false",
            "xla_tpu_vmem_scavenging_mode": "NONE",
        },
    )
    def fn(x):
        x_hbm = jax.new_ref(x, memory_space=pltpu.HBM)
        o_hbm = jax.empty_ref(jax.ShapeDtypeStruct(x.shape, x.dtype), memory_space=pltpu.HBM)

        @pl.core_map(mesh=single_core_mesh, name="4_byte_dma_repro")
        def _():
            sem = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
            x_vmem = jax.empty_ref(jax.ShapeDtypeStruct(x.shape, x.dtype), memory_space=pltpu.VMEM)

            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return jax.freeze(o_hbm)

    x = jnp.ones(SHAPE, dtype=jnp.int32) * 42
    result = fn(x)

    print("expected:", x)
    print("actual:  ", result)
    assert result.tolist() == x.tolist()

if __name__ == "__main__":
    main()
```

### Expected Behaviour

The output should be equal to the input because the DMA round trip should preserve the value.

### Actual Behaviour

Shape | Payload size | Result
-- | -- | --
(1,) | 4 bytes | Incorrect: returns zero
(1, 1) | 4 bytes | Incorrect: returns zero
(1, 2) | 8 bytes | Correct
(2,) | 8 bytes | Correct
(8, 128) | 4096 bytes | Correct

### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.11.1.dev20260728+f8e48d5d9
jaxlib: 0.11.0
numpy:  2.5.1
python: 3.14.6 (main, Jun 23 2026, 15:18:23) [Clang 22.1.3 ]
device info: TPU v4-4, 4 local devices"
process_count: 1
platform: uname_result(system='Linux', node='t1v-n-4de715bb-w-0', release='5.19.0-1022-gcp', version='#24~22.04.1-Ubuntu SMP Sun Apr 23 09:51:08 UTC 2023', machine='x86_64')
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises pallas TPU DMA round trips for four-byte HBM-to-VMEM payloads.

## Observed implementation decision

The DMA path mishandles the minimum transfer size/alignment relation.

## Physical effect

The returned VMEM payload is zero instead of the copied bytes.

## Observed consequence

A four-byte round trip silently loses data.

## Reported proposal or workaround

The report provides a hardware reproducer.

## Repair trace

No source-localized DMA repair, landed patch, or fixed regression is linked.

## Archived evidence

- Raw report: `data/raw/github/jax/39505/report.json`
- SHA-256: `fb7ef1c8fdf90ab2540a75f640ce05903f087e4baf500d0eb6614c2d3ac8f16b`
