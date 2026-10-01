# Pallas Mosaic-TPU: DMA silently returns zeros when `pallas_call` has an unused operand computed inside the jit (libtpu 0.0.43 regression)

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/39744](https://github.com/jax-ml/jax/issues/39744)
- Opened: 2026-08-05T10:34:54Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

A `pltpu.sync_copy` (equally `make_async_copy`) out of an HBM operand silently returns **all zeros** when the `pallas_call` is given an additional operand that the kernel never reads, and that operand is produced *inside* the jitted computation.

Correct on libtpu **0.0.41**; wrong on **0.0.43, 0.0.44 and 0.0.45** with the same jax (0.10.1).

### Reproducer

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

SOURCE, UNUSED, OUT = (2, 256), (16, 128), (1, 256)


def kernel(source_hbm, unused_hbm, out_hbm, staging):
    pltpu.sync_copy(source_hbm.at[pl.ds(0, 1), :], staging)
    pltpu.sync_copy(staging, out_hbm)


def hbm(shape):
    return pl.BlockSpec(shape, memory_space=pltpu.HBM)


copy_first_row = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(OUT, jnp.float32),
    in_specs=[hbm(SOURCE), hbm(UNUSED)],
    out_specs=hbm(OUT),
    scratch_shapes=[pltpu.VMEM(OUT, jnp.float32)],
    grid=(1,),
)


@jax.jit
def run(source):  # the second operand is never read by the kernel
    return copy_first_row(source, jnp.ones(UNUSED, jnp.float32))


source = np.random.default_rng(0).standard_normal(SOURCE, dtype=np.float32)
got = np.asarray(run(jnp.asarray(source)))
print("max abs error :", np.abs(got - source[:1]).max())
print("all zeros     :", not np.any(got))
```

Output:

```
libtpu 0.0.41 : max abs error : 0.0          all zeros : False
libtpu 0.0.45 : max abs error : 2.7962992    all zeros : True
```

### Note

Requires all four:

1. the unused operand at least **4 rows** of 128 lanes.
2. `jax.jit`
3. source rows at least **256** floats wide;
4. a DMA of a **strict sub-slice** of an HBM operand into VMEM scratch.

### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.10.1
jaxlib: 0.10.1
numpy:  2.4.4
python: 3.13.13 (main, Jun  2 2026, 22:27:49) [Clang 22.1.3 ]
device info: TPU v4-1, 1 local devices"
process_count: 1
platform: uname_result(system='Linux', node='t1v-n-a2361288-w-0', release='5.8.0-1035-gcp', version='#37~20.04.1-Ubuntu SMP Thu Jun 17 16:04:29 UTC 2021', machine='x86_64')
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises pallas Mosaic-TPU custom-call inputs when one operand is unused and placed in CMEM.

## Observed implementation decision

Memory-space assignment moves an operand to CMEM while the Mosaic boundary still describes it as HBM.

## Physical effect

The custom call issues DMA using a stale memory-space contract.

## Observed consequence

TPU v4 execution returns zeros.

## Reported proposal or workaround

PRs 39811 and 39838 propose retaining/constraining HBM inputs.

## Repair trace

Both proposals were unmerged at the study boundary, so no landed repair chain exists.

## Archived evidence

- Raw report: `data/raw/github/jax/39744/report.json`
- SHA-256: `93d007fa9b242f9e987edcaf1a368cbd22a287b1e5cdb40eb944a8042bb28fb3`
