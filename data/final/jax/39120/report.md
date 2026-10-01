# XLA:CPU float32 miscompilation: zero-padding read back as garbage under jit.

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/39120](https://github.com/jax-ml/jax/issues/39120)
- Opened: 2026-07-13T14:52:51Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

As I wrote tests for an overlap-save fft convolution, I realized that `jit` compiling the function resulted in flaky, sometimes very large, deviation from the correct result.

By doing some investigation, I think I figured out what could have been the cause: `jnp.pad` followed by nested `lax.scan`s that `dynamic_slice` overlapping blocks returns uninitialized memory (NaN, denormals, huge values) where the pad row should be exactly 0.0. Eager execution (`jax.disable_jit`) is always correct.

Below a small script that reproduce (in most runs) the bug.

```python

import sys

import jax
import jax.numpy as jnp
import numpy as np

N, C = 60, 4  # input rows, columns
BLOCK, STEP, NBLOCKS = 23, 19, 3  # overlapping blocks: rows [i*19, i*19 + 23)
NPAD = 61  # (NBLOCKS - 1) * STEP + BLOCK; one zero row appended
CHUNK = 2  # inner scan slices column pairs


@jax.jit
def f(x):
    x = jnp.pad(x, ((0, NPAD - N), (0, 0)))  # append one row of zeros

    def outer(_, i):
        blk = jax.lax.dynamic_slice(x, (i * STEP, 0), (BLOCK, C))

        def inner(_, j):
            return None, jax.lax.dynamic_slice(blk, (0, j * CHUNK), (BLOCK, CHUNK))

        _, ys = jax.lax.scan(inner, None, jnp.arange(C // CHUNK))
        return None, ys

    _, out = jax.lax.scan(outer, None, jnp.arange(NBLOCKS))
    return out


x = jax.random.normal(jax.random.PRNGKey(0), (N, C))
result = np.asarray(f(x))

# the last row of the last block is the appended zero row: it must be exactly 0
pad_row = result[-1, :, -1, :]
if np.all(pad_row == 0.0):
    print("ok: pad row is zero")
    sys.exit(0)
print(f"BUG: pad row read as {pad_row.ravel()} (expected exact zeros)")
sys.exit(1)

```

And blow a few output I obtained, note the presence of NaNs but also of large values (1.2606397e+21).

```
Observed output (12 fresh processes):
    BUG: pad row read as [1.4739792e-35 1.6815582e-44 2.7404502e-34 1.6815582e-44]
    BUG: pad row read as [   nan    nan 1.e-45 1.e-45]
    BUG: pad row read as [0.0000000e+00 0.0000000e+00 2.8069184e+12 9.8090893e-45]
    BUG: pad row read as [nan nan nan nan]
    BUG: pad row read as [3.e-45 0.e+00 0.e+00 0.e+00]
    ok: pad row is zero
    BUG: pad row read as [1.2606397e+21 1.5414283e-44 4.9011089e+22 1.5414283e-44]
    BUG: pad row read as [nan nan nan nan]
    BUG: pad row read as [-7.544458e-04  1.261169e-44  4.203895e-45  4.203895e-45]
    BUG: pad row read as [7.631201e+24 1.401298e-44 1.301149e+23 1.401298e-44]
    BUG: pad row read as [-2.5189768e-13  1.4012985e-44 -2.7766133e-14  1.4012985e-44]
    BUG: pad row read as [   nan    nan 1.e-45 1.e-45]
    === reproduced in 11/12 processes ===
```

### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.9.0
jaxlib: 0.9.0
numpy:  2.3.5
python: 3.13.11 (main, Jan 28 2026, 21:41:04) [Clang 21.1.8 ]
device info: cpu-1, 1 local devices"
process_count: 1
platform: uname_result(system='Darwin', node='PHH9L02RFN', release='25.5.0', version='Darwin Kernel Version 25.5.0: Mon Apr 27 20:39:09 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6020', machine='arm64')
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

A float32 overlap-save convolution pads one zero row and reads overlapping blocks through nested scans under jax.jit on CPU.

## Observed implementation decision

An unidentified XLA:CPU optimization or buffer assignment fails to preserve the padded row's initialized zero value.

## Physical effect

A dynamic slice reads uninitialized bytes where the pad row should reside.

## Observed consequence

JIT execution intermittently produces NaN, denormal, or very large deviations while eager execution is correct.

## Reported proposal or workaround

NeMoS PR 589 re-zeros statically known pad rows with jnp.where before consumption.

## Repair trace

The downstream workaround masks the bad row but the public record does not identify the first invalid XLA source decision, an upstream repair, or a fixed-version compiler regression.

## Archived evidence

- Raw report: `data/raw/github/jax/39120/report.json`
- SHA-256: `baf9d5aa7ef1da586abcd67ba6039733a7b9f55884e3b77c298ced7a3c63bf83`
