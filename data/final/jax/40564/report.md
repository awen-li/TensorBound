# `jnp.maximum` JVP silently drops a gradient branch on GPU when the fusion autotuner emits the operand twice with different rounding

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/40564](https://github.com/jax-ml/jax/issues/40564)
- Opened: 2026-09-09T09:04:00Z
- Closed: 2026-09-09T20:05:41Z
- Labels: `bug`

## Reported behavior

### Description

> [!NOTE]
> The issue was identified by Claude. It came up during numerical stability work in hyperbolic deep learning, where max is often used to enforce norm floors, e.g., for denominators to prevent division by zero `denom = jnp.maximum(c_norm_prod, MIN_NORM)`.

## Description

On GPU, `jax.grad` through a `jnp.maximum`/`jnp.minimum` can silently return `0.0` for the true
winning branch with no error and no NaN. `lax.max_p`'s JVP rule (`_balanced_eq` in `jax/_src/lax/lax.py`) decides which operand a gradient flows through by testing *bitwise* equality between each operand and `z = max(x, y)`. This works only if the winning operand is the same value everywhere it appears in the compiled
graph.

On XLA:GPU, that's not always the case: when the winning operand is a computed expression, XLA
rematerializes it in the backward pass, and its fusion autotuner picks each occurrence's code-generation
backend independently. This could be potentially a different code generator (native vs. Triton) or, in the reproducer below, the same generator with different internal tuning (e.g., different Triton block/tile configurations). Either way, the two occurrences can round the last bit differently. When they do,
neither `x_recomputed == z` nor `y == z` holds, both branches of `_balanced_eq` evaluate to `0`,
and the gradient through that `max` call vanishes.

### Reproduction

Pure JAX, fixed inputs embedded, `float64` used only as the reference. **Different hardware may require different numbers, see below**!

```python
import base64, json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

X_B64 = "ELHRv7ZM3z4EdVc/2Gr3PqIYTj+bXU6/A26LPkcphT8wbaY/ie6avuVKXj9bOHC8F1fEPldM+z0kEbS/u99sv9k7ML//GGu9UBF6P4Cy5Du20pE+hXVDv9ZJmj7P4Ya/Morsvg0uwj8IHSXAQEa+P74dgD9Ff+6//IqDP1q8RD93Bws+A8/oPLBRxr2xP5W+lXLoP8Q1r7/yPhlA/kzoPg=="
X_SHAPE = (8, 5)
Z_B64 = "C9iXPgWdxL70Oxq/RQHxPaA80L6AGy28VfMOvQGrxT3RZxm+AluWPr474L2RrQ++ckmPvnxlDL5fy6S9zLegPSjeib7Rw1U8VqKRPlPrOr46HL69aAA3Pvnhhz6xGQC/3supPV/fe74Di4O+Yf0WPLxXDz65fUK+3MiGPr2ecT5Pm8Y+mdnSvdzkfb4PiX6+N0UZv0n1jL22HhE+SFSrvf5QOr5yDmc+hhGlvmtB2b7LUaW9Mn+nvn+FiL8xNmg9PoqcvepjHT1jhPk+UpWcvjOpbL41xC2+i7ABv33KtL51TQK+ZrWWvj4R8D0W8pK7RPDqPQxNQ70yh7y9/1LjvaL8zj26mCc++GQiPtYrt77KcoC+Ie10PaDqET8S67W+dXuVvaT1Dz4qRDG/IU1OvV5OTz14AlQ9gNcFv7rtSD4Urz09oejaPMwR7D7GfZy+LnUsvqjjOT7riSe+zULrvpiThznzkZY99guLvvVQKT4fJZS5h/56vcJGvT79MiS/H3AOvkYHi74jGJw+bB8dPtcafr0MR5o+RUfsvXzdUT2zzoW+PwGnO4AwJb30pJM8UE4oPoQmnD1B1Ai/yYb5veSGkr77p6A+AZmOvlYUEryWHI0+PQK/vRYger6I2Ea9SbaRPoCsMr+u/h+8GA7jPbs9cz537Wg6n4f4PfBjLD5W/4M+ulG4PfPiY77Gt1k+CBUcv1kVuD7BR8M9ggy2PfO9Tr2kPqC+TYp+PHmsRL4+ASy+hR/IvpBwgL0dSAU+zFX2vXmiA71zx8Q+1RsWPlLdrz30gPu+3JIWPuuGET7S7am+jax5vhbIvL21a5M+4rPTPgVMHD/yE/Q+ayZdPA=="
Z_SHAPE = (32, 5)

def load(b64, shape):
    return np.frombuffer(base64.b64decode(b64), np.float32).reshape(shape)

x32 = jnp.asarray(load(X_B64, X_SHAPE))   # (8, 5)
z32 = jnp.asarray(load(Z_B64, Z_SHAPE))   # (32, 5)

def loss(z, x):
    zn = jnp.maximum(jnp.linalg.norm(z, ord=2, axis=-1, keepdims=True), 1e-15).T
    dot = jnp.einsum("bi,oi->bo", x, z, precision=jax.lax.Precision.HIGHEST)
    return jnp.sum((zn * jnp.arcsinh(dot / zn)) ** 2)

g_jit = np.asarray(jax.jit(jax.grad(loss))(z32, x32), np.float64)
g_ref = np.asarray(jax.grad(loss)(z32.astype(jnp.float64), x32.astype(jnp.float64)), np.float64)
relerr = float(np.abs(g_jit - g_ref).max() / np.abs(g_ref).max())
print(json.dumps({"relerr": relerr, "verdict": "PASS" if relerr < 1e-4 else "FAIL"}))
```

### Expected output

```
{"relerr": 1.5e-07, "verdict": "PASS"}
```

### Actual output

Deterministic on every launch tested (8/8), (see configuration below):

```
{"relerr": 0.2972045701533697, "verdict": "FAIL"}
```

`jax.grad` in eager mode is correct (1e-07); only the `jit`-compiled backward is wrong.

### Root cause

A relative error of 0.30 is far above anything a float32 rounding difference can produce (~1e-7). Consequently, this is a dropped term, not a rounding error. The compiled HLO confirms it: the duplicated `sqrt(sum(z*z))` fusion has no native (`kind=kInput`) kernel here at all. Both occurrences compile to Triton (`kind=kCustom`, 3 kernels total for the surrounding fusions). Yet, they still round differently, which the backward graph then feeds into two `_balanced_eq`s. The divergence isn't specifically "native vs. Triton emitter", but rather any two independently autotuned emitter choices for the same expression.

**This is not the same bug as #37821**: that one is the deliberate `0.5/0.5` split at an *exact* tie,
`x == y`. Here `x` and `y` are never tied.

### Workaround

Replace `jnp.maximum(n, eps)` with `jnp.where(n > eps, n, eps)`. The comparison is then against a
constant, which no inter-emitter ULP disagreement can flip.

### Suggested fix upstream

Make `_balanced_eq` robust to the operand and `z` not being bitwise identical, e.g. lower `max`'s
JVP to a `select` on the same `>` predicate of the forward pass, rather than re-deriving it from an equality test. That's strictly safer than the status quo, where the failure mode is a silently zeroed gradient rather than a NaN.

### System info (python version, jaxlib version, accelerator, etc.)

> [!WARNING]
> I wasn't able to reproduce with these exact numbers on an H100/H200. Most likely, it still exists, but requires different numbers.

```
jax:    0.11.1
jaxlib: 0.11.1
numpy:  2.4.3
python: 3.12.7 (main, Oct 16 2024, 04:37:19) [Clang 18.1.8 ]
device info: NVIDIA A100-PCIE-40GB-1, 1 local devices"
process_count: 1
platform: uname_result(system='Linux', node='rey', release='6.8.0-54-generic', version='#56-Ubuntu SMP PREEMPT_DYNAMIC Sat Feb  8 00:37:57 UTC 2025', machine='x86_64')
XLA_PYTHON_CLIENT_PREALLOCATE=false

$ nvidia-smi
Wed Sep  9 10:11:10 2026
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.216.03             Driver Version: 535.216.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-PCIE-40GB          Off | 00000000:5E:00.0 Off |                    0 |
| N/A   26C    P0              36W / 250W |    425MiB / 40960MiB |      2%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-PCIE-40GB          Off | 00000000:86:00.0 Off |                    0 |
| N/A   28C    P0              34W / 250W |      3MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A   1161061      C   ...che/builds-v0/.tmpylj4vx/bin/python      416MiB |
+---------------------------------------------------------------------------------------+
```

## Trigger

The forward and backward graphs contain independently tuned copies of a computed maximum operand that differ in the last bit.

## Relevant entities

max/min operands; forward result; rematerialized operand; branch mask; input tangents; output gradient

## Implemented relation

The derivative selects the winning operand from a direct comparison of the two operands, with equal values sharing the tangent.

## Pre-state

One operand is the true winner, but its backward rematerialization is not bitwise equal to the stored forward result.

## Binding transition

The old rule separately tests each rematerialized operand for equality with the forward max/min result.

## Post-state

Neither equality holds, both tangent masks become zero, and the winning gradient branch disappears.

## Validity obligation

Derivative branch selection remains valid when an operand and the forward answer are numerically equivalent but not bitwise identical across rematerialization.

## Invalid decision

The pre-fix `_balanced_eq` helper in `jax/_src/lax/lax.py` uses
`select(_eq_meet(x, z), _ones(z), _zeros(z))` to select the tangent mask.
This derives branch identity from equality between a rematerialized operand
and the separately computed forward result. The repair introduces
`_balanced_cmp`; that new helper is not the historical invalid operation.
See the revision-time
[source-location correction](../../../../results/source-location-corrections.json).

## Physical effect

The compiled backward pass multiplies the winning tangent by zero.

## Observed consequence

The jitted gradient has relative error about 0.297 while eager and float64 reference gradients agree.

## Repair site

jax/_src/lax/lax.py max_p/min_p JVP registration and _balanced_eq

## Repair

PR 40578 replaces answer-equality masks with _balanced_cmp on the two operands, keeps half gradients for ties and zero gradients for NaNs, and adds the reported GPU regression.

## Why the repair works

One direct operand comparison determines the winner without requiring a rematerialized operand to reproduce the forward result bit for bit.

## Causal basis

The report traces the missing term to two failed equality masks, and the merged patch changes exactly those masks and confirms the supplied A100 regression fails before the change.

## Analysis implication

Autodiff checking needs the provenance of branch masks and whether they compare corresponding operands or independently rematerialized values.

## Fix patches

- [`fixes/jax-40564-pr40578.patch`](fixes/jax-40564-pr40578.patch) — https://github.com/jax-ml/jax/pull/40578
