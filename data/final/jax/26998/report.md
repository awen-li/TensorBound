# jax.debug.print Displays Misleading Output for XLA 4-bit Tensors

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/26998](https://github.com/jax-ml/jax/issues/26998)
- Opened: 2025-03-07T20:18:27Z
- Closed: 2025-05-16T10:22:54Z
- Labels: `bug`

## Reported behavior

### Description

In JAX, a 4-bit float tensor (e.g., float4_e2m1fn) is padded to 1 byte per element. However, XLA stores two 4-bit elements compactly in a single byte. As a result, jax.debug.print does not correctly display the packed tensor, leading to inconsistent output between JIT-compiled and non-JIT-compiled runs.

The issue is demonstrated in the following example, where the outputs differ depending on whether JIT compilation is used:
```
import jax.numpy as jnp
import jax
output_type = jnp.float32
shape = (64, 32)
k1 = jax.random.key(0)

a = jax.random.uniform(k1, shape, minval=-1.0, dtype=output_type)
a_e2m1 = a.astype(jnp.float4_e2m1fn)

print(f"py print ref:{a_e2m1}")

def fn(x):
  jax.debug.print("jax.debug.print {x}", x=x)
  return x

print("In jit-compiled")
out = jax.jit(fn)(a_e2m1)
print("Outside jit-compiled")
_ = fn(a_e2m1)
```

### System info (python version, jaxlib version, accelerator, etc.)

jax:    0.5.2
jaxlib: 0.5.1
numpy:  1.26.4
python: 3.12.7
device info: cpu-1, 1 local devices"

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

A jitted debug.print transfers a packed float4_e2m1fn array through the callback path.

## Observed implementation decision

The callback constructs a NumPy array whose element representation has no matching sub-byte dtype, so packed bytes are interpreted through an incompatible host layout.

## Physical effect

The callback's host values no longer correspond to the logical four-bit tensor elements.

## Observed consequence

Jitted debug.print displays values inconsistent with non-jitted printing.

## Reported proposal or workaround

PR 27013 proposes converting float4 values to float32 before invoking debug_callback, and the reporter confirms the proposal fixes the example.

## Repair trace

PR 27013 was closed without merge. Reporter confirmation of an unmerged proposal does not establish a landed JAX repair or a durable fixed regression in the studied source history.

## Archived evidence

- Raw report: `data/raw/github/jax/26998/report.json`
- SHA-256: `473883422663cad14d0ee06a2a778b484871f80f6cf3668ef8cae23ada8500df`
