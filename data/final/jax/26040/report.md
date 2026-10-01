# zeros_like does not match device when used under JIT

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/26040](https://github.com/jax-ml/jax/issues/26040)
- Opened: 2025-01-22T19:27:13Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

Given our current model of how device assignment as it interacts with JIT, this may be working as expected, but I suspect some users would find this to be a surprising breakage of JIT-invariance. Is there anything we can do to change this?

Related to #26000.

```python
import jax
import jax.numpy as jnp
jax.config.update('jax_num_cpu_devices', 2)

cpu1, cpu2 = jax.devices()

x1 = jnp.arange(4, device=cpu1)
x2 = jnp.arange(4, device=cpu2)

func = jnp.zeros_like

assert func(x1).device == x1.device  # passes
assert jax.jit(func)(x1).device == x1.device  # passes

assert func(x2).device == x2.device  # passes
assert jax.jit(func)(x2).device == x2.device  # fails
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises zeros_like placement under jit when the reference value has a device binding.

## Observed implementation decision

Lowering constructs the zero constant without carrying the reference array's device/placement relation.

## Physical effect

The result is placed according to compilation defaults rather than the reference.

## Observed consequence

zeros_like returns a tensor on a different device.

## Reported proposal or workaround

PR 26045 proposes a primitive-based full_like implementation.

## Repair trace

PR 26045 was not merged and no alternative landed repair is linked.

## Archived evidence

- Raw report: `data/raw/github/jax/26040/report.json`
- SHA-256: `90007aa1cbe974d90153ab30de0515559d51dd7e7ac43007cd3fcf0ab1f41929`
