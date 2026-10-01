# Unexpected device placement for arrays created within a jax.default_device context

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/10959](https://github.com/jax-ml/jax/issues/10959)
- Opened: 2022-06-03T00:30:23Z
- Closed: 2022-06-15T19:29:30Z
- Labels: `bug`

## Reported behavior

This is related to https://github.com/google/jax/pull/9118 and has two parts. A colab notebook reproducing the behavior show in the code cells below can be found [here](https://colab.research.google.com/drive/1vsOpMEAn0lKUBjZPOOty4cBLqhgqNsEH?usp=sharing).

### 1. Arrays with a specific shape created by `jnp.zeros`, `jnp.ones` or `jnp.arange` are always placed on the first device that they were created on

```python
import jax
import jax.numpy as jnp
NON_DEFAULT_BACKEND = "interpreter"  # Should work with any non-default backend.
```

```python
# Create ones(1) using the default backend.
print(jnp.ones(1).device_buffer.device(), "(should be 'TFRT_CPU_0')")

# Create ones(1) in a jax.default_device context with a different device.
with jax.default_device(jax.devices(NON_DEFAULT_BACKEND)[0]):
  print(jnp.ones(1).device_buffer.device(), "(should be 'interpreter:0')")
```

```
Output:
TFRT_CPU_0 (should be 'TFRT_CPU_0')
TFRT_CPU_0 (should be 'interpreter:0')
```

This also applies to arrays initially created within a `jax.default_device` context:


```python
# Create ones(2) in a jax.default_device context with a different device.
with jax.default_device(jax.devices(NON_DEFAULT_BACKEND)[0]):
  print(jnp.ones(2).device_buffer.device(), "(should be 'interpreter:0')")

# Create ones(2) using the default backend.
print(jnp.ones(2).device_buffer.device(), "(should be 'TFRT_CPU_0')")
```

```
Output:
interpreter:0 (should be 'interpreter:0')
interpreter:0 (should be 'TFRT_CPU_0')
```

This might be an issue with memoization.

This also applies to identical `jnp.array` calls.

### 2. Passing the module-level default device to `jax.default_device` causes unexpected device placement for all subsequent `jnp.zeros` and `jnp.ones` calls

**Note: The following code should be executed in a fresh runtime.**

```python
import jax
import jax.numpy as jnp
NON_DEFAULT_BACKEND = "interpreter"  # Should work with any non-default backend.
```

```python
# If the first array created by jnp.ones or jnp.zeros is created within a
# jax.default_device context that was passed the module-level
# default device, then all subsequent arrays created by those functions will be
# placed on the module-level default device.
with jax.default_device(jax.devices(jax.default_backend())[0]):
  print(jnp.ones(3).device_buffer.device(), "(should be 'TFRT_CPU_0')")

# Note that ones(4) has not already been created, so this is a separate issue
# from the one described above.
with jax.default_device(jax.devices(NON_DEFAULT_BACKEND)[0]):
  print(jnp.ones(4).device_buffer.device(), "(should be 'interpreter:0')")

# jnp.zeros is affected as well.
with jax.default_device(jax.devices(NON_DEFAULT_BACKEND)[0]):
  print(jnp.zeros(5).device_buffer.device(), "(should be 'interpreter:0')")
```

```
Output:
TFRT_CPU_0 (should be 'TFRT_CPU_0')
TFRT_CPU_0 (should be 'interpreter:0')
TFRT_CPU_0 (should be 'interpreter:0')
```

```python
# Arrays created by jnp.arange are not affected.
with jax.default_device(jax.devices(NON_DEFAULT_BACKEND)[0]):
  print(jnp.arange(4).device_buffer.device(), "(should be 'interpreter:0')")
```

```
Output:
interpreter:0 (should be 'interpreter:0')
```

## Trigger

The same jnp.ones or jnp.zeros primitive and shape is evaluated first on one device and then inside a different default-device context.

## Relevant entities

array-construction primitive; trace context; in-memory cache key; default device; cached executable; result array

## Implemented relation

An unspecified array placement is resolved from the default device active for that primitive evaluation.

## Pre-state

A primitive evaluation is cached while one default device is active.

## Binding transition

The default device changes, but the old trace-context cache key does not contain that setting and reuses the prior cached evaluation.

## Post-state

The cached result-placement decision still names the first device.

## Validity obligation

Cache equivalence includes every configuration value that changes primitive placement semantics.

## Invalid decision

The in-memory primitive cache omitted jax_default_device from its trace-context key.

## Physical effect

A constructor inside the new context reuses an executable or value placement bound to the previous device.

## Observed consequence

Arrays appear on the first device used for that shape instead of the device selected by the current context.

## Repair site

jax/_src/config.py::_trace_context

## Repair

Commit 7098088 adds jax_default_device to the trace-context tuple and adds cached, uncached, and reset placement regressions.

## Why the repair works

Changing the default device now produces a different cache key, forcing placement to be resolved under the current context.

## Causal basis

The report identifies memoization, and the fixing commit states that a cache hit ignored the changed default device and adds the reported context transition to its tests.

## Analysis implication

Compilation-cache analysis needs all dynamic placement configuration, not only primitive operands, shapes, and dtype settings.

## Fix patches

- [`fixes/jax-10959-commit7098088.patch`](fixes/jax-10959-commit7098088.patch) — https://github.com/jax-ml/jax/commit/7098088f4eb15cf750398889e4341dbc15cda1b3
