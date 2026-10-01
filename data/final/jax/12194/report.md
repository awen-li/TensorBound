# pure_callback passes jax.DeviceArray to the callback on CPU when not jitted

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/12194](https://github.com/jax-ml/jax/issues/12194)
- Opened: 2022-09-01T09:35:30Z
- Closed: 2022-09-02T00:49:25Z
- Labels: `bug`

## Reported behavior

### Description

As title.
This only happens when the callback is executed outside of jit boundaries.

This is different from the documented behaviour.

```python
➜ python
Python 3.10.6 (main, Aug 23 2022, 11:35:18) [Clang 13.1.6 (clang-1316.0.21.2.5)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import jax
>>> def test(x):
...   print(type(x))
...   return x
...
>>> def f(x):
...   return jax.pure_callback(test, x, x)
...
>>> x= jax.numpy.ones(3)
>>> f(x)
<class 'jaxlib.xla_extension.DeviceArray'>
DeviceArray([1., 1., 1.], dtype=float32)
>>> jax.jit(f)(x)
<class 'numpy.ndarray'>
DeviceArray([1., 1., 1.], dtype=float32)
```

### What jax/jaxlib version are you using?

jax 0.3.17

### Which accelerator(s) are you using?

CPU

### Additional System Info

MacOs

## Trigger

A pure callback receives a JAX array in eager execution while the same callback under jit receives a NumPy array.

## Relevant entities

JAX input array; pure_callback primitive; eager impl rule; lowering rule; NumPy host value

## Implemented relation

The callback boundary presents the documented NumPy host representation in eager and compiled execution.

## Pre-state

The operand is a device-backed JAX array and the callback API promises a host NumPy argument.

## Binding transition

The old eager primitive implementation invokes the Python callback directly with its JAX operand.

## Post-state

The callback receives DeviceArray and host NumPy APIs such as default_rng reject it.

## Validity obligation

All execution modes cross the same device-to-host callback-lowering boundary before invoking user code.

## Invalid decision

pure_callback defined a separate eager implementation that bypassed its lowering semantics.

## Physical effect

The callback's argument representation depends on whether jit surrounds the call.

## Observed consequence

Outside jit, the callback observes DeviceArray instead of the documented numpy.ndarray and downstream NumPy code fails.

## Repair site

jax/_src/callback.py pure_callback_p implementation

## Repair

PR 12203 makes dispatch.apply_primitive and the primitive lowering the eager implementation and adds a non-jit ndarray regression.

## Why the repair works

Eager execution now uses the same host-callback transfer and conversion path as compiled execution.

## Causal basis

The report contrasts eager and jitted argument types, and the linked fix removes the separate eager callback path and asserts ndarray input without jit.

## Analysis implication

Foreign-callback analysis needs execution mode, transfer boundary, host representation, and user callback contract.

## Fix patches

- [`fixes/jax-12194-pr12203.patch`](fixes/jax-12194-pr12203.patch) — https://github.com/jax-ml/jax/pull/12203
