# device_put leaks references on numpy inputs

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/14882](https://github.com/jax-ml/jax/issues/14882)
- Opened: 2023-03-09T19:05:29Z
- Closed: 2023-03-10T19:54:10Z
- Labels: `bug`

## Reported behavior

First reported in #14876; here's a minimal repro that reproduces on CPU and GPU:
```python
import sys
import jax
import numpy as np

x = np.arange(1000, dtype='float32')
print(sys.getrefcount(x))
# 2

y = jax.device_put(x)
print(sys.getrefcount(x))
# 3

del y
print(sys.getrefcount(x))
# 3
```

## Trigger

A NumPy array is transferred with device_put and the resulting JAX array is deleted before another JAX dispatch.

## Relevant entities

NumPy array; device_put; device array; asynchronous transfer; XLA garbage collector; Python garbage collector

## Implemented relation

The runtime reference that keeps a host input alive for an asynchronous device transfer is released after the transfer-backed device value becomes unreachable.

## Pre-state

device_put retains the NumPy input while creating and transferring the device value.

## Binding transition

The user deletes the device value and runs Python garbage collection, but the old integration does not invoke XLA's collector.

## Post-state

XLA's deferred object still holds the NumPy input even though no Python-visible device value remains.

## Validity obligation

Host references retained by XLA are reclaimed when Python performs garbage collection, independent of a later JAX dispatch.

## Invalid decision

The release path depended on incidental XLA garbage collection triggered by subsequent execution.

## Physical effect

The NumPy object's reference count stays elevated after the device result is deleted.

## Observed consequence

Repeated device_put operations retain host inputs and appear as a reference leak.

## Repair site

jax/_src/lib/__init__.py XLA garbage-collection integration

## Repair

PR 14887 registers an XLA collection callback with Python's gc.callbacks and adds a weak-reference regression for device_put input release.

## Why the repair works

Every Python collection cycle now gives XLA an opportunity to delete unreachable transfer objects and release their host references.

## Causal basis

The report demonstrates that explicit XLA collection releases the input; the merged repair connects Python collection to that same operation and tests the host weak reference.

## Analysis implication

Transfer-lifetime analysis needs host ownership, asynchronous transfer completion, device-value reachability, deferred runtime objects, and both garbage collectors.

## Fix patches

- [`fixes/jax-14882-pr14887.patch`](fixes/jax-14882-pr14887.patch) — https://github.com/jax-ml/jax/pull/14887
