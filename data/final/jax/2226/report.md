# `np.stack` does not respect stickied arrays

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/2226](https://github.com/jax-ml/jax/issues/2226)
- Opened: 2020-02-14T06:03:20Z
- Closed: 2020-05-01T04:54:22Z
- Labels: `bug`

## Reported behavior

In the snipped below, I stick `a` to the CPU. Addition and concatenation respects this, but `np.stack`, for some reason, does not:

```python

import jax

a = jax.random.normal(jax.random.PRNGKey(2), (10, 10))
b = jax.random.normal(jax.random.PRNGKey(2), (10, 10))

a.device_buffer.device(), b.device_buffer.device()
(GpuDevice(id=0), GpuDevice(id=0))

a = jax.device_put(a, jax.devices('cpu')[0])
a.device_buffer.device()
CpuDevice(id=0)

jax.np.stack([a, b]).device_buffer.device()
GpuDevice(id=0)  # Must be CPU!

(a + b).device_buffer.device()
CpuDevice(id=0)

jax.np.concatenate([a, b]).device_buffer.device()
CpuDevice(id=0)
```

## Trigger

jnp.stack expands dimensions on an array committed to CPU and an uncommitted array resident on the default GPU.

## Relevant entities

DeviceArray; physical device; device commitment; lazy reshape; expanded array; concatenate placement

## Implemented relation

Lazy view operations preserve both the source buffer's physical device and its explicit device commitment.

## Pre-state

The first input is resident on and committed to CPU, so an operation combining it with uncommitted data follows CPU placement.

## Binding transition

The old singleton reshape used by expand_dims creates a new DeviceArray over the same buffer but passes None as its device commitment.

## Post-state

Concatenation sees only uncommitted operands and selects the default GPU for stack.

## Validity obligation

A metadata-only shape or axis transformation retains the placement commitment of the unchanged underlying buffer.

## Invalid decision

The lazy broadcast, reshape, and transpose constructors copied the buffer but discarded operand._device.

## Physical effect

The committed CPU operand is treated as movable and stack transfers or executes on the default GPU.

## Observed consequence

jnp.stack returns a GPU array although addition and concatenate over the same explicitly placed input follow CPU.

## Repair site

jax/lax/lax.py lazy broadcast_in_dim, reshape, and transpose implementations

## Repair

Commit ac023bf passes operand._device into each lazy DeviceArray view constructor and adds placement tests distinguishing physical residence from commitment.

## Why the repair works

The expanded input remains marked as CPU-committed, so the downstream concatenate placement rule selects CPU instead of treating every operand as uncommitted.

## Causal basis

The issue traces stack to singleton reshape losing the DeviceArray device field; the maintainer states PR 2882 covers it, and the merged patch changes that exact constructor and its placement tests.

## Analysis implication

Placement analysis needs physical buffer device and commitment as separate facts and must propagate both through metadata-only views.

## Fix patches

- [`fixes/jax-2226-commit-ac023bf.patch`](fixes/jax-2226-commit-ac023bf.patch) — https://github.com/jax-ml/jax/commit/ac023bf28fc368eebe70ab99ef691fc831422d75
