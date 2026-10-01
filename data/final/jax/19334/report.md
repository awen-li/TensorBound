# `broadcast_to` raise Disallowed host-to-device transfer when used inside jax.jit

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/19334](https://github.com/jax-ml/jax/issues/19334)
- Opened: 2024-01-12T11:43:42Z
- Closed: 2024-01-26T18:41:06Z
- Labels: `bug`

## Reported behavior

### Description

Reproduction:

```python
@jax.jit
def test():
  return jnp.broadcast_to(0.1, (4,))

with jax.transfer_guard('disallow'):
  test()
```
Fail with:
```
XlaRuntimeError: INVALID_ARGUMENT: Disallowed host-to-device transfer: aval=ShapedArray(float32[]), dst_sharding=SingleDeviceSharding(device=TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0))
```

The `0.1` is a constant in the XLA graph, so this should not trigger any host-to-device transfer.

It's possible to silence the issue with `jnp.broadcast_to(jnp.asarray(0.1), (4,))` but original issue is a bug.

### What jax/jaxlib version are you using?

HEAD

### Which accelerator(s) are you using?

TPU

## Trigger

A jitted broadcast_to closes over Python scalar 0.1 while host-to-device transfers are disallowed.

## Relevant entities

Python scalar; lax.asarray; traced constant; device_put implementation; compiled graph

## Implemented relation

A scalar captured during tracing becomes a constant in the compiled graph rather than a runtime host-to-device transfer.

## Pre-state

The scalar is available to the tracer as a closure constant before execution.

## Binding transition

The old lax.asarray directly invokes _device_put_impl for non-JAX scalars.

## Post-state

Transfer guard observes a host-to-device placement operation for a value that belongs in the graph constant pool.

## Validity obligation

Trace-time literal conversion preserves compile-time constant status and binds no runtime transfer operation.

## Invalid decision

lax.asarray used the eager device-placement implementation for a scalar captured under jit.

## Physical effect

The scalar follows a host-to-device transfer path instead of constant lowering.

## Observed consequence

broadcast_to raises an INVALID_ARGUMENT disallowed host-to-device transfer error on TPU.

## Repair site

jax/_src/lax/lax.py::asarray

## Repair

PR 19511 routes scalar and NumPy inputs through no-op _convert_element_type, verifies an empty jaxpr, and adds the transfer-guard regression.

## Why the repair works

Tracing represents the scalar as a typed graph literal, so compiled execution requires no host buffer transfer.

## Causal basis

The maintainer reduces the failure to lax.asarray, and the merged patch replaces its device_put path and asserts both zero bound primitives and no guarded transfer.

## Analysis implication

Placement checks need to distinguish literal materialization during tracing from runtime data movement.

## Fix patches

- [`fixes/jax-19334-pr19511.patch`](fixes/jax-19334-pr19511.patch) — https://github.com/jax-ml/jax/pull/19511
