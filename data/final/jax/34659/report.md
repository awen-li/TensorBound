# Case where `device_put` creates a malformed McJAX array

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/34659](https://github.com/jax-ml/jax/issues/34659)
- Opened: 2026-01-27T19:58:13Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

In the call to `device_put(x, sharding)` we check if `x.sharding` is the same as `sharding` by checking whether they have the same shard shapes and number of addressable devices: https://github.com/jax-ml/jax/blob/191504b6e3ff70b38a09ad83f61c590498c1182b/jax/_src/array.py#L1240

If so, no resharding is needed, and a fast path is dispatched. This fast path eventually builds the output PyArray re-using the IFRT array from `x`: https://github.com/jax-ml/jax/blob/191504b6e3ff70b38a09ad83f61c590498c1182b/jaxlib/py_array.cc#L1300

Under certain circumstances, this can result in a malformed array. For example, say we're in McJAX with four processes and one device per process, and this code is run in each process

```
x = jnp.arange(64).reshape(8, 8)  #  uncommitted array, sharding has 1 device

# fully replicated sharding over 2 devices, but just 1 addressable device so `same_sharding` is True
n_dev = jax.device_count() // 2
src_sharding = jax.sharding.NamedSharding(
    jax.make_mesh((n_dev,), ("x",), devices=jax.devices()[:n_dev],
                  axis_types=(jax.sharding.AxisType.Explicit,)),
    P()
)

# y has the same underlying IFRT array as x, so its IFRT sharding has 1 device when it should have 2
# This line doesn't error, it's just malformed.
y = jax.device_put(x, src_sharding)

dst_sharding = jax.sharding.NamedSharding(
    jax.make_mesh((n_dev,), ("x",), devices=jax.devices()[n_dev:],
                  axis_types=(jax.sharding.AxisType.Explicit,)),
    P()
)

# This errors in xla::ifrt::CopyArrays, since y's IFRT sharding has the wrong number of devices
z = jax.device_put(y, dst_sharding)
```

We should rebuild the output IFRT array so that its sharding has the right number of devices. The test case above uses shardings that are nonaddressable by some processes, which is a new feature. This issue exists for "standard" McJAX arrays as well (where the array is sharded over all global devices), but doesn't seem to have posed a problem (I don't immediately know how to write a failing test case for standard McJAX arrays).

### System info (python version, jaxlib version, accelerator, etc.)

-

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises multi-controller JAX device_put construction of a global array.

## Observed implementation decision

The transfer assembles per-process shards into an inconsistent McJAX array/device mapping.

## Physical effect

Array metadata and addressable buffers disagree about shard ownership.

## Observed consequence

The resulting array is malformed.

## Reported proposal or workaround

PR 35808 is proposed in the discussion.

## Repair trace

No merged repair and fixed multiprocess regression are present in the evidence.

## Archived evidence

- Raw report: `data/raw/github/jax/34659/report.json`
- SHA-256: `a6cf2dcbabfc47cbd81dde5faaacf374904363d239a092ed11c9149e405a254e`
