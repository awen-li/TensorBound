# `while-loop-all-reduce-code-motion` miscompiles scatter gradient accumulation in `scan`

- Repository: `openxla/xla`
- Report: [https://github.com/openxla/xla/issues/49143](https://github.com/openxla/xla/issues/49143)
- Opened: 2026-09-21T04:49:23Z
- Closed: 2026-09-22T11:49:32Z
- Labels: `bug`, `Google-TPU`, `sharding`

## Reported behavior

I first observed this while comparing full-batch and microbatch gradients in Gemma 4 post-training correctness tests on TPU v4-8, then reduced it to a standalone JAX computation that reproduces the issue without the training pipeline.

The default TPU execution produces 109628 incorrect elements, with a maximum absolute error of 1.3125, compared with an independent NumPy `add.at` reference. Disabling only `while-loop-all-reduce-code-motion` makes the result exactly match the reference. The same program also matches the reference on four CPU devices.

## Environment

- Python 3.12
- JAX 0.11.2
- jaxlib 0.11.2
- libtpu 0.0.48
- TPU v4-8
- NumPy 2.5.2

## Reproducer

Save the following as `repro_scatter_scan.py`. It accumulates embedding gradients across two microbatches using lax.scan. It uses a float32 table of shape [96, 35, 256], 16 input rows, a four-device mesh sharding the vocabulary and batch dimensions, and integer cotangents scaled by 1/16.

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

assert jax.device_count() == 4
mesh = Mesh(np.array(jax.devices()), ("d",))
sharding = NamedSharding(mesh, P("d"))

ids = np.tile(np.array([
    [0, 2, 13, 9, 41, 8, 1, 0],
    [2, 5, 16, 12, 6, 44, 18, 1],
], dtype=np.int32), (8, 1))
weights = (np.arange(16 * 8 * 35 * 256) % 17 - 8).astype(np.float32)
weights = weights.reshape(16, 8, 35, 256)
expected = np.zeros((96, 35, 256), dtype=np.float32)
np.add.at(expected, ids, weights / 16)

def loss(table, indices, cotangent):
    return jnp.sum(jnp.take(table, indices, axis=0) * cotangent) / 16

@jax.jit
def accumulated_gradient(table, indices, cotangent):
    def body(carry, step):
        i = jax.lax.dynamic_slice_in_dim(indices, step * 8, 8, axis=0)
        w = jax.lax.dynamic_slice_in_dim(cotangent, step * 8, 8, axis=0)
        i = jax.lax.with_sharding_constraint(i, sharding)
        w = jax.lax.with_sharding_constraint(w, sharding)
        return carry + jax.grad(loss)(table, i, w), None
    return jax.lax.scan(body, jnp.zeros_like(table), jnp.arange(2))[0]

actual = np.asarray(accumulated_gradient(
    jax.device_put(np.zeros_like(expected), sharding),
    jax.device_put(ids, sharding),
    jax.device_put(weights, sharding),
))
print("incorrect elements:", np.count_nonzero(actual != expected), flush=True)
print("max absolute error:", np.max(np.abs(actual - expected)), flush=True)
assert np.array_equal(actual, expected), "Gradient accumulation is incorrect"
```

Run each command in a fresh process. The TPU commands require exclusive access to four devices. Persistent compilation caching is disabled.

### Default TPU pipeline

```bash
JAX_PLATFORMS=tpu JAX_ENABLE_COMPILATION_CACHE=false \
python repro_scatter_scan.py
```

Observed output:

```text
incorrect elements: 109628
max absolute error: 1.3125
```

The final assertion fails with `AssertionError: Gradient accumulation is incorrect`; the process exits with status 1.

### Disable only the suspected pass

```bash
JAX_PLATFORMS=tpu JAX_ENABLE_COMPILATION_CACHE=false \
XLA_FLAGS=--xla_disable_hlo_passes=while-loop-all-reduce-code-motion \
python repro_scatter_scan.py
```

Observed output:

```text
incorrect elements: 0
max absolute error: 0.0
```

The assertion passes and the process exits with status 0.

### Four-device CPU comparison

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_COMPILATION_CACHE=false \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
python repro_scatter_scan.py
```

This also reports zero mismatches and exits with status 0.

## Expected behavior

Accumulating the gradients from both microbatches should match the NumPy reference, including contributions from all partitions and both iterations.

## Before/after HLO evidence

Adjacent HLO dumps immediately before and after `while-loop-all-reduce-code-motion` show the following transformation. The pseudocode below describes the relevant HLO semantics, not the original Python program.

Before the pass:

```python
carry = 0
for microbatch in microbatches:
    base = carry if partition_id == 0 else 0
    local = scatter_add(base, indices, updates)
    carry = all_reduce_sum(local)
```

Here, each partition receives the same globally reduced carry after every iteration. Including the old carry on only one partition avoids counting it multiple times in the next all-reduce.

After the pass:

```python
carry = 0
for microbatch in microbatches:
    base = carry if partition_id == 0 else 0
    carry = scatter_add(base, indices, updates)

result = all_reduce_sum(carry)
```

The all-reduce has moved outside the loop, but the partition-dependent zeroing remains. Consequently, partitions 1–3 discard their previous local contributions on the next iteration.

For these inputs, this mechanism predicts an output containing only the contributions from rows `0, 1, 8, ..., 15`, rather than all 16 rows. In the extended diagnostic probe, an independently computed NumPy result for exactly those retained rows matches the incorrect TPU output element-for-element.

To generate the relevant HLO dumps from the standalone program, use a fresh dump directory:

```bash
JAX_PLATFORMS=tpu JAX_ENABLE_COMPILATION_CACHE=false \
XLA_FLAGS="--xla_dump_to=/tmp/scatter-hlo --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=while-loop-all-reduce-code-motion" \
python repro_scatter_scan.py
```

## Results and compare results

The standalone program above was tested with the following results:

| Configuration | Incorrect elements | Maximum absolute error | Exit status |
|---|---:|---:|---:|
| Four CPU devices | 0 | 0 | 0 |
| Four TPU devices, default pipeline | 109628 | 1.3125 | 1 |
| Four TPU devices, suspected pass disabled | 0 | 0 | 0 |
| Four TPU devices, default pipeline, second fresh process | 109628 | 1.3125 | 1 |

The two default TPU runs were compiled independently with persistent compilation caching disabled. HLO dumping was enabled during these validation runs, including both sides of the pass-disable comparison.

Separately, an extended diagnostic version of the program also produced zero mismatches on TPU when either flattening the trailing table dimensions before lookup or placing an `optimization_barrier` on the table tangent. Those controls are not included in the short program above.

## Trigger

A scan carry uses select(partition_id == 0, prior_carry, zero) before scatter and all-reduce.

## Relevant entities

while-loop carry; device-dependent predicate; scatter; all-reduce group; hoisted collective

## Implemented relation

Moving an all-reduce out of a loop preserves every device's carried contribution under the same predicate semantics.

## Pre-state

The select retains the prior carry on one partition and resets it on other partitions before each in-loop reduction.

## Binding transition

The scatter-specialized eligibility check ignores any select between the carry buffer and zero without checking predicate replication.

## Post-state

The all-reduce moves outside the loop while the device-dependent select remains inside.

## Validity obligation

A carry select is transparent to collective motion only when its predicate is replicated within each all-reduce group.

## Invalid decision

IsScatterBufferUsed treated carry-versus-zero selects as harmless for both replicated and per-device predicates.

## Physical effect

Nonzero partitions discard earlier contributions that the moved collective can no longer restore at each iteration.

## Observed consequence

The TPU gradient accumulation has 109,628 incorrect elements and maximum absolute error 1.3125.

## Repair site

xla/hlo/transforms/collectives/while_loop_all_reduce_code_motion.cc

## Repair

PR 49236 removes the weaker scatter-only checker and reuses the generic buffer-use analysis, which tests predicate replication; eight parameterized regressions cover safe and unsafe cases.

## Why the repair works

Device-dependent selects now keep the collective inside the loop, while group-replicated predicates remain eligible for motion.

## Causal basis

The imported patch explicitly states it fixes 49143 and documents the lost-contribution mechanism and four failing negative controls.

## Analysis implication

Collective-motion checks need predicate values across the collective group, carry uses, scatter semantics, and the before/after reduction location.

## Fix patches

- [`fixes/xla-49143-pr49236.patch`](fixes/xla-49143-pr49236.patch) — https://github.com/openxla/xla/pull/49236
