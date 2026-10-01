# SPMD partitioner produces incorrect all-reduce ordering for scatter-add backward

- Repository: `openxla/xla`
- Report: [https://github.com/openxla/xla/issues/42481](https://github.com/openxla/xla/issues/42481)
- Opened: 2026-05-12T21:11:16Z
- Closed: 2026-07-09T18:42:46Z
- Labels: `bug`, `NVIDIA-GPU`, `err:performance`

## Reported behavior

## Bug Summary
  The SPMD partitioner incorrectly orders collectives in the backward pass of
  jnp.take (scatter-add) when the gather operand has a `reduced` axis annotation.
  It performs dynamic-slice BEFORE the all-reduce across TP, which conflates
  unrelated vocab positions that happen to share the same local index.

  Correct order:  scatter → all-reduce across TP → dynamic-slice (extract shard)
  Buggy order:    scatter → dynamic-slice (extract shard) → all-reduce across TP

  The all-reduce is meant to combine partial scatter contributions from different
  TP devices (which each processed different sequence positions). It must happen on
  the full-vocab buffer so that global indices align. When it happens after the
  slice, local row i from shard 0 (global row i) gets summed with local row i from
  shard 1 (global row i + shard_size), producing incorrect gradients.

## Reproduction
  python scatter_add_sharding_bug.py

  Expected: only vocab rows present in input_ids have non-zero gradient.
  Actual: gradient leaks to unrelated rows due to the wrong collective ordering.

## Environment
  - JAX 0.10.0.dev20260512 (jaxlib 0.10.0.dev20260512+cuda12.cudnn101)
  - 8x NVIDIA H100 80GB
  - CUDA 12, cuDNN 10.1

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, PartitionSpec as PS, auto_axes

VOCAB_SIZE = 256
HIDDEN_SIZE = 128
BATCH_SIZE = 8
SEQ_LEN = 32


def simulate_wrong_reduction(input_ids, dp, tp):
    """Manually simulate the incorrect HLO pattern for validation.

    The buggy HLO does:
      1. Each device scatters gradients into a full [VOCAB_SIZE, HIDDEN_SIZE] buffer
         using global indices (but only processes its local seq slice).
      2. dynamic-slice extracts the local TP shard (VOCAB_SIZE/TP rows).
      3. all-reduce across TP within each DP group.  <-- BUG: wrong order
      4. all-reduce across DP.

    Step 3 is incorrect because it sums local row i from different TP shards,
    which correspond to different global vocab positions.
    """
    shard_size = VOCAB_SIZE // tp
    ids = np.array(input_ids)
    bs_per_dp = BATCH_SIZE // dp
    seq_per_tp = SEQ_LEN // tp

    device_grads = np.zeros((dp, tp, shard_size, HIDDEN_SIZE))
    for d in range(dp):
        for t in range(tp):
            local_ids = ids[
                d * bs_per_dp : (d + 1) * bs_per_dp,
                t * seq_per_tp : (t + 1) * seq_per_tp,
            ]
            full_buf = np.zeros((VOCAB_SIZE, HIDDEN_SIZE))
            for tok in local_ids.flatten():
                full_buf[tok, :] += 1.0
            device_grads[d, t] = full_buf[t * shard_size : (t + 1) * shard_size]

    # Buggy all-reduce across TP (sums unrelated global rows)
    after_tp_reduce = np.sum(device_grads, axis=1)
    # All-reduce across DP
    after_dp_reduce = np.sum(after_tp_reduce, axis=0)

    full_grad = np.zeros((VOCAB_SIZE, HIDDEN_SIZE))
    for t in range(tp):
        full_grad[t * shard_size : (t + 1) * shard_size] = after_dp_reduce
    return np.any(full_grad != 0, axis=-1)


def main():
    num_devices = jax.device_count()
    assert num_devices >= 4, f"Need >= 4 devices, got {num_devices}"
    dp, tp = 2, num_devices // 2

    print(f"JAX {jax.__version__}")
    print(f"Devices: {num_devices}x {jax.devices()[0].device_kind}")
    print(f"Mesh: DP={dp}, TP={tp}")
    print(f"Vocab={VOCAB_SIZE}, Hidden={HIDDEN_SIZE}, Batch={BATCH_SIZE}, Seq={SEQ_LEN}")
    print(f"Token ID range: [1, {VOCAB_SIZE // 4})")
    print()

    mesh = jax.make_mesh(
        (dp, tp),
        ("dp", "tp"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )

    embedding_ps = PS("tp", None)
    residual_ps = PS("dp", "tp", None)
    batch_ps = PS("dp", None)
    compute_ps = PS("tp", None, reduced=frozenset({"dp"}))

    params = jax.random.normal(jax.random.PRNGKey(42), (VOCAB_SIZE, HIDDEN_SIZE))
    input_ids = jax.random.randint(
        jax.random.PRNGKey(0), (BATCH_SIZE, SEQ_LEN), 1, VOCAB_SIZE // 4
    )

    with jax.set_mesh(mesh):

        @jax.jit
        def compute(params_in, ids_in):
            emb = jax.sharding.reshard(params_in, compute_ps)
            ids = jax.sharding.reshard(ids_in, batch_ps)
            grad = jax.grad(
                lambda e: jnp.sum(
                    auto_axes(jnp.take, out_sharding=residual_ps)(e, ids, axis=0)
                )
            )(emb)
            grad = jax.sharding.reshard(grad, embedding_ps)
            return grad

        grad_result = np.array(compute(params, input_ids))

    actual_active = np.any(grad_result != 0, axis=-1)
    simulated_active = simulate_wrong_reduction(input_ids, dp, tp)

    used_ids = set(np.array(input_ids).flatten().tolist())
    actual_leaked = [i for i in range(VOCAB_SIZE) if i not in used_ids and actual_active[i]]

    print("--- Results ---")
    print(f"  Expected non-zero rows: {len(used_ids)}/{VOCAB_SIZE}")
    print(f"  Actual non-zero rows:   {int(np.sum(actual_active))}/{VOCAB_SIZE}")
    print(f"  Leaked (should be 0):   {len(actual_leaked)}")
    if actual_leaked:
        print(f"  Leaked rows (first 20): {actual_leaked[:20]}")
    print()

    match = np.array_equal(actual_active, simulated_active)
    print(f"--- Matches simulated wrong-order pattern: {match} ---")
    print()

    if actual_leaked:
        print("BUG CONFIRMED.")
        print()
        print("The SPMD partitioner produces this backward for the TP-sharded gather:")
        print("  1. scatter-add into [256, 128] buffer (correct)")
        print("  2. dynamic-slice to extract local shard [64, 128] (too early)")
        print("  3. all-reduce across TP on [64, 128] (wrong: mixes unrelated rows)")
        print("  4. all-reduce across DP (correct)")
        print()
        print("Correct ordering should be:")
        print("  1. scatter-add into [256, 128] buffer")
        print("  2. all-reduce across TP on [256, 128] (combine seq contributions)")
        print("  3. dynamic-slice to extract local shard [64, 128]")
        print("  4. all-reduce across DP")
    else:
        print("PASS: no gradient leak detected.")


if __name__ == "__main__":
    main()
```

## Trigger

The scatter operand is sharded on a vocabulary dimension while updates use the same mesh axis as a reduction dimension.

## Relevant entities

scatter operand; updates; global vocabulary rows; dynamic slice; all-reduce; device groups; mesh axis

## Implemented relation

Partial updates for one global scatter destination are reduced while global indices still align, before each device extracts its operand shard.

## Pre-state

Each device holds partial updates indexed in the full global vocabulary space.

## Binding transition

The old index-passthrough partitioning path dynamically slices the operand result before reducing updates across the conflicting device groups.

## Post-state

Equal local row numbers from different global vocabulary shards become operands of the same all-reduce position.

## Validity obligation

When operand and update shardings share a non-independent mesh axis, scatter reduction precedes operand slicing or the strategy falls back to a safe representation.

## Invalid decision

GSPMD accepted index-passthrough scatter partitioning without checking the operand/update reduction-sharding conflict.

## Physical effect

The all-reduce sums unrelated global vocabulary rows that share a local shard offset.

## Observed consequence

The jnp.take gradient leaks nonzero values into vocabulary rows absent from the input IDs.

## Repair site

xla/service/spmd/gather_scatter_handler.cc scatter partition selection

## Repair

Commit 23897cd detects overlapping operand scatter and update reduction groups and rejects the index-passthrough strategy so partitioning falls back to a safe scatter form.

## Why the repair works

The unsafe slice-before-reduce schedule is no longer generated when its device groups conflate distinct global indices.

## Causal basis

The report derives the wrong schedule, and the fixing commit adds the exact conflict predicate plus tests requiring replicated-scatter fallback instead of slice-before-reduce.

## Analysis implication

Collective-order analysis needs scatter dimension numbers, logical global indices, operand and update shardings, grouped device sets, slice position, and reduction position.

## Fix patches

- [`fixes/xla-42481-commit23897cd.patch`](fixes/xla-42481-commit23897cd.patch) — https://github.com/openxla/xla/commit/23897cddcd49450d0a808df453be06e68cabbc7d
