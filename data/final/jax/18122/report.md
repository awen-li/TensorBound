# jax.lax.all_to_all gives wrong result on complex arrays.

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/18122](https://github.com/jax-ml/jax/issues/18122)
- Opened: 2023-10-15T01:06:09Z
- Closed: 2023-10-16T23:02:29Z
- Labels: `bug`, `XLA`, `NVIDIA GPU`

## Reported behavior

### Description

jax.lax.all_to_all is correct on real arrays, but gives wrong result on complex arrays. The code reproduce the issue is the following.
```python
import jax
import jax.numpy as jnp
def all_to_all(xx):
    """twice all_to_all is expected to give an identical output as input"""
    yy = jax.lax.all_to_all(xx, "p", 1, 0)
    zz = jax.lax.all_to_all(yy, "p", 0, 1)
    return zz
pa2a = jax.pmap(all_to_all, axis_name="p", in_axes=0, out_axes=0)
# test on real array
aa = jnp.arange(2*3*2*4).reshape(2,3,2,4)
bb = pa2a(aa)
print('diff in real case', jnp.max(jnp.abs(aa-bb)))
# test on complex array
aa = jnp.arange(2*3*2*4).reshape(2,3,2,4)
aa = aa + 1.J*aa
bb = pa2a(aa)
print('diff in complex case', jnp.max(jnp.abs(aa-bb)))
```

I run the code on two GPUs with jax version is 0.4.18. The output is
```
diff in real case 0
diff in complex case 66.46803
```

### What jax/jaxlib version are you using?

0.4.18

### Which accelerator(s) are you using?

GPU

### Additional system info

python 3.10 Linux

### NVIDIA GPU info

_No response_

## Trigger

A complex64 array is exchanged by lax.all_to_all across two GPUs using a split dimension.

## Relevant entities

logical complex elements; NCCL scalar datatype; count multiplier; participant chunks; send byte offsets; receive byte offsets

## Implemented relation

Each participant exchanges one disjoint byte chunk representing its share of the logical complex elements.

## Pre-state

The buffer contains element_count logical complex values, each represented to NCCL by two real scalar values.

## Binding transition

The old split-dimension path multiplies element_count by the complex scalar multiplier and then derives chunk_bytes from that inflated count using the logical complex byte width.

## Post-state

Chunk byte offsets advance twice as far as the logical partition while NCCL counts and offsets no longer describe the same chunk.

## Validity obligation

The complex-to-NCCL multiplier is applied once to transfer counts, while byte offsets partition the original logical buffer exactly once.

## Invalid decision

RunAllToAll incorporated the complex multiplier into element_count before computing chunk_elements and chunk_bytes.

## Physical effect

Participants send and receive bytes from incorrect chunk boundaries, leaving the reconstructed complex array with misplaced or invalid values.

## Observed consequence

Applying complementary all_to_all operations fails to reconstruct the input and produces a maximum difference of 66.46803, while the real-valued control is exact.

## Repair site

xla/service/gpu/nccl_all_to_all_thunk.cc::RunAllToAll split-dimension branch

## Repair

Commit 92d3454 keeps element_count in logical elements, computes byte offsets from logical chunk_elements, and applies the complex multiplier only to the NCCL send and receive counts.

## Why the repair works

The byte ranges now partition the physical complex buffer once and each NCCL call still transfers both real scalar components of every logical complex element.

## Causal basis

The XLA commit explicitly fixes JAX issue 18122, names the double-applied multiplier, and changes the exact count and offset calculations exercised by the report.

## Analysis implication

Collective analysis needs logical element type and count, transport datatype and multiplier, participant count, per-peer scalar count, byte width, and byte offsets.

## Fix patches

- [`fixes/jax-18122-commit92d3454.patch`](fixes/jax-18122-commit92d3454.patch) — https://github.com/openxla/xla/commit/92d3454475dc1413517ff60b999b14d0a308e994
