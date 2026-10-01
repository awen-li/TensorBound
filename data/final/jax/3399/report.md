# vmap(pmap) different results than map(pmap)

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/3399](https://github.com/jax-ml/jax/issues/3399)
- Opened: 2020-06-10T22:41:17Z
- Closed: 2020-06-15T02:26:16Z
- Labels: `bug`

## Reported behavior

Likely I'm confused, but I recently tracked a bug in my program to a different output when I run (what I expect to be) the same computation with vmap vs. map.

This was run on TPUv2.

```
import functools
import jax
import jax.numpy as jnp
print(jax.local_device_count())

# In the code below, pts only exists to allow us to execute with pmap.

def map_version(qs, pts):
  return jax.lax.map(lambda x: func(x, pts), qs)

def vmap_version(qs, pts):
  return jax.vmap(func, in_axes=(0, None))(qs, pts)

def func(q, pts):
"""Returns q and also the view of q that the pmap'd lambda gets"""
  q_from_pmap = jax.pmap(lambda x, y: y, in_axes=(0, None))(pts, q)
  return q, q_from_pmap

device_count = 2
pts = jnp.ones(device_count)
qs = jnp.asarray(((0,0), (3,3), (2,2)))

print(f'qs.shape: {qs.shape} qs:\n{qs}')
print(pts.shape: {pts.shape} {pts}')

print(f'map version\n-----------')
q, q_from_pmap = map_version(qs, pts)
print(f'q from func:{q.shape}\n{q}')
print(f'q_from_pmap:{q_from_pmap.shape}\n{q_from_pmap}')

print(f'vmap version\n-----------')
q, q_from_pmap = vmap_version(qs, pts)
print(f'q from func:{q.shape}\n{q}')
print(f'q_from_pmap:{q_from_pmap.shape}\n{q_from_pmap}')
```

Output:
```
8
qs.shape: (3, 2) qs:
[[0 0]
 [3 3]
 [2 2]]
points.shape: (2,) [1. 1.]
map version
-----------
q from func:(3, 2)
[[0 0]
 [3 3]
 [2 2]]
q_from_pmap:(3, 2, 2)
[[[0 0]
  [0 0]]

 [[3 3]
  [3 3]]

 [[2 2]
  [2 2]]]
vmap version
-----------
q from func:(3, 2)
[[0 0]
 [3 3]
 [2 2]]
q_from_pmap:(2, 2, 3)
[[[0 3 2]
  [0 3 2]]

 [[0 3 2]
  [0 3 2]]]
```

The map version does exactly what I would expect. The vmap version does not.

If I had to guess, it would be that vmap does some sort of virtual slicing, but that slicing is not maintained once qs is passed through to the pmapped lambda. The pmapped lambda seems to get all the rows of qs, when I would expect it to only get one row per batch.

Please let me know if I'm abusing vmap here, or if it's a bug.


**Update**

If we transpose qs in vmap_version, everything works:
```
def vmap_version(qs, pts):
  return jax.vmap(func, in_axes=(0, None))(qs.T, pts)
```
This is true for my actual program as well, which is doing quite a bit more sophisticated work than this example. Doesn't feel right. :)

## Trigger

vmap batches q while an inner pmap maps pts on axis zero and treats q as unmapped.

## Relevant entities

vmap batch axis; pmap device axis; mapped_invars; q; pts; output axes

## Implemented relation

Axis movement across nested map transformations depends on whether each pmap operand is mapped by the inner map.

## Pre-state

q has a vmap axis but is explicitly unmapped for pmap; pts is mapped by pmap.

## Binding transition

BatchTrace.process_map moves and renumbers every batched operand as if pmap mapped all inputs.

## Post-state

q's vmap dimension is reinterpreted relative to the pmap dimension and emerges in the wrong output position.

## Validity obligation

Nested-map axis adjustment uses each operand's mapped_invars flag when inserting or removing the inner device axis.

## Invalid decision

The old batching rule ignored pmap's mapped_invars and applied one axis transformation to mapped and unmapped operands.

## Physical effect

The logical batch, device, and value dimensions are permuted in the pmap result.

## Observed consequence

vmap(pmap) returns shape 2x2x3 with transposed values instead of the map(pmap) result of shape 3x2x2.

## Repair site

jax/interpreters/batching.py::BatchTrace.process_map

## Repair

PR 3439 reads mapped_invars, moves an axis only for a mapped inner operand, adjusts each batch dimension by that flag, and adds the reported vmap-of-pmap comparison.

## Why the repair works

Each operand's outer batch axis is renumbered relative to an inner device axis only when that inner axis actually exists for the operand.

## Causal basis

The maintainer diagnosis names mapped_invars, and the linked patch uses that field in the failing BatchTrace path and adds the issue reproducer.

## Analysis implication

Nested mapping analysis needs per-operand axis identity and mapped status at every transformation boundary.

## Fix patches

- [`fixes/jax-3399-pr3439.patch`](fixes/jax-3399-pr3439.patch) — https://github.com/jax-ml/jax/pull/3439
