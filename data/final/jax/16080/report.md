# zeros_like does not preserve sharding information

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/16080](https://github.com/jax-ml/jax/issues/16080)
- Opened: 2023-05-21T06:24:24Z
- Closed: 2023-05-21T13:50:40Z
- Labels: `bug`

## Reported behavior

### Description

It looks like zeros_like doesn't preserve sharding information (@sholtodouglas found something different a while back here: https://github.com/google/jax/discussions/12799#discussioncomment-3887579). I feel like it probably should, as Sholto indicated in his post as well.

```python
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PositionalSharding


@pjit
def init_zeros_like(arr):
    zz = jnp.zeros_like(arr)
    return zz

pos_sharding = PositionalSharding(jax.devices()).reshape((-1, 1))

arr0 = jnp.ones((16, 2))
arr0 = jax.device_put(arr0, pos_sharding)

print("initial array, sharded")
jax.debug.visualize_array_sharding(arr0)

arr1 = init_zeros_like(arr0)

print("result array, should be sharded but isn't")
jax.debug.visualize_array_sharding(arr1)
assert arr0.sharding.is_equivalent_to(arr1.sharding, 2), (arr0.sharding, arr1.sharding)  # fails
```

<details><summary>Output showing failure</summary>
<pre>
/opt/homebrew/Caskroom/miniforge/base/envs/levanter/bin/python /Users/dlwh/src/levanter/zeros_like_issue.py 
initial array, sharded
┌───────┐
│ CPU 0 │
├───────┤
│ CPU 1 │
├───────┤
│ CPU 2 │
├───────┤
│ CPU 3 │
├───────┤
│ CPU 4 │
├───────┤
│ CPU 5 │
├───────┤
│ CPU 6 │
├───────┤
│ CPU 7 │
└───────┘
result array, should be sharded but isn't
┌───────┐
│       │
│       │
│       │
│       │
│ CPU 0 │
│       │
│       │
│       │
│       │
└───────┘
Traceback (most recent call last):
  File "/Users/dlwh/src/levanter/zeros_like_issue.py", line 24, in <module>
    assert arr0.sharding.is_equivalent_to(arr1.sharding, 2), (arr0.sharding, arr1.sharding)
AssertionError: (PositionalSharding([[{CPU 0}]
                    [{CPU 1}]
                    [{CPU 2}]
                    [{CPU 3}]
                    [{CPU 4}]
                    [{CPU 5}]
                    [{CPU 6}]
                    [{CPU 7}]]), SingleDeviceSharding(device=CpuDevice(id=0)))

Process finished with exit code 1
</pre>
</details>


### What jax/jaxlib version are you using?

jax 0.4.10 jaxlib 0.4.10

### Which accelerator(s) are you using?

CPU (haven't checked TPU/GPU yet)

### Additional system info

Python 3.10.8, MacOS

### NVIDIA GPU info

_No response_

## Trigger

A jitted computation creates an array with the shape and dtype of an input that was explicitly sharded across devices.

## Relevant entities

input array; constructor result; input sharding; SPMD sharding propagation; shard group identifier

## Implemented relation

A constructor result requested to be sharded like an input carries an explicit same-sharding relation into SPMD propagation.

## Pre-state

The input array has an explicit multi-device sharding, while zeros_like derives the new array's shape and dtype from that input.

## Binding transition

The old constructor creates a constant-shaped result without representing a sharding relation between the input and result.

## Post-state

SPMD propagation sees no relation that constrains the result to the input's sharding and may place the result on the default device.

## Validity obligation

When two values are required to share a placement, that equality remains explicit until the SPMD partitioner chooses their common sharding.

## Invalid decision

The constructor preserved shape and dtype but omitted the input-to-result sharding relation needed by the partitioner.

## Physical effect

The input spans multiple devices while the constructed result is placed on one default device.

## Observed consequence

The reported zeros_like result is single-device sharded and fails the equivalence check against the eight-device input sharding.

## Repair site

jax/_src/shard_alike.py::_group_shard and jax.experimental.shard_alike

## Repair

Commit 72fbdb2 adds shard_alike, which lowers both values to side-effecting Sharding custom calls with one shard-group identifier and the AS group type; its tests check that jitted constructor-like results inherit the input's multi-device sharding.

## Why the repair works

The shared group identifier makes equal sharding a relation visible to bidirectional SPMD propagation, so the partitioner selects one compatible sharding for both values instead of placing the new value independently.

## Causal basis

A JAX maintainer closes report 16080 as a duplicate of 15600. Report 15600 proposes a shard_like primitive as the fix, and the merged commit declares that it fixes 15600, implements that relation, and tests its placement effect.

## Analysis implication

Placement analysis needs value provenance, existing sharding, same-sharding group identity, propagation direction, and the final device assignment together.

## Fix patches

- [`fixes/jax-16080-commit-72fbdb2.patch`](fixes/jax-16080-commit-72fbdb2.patch) — https://github.com/jax-ml/jax/commit/72fbdb2eb5589cf6d84899034d04b01722a64718
