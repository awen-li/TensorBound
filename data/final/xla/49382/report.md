# [SPMD] Spatially partitioned convolution all-reduces over all partitions, counting partially replicated operands once per replica

- Repository: `openxla/xla`
- Report: [https://github.com/openxla/xla/issues/49382](https://github.com/openxla/xla/issues/49382)
- Opened: 2026-09-24T05:48:32Z
- Closed: Open at collection
- Labels: `bug`, `CPU`, `sharding`

## Reported behavior

### Description

If a convolution's LHS and RHS are both split on their spatial dimensions and also partially replicated (`last_tile_dim_replicate`), the SPMD partitioner does three things:

1. It exchanges halos along the spatial split.
2. It convolves each shard.
3. It all-reduces the partial results over **every** partition.

Partitions that are replicas of each other hold the same partial sum. So the result comes out multiplied by the size of the replicated dimension, with no error or warning.

In JAX this is the kernel (filter) gradient of an ordinary 2D convolution. It shows up when the input's image rows are split over one mesh axis and replicated over another. The gradient comes back 2x. In every layout we tried the input gradient was exact.

### HLO before `spmd-partitioning`

4 partitions. Both operands are split in two on H (dimension 1) and replicated over the other two devices. This module is the one JAX emits with `jax_use_shardy_partitioner=False`. The Shardy path (the default) hands the partitioner the equivalent module, with named shardings (`{mesh['a'=2,'b'=2], [{}, {'a'}, {}, {}]}`) and no sharding on the root. It gets the same all-reduce.

```
ENTRY %main.1 (x.1: f32[4,8,8,8], cotangent.1: f32[4,8,8,16]) -> f32[3,3,8,16] {
  %x.1 = f32[4,8,8,8]{3,2,1,0} parameter(0), sharding={devices=[1,2,1,1,2]<=[4] last_tile_dim_replicate}
  %cotangent.1 = f32[4,8,8,16]{3,2,1,0} parameter(1), sharding={devices=[1,2,1,1,2]<=[4] last_tile_dim_replicate}
  ROOT %conv_general_dilated.1 = f32[3,3,8,16]{3,2,1,0} convolution(%x.1, %cotangent.1), window={size=8x8 pad=1_1x1_1}, dim_labels=f01b_i01o->01bf, sharding={replicated}
}
```

### HLO after `spmd-partitioning`

Metadata and frontend attributes removed:

```
ENTRY %main.1_spmd (param: f32[4,4,8,8], param.1: f32[4,4,8,16]) -> f32[3,3,8,16] {
  %param = f32[4,4,8,8]{3,2,1,0} parameter(0), sharding={devices=[1,2,1,1,2]<=[4] last_tile_dim_replicate}
  %slice = f32[4,1,8,8]{3,2,1,0} slice(%param), slice={[0:4], [3:4], [0:8], [0:8]}
  %collective-permute = f32[4,1,8,8]{3,2,1,0} collective-permute(%slice), channel_id=1, source_target_pairs={{0,2},{1,3}}
  %slice.1 = f32[4,1,8,8]{3,2,1,0} slice(%param), slice={[0:4], [0:1], [0:8], [0:8]}
  %collective-permute.1 = f32[4,1,8,8]{3,2,1,0} collective-permute(%slice.1), channel_id=2, source_target_pairs={{2,0},{3,1}}
  %concatenate = f32[4,6,8,8]{3,2,1,0} concatenate(%collective-permute, %param, %collective-permute.1), dimensions={1}
  %param.1 = f32[4,4,8,16]{3,2,1,0} parameter(1), sharding={devices=[1,2,1,1,2]<=[4] last_tile_dim_replicate}
  %convolution = f32[3,3,8,16]{3,2,1,0} convolution(%concatenate, %param.1), window={size=4x8 pad=0_0x1_1}, dim_labels=f01b_i01o->01bf
  ROOT %all-reduce = f32[3,3,8,16]{3,2,1,0} all-reduce(%convolution), channel_id=3, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%add.clone
}
```

With `<=[4]` over `[1,2,1,1,2]`, H shard 0 is on partitions {0,1} and H shard 1 is on {2,3}. The halo exchange correctly pairs {0,2} and {1,3}. The all-reduce should use those same groups, `replica_groups={{0,2},{1,3}}`. Over `{{0,1,2,3}}` it adds each shard's partial sum twice.

### Where

Both halo-exchange paths for "LHS and RHS partitioned at spatial dimensions" create the all-reduce with an empty `CollectiveDeviceList()`, which means every partition:

- `PartitionConvolutionWithSpatialDimensionHaloExchangeOnRHS`, [convolution_handler.cc#L519-L521](https://github.com/openxla/xla/blob/73d973412746940ae206c6c5190d3a0a19ffd225/xla/service/spmd/convolution_handler.cc#L519-L521)
- `PartitionConvolutionWithSpatialDimensionHaloExchangeOnLHS`, [convolution_handler.cc#L749-L751](https://github.com/openxla/xla/blob/73d973412746940ae206c6c5190d3a0a19ffd225/xla/service/spmd/convolution_handler.cc#L749-L751)

A suggestion we have not tested: the dot handler reduces partial sums with `SpmdPartitioner::AllReduceAlongShardingDims` over the partitioned dimensions. Doing the same here over the LHS's partitioned spatial dimensions would leave the replicas out of the sum.

### JAX reproducer

Stock `jax==0.11.2` and `jaxlib==0.11.2` from PyPI, CPU:

```python
# JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 python params.py
# The kernel gradient alone, from x and the output cotangent passed in with
# their image rows (H) split over "a" of a 2x2 mesh and replicated over "b".
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((2, 2), ("a", "b"), axis_types=(AxisType.Auto, AxisType.Auto))
x = jax.random.normal(jax.random.key(0), (4, 8, 8, 8))  # NHWC
kernel = jax.random.normal(jax.random.key(1), (3, 3, 8, 16))  # HWIO
cotangent = jax.random.normal(jax.random.key(2), (4, 8, 8, 16))


def kernel_gradient(x, cotangent):
    def loss(kernel):
        y = jax.lax.conv_general_dilated(x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
        return jnp.sum(y * cotangent)
    return jax.grad(loss)(kernel)


rows = NamedSharding(mesh, P(None, "a"))
expected = kernel_gradient(x, cotangent)
got = jax.jit(kernel_gradient)(jax.device_put(x, rows), jax.device_put(cotangent, rows))
print(f"x and cotangent P(None, 'a'): |grad| / |expected| = {float(jnp.linalg.norm(got) / jnp.linalg.norm(expected)):.4f}")
```

It prints the same with `JAX_USE_SHARDY_PARTITIONER=true` (the default) and `false`:

```
x and cotangent P(None, 'a'): |grad| / |expected| = 2.0000
```

The expected ratio is 1.0000. JAX's transpose rule emits a plain `convolution`, and the wrong all-reduce carries `is_spmd_generated="true"`, so this does not look like a JAX issue.

### Other layouts

The same shapes on the same 2x2 mesh, with its axes named `rows` and `other` (`layouts.py` below). `x` is passed in with the placement shown, and the convolution's output is constrained as shown. The cotangent is a constant. The table gives the kernel gradient's norm relative to one device's. We traced only the first row to the code above. The 0.5 and grouped-convolution rows may go through other paths.

| operation | `x` | output constrained to | kernel gradient |
|---|---|---|---|
| 3x3 conv | `P(None, "rows")` | none | x2 |
| 3x3 conv | `P()` | `P(None, "rows")` | x0.5 |
| 3x3 conv | `P()` | `P(None, "other")` | x0.5 |
| 2x2 stride-2 conv (a patch embedding) | `P()` | `P(None, "rows")` | x0.5 |
| 3x3 depthwise conv (`feature_group_count=8`) | `P("rows")` | none | x2 |
| 3x3 conv | `P()` | `P("rows")` | x1 |
| 3x3 conv | `P("rows")` | `P("rows")` or none | x1 |
| 3x3 conv | `P(None, "rows")` | `P(None, "rows")` | x1 |
| 3x3 conv | `P()` or `P("rows", "other")` | `P("rows", "other")` | x1 |
| 3x3 conv | `P()` | `P()` | x1 |
| 1x1 conv | `P()` | `P(None, "rows")` | x1 |
| `einsum("bhwc,cd->bhwd")` | `P()` | `P(None, "rows")` | x1 |

In all of them the input gradient matched one device's exactly. In the depthwise row only the batch is split. On a mesh with a second axis, that is the plain data-parallel layout, and it gives a 2x kernel gradient.

<details>
<summary>layouts.py</summary>

```python
# JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 python layouts.py
# Which placements of a convolution's input and output give the wrong kernel
# gradient: the input x placed (or not), the output constrained (or not).
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((2, 2), ("rows", "other"), axis_types=(AxisType.Auto, AxisType.Auto))
x = jax.random.normal(jax.random.key(0), (4, 8, 8, 8))


def conv(x, kernel):
    return jax.lax.conv_general_dilated(x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))


def patch(x, kernel):
    return jax.lax.conv_general_dilated(x, kernel, (2, 2), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC"))


def depthwise(x, kernel):
    return jax.lax.conv_general_dilated(x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"),
                                        feature_group_count=8)


def dot(x, kernel):
    return jnp.einsum("bhwc,cd->bhwd", x, kernel[0, 0])


CASES = [  # (operator, kernel shape, x's placement, the output's constraint)
    (conv, (3, 3, 8, 16), P(), P("rows")),
    (conv, (3, 3, 8, 16), P(), P(None, "rows")),
    (conv, (3, 3, 8, 16), P(), P("rows", "other")),
    (conv, (3, 3, 8, 16), P(), P()),
    (conv, (3, 3, 8, 16), P("rows"), P("rows")),
    (conv, (3, 3, 8, 16), P("rows"), None),
    (conv, (3, 3, 8, 16), P("rows"), P("rows", "other")),
    (conv, (3, 3, 8, 16), P("rows", "other"), P("rows", "other")),
    (conv, (3, 3, 8, 16), P(None, "rows"), P(None, "rows")),
    (conv, (3, 3, 8, 16), P(None, "rows"), None),
    (conv, (3, 3, 8, 16), P(), P(None, "other")),
    (conv, (3, 3, 8, 16), P(), P("other", "rows")),
    (patch, (2, 2, 8, 16), P(), P(None, "rows")),
    (patch, (2, 2, 8, 16), P(), P("rows")),
    (depthwise, (3, 3, 1, 8), P("rows"), None),
    (conv, (1, 1, 8, 16), P(), P(None, "rows")),
    (dot, (1, 1, 8, 16), P(), P(None, "rows")),
]
for operator, shape, placed, constrained in CASES:
    kernel = jax.random.normal(jax.random.key(2), shape)

    def loss(kernel, x):
        y = operator(x, kernel)
        if constrained is not None:
            y = jax.lax.with_sharding_constraint(y, NamedSharding(mesh, constrained))
        return jnp.sum(y * cotangent)

    cotangent = jax.random.normal(jax.random.key(1), jax.eval_shape(operator, x, kernel).shape)
    expected = jax.grad(loss, argnums=(0, 1))(kernel, x)
    got = jax.jit(jax.grad(loss, argnums=(0, 1)))(kernel, jax.device_put(x, NamedSharding(mesh, placed)))
    ratio = float(jnp.linalg.norm(got[0]) / jnp.linalg.norm(expected[0]))
    input_gap = float(jnp.max(jnp.abs(got[1] - expected[1])) / jnp.max(jnp.abs(expected[1])))
    print(f"{operator.__name__} {shape[0]}x{shape[1]}, x {placed}, output constrained to {constrained}: "
          f"kernel gradient x{ratio:.4f}, input gradient off by {input_gap:.1e} of its largest entry")
```

</details>

### Impact

Hybrid layouts silently train convolutional models with wrong gradients. Examples are a UNet, a VAE or a DiT patch embedding under data x sequence, fsdp x tensor or tensor x sequence. The loss and the forward pass agree with one device, so only a gradient comparison catches it. We found it on CUDA (4x RTX 3090), where a Stable Diffusion 3-shaped model's patch-embedding kernel gradient doubled under a `tensor=2 x sequence=2` mesh.

Our workaround is to constrain each convolution's input and output so that every mesh axis above size one splits them, or none does. No layout of that form has given a wrong gradient.

### System information

- jax 0.11.2, jaxlib 0.11.2 (PyPI wheels), Python 3.13.15, Linux x86-64
- CPU backend, 4 host devices (`--xla_force_host_platform_device_count=4`)
- `convolution_handler.cc` read at 73d973412746940ae206c6c5190d3a0a19ffd225

Related, performance only: #47945, where a batch-sharded grouped convolution all-gathers the batch.

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises xLA SPMD convolution with spatial partitioning and partially replicated operands.

## Observed implementation decision

Collective group construction all-reduces over all partitions rather than one representative per replica group.

## Physical effect

Partial replicas contribute the same logical operand multiple times.

## Observed consequence

The convolution output is over-counted.

## Reported proposal or workaround

The report identifies the all-reduce grouping symptom.

## Repair trace

No landed partition-group repair or fixed regression is linked.

## Archived evidence

- Raw report: `data/raw/github/xla/49382/report.json`
- SHA-256: `06c9d068306347633f5efe8f37d6947e8e00d39a4ddc4c4bc81b726c2eb437ae`
