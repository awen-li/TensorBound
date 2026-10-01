# Sharded FFT with JIT gives incorrect results on GPU since 0.4.9

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/16909](https://github.com/jax-ml/jax/issues/16909)
- Opened: 2023-07-31T16:35:32Z
- Closed: 2023-10-23T13:53:07Z
- Labels: `bug`, `XLA`, `NVIDIA GPU`

## Reported behavior

### Description

(this is not the same issue as #15680, but it gives helpful information on this topic)

I am trying to calculate the 3D-FFT of a 1024^3 matrix using 2 GPUs. This should work using code similar to https://jax.readthedocs.io/en/latest/jax.experimental.custom_partitioning.html (but adapted to calculate a 3D-FFT instead of a 1D) or the JAX-only code from https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuFFTMp/JAX_FFT (as shared in #15680).

My code seems to give the correct result when not using JIT, when only using GPU devices or when not sharding the memory.
But in the combination I need (JIT with GPU on sharded memory), the results differ from the other results (and what scipy/numpy calculates).


main code as a minimal example:
```python
import os
from pathlib import Path

import jax
import numpy as np
import scipy
from jax import jit
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

import sharded_fft_minimal as sharded_fft

os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count=2'

jax.config.update("jax_enable_x64", True)


def print_subset(x):
    print(x[0, :4, :4])


size = 4
# xs = jax.random.normal(jax.random.PRNGKey(0), (size, size, size), dtype=jax.numpy.complex64)
xs = jax.numpy.ones((size, size, size), dtype=jax.numpy.complex64)

x_np = np.asarray(xs)
out_ref = scipy.fft.fftn(x_np, workers=128)

devices = mesh_utils.create_device_mesh((2,))

mesh = Mesh(devices, axis_names=('gpus',))

with mesh:
    xshard = jax.device_put(xs, NamedSharding(mesh, P("gpus", None)))

    def two_fft(x):
        out = sharded_fft.fft(x)
        out = sharded_fft.fft_X(out)
        return out


    two_fft_jit = jit(
        two_fft,
        in_shardings=(NamedSharding(mesh, P("gpus", None))),
        out_shardings=(NamedSharding(mesh, P(None, "gpus")))
    )

    out = two_fft(xshard).block_until_ready()
    out_jit = two_fft_jit(xshard).block_until_ready()

    print("JAX output without JIT:")
    print_subset(out)
    print("JAX output with JIT:")
    print_subset(out_jit)
    print("Reference output:")
    print_subset(out_ref)

    is_equal = np.allclose(out_ref, out_jit, rtol=1.e-2, atol=1.e-4)

    print(f"Are the jax (JIT) and scipy results the same? {is_equal}")

    a = Path("compiled.txt")
    a.write_text(two_fft_jit.lower(xshard).compile().as_text())
```
with `shared_fft_minimal.py`:

```python
import jax
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import PartitionSpec as P, NamedSharding


def _fft(x):
    return jax.numpy.fft.fftn(x, axes=list(range(1, len(x.shape))))


def _fftX(x):
    return jax.numpy.fft.fft(x, axis=0)


def supported_sharding(sharding, shape):
    return NamedSharding(sharding.mesh, P("gpus",None))


def partition(arg_shapes, result_shape):
    result_shardings = jax.tree_map(lambda x: x.sharding, result_shape)
    arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
    return _fft, supported_sharding(arg_shardings[0], arg_shapes[0]), (
        supported_sharding(arg_shardings[0], arg_shapes[0]),)


def infer_sharding_from_operands(arg_shapes, result_shape):
    arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
    return supported_sharding(arg_shardings[0], arg_shapes[0])


def supported_sharding_X(sharding, shape):
    return NamedSharding(sharding.mesh, P(None, "gpus"))


def partition_X(arg_shapes, result_shape):
    result_shardings = jax.tree_map(lambda x: x.sharding, result_shape)
    arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
    return _fftX, supported_sharding_X(arg_shardings[0], arg_shapes[0]), (
        supported_sharding_X(arg_shardings[0], arg_shapes[0]),)


def infer_sharding_from_operands_X(arg_shapes, result_shape):
    arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
    return supported_sharding_X(arg_shardings[0], arg_shapes[0])


def fft(x):
    @custom_partitioning
    def _fft_(x):
        return _fft(x)

    _fft_.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition)
    return _fft_(x)


def fft_X(x):
    @custom_partitioning
    def _fft_(x):
        return _fftX(x)

    _fft_.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands_X,
        partition=partition_X)
    return _fft_(x)
```

When I run this on a host with 2 GPUs (with CPUs all three outputs are the same), I get:
```text
2023-07-31 18:28:38.496970: W external/xla/xla/service/gpu/nvptx_compiler.cc:698] The NVIDIA driver's CUDA version is 11.6 which is older than the ptxas CUDA version (11.8.89). Because the driver is older than the ptxas version, XLA is disabling parallel compilation, which may slow down compilation. You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.
JAX output without JIT:
[[64.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]]
JAX output with JIT:
[[32.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]]
Reference output:
[[64.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]]
Are the jax (JIT) and scipy results the same? False
```


<details><summary>HLO</summary>
<p>

```
HloModule jit_two_fft, is_scheduled=true, entry_computation_layout={(c64[2,4,4]{2,1,0})->c64[4,2,4]{2,1,0}}

%fused_computation (param_0.1: c64[2,2,2,4]) -> c64[2,4,4] {
  %param_0.1 = c64[2,2,2,4]{3,2,0,1} parameter(0)
  %bitcast.30 = c64[4,2,4]{2,1,0} bitcast(c64[2,2,2,4]{3,2,0,1} %param_0.1)
  ROOT %transpose.5 = c64[2,4,4]{2,1,0} transpose(c64[4,2,4]{2,1,0} %bitcast.30), dimensions={1,2,0}, metadata={op_name="jit(two_fft)/jit(main)/custom_partitioning[partition=<function partition_X at 0x1510182a8310> propagate_user_sharding=None infer_sharding_from_operands=<function infer_sharding_from_operands_X at 0x1510182a83a0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[]]" source_file="/jax-testing/sharded_fft_minimal.py" source_line=65}
}

%fused_computation.1 (param_0.4: c64[2,4,4]) -> c64[2,2,2,4] {
  %param_0.4 = c64[2,4,4]{2,1,0} parameter(0)
  %bitcast.32 = c64[2,2,2,4]{3,2,1,0} bitcast(c64[2,4,4]{2,1,0} %param_0.4)
  %transpose.6 = c64[2,2,2,4]{3,2,1,0} transpose(c64[2,2,2,4]{3,2,1,0} %bitcast.32), dimensions={1,0,2,3}, metadata={op_name="jit(two_fft)/jit(main)/custom_partitioning[partition=<function partition_X at 0x1510182a8310> propagate_user_sharding=None infer_sharding_from_operands=<function infer_sharding_from_operands_X at 0x1510182a83a0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[]]" source_file="/jax-testing/sharded_fft_minimal.py" source_line=65}
  ROOT %bitcast.31 = c64[2,2,2,4]{3,2,0,1} bitcast(c64[2,2,2,4]{3,2,1,0} %transpose.6), metadata={op_name="jit(two_fft)/jit(main)/custom_partitioning[partition=<function partition_X at 0x1510182a8310> propagate_user_sharding=None infer_sharding_from_operands=<function infer_sharding_from_operands_X at 0x1510182a83a0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[]]" source_file="/jax-testing/sharded_fft_minimal.py" source_line=65}
}

ENTRY %main.6_spmd (param: c64[2,4,4]) -> c64[4,2,4] {
  %param = c64[2,4,4]{2,1,0} parameter(0), sharding={devices=[2,1,1]0,1}
  %fft.0 = c64[2,4,4]{2,1,0} fft(c64[2,4,4]{2,1,0} %param), fft_type=FFT, fft_length={4,4}, metadata={op_name="jit(main)/jit(fft)/fft[fft_type=FftType.FFT fft_lengths=(4, 4)]" source_file="/jax-testing/sharded_fft_minimal.py" source_line=7}
  %fusion.1 = c64[2,2,2,4]{3,2,0,1} fusion(c64[2,4,4]{2,1,0} %fft.0), kind=kLoop, calls=%fused_computation.1, metadata={op_name="jit(two_fft)/jit(main)/custom_partitioning[partition=<function partition_X at 0x1510182a8310> propagate_user_sharding=None infer_sharding_from_operands=<function infer_sharding_from_operands_X at 0x1510182a83a0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[]]" source_file="/jax-testing/sharded_fft_minimal.py" source_line=65}
  %all-to-all-start = ((c64[2,2,2,4]{3,2,0,1}), c64[2,2,2,4]{3,2,0,1}) all-to-all-start(c64[2,2,2,4]{3,2,0,1} %fusion.1), channel_id=1, replica_groups={{0,1}}, dimensions={1}, metadata={op_name="jit(two_fft)/jit(main)/custom_partitioning[partition=<function partition_X at 0x1510182a8310> propagate_user_sharding=None infer_sharding_from_operands=<function infer_sharding_from_operands_X at 0x1510182a83a0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[]]" source_file="/jax-testing/sharded_fft_minimal.py" source_line=65}, backend_config={"is_sync":true}
  %all-to-all-done = c64[2,2,2,4]{3,2,0,1} all-to-all-done(((c64[2,2,2,4]{3,2,0,1}), c64[2,2,2,4]{3,2,0,1}) %all-to-all-start), channel_id=1, replica_groups={{0,1}}, dimensions={1}, metadata={op_name="jit(two_fft)/jit(main)/custom_partitioning[partition=<function partition_X at 0x1510182a8310> propagate_user_sharding=None infer_sharding_from_operands=<function infer_sharding_from_operands_X at 0x1510182a83a0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[]]" source_file="/jax-testing/sharded_fft_minimal.py" source_line=65}
  %fusion = c64[2,4,4]{2,1,0} fusion(c64[2,2,2,4]{3,2,0,1} %all-to-all-done), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(two_fft)/jit(main)/custom_partitioning[partition=<function partition_X at 0x1510182a8310> propagate_user_sharding=None infer_sharding_from_operands=<function infer_sharding_from_operands_X at 0x1510182a83a0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[]]" source_file="/jax-testing/sharded_fft_minimal.py" source_line=65}
  %fft.1 = c64[2,4,4]{2,1,0} fft(c64[2,4,4]{2,1,0} %fusion), fft_type=FFT, fft_length={4}, metadata={op_name="jit(main)/jit(fft)/fft[fft_type=FftType.FFT fft_lengths=(4,)]" source_file="/jax-testing/sharded_fft_minimal.py" source_line=11}
  ROOT %transpose.4 = c64[4,2,4]{2,1,0} transpose(c64[2,4,4]{2,1,0} %fft.1), dimensions={2,0,1}, frontend_attributes={fingerprint_before_lhs="902ad6c9909648982a9ff9203157ab0d"}, metadata={op_name="jit(two_fft)/jit(main)/custom_partitioning[partition=<function partition_X at 0x1510182a8310> propagate_user_sharding=None infer_sharding_from_operands=<function infer_sharding_from_operands_X at 0x1510182a83a0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[]]" source_file="/jax-testing/sharded_fft_minimal.py" source_line=65}
}
```

</p>
</details>

### What jax/jaxlib version are you using?

jax==0.4.14, jaxlib==0.4.14+cuda11.cudnn86

### Which accelerator(s) are you using?

GPU (2x Nvidia A40)

### Additional system info

Python 3.9.15, AlmaLinux 8.5

### NVIDIA GPU info

```text
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.39.01    Driver Version: 510.39.01    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A40          Off  | 00000000:41:00.0 Off |                    0 |
|  0%   32C    P0    74W / 300W |      0MiB / 46068MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A40          Off  | 00000000:A1:00.0 Off |                    0 |
|  0%   31C    P0    75W / 300W |      0MiB / 46068MiB |      3%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Duplicate mapping

- Primary report: `github-jax-ml-jax-18122`
- Shared cause: `cause-jax-18122-complex-all-to-all-double-applies-count-multiplier`

The report bisects the regression to XLA all-to-all support; the maintainer identifies commit 92d3454 as its repair, and two reporters confirm the sharded FFT works after jaxlib 0.4.19.

The sections below describe the shared cause reconstructed from the primary report. The reported behavior above remains the observation specific to this report.

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

- [`fixes/jax-16909-commit92d3454.patch`](fixes/jax-16909-commit92d3454.patch) — https://github.com/openxla/xla/commit/92d3454475dc1413517ff60b999b14d0a308e994
