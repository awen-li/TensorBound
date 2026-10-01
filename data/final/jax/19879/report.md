# GPU not partitioning RNG with jax_threefry_partitionable=True (vmap/scan layers)

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/19879](https://github.com/jax-ml/jax/issues/19879)
- Opened: 2024-02-19T17:48:36Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

Mostly minimized? This shouldn't OOM on the 4 a6000's I'm using (and the unminimized version works fine on TPU)

```python
import jax
import jax.random as jrandom

from jax.sharding import PartitionSpec as P, Mesh, NamedSharding

from jax.lax import with_sharding_constraint

UP_PSPEC = P("data", None)
DOWN_PSPEC = P(None, "data")
VMAP_UP_PSPEC = P(None, "data", None)
VMAP_DOWN_PSPEC = P(None, "data", None)

MLP = 14336
EMBED = 4096
LAYERS = 48


def linear_init(dims, key, pspec):
    return with_sharding_constraint(jrandom.normal(key, dims), pspec) * 0.02


def init_module(key):
    k_fc, k_up_proj, k_down_proj = jrandom.split(key, 3)
    gate_proj = linear_init((MLP, EMBED), k_fc, UP_PSPEC)

    return gate_proj


def init_stack(key):
    return jax.vmap(init_module, spmd_axis_name=None)(jrandom.split(key, LAYERS))


DEFAULT_JAX_CONFIG = {
    "jax_threefry_partitionable": True,
}

def main():
    for k, v in DEFAULT_JAX_CONFIG.items():
        jax.config.update(k, v)

    mesh = Mesh(jax.devices("gpu"), ("data",))

    with mesh:
        key = jrandom.PRNGKey(0)
        jit_init_stack = jax.jit(init_stack,
                                 out_shardings=(NamedSharding(mesh, VMAP_UP_PSPEC)))

        jit_init_stack(key)

main()
```

```
Peak buffers:
	Buffer 1:
		Size: 10.50GiB
		Operator: op_name="jit(init_stack)/jit(main)/vmap(jit(_normal))/jit(_normal_real)/jit(_uniform)/threefry2x32" source_file="/sailhome/dlwh/vmap_rand_oom.py" source_line=19
		XLA Label: custom-call
		Shape: u32[48,14336,4096]
		==========================

	Buffer 2:
		Size: 10.50GiB
		Operator: op_name="jit(init_stack)/jit(main)/vmap(jit(_normal))/jit(_normal_real)/jit(_uniform)/threefry2x32" source_file="/sailhome/dlwh/vmap_rand_oom.py" source_line=19
		XLA Label: custom-call
		Shape: u32[48,14336,4096]
		==========================

	Buffer 3:
		Size: 10.50GiB
		Operator: op_name="jit(init_stack)/jit(main)/vmap(jit(_normal))/jit(_normal_real)/jit(_uniform)/threefry2x32" source_file="/sailhome/dlwh/vmap_rand_oom.py" source_line=19
		XLA Label: fusion
		Shape: u32[48,14336,4096]
		==========================
```

cc @froystig @mattjj

### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.4.23
jaxlib: 0.4.23
numpy:  1.26.2
python: 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0]
jax.devices (4 total, 4 local): [cuda(id=0) cuda(id=1) cuda(id=2) cuda(id=3)]
process_count: 1

$ nvidia-smi
Mon Feb 19 09:37:48 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA RTX A6000               On  | 00000000:01:00.0 Off |                  Off |
| 30%   32C    P2              24W / 300W |    270MiB / 49140MiB |      1%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA RTX A6000               On  | 00000000:25:00.0 Off |                  Off |
| 30%   31C    P2              20W / 300W |    270MiB / 49140MiB |      2%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA RTX A6000               On  | 00000000:41:00.0 Off |                  Off |
| 30%   33C    P2              31W / 300W |    270MiB / 49140MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA RTX A6000               On  | 00000000:61:00.0 Off |                  Off |
| 30%   30C    P2              30W / 300W |    270MiB / 49140MiB |      2%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A   1699541      C   python                                      262MiB |
|    1   N/A  N/A   1699541      C   python                                      262MiB |
|    2   N/A  N/A   1699541      C   python                                      262MiB |
|    3   N/A  N/A   1699541      C   python                                      262MiB |
+---------------------------------------------------------------------------------------+
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

A large random tensor is generated under jit with a named output sharding and jax_threefry_partitionable enabled.

## Observed implementation decision

The compiled random-generation custom call is reported as producing full-size peak buffers rather than the expected partitioned storage.

## Physical effect

Each participating device is associated with a 10.5 GiB intermediate for the global random tensor.

## Observed consequence

Initialization exhausts GPU memory while a list-comprehension formulation avoids the failure.

## Reported proposal or workaround

A later user reports that the reproducer finishes on JAX 0.4.28 with four T4 GPUs.

## Repair trace

A bounded review of the 0.4.27-to-0.4.28 history did not identify a random-placement repair tied to the report, and the issue remains open without maintainer confirmation or a fixed regression. The version observation alone cannot identify the invalid source decision or repair delta.

## Archived evidence

- Raw report: `data/raw/github/jax/19879/report.json`
- SHA-256: `0e300bdfc368461ae965c103f83d8ddc4b763cb62b59512916ddfcfb3374ecf6`
