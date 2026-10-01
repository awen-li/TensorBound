# Silent wrong results on 4 GPUs: multi-output fusion aliases two in-place dynamic-update-slice outputs to the same buffer

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/39100](https://github.com/jax-ml/jax/issues/39100)
- Opened: 2026-07-11T22:01:07Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

A jitted computation on a 1D-sharded array returns silently wrong results on 4 CUDA devices. The same program is correct on CPU (any device count), on 1–3 GPUs, in eager mode, and on 4 GPUs with --xla_disable_hlo_passes=priority-fusion.

I hit this in a ocean model, where it silently corrupted multi-GPU simulations; the reproducer below is distilled to pure jax. Requires exactly 4 (or more?)  CUDA devices (does not reproduce with 2 or 3).

```python
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)  # not required: float32 fails too

N = 4                # devices
R, LO, HI = 2, 1, 2  # interior rows per piece, low/high padding
PER = R + LO + HI    # rows per padded piece (5)


def c2p(x):  # "compact" f64[8] -> "padded" f64[20]
    return jnp.concatenate(
        [jnp.pad(x[R * k:R * (k + 1)], (LO, HI)) for k in range(N)])


def p2c(x):  # "padded" f64[20] -> "compact" f64[8]
    return jnp.concatenate(
        [x[PER * k + LO:PER * k + LO + R] for k in range(N)])


def f(x, z):
    term_a = p2c(c2p(x) * z)      # reassembly chain A
    term_b = p2c(c2p(p2c(z)))     # reassembly chain B
    return term_a + term_b


mesh = Mesh(np.array(jax.devices()[:N]), ("d",))
shd = NamedSharding(mesh, P("d"))
rng = np.random.default_rng(0)
x = jax.device_put(rng.standard_normal(N * R), shd)
z = jax.device_put(rng.standard_normal(N * PER), shd)

ref = np.asarray(f(x, z))            # eager op-by-op reference
got = np.asarray(jax.jit(f)(x, z))   # wrong on 4 GPUs
print("max abs err:", np.max(np.abs(got - ref)))
```

with output

```
JAX_PLATFORMS=cuda python repr
  max abs err: 1.1063197497151054        # first shard's elements wrong

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 python repro.py
  max abs err: 0.0

JAX_PLATFORMS=cuda XLA_FLAGS=-ity-fusion python repro.py
  max abs err: 0.0
```

Below is claude's opinion

```
Root cause (from HLO / buffer-assignment dumps)

The SPMD partitioner lowers each slice -> pad -> concatenate reassembly chain to the usual idiom: each device dynamic-updro-initialized piece buffer(offset = partition-id, select-masked), the pieces are all-reduce-summed, and each device slices its shard back out.

Three steps then interact:

1. CSE shares the zero-initialstruction) between both chains'dynamic-update-slices.
2. priority-fusion (+ multi-ouns into one kLoop fusion whoseroot tuple contains two dynamic-update-slice outputs updating the same parameter:

%fused_dynamic_update_slice_slice (param_0.72: f64[2], ...) -> (f64[2], f64[2], f64[2],
f64[2]) {
  %dynamic-update-slice.4.1       = f64[2]{0} dynamic-update-slice(%param_0.72,
%select.19.3, %convert.18)
  %dynamic-update-slice.9.clone.1 = f64[2]{0} dynamic-update-slice(%param_0.72,
%select.11.3.clone.1, %convert
  ROOT %tuple.6 = (...) tuple(%dynamic-update-slice.4.1, %slice.53.1,
                              .1, %slice.42.1.clone.1.clone.1)
}
3. Buffer assignment applies tice optimization to both tupleoutputs, aliasing them onto the one shared operand buffer — legal for one in-place DUS, not
for two:

value: <377 copy.12 @0>       size=16, offset=0): f64[2]{0}
value: <392 loop_dynamic_update_slice_slice_fusion{0} @0> (size=16, offset=0): f64[2]{0}
value: <394 loop_dynamic_updatsize=16, offset=0): f64[2]{0}

3. Outputs {0} and {2} — chaine — are the same physicalmemory, so two logically different arrays become one. Downstream both chains read "piece 0"
from the same bytes, which corion 0. (In our original program the downstream op was a + (-b) with aliased a == b, producing exactly-zero rows.)

The pattern is fragile, consistent with a buffer-assignment/fusion-shape condition: it
needs exactly 4 devices, the ametric (1, 1) and reversed (2,1) compile correctly), and the two chains related enough to share the CSE'd zero base but
not be CSE-identical.

Not involved (all ruled out byt (NCCL_P2P_DISABLE=1unchanged), async collectives (--xla_gpu_disable_async_collectives=ALLCOLLECTIVES
unchanged), latency-hiding schmbiner, command buffers, Shardyvs GSPMD (JAX_USE_SHARDY_PARTITIONER=false unchanged).
```

### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.10.2
jaxlib: 0.10.2
numpy:  2.5.0
python: 3.12.11 (main, Aug  8 2025, 17:06:48) [Clang 20.1.4 ]
device info: NVIDIA A100-SXM4-80GB-4, 4 local devices"
process_count: 1
platform: uname_result(system='Linux', node='l50042.lvt.dkrz.de', release='4.18.0-553.141.1.el8.x86_64', version='#1 SMP Thu Jul 9 20:00:37 CEST 2026', machine='x86_64')
JAX_PLATFORMS=cuda
XLA_PYTHON_CLIENT_PREALLOCATE=false

$ nvidia-smi
Sat Jul 11 23:59:21 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.159.04             Driver Version: 580.159.04     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                    Disp.A | Volatile Uncorr.ECC |
| Fan  Temp   Perf         mory-Usage | GPU-Util  ComputeM. |
|                                     |               MIGM. |
|=====================================+======================|
|   0  NVIDIA A100-SXM4-80G3:00.0 Off |0 |
| N/A   43C    P0            81920MiB |      1%Default |
|                                     |Disabled |
+-------------------------------------+----------------------+
|   1  NVIDIA A100-SXM4-80G4:00.0 Off |0 |
| N/A   44C    P0            81920MiB |      0%Default |
|                                     |Disabled |
+-------------------------------------+----------------------+
|   2  NVIDIA A100-SXM4-80G4:00.0 Off |0 |
| N/A   44C    P0            81920MiB |      1%Default |
|                                     |Disabled |
+-------------------------------------+----------------------+
|   3  NVIDIA A100-SXM4-80G4:00.0 Off |0 |                                                                                | N/A   44C    P0            81920MiB |      2%Default |                                                                          |                                     |Disabled |                                                                         +-------------------------------------+----------------------+
+------------------------------------------------------------------------------------+
| Processes:                                                                        |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          800461      C   .../fridom-dev/.venv/bin/python3        424MiB |
|    1   N/A  N/A          800461      C   .../fridom-dev/.venv/bin/python3        418MiB |
|    2   N/A  N/A          800461      C   .../fridom-dev/.venv/bin/python3        418MiB |
|    3   N/A  N/A          800461      C   .../fridom-dev/.venv/bin/python3             418MiB |
+-----------------------------------------------------------------------------------------+
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises xLA multi-output fusion of two in-place dynamic-update-slice results on four GPUs.

## Observed implementation decision

Buffer assignment aliases distinct fusion outputs to one physical allocation.

## Physical effect

One update overwrites storage required by the other output.

## Observed consequence

Multi-device execution silently returns wrong values.

## Reported proposal or workaround

The report has no completed repair proposal.

## Repair trace

No responsible buffer-assignment patch or fixed regression is available.

## Archived evidence

- Raw report: `data/raw/github/jax/39100/report.json`
- SHA-256: `995f6889206feb6120b123307368321d4b82bc1a8aa01dfba9c7d3ec42e18b3f`
