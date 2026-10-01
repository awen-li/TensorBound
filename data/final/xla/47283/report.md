# RaggedAllToAll one-shot path crashes with CUDA_ERROR_ILLEGAL_ADDRESS above 32 ranks

- Repository: `openxla/xla`
- Report: [https://github.com/openxla/xla/issues/47283](https://github.com/openxla/xla/issues/47283)
- Opened: 2026-08-15T14:45:52Z
- Closed: Open at collection
- Labels: `bug`, `NVIDIA-GPU`

## Reported behavior

The one-shot ragged-all-to-all crashes when the clique has more than 32 devices. We see this on GB200 NVL72 (16 nodes × 4 GPUs, one 64-device clique, one process per GPU). The first execution fails on many ranks at once with `CUDA_ERROR_ILLEGAL_ADDRESS`. The crash is deterministic (5/5 training runs). The identical program at 16 ranks runs correctly.

## Minimal repro

The repro needs more than 32 GPUs in one NVLink domain. Smaller domains (for example, B200 NVL8) never reach the affected path. Verified on GB200 NVL72 with stock nightly `0.11.1.dev20260811`: 64 devices crash, 16 devices print `ok`.

```python
# One process per GPU (or per node) across a >32-GPU NVLink domain.
#   export XLA_FLAGS='--xla_gpu_ragged_all_to_all_mode=symmetric'
#   python repro.py <coordinator_host:port> <num_processes> <process_id>
# If the one-shot declines (INVALID_ARGUMENT: "fallback to NCCL is not allowed"),
# add --xla_gpu_unsupported_override_fast_interconnect_slice_size=<total GPUs>.
import sys
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, PartitionSpec as P

coord, n_proc, pid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
jax.distributed.initialize(coord, num_processes=n_proc, process_id=pid)
n = jax.device_count()
mesh = Mesh(np.array(jax.devices()), ("x",))

def f(operand, output):
    i = jax.lax.axis_index("x").astype(jnp.int64)
    in_off = jnp.arange(n, dtype=jnp.int64)       # local row j goes to peer j
    sizes = jnp.ones(n, dtype=jnp.int64)
    out_off = jnp.full((n,), i, dtype=jnp.int64)  # receiver row = sender index
    recv = jnp.ones(n, dtype=jnp.int64)
    return jax.lax.ragged_all_to_all(operand, output, in_off, sizes, out_off, recv, axis_name="x")

operand = jnp.arange(n * n * 4, dtype=jnp.bfloat16).reshape(n * n, 4)
output = jnp.zeros((n * n, 4), dtype=jnp.bfloat16)
result = jax.jit(jax.shard_map(f, mesh=mesh, in_specs=(P("x"), P("x")), out_specs=P("x")))(operand, output)
jax.block_until_ready(result)
print(f"process {jax.process_index()}: ok ({n} devices)", flush=True)
```

## Root cause hypothesis

The one-shot barrier supports at most [`kMaxPeers = 32`](https://github.com/openxla/xla/blob/3a767ceae167531faac975b92aa6d1df1a7bcedd/xla/stream_executor/gpu/multi_gpu_barrier_kernel.h#L52) peers ([both kernel variants](https://github.com/openxla/xla/blob/3a767ceae167531faac975b92aa6d1df1a7bcedd/xla/stream_executor/gpu/multi_gpu_barrier_kernel.h#L69)). [`LaunchMultiGpuBarrier`](https://github.com/openxla/xla/blob/3a767ceae167531faac975b92aa6d1df1a7bcedd/xla/backends/gpu/runtime/collective_kernel_api.cc#L88) rejects larger cliques with a [`TF_RET_CHECK`](https://github.com/openxla/xla/blob/3a767ceae167531faac975b92aa6d1df1a7bcedd/xla/backends/gpu/runtime/collective_kernel_api.cc#L94-L95). The NCCL-window variant [`LaunchMultiGpuBarrierWithNccl`](https://github.com/openxla/xla/blob/3a767ceae167531faac975b92aa6d1df1a7bcedd/xla/backends/gpu/runtime/collective_kernel_api.cc#L120) has no such check. It launches with a [32-slot signal layout](https://github.com/openxla/xla/blob/3a767ceae167531faac975b92aa6d1df1a7bcedd/xla/backends/gpu/runtime/collective_kernel_api.cc#L137) and passes `num_devices = 64` to the device kernel. Ranks then index signal slots past the 32-slot region of the symmetric window. The result is device memory corruption instead of a clean error.

The [thunk selects this path](https://github.com/openxla/xla/blob/3a767ceae167531faac975b92aa6d1df1a7bcedd/xla/backends/gpu/runtime/ragged_all_to_all_thunk.cc#L1008-L1015) whenever `lsa_size == num_devices`. That condition is always true for a rack-wide NVL72 clique.

## Fix validation

We rebuilt `jax-cuda13-pjrt` at the dev20260811 pins with one change: `kMaxPeers = 32 → 128`. The kernel is fully parameterized on the constant. The crash at 64 ranks is gone. A 25-step expert-parallel MoE training benchmark (384 experts, top-8 routing) completes 25/25 steps. At 16 ranks, the patched wheel matches stock throughput (22.48% vs 22.46% MFU).

## Environment

- GB200 NVL72, 16 nodes × 4 GPUs, one process per GPU, CUDA 13
- jax/jaxlib/plugin/pjrt `0.11.1.dev20260811` (stock nightly), NCCL 2.30.7
- Crash analyzed at jax `fe0f1193` / xla `3a767cea`; fix validated on a rebuild at jax `8d1be7d7` / xla `60f8069e`
- Related: [#47146](https://github.com/openxla/xla/issues/47146) (ragged-all-to-all device-kernel peer-access gate on the same topology)

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

RaggedAllToAll one-shot path crashes with CUDA_ERROR_ILLEGAL_ADDRESS above 32 ranks

## Observed implementation decision

The NCCL-window launch path passes num_devices without enforcing kMaxPeers, and the ragged-all-to-all thunk selects this path whenever local-slice size equals the clique size even when the clique exceeds the layout bound.

## Physical effect

A 64-rank barrier kernel indexes peer signal slots beyond a symmetric-memory region whose layout reserves only 32 slots.

## Observed consequence

Ranks write outside the signal region and deterministically fail with CUDA illegal-address errors on a 64-device NVL72 clique.

## Reported proposal or workaround

The same workload succeeds at 16 ranks; increasing kMaxPeers from 32 to 128 removes the 64-rank crash for the reproducer and 25 training steps while retaining measured 16-rank throughput.

## Repair trace

Increasing kMaxPeers removes the crash in the reporter's local experiment, but no merged bounds/selection repair or independently fixed repository revision is available.

## Archived evidence

- Raw report: `data/raw/github/xla/47283/report.json`
- SHA-256: `630c3fd48e000dffe14726a9b58e3428b1cd461925a1143b0a28d4a858363ea6`
