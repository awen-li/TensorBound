# Probabilistic illegal memory access from a kernel with multiple tl.dot on H100 (sm_90)

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/10486](https://github.com/triton-lang/triton/issues/10486)
- Opened: 2026-06-04T21:45:20Z
- Closed: 2026-08-19T12:21:39Z
- Labels: `bug`, `needs reproducer`

## Reported behavior

### Describe the bug

A single `@triton.jit` kernel with four `tl.dot` calls intermittently triggers `CUDA error: an illegal memory access was encountered` on an H100 (sm_90). Across fresh processes it crashes ~40-60% of the time; within a single process it is all-or-nothing (a process either faults on its first launch or never faults), so each subprocess is one trial and the reproducers launch fresh subprocesses to measure the crash rate.

Below is the minimal kernel. It can be reduced a bit further (e.g. dropping one `tl.dot`), but reducing it flips the crash from probabilistic to ~100% almost-deterministic, so it is kept at this point where it crashes only part of the time.

Second is a copy kernel with byte-identical load/store pointer arithmetic but the four `tl.dot` calls replaced by reductions. It runs 0/20, confirming the illegal access is not coming from the loads/stores.

## Minimal kernel that crashes (~40-60% of fresh processes)

```python
import subprocess
import sys

import torch
import triton
import triton.language as tl

N = 20
BHN, C, D, DV = 1024, 64, 128, 128
GRID = (BHN * (D // 32),)  # 4096


@triton.jit
def chunk_bwd_dqk(do, v, h, dh, k, dq_out, dk_out, C: tl.constexpr):
    D: tl.constexpr = 128
    DV: tl.constexpr = 128
    BLOCK_D: tl.constexpr = 32
    BHN: tl.constexpr = 1024
    bhn = tl.program_id(0) % BHN
    d_tile = tl.program_id(0) // BHN
    c = tl.arange(0, C)
    dv = tl.arange(0, DV)
    d = (d_tile * BLOCK_D + tl.arange(0, BLOCK_D)).to(tl.int32)

    do_t = tl.load(do + bhn * (C * DV) + c[:, None] * DV + dv[None, :])  # do[bhn, c, dv]  -> [C, DV]
    v_t = tl.load(v + bhn * (C * DV) + c[:, None] * DV + dv[None, :])    # v[bhn, c, dv]   -> [C, DV]
    h_t = tl.load(h + bhn * (D * DV) + d[:, None] * DV + dv[None, :])    # h[bhn, d, dv]   -> [BLOCK_D, DV]
    dh_t = tl.load(dh + bhn * (D * DV) + d[:, None] * DV + dv[None, :])  # dh[bhn, d, dv]  -> [BLOCK_D, DV]

    dA_raw = tl.dot(do_t, tl.trans(v_t), input_precision="tf32", out_dtype=tl.float32)
    dq_cross = tl.dot(do_t, tl.trans(h_t), input_precision="tf32", out_dtype=tl.float32)
    dk_state = tl.dot(v_t, tl.trans(dh_t), input_precision="tf32", out_dtype=tl.float32)

    causal = tl.cast(c[:, None] >= c[None, :], tl.float32)
    dA = tl.cast(dA_raw * causal, tl.bfloat16)

    kt = tl.load(k + bhn * (C * D) + c[:, None] * D + d[None, :])  # k[bhn, c, d]  -> [C, BLOCK_D]
    dq = tl.dot(dA, kt, input_precision="tf32", out_dtype=tl.float32) + dq_cross

    tl.store(dq_out + bhn * (C * D) + c[:, None] * D + d[None, :], dq)        # dq_out[bhn, c, d]
    tl.store(dk_out + bhn * (C * D) + c[:, None] * D + d[None, :], dk_state)  # dk_out[bhn, c, d]


def run():
    g = lambda *s: torch.randn(*s, dtype=torch.bfloat16, device="cuda")
    k, v = g(BHN, C, D), g(BHN, C, DV)
    do = g(BHN, C, DV)
    h, dh = g(BHN, D, DV), g(BHN, D, DV)
    dq_out = torch.empty(BHN, C, D, dtype=torch.float32, device="cuda")
    dk_out = torch.empty(BHN, C, D, dtype=torch.float32, device="cuda")
    chunk_bwd_dqk[GRID](do, v, h, dh, k, dq_out, dk_out, C, num_warps=4, num_stages=2)
    torch.cuda.synchronize()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "CHILD":
        torch.manual_seed(0)
        run()
    else:
        crashed = 0
        for _ in range(N):
            crashed += subprocess.run([sys.executable, __file__, "CHILD"]).returncode != 0
        print(f"{crashed}/{N} crashed")
```

Observed:

```
$ python repro.py
12/20 crashed     # varies run to run, ~40-60%
```

A single crashing process prints:

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

## Same loads/stores, no matmul -> never crashes (0/20)

Byte-identical pointer arithmetic; the four `tl.dot` calls are replaced by reductions that keep every input live and produce the same `[C, BLOCK_D]` output.

```python
import subprocess
import sys

import torch
import triton
import triton.language as tl

N = 20
BHN, C, D, DV = 1024, 64, 128, 128
GRID = (BHN * (D // 32),)  # 4096


@triton.jit
def chunk_bwd_dqk_copy(do, v, h, dh, k, dq_out, dk_out, C: tl.constexpr):
    D: tl.constexpr = 128
    DV: tl.constexpr = 128
    BLOCK_D: tl.constexpr = 32
    BHN: tl.constexpr = 1024
    bhn = tl.program_id(0) % BHN
    d_tile = tl.program_id(0) // BHN
    c = tl.arange(0, C)
    dv = tl.arange(0, DV)
    d = (d_tile * BLOCK_D + tl.arange(0, BLOCK_D)).to(tl.int32)

    # Identical load pointers to the minimal kernel.
    do_t = tl.load(do + bhn * (C * DV) + c[:, None] * DV + dv[None, :])
    v_t = tl.load(v + bhn * (C * DV) + c[:, None] * DV + dv[None, :])
    h_t = tl.load(h + bhn * (D * DV) + d[:, None] * DV + dv[None, :])
    dh_t = tl.load(dh + bhn * (D * DV) + d[:, None] * DV + dv[None, :])
    kt = tl.load(k + bhn * (C * D) + c[:, None] * D + d[None, :])

    # No matmul: fold every input into the output shape so nothing is DCE'd.
    row = tl.sum(tl.cast(do_t, tl.float32), axis=1) + tl.sum(tl.cast(v_t, tl.float32), axis=1)
    col = tl.sum(tl.cast(h_t, tl.float32), axis=1) + tl.sum(tl.cast(dh_t, tl.float32), axis=1)
    out = tl.cast(kt, tl.float32) + row[:, None] + col[None, :]

    # Identical store pointers to the minimal kernel.
    tl.store(dq_out + bhn * (C * D) + c[:, None] * D + d[None, :], out)
    tl.store(dk_out + bhn * (C * D) + c[:, None] * D + d[None, :], out)


def run():
    g = lambda *s: torch.randn(*s, dtype=torch.bfloat16, device="cuda")
    k, v = g(BHN, C, D), g(BHN, C, DV)
    do = g(BHN, C, DV)
    h, dh = g(BHN, D, DV), g(BHN, D, DV)
    dq_out = torch.empty(BHN, C, D, dtype=torch.float32, device="cuda")
    dk_out = torch.empty(BHN, C, D, dtype=torch.float32, device="cuda")
    chunk_bwd_dqk_copy[GRID](do, v, h, dh, k, dq_out, dk_out, C, num_warps=4, num_stages=2)
    torch.cuda.synchronize()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "CHILD":
        torch.manual_seed(0)
        run()
    else:
        crashed = 0
        for _ in range(N):
            crashed += subprocess.run([sys.executable, __file__, "CHILD"]).returncode != 0
        print(f"{crashed}/{N} crashed")
```

Observed:

```
$ python repro_copy.py
0/20 crashed
```

## compute-sanitizer racecheck on a crashing run of the minimal kernel

`$ compute-sanitizer --tool racecheck python repro.py CHILD`

```
========= COMPUTE-SANITIZER
========= Error: Race reported between Write access at chunk_bwd_dqk+0x5c0 in repro.py:25
=========     and Read access at chunk_bwd_dqk+0x1c50 in repro.py:41 [5632 hazards]
=========     and Read access at chunk_bwd_dqk+0x1c80 in repro.py:41 [5456 hazards]
=========
========= Error: Race reported between Write access at chunk_bwd_dqk+0x5d0 in repro.py:25
=========     and Read access at chunk_bwd_dqk+0x1c50 in repro.py:41 [5368 hazards]
=========     and Read access at chunk_bwd_dqk+0x1c80 in repro.py:41 [5632 hazards]
=========
========= Error: Race reported between Write access at chunk_bwd_dqk+0x5e0 in repro.py:25
=========     and Read access at chunk_bwd_dqk+0x1c60 in repro.py:41 [5632 hazards]
=========     and Read access at chunk_bwd_dqk+0x1c70 in repro.py:41 [5368 hazards]
=========
========= Error: Race reported between Write access at chunk_bwd_dqk+0x630 in repro.py:25
=========     and Read access at chunk_bwd_dqk+0x1c60 in repro.py:41 [5368 hazards]
=========     and Read access at chunk_bwd_dqk+0x1c70 in repro.py:41 [5632 hazards]
=========
========= Error: Race reported between Read access at chunk_bwd_dqk+0x1c90 in repro.py:41
=========     and Write access at chunk_bwd_dqk+0x5f0 in repro.py:25 [4864 hazards]
=========     and Write access at chunk_bwd_dqk+0x610 in repro.py:25 [4800 hazards]
=========
========= Error: Race reported between Read access at chunk_bwd_dqk+0x1ca0 in repro.py:41
=========     and Write access at chunk_bwd_dqk+0x600 in repro.py:25 [4656 hazards]
=========     and Write access at chunk_bwd_dqk+0x620 in repro.py:25 [4736 hazards]
=========
========= Error: Race reported between Read access at chunk_bwd_dqk+0x1cb0 in repro.py:41
=========     and Write access at chunk_bwd_dqk+0x600 in repro.py:25 [3840 hazards]
=========     and Write access at chunk_bwd_dqk+0x620 in repro.py:25 [3784 hazards]
=========
========= Error: Race reported between Read access at chunk_bwd_dqk+0x1cc0 in repro.py:41
=========     and Write access at chunk_bwd_dqk+0x5f0 in repro.py:25 [3384 hazards]
=========     and Write access at chunk_bwd_dqk+0x610 in repro.py:25 [3456 hazards]
=========
Traceback (most recent call last):
  File "repro.py", line 58, in <module>
    run()
  File "repro.py", line 52, in run
    torch.cuda.synchronize()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
========= Target application returned an error
========= RACECHECK SUMMARY: 8 hazards displayed (8 errors, 0 warnings)
```

## compute-sanitizer memcheck on a crashing run of the minimal kernel

`$ compute-sanitizer --tool memcheck python repro.py CHILD`

```
========= COMPUTE-SANITIZER
========= Program hit cudaErrorIllegalAddress (error 700) due to "an illegal memory access was encountered" on CUDA API call to cudaDeviceSynchronize.
=========     Saved host backtrace up to driver entry point at error
=========         Host Frame: c10::cuda::device_synchronize() in libc10_cuda.so
=========         Host Frame: THCPModule_cudaSynchronize(_object*, _object*) in libtorch_python.so
=========         ...
=========         Host Frame: synchronize in __init__.py:1181
=========         Host Frame: run in repro.py:52
=========         Host Frame: <module> in repro.py:58
=========
========= Program hit cudaErrorIllegalAddress (error 700) due to "an illegal memory access was encountered" on CUDA API call to cudaGetLastError.
=========     Saved host backtrace up to driver entry point at error
=========         Host Frame: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, unsigned int, bool) in libc10_cuda.so
=========         Host Frame: c10::cuda::device_synchronize() in libc10_cuda.so
=========         ...
=========         Host Frame: synchronize in __init__.py:1181
=========         Host Frame: run in repro.py:52
=========         Host Frame: <module> in repro.py:58
=========
Traceback (most recent call last):
  File "repro.py", line 58, in <module>
    run()
  File "repro.py", line 52, in run
    torch.cuda.synchronize()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
========= Target application returned an error
========= ERROR SUMMARY: 2 errors
```


### Environment details

- Triton: 3.7.0
- PyTorch: 2.12.0+cu130 (CUDA 13.0)
- GPU: NVIDIA H100 80GB HBM3 (compute capability sm_90)
- NVIDIA driver: 595.71.05
- CUDA toolkit (nvcc): 12.8
- OS: Linux x86_64

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

Four dot operations execute in one sm_90 kernel across fresh processes with identical input shapes and launch parameters.

## Observed implementation decision

The report isolates the failure to the dot-containing compilation path but does not identify the responsible compiler transformation or generated invalid address.

## Physical effect

Some compiled-process instances perform an illegal GPU memory access although a load/store-identical reduction variant does not.

## Observed consequence

Approximately 40 to 60 percent of fresh H100 processes fail with cudaErrorIllegalAddress.

## Reported proposal or workaround

Maintainers cannot reproduce on Triton main or GB200 and close the report as likely fixed.

## Repair trace

No fixing commit, repaired transformation, before/after code contrast, or issue-specific regression is linked to the non-reproduction on main. The observed recovery therefore cannot support an implemented-relation-to-repair chain.

## Archived evidence

- Raw report: `data/raw/github/triton/10486/report.json`
- SHA-256: `0d702038da027848024397ad4787470d7c079d661738e074a612a558cf0e3ecd`
