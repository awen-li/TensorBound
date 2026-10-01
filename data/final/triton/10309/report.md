# [Blackwell SM100] RuntimeError: misaligned address on tcgen05.mma

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/10309](https://github.com/triton-lang/triton/issues/10309)
- Opened: 2026-05-14T03:15:24Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Describe the bug

I hit a runtime misaligned address error on Blackwell / SM100 when compiling the kernel with Triton/main.

# Reproducer
```Python
import torch
import triton
import triton.language as tl

@triton.jit
def fwd_kernel(q_ptr, k_ptr, y_ptr, BM: tl.constexpr, BN: tl.constexpr, DH: tl.constexpr):
    m = tl.arange(0, BM)
    n = tl.arange(0, BN)
    h = tl.arange(0, DH)

    q = tl.load(q_ptr + m[:, None] * DH + h[None, :], mask=m[:, None] < 1, other=0.0)
    k = tl.load(k_ptr + n[None, :] * DH + h[:, None], mask=n[None, :] < 1, other=0.0)

    score = tl.dot(q, k, allow_tf32=True)
    y = tl.dot(score.to(q_ptr.dtype.element_ty), tl.trans(k), allow_tf32=True)

    tl.store(y_ptr + m[:, None] * DH + h[None, :], y, mask=m[:, None] < 1)

BM, BN, DH = 64, 16, 512

q = torch.ones((1, DH), device="cuda", dtype=torch.bfloat16)
k = torch.ones((1, DH), device="cuda", dtype=torch.bfloat16)
y = torch.empty_like(q)

fwd_kernel[(1,)](q, k, y, BM, BN, DH, num_warps=4, num_stages=2)
torch.cuda.synchronize()
```

# Error Message
RuntimeError: Triton Error [CUDA]: misaligned address

# CUDA GDB
```
[Current focus set to CUDA kernel 0, grid 3, block (0,0,0), thread (1,0,0), device 0, sm 142, warp 2, lane 1]

CUDA Exception: Warp Misaligned Address
The exception was triggered at PC 0x7fc0235be990  fwd_kernel  (minimal_misaligned_address_repro.py:15)


*> 0x00007fc0235be990 <+81808>: UTCHMMA tmem[UR37],gdesc[UR8],tmem[UR4],tmem[UR6],idesc[UR7],UPT
   0x00007fc0235be9a0 <+81824>: NOP
=> 0x00007fc0235be9b0 <+81840>: BSYNC.RECONVERGENT B6

UR37           0x100               256
UR8            0x801240            8393280
UR4            0x100000            1048576
UR6            0x0                 0
UR7            0x4410490           71369872
UPT            0x1                 1
```

# Compute Sanitizer Report
```
========= Misaligned shared or local address
=========     at fwd_kernel+0x13f90 in minimal_misaligned_address_repro.py:15
=========     by thread (1,0,0) in block (0,0,0)
========= Misaligned shared or local address
=========     at fwd_kernel+0x13f90 in minimal_misaligned_address_repro.py:15
=========     by thread (2,0,0) in block (0,0,0)
...
========= Misaligned shared or local address
=========     at fwd_kernel+0x13f90 in minimal_misaligned_address_repro.py:15
=========     by thread (31,0,0) in block (0,0,0)
========= Program hit CUDA_ERROR_MISALIGNED_ADDRESS (error 716) due to "misaligned address" on CUDA API call to cuLaunchKernelEx.
```

### Environment details

Triton: main
GPU: Blackwell / SM100

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises triton SM100 tensor-memory allocation for a two-dot kernel that promotes the left operand to tensor memory.

## Observed implementation decision

The allocation/layout path supplies tcgen05.mma with an address that violates its required alignment.

## Physical effect

UTCHMMA executes with a misaligned tensor-memory operand.

## Observed consequence

The kernel raises CUDA_ERROR_MISALIGNED_ADDRESS.

## Reported proposal or workaround

PR 10320 proposes disallowing the affected split-N 64-row layout; a downstream Helion workaround merged.

## Repair trace

The Triton repair PR was not merged, and the downstream workaround does not correct Triton's allocator, so no landed upstream repair chain exists.

## Archived evidence

- Raw report: `data/raw/github/triton/10309/report.json`
- SHA-256: `c9b94804df9dd1a50ea4741e3d3fc624771dcdbb1e450c6aec2ff37472077edf`
