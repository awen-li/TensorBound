# [Bug] Two `tl.dot` chained through one accumulator in a K-loop compute wrong results on sm_103 (B300)

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/10821](https://github.com/triton-lang/triton/issues/10821)
- Opened: 2026-07-08T22:24:11Z
- Closed: 2026-07-17T21:14:51Z
- Labels: `bug`

## Reported behavior

### Describe the bug

On NVIDIA B300 (sm_103), a K-loop that issues **two `tl.dot` calls chained through a single accumulator** (`acc = tl.dot(a1, b1, acc); acc = tl.dot(a2, b2, acc)`) produces wrong results for tile sizes that take the tcgen05 MMA path (BLOCK_M ≥ 64). Max abs error is ~50–70 on values of magnitude ~25 (reference: `A1 @ B1 + A2 @ B2` in fp32 torch). The wrong results are **deterministic** — bit-identical across runs — so this is a miscompile, not a race (distinct from #9871, whose fix is present in the tested builds).

Controls that are all correct: the same loop with a single `tl.dot`; the same two dots into **two separate accumulators** (summed after the loop); and `BLOCK_M = 16` (non-tcgen05 lowering). Both fp32 and bf16 inputs are affected.

Reproduces on **triton 3.6.0, 3.7.1, and current `main` (3.8.0, be7e23cc653162903be98fc24ae8944cbe42810a)**, each in a clean venv.

## Reproducer

```python
import torch
import triton
import triton.language as tl


@triton.jit
def two_dots_one_acc(a1, a2, b1, b2, c, M, N, K,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for kk in tl.range(0, K, BK):
        k = kk + tl.arange(0, BK)
        acc = tl.dot(tl.load(a1 + m[:, None] * K + k[None, :]),
                     tl.load(b1 + k[:, None] * N + n[None, :]), acc=acc)
        acc = tl.dot(tl.load(a2 + m[:, None] * K + k[None, :]),
                     tl.load(b2 + k[:, None] * N + n[None, :]), acc=acc)
    tl.store(c + m[:, None] * N + n[None, :], acc.to(c.dtype.element_ty))


def run(dtype, BM, BN, BK, M=256, N=256, K=256):
    torch.manual_seed(0)
    a1 = torch.randn(M, K, device="cuda", dtype=dtype)
    a2 = torch.randn(M, K, device="cuda", dtype=dtype)
    b1 = torch.randn(K, N, device="cuda", dtype=dtype)
    b2 = torch.randn(K, N, device="cuda", dtype=dtype)
    ref = a1.float() @ b1.float() + a2.float() @ b2.float()
    c = torch.empty(M, N, device="cuda", dtype=dtype)
    two_dots_one_acc[(M // BM, N // BN)](a1, a2, b1, b2, c, M, N, K, BM, BN, BK)
    torch.cuda.synchronize()
    return (c.float() - ref).abs().max().item()  # rounding noise < 0.5; bug ~50-70


for dtype in (torch.float32, torch.bfloat16):
    for BM, BN, BK in [(128, 64, 32), (64, 64, 64), (16, 64, 32)]:
        print(dtype, f"BM={BM} BN={BN} BK={BK}", "max_err =", run(dtype, BM, BN, BK))
```

## Output (triton main @ be7e23c, B300)

```
torch.float32 BM=128 BN=64 BK=32 max_err = 64.53   <- WRONG
torch.float32 BM=64  BN=64 BK=64 max_err = 53.57   <- WRONG
torch.float32 BM=16  BN=64 BK=32 max_err = 0.08    (ok, non-tcgen05 tile)
torch.bfloat16 BM=128 BN=64 BK=32 max_err = 57.23  <- WRONG
torch.bfloat16 BM=64  BN=64 BK=64 max_err = 61.79  <- WRONG
torch.bfloat16 BM=16  BN=64 BK=32 max_err = 0.25   (ok)
```

Replacing the shared accumulator with two accumulators summed after the loop, or merging the two dots into one GEMM, gives correct results at all tile sizes (max_err ≤ 0.25).

@atalman @lezcano @ThomasRaoux

### Environment details

## Environment

| | |
|---|---|
| GPU | NVIDIA B300 SXM6 (sm_103, compute capability (10, 3)) |
| Driver / CUDA | 595.71.05 / 13.2 |
| torch | 2.12.1+cu130 |
| triton | 3.6.0, 3.7.1, and main @ be7e23cc653162903be98fc24ae8944cbe42810a — all reproduce |
| OS / Python | Ubuntu (linux 6.17), Python 3.12 |

sm_100 (B200) not tested; H100 (sm_90) is unaffected by the same code.

## Trigger

Two tcgen05 MMA operations update one logical accumulator in each K-loop iteration.

## Relevant entities

first MMA; second MMA; loop-carried accumulator; temporary tensor-memory allocations; pipeline stages

## Implemented relation

MMA2 consumes MMA1's result in the same iteration, and MMA1 of the next iteration consumes the accumulator after MMA2 completes.

## Pre-state

The program expresses a strict cyclic chain MMA1(i) to MMA2(i) to MMA1(i+1).

## Binding transition

Acceleration creates separate accumulator allocations and latency assignment places the two MMAs in different pipeline stages.

## Post-state

Pipeline expansion can schedule MMA1(i+1) before MMA2(i) has updated the loop-carried accumulator.

## Validity obligation

Pipeline staging preserves every same-iteration and loop-carried accumulator dependency between chained MMA operations.

## Invalid decision

AssignLatencies treated the chained MMAs as overlap candidates without detecting the accumulator's transfer cycle.

## Physical effect

The next iteration reads an accumulator state that omits the second MMA from the preceding iteration.

## Observed consequence

Blackwell kernels show deterministic maximum absolute errors around 50 to 70, while single-MMA and separate-accumulator controls are correct.

## Repair site

AccelerateMatmul, OptimizeAccumulatorInit, and Pipeliner/AssignLatencies

## Repair

PR 10858 forwards chained MMAs to one accumulator where valid and detects loop-carried accumulator cycles so their MMA operations receive no movable latency; tests cover shared accumulators and cyclic transfers.

## Why the repair works

The pipeline no longer moves a dependent MMA across the operation that produces its logical accumulator state.

## Causal basis

The issue's controls isolate a shared accumulator on tcgen05, and the merged patch explicitly reconstructs that identity and blocks stage movement when its dependency graph crosses the loop boundary.

## Analysis implication

A pipeline check needs accumulator identity across temporary allocations, transfer edges, same-iteration order, loop-carried cycles, and assigned stages.

## Fix patches

- [`fixes/triton-10821-pr10858.patch`](fixes/triton-10821-pr10858.patch) — https://github.com/triton-lang/triton/pull/10858
