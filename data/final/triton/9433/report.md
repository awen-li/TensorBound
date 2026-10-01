# RAW hazard with pipelined wgmma and LHS in register

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/9433](https://github.com/triton-lang/triton/issues/9433)
- Opened: 2026-02-11T15:53:40Z
- Closed: 2026-05-29T01:06:30Z
- Labels: `bug`

## Reported behavior

We hit a RAW hazard when the `wgmma.mma_async` LHS operand is passed in registers.

From the [`wgmma.wait_group`](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group) documentation:

> Accessing the accumulator register or the input register containing the fragments of matrix A of a `wgmma.mma_async` instruction without first performing a `wgmma.wait_group` instruction that waits on a wgmma-group including that `wgmma.mma_async` instruction is undefined behavior.

Triton [inserts](https://github.com/triton-lang/triton/blob/088fbe15e4868506ac0ad42a3cfbbc923971b8d4/lib/Dialect/TritonGPU/Transforms/Pipeliner/WGMMAPipeline.cpp#L666) a `warp_group_dot_wait` for this, which lowers to `wgmma.wait_group`. The `pendings` value (explained in [this comment](https://github.com/triton-lang/triton/blob/088fbe15e4868506ac0ad42a3cfbbc923971b8d4/lib/Dialect/TritonGPU/Transforms/Pipeliner/WGMMAPipeline.cpp#L657-L660)) however does not prevent the RAW hazard: `properlyAsyncDots.size() - 1` only guarantees that the first `wgmma.mma_async` has finished during the next loop iteration. As a result, dot prologue instructions may write to the input registers of the other `wgmma.mma_async` instructions from the previous iteration before they have executed.

We have only seen the issue manifest itself with an 'empty' prologue, where the results of a `ldmatrix` instruction are passed directly to `wgmma.mma_async`. If the `ldmatrix` results are first copied to another register (for example through negating the input), the issue seems to disappear. Passing the LHS input in register directly from `ldmatrix` requires a bit of a dance (see repro below), but we do see the same issue in real workloads where the `if` statement is the result of a concat.

Changing `properlyAsyncDots.size() - 1` to `0` fixes the issue, but has significant performance impact, because the `wgmma.wait_group` follows right after the `wgmma.mma_async`, instead of before the input register is written.

```
import sys
import torch
import triton
import triton.language as tl


@triton.jit
def wgmma_register_sync_repro_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
  pid = tl.program_id(axis=0)

  offs_m = tl.arange(0, BLOCK_M)
  offs_n = tl.arange(0, BLOCK_N)
  offs_k = tl.arange(0, BLOCK_K)

  a_ptrs = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
  b_ptrs = b_ptr + (offs_k[:, None] * 1 + offs_n[None, :] * K)

  acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

  for k in range(0, K, BLOCK_K):
    # Force A to be passed in register directly from ldmatrix.
    if pid >= 0:
      a = tl.load(a_ptrs)
    else:
      a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    # This produces the following PTX code:
    # 4x ldmatrix.sync.aligned.m8n8.x4.shared.b16 {<A registers>}, [...];
    # 4x {
    #   wgmma.fence.sync.aligned;
    #   wgmma.mma_async.sync.aligned {<ACC registers>}, {<A registers>}, ...;
    #   wgmma.commit_group.sync.aligned;
    #   wgmma.wait_group.sync.aligned 3;
    # }
    # RAW hazard: the ldmatrix instructions write <A registers> before the last
    # 3 wgmma.mma_sync instructions have read them.
    acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    a_ptrs += BLOCK_K
    b_ptrs += BLOCK_K

  c_ptrs = c_ptr + (pid * M * N) + (offs_m[:, None] * N + offs_n[None, :])
  tl.store(c_ptrs, acc)


def test_repro():
  torch.manual_seed(0)

  M = 64
  N = 128
  K = 4096
  BLOCK_M = 64
  BLOCK_N = 128
  BLOCK_K = 64

  NUM_BLOCKS = 512

  a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
  b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).t()
  c = torch.empty((NUM_BLOCKS, M, N), device="cuda", dtype=torch.float32)

  print(
      f"Running on {torch.cuda.get_device_name(0)} with {NUM_BLOCKS} blocks..."
  )

  wgmma_register_sync_repro_kernel[(NUM_BLOCKS,)](
      a, b, c, M=M, N=N, K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
  )

  torch.cuda.synchronize()

  target = torch.matmul(a.float(), b.float())
  diff = torch.abs(c - target.unsqueeze(0))
  max_diff = torch.max(diff)

  print(f"Max Abs Error: {max_diff.item()}")

  if max_diff.item() > 1.0:
    print("FAILED: Output corrupted (Race Condition triggered).")
    sys.exit(1)
  else:
    print("PASSED: Output matches.")


if __name__ == "__main__":
  test_repro()
```

### Environment details

Triton version: 84214b6
GPU: H100

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

RAW hazard with pipelined wgmma and LHS in register

## Observed implementation decision

The pipeline emits wait_group N-1 while reusing A registers and relies on register moves/liveness scheduling that older ptxas versions may remove or place before the dependency barrier.

## Physical effect

Pipelined WGMMA allows the next iteration's ldmatrix or register move to overwrite register-resident A fragments before all prior asynchronous matrix operations have consumed them.

## Observed consequence

The H100 kernel races fragment production with asynchronous consumption and produces large numerical corruption.

## Reported proposal or workaround

Triton PRs #9514/#9530 materialize moves in the correct position; CUDA 13.3 ptxas then staggers them around DEPBAR, and the reporter confirms the reproducer is fixed.

## Repair trace

PRs 9514 and 9530 make the Triton PTX relation explicit and place moves before the fence, yet the report still observes a SASS hazard with ptxas 13.1 and attributes final success to ptxas 13.3. The ptxas repair is not archived, so the combined repair chain remains incomplete.

## Archived evidence

- Raw report: `data/raw/github/triton/9433/report.json`
- SHA-256: `bda6cf1737c294689562899673bfb2d7415408a90c115d974c064c12c2f091ea`
