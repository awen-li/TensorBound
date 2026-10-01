# `make_tensor_descriptor` + `num_stages >= 3` produces `CUDA_ERROR_MISALIGNED_ADDRESS`

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/10474](https://github.com/triton-lang/triton/issues/10474)
- Opened: 2026-06-04T15:15:45Z
- Closed: 2026-07-17T08:07:59Z
- Labels: `bug`

## Reported behavior

### Describe the bug

## Summary

`tl.make_tensor_descriptor(...).load([offset])` inside a `tl.range` loop crashes at runtime with `misaligned address` when `num_stages >= 3`.

## Repro
I tried 2 variants, both crash with `num_stages=3`
```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import triton
import triton.language as tl

triton.set_allocator(
    lambda size, align, stream: torch.empty(size, dtype=torch.int8, device='cuda')
)


@triton.jit
def kernel_iter_arg(in_ptr, out_ptr, xnumel,
                    XBLOCK: tl.constexpr, RSPLIT_SIZE: tl.constexpr,
                    NUM_STAGES: tl.constexpr):
    """xoffset as a loop-carried variable (iter arg)."""
    xoffset = tl.program_id(0) * RSPLIT_SIZE
    split_size = min(RSPLIT_SIZE, xnumel - xoffset)
    for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
        val = tl.make_tensor_descriptor(in_ptr, shape=[xnumel], strides=[1],
                                        block_shape=[XBLOCK]).load([xoffset])
        tl.make_tensor_descriptor(out_ptr, shape=[xnumel], strides=[1],
                                  block_shape=[XBLOCK]).store([xoffset], val)
        xoffset += XBLOCK


@triton.jit
def kernel_iv_based(in_ptr, out_ptr, xnumel,
                    XBLOCK: tl.constexpr, RSPLIT_SIZE: tl.constexpr,
                    NUM_STAGES: tl.constexpr):
    """xoffset derived directly from the loop induction variable."""
    xoffset_base = tl.program_id(0) * RSPLIT_SIZE
    split_size = min(RSPLIT_SIZE, xnumel - xoffset_base)
    for xoff in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
        xoffset = xoffset_base + xoff
        val = tl.make_tensor_descriptor(in_ptr, shape=[xnumel], strides=[1],
                                        block_shape=[XBLOCK]).load([xoffset])
        tl.make_tensor_descriptor(out_ptr, shape=[xnumel], strides=[1],
                                  block_shape=[XBLOCK]).store([xoffset], val)


M = 1024000
grid = (M // 128,)
inp = torch.randn(M, dtype=torch.float32, device='cuda')
out = torch.zeros_like(inp)

# Both crash with num_stages=3:
kernel_iter_arg[grid](inp, out, M, XBLOCK=4, RSPLIT_SIZE=128, NUM_STAGES=3)
# kernel_iv_based[grid](inp, out, M, XBLOCK=4, RSPLIT_SIZE=128, NUM_STAGES=3)

torch.cuda.synchronize()
```


### Environment details

Triton: 3.7.0+git88b227e2
GPU: NVIDIA H100 80GB

## Trigger

A 16-byte tensor-descriptor load is software-pipelined with at least three stages.

## Relevant entities

tensor descriptor; TMA transaction; shared-memory ring buffer; per-stage allocation size; stage destination address

## Implemented relation

Each pipelined TMA transaction writes to the shared-memory slice assigned to its stage.

## Pre-state

The base shared-memory allocation is aligned, but each logical stage occupies only 16 bytes.

## Binding transition

The pipeliner converts the descriptor load to TMA and indexes consecutive ring-buffer stages by the logical stage size.

## Post-state

The second stage begins 16 bytes after the aligned base and therefore violates TMA's 128-byte shared-memory address requirement.

## Validity obligation

Every stage destination selected for a TMA transaction satisfies the required shared-memory alignment.

## Invalid decision

Descriptor-to-TMA selection checked the operation form but not whether the per-stage block size preserves alignment across the ring buffer.

## Physical effect

A TMA transaction targets a misaligned shared-memory address.

## Observed consequence

Both reported kernels fail at runtime with CUDA_ERROR_MISALIGNED_ADDRESS when num_stages is three.

## Repair site

Triton descriptor-to-pointer rewriting and TritonGPU pipelining selection

## Repair

PR 10898 checks the descriptor load block size against 128-byte alignment, excludes unaligned loads from TMA pipelining, lowers them through the pointer path, and adds runtime and IR regressions.

## Why the repair works

The failing load no longer creates TMA transactions for ring-buffer addresses whose stage stride cannot preserve alignment.

## Causal basis

The reported failure begins only with multistaging, and the patch's alignment gate plus end-to-end test directly distinguish the unsafe TMA path from the repaired non-TMA path.

## Analysis implication

A staged-transfer check needs base alignment, per-stage physical size, ring-buffer index, transaction kind, and required address alignment.

## Fix patches

- [`fixes/triton-10474-pr10898.patch`](fixes/triton-10474-pr10898.patch) — https://github.com/triton-lang/triton/pull/10898
