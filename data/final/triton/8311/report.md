# Incorrect results from `warp_specialize=True`

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/8311](https://github.com/triton-lang/triton/issues/8311)
- Opened: 2025-09-28T04:28:15Z
- Closed: 2025-12-11T01:30:35Z
- Labels: `bug`

## Reported behavior

### Describe the bug

The following kernel gives incorrect output when I set `WARP_SPECIALIZE=True`:
```python
import torch
import triton
import triton.language as tl


def _alloc_fn(size: int, alignment: int, stream: int | None) -> torch.Tensor:
    return torch.empty(size, device='cuda', dtype=torch.int8)

triton.set_allocator(_alloc_fn)

@triton.jit
def bad_kernel(x, y, out, SIZE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, WARP_SPECIALIZE: tl.constexpr):
    desc = tl.make_tensor_descriptor(y, [SIZE, SIZE], [SIZE, 1], [BLOCK_K, BLOCK_N])
    for mo in tl.range(0, SIZE, BLOCK_M):
        for no in tl.range(0, SIZE, BLOCK_N):
            rows = mo + tl.arange(0, BLOCK_M)
            cols = no + tl.arange(0, BLOCK_N)
            acc = tl.full([BLOCK_M, BLOCK_N], 0.0, tl.float32)
            for ko in tl.range(0, SIZE, BLOCK_K, warp_specialize=WARP_SPECIALIZE):
                ks = ko + tl.arange(0, BLOCK_K)
                x_tile = tl.load(x + (rows[:, None] * SIZE + ks[None, :]))
                y_tile = desc.load([ko, no])
                acc = tl.dot(tl.cast(x_tile, tl.float16), tl.cast(y_tile, tl.float16), acc=acc, input_precision='tf32', out_dtype=tl.float32)
            tl.store(out + (rows[:, None] * SIZE + cols[None, :]), tl.cast(acc, tl.float16))


def main():
    SIZE = 1024
    BLOCK_M = 2
    BLOCK_N = 8
    BLOCK_K = 512
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    x = torch.randn((SIZE, SIZE), dtype=torch.float16, device='cuda')
    y = torch.randn((SIZE, SIZE), dtype=torch.float16, device='cuda')
    out = torch.empty((SIZE, SIZE), dtype=torch.float16, device='cuda')
    ref = torch.matmul(x, y)

    print("Running bad kernel with warp_specialize=False")
    bad_kernel[(1,)](x, y, out, SIZE, BLOCK_M, BLOCK_N, BLOCK_K, False, num_warps=16, num_stages=1)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("Passed!")

    print()
    print("Running bad kernel with warp_specialize=True")
    bad_kernel[(1,)](x, y, out, SIZE, BLOCK_M, BLOCK_N, BLOCK_K, True, num_warps=16, num_stages=1)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("Passed!")


if __name__ == '__main__':
    main()
```

Output:
```
$ python warp_specialize_bug.py
Running bad kernel with warp_specialize=False
Passed!

Running bad kernel with warp_specialize=True
Traceback (most recent call last):
  File "/home/jansel/helion/warp_specialize_bug.py", line 52, in <module>
    main()
  File "/home/jansel/helion/warp_specialize_bug.py", line 47, in main
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
  File "/home/jansel/pytorch/torch/testing/_comparison.py", line 1589, in assert_close
    raise error_metas[0].to_error(msg)
AssertionError: Tensor-likes are not close!

Mismatched elements: 1041126 / 1048576 (99.3%)
Greatest absolute difference: 139.0 at index (673, 708) (up to 0.01 allowed)
Greatest relative difference: inf at index (77, 503) (up to 0.01 allowed)
```

### Environment details

Triton: 8ee584014e9570ba608809c42dc2060fdd214a98
GPU: RTX 5090

## Trigger

The consumer partition loads a shared-memory Y tile and then signals through arrive_barrier that the tile can be reused.

## Relevant entities

TMA producer partition; shared-memory tile; local_load; arrive_barrier; instruction reorder pass; dot consumer

## Implemented relation

A consumer reads the current shared-memory tile before publishing the barrier arrival that permits the producer to refill that storage.

## Pre-state

local_load precedes arrive_barrier, so the consumer has captured the tile before reuse is announced.

## Binding transition

TritonGPUReorderInstructions moves local_load below arrive_barrier because it does not treat the intervening side effect as a reordering boundary.

## Post-state

The producer can refill the shared tile while the consumer is still loading it.

## Validity obligation

A memory read does not cross a write or synchronization side effect that changes when its storage may be modified.

## Invalid decision

The reorder pass moved local_load across a write-side-effecting barrier operation.

## Physical effect

Producer and consumer partitions race on the same shared-memory tile.

## Observed consequence

warp_specialize=True returns incorrect values for about 99.3% of the reported output elements.

## Repair site

lib/Dialect/TritonGPU/Transforms/ReorderInstructions.cpp

## Repair

PR 8423 detects write side effects on the path between a local load and its proposed position and cancels the reorder when such an operation intervenes.

## Why the repair works

local_load remains before arrive_barrier, so the consumer finishes reading before the producer receives permission to overwrite the shared tile.

## Causal basis

The issue shows the exact before/after IR swap and confirms that disabling the pass restores results; the merged patch blocks local-load movement across write side effects and adds that barrier pattern as a regression.

## Analysis implication

Scheduling analysis needs aliasing shared storage, operation side effects, barrier meaning, cross-partition producer-consumer roles, and the proposed instruction movement path.

## Fix patches

- [`fixes/triton-8311-pr8423.patch`](fixes/triton-8311-pr8423.patch) — https://github.com/triton-lang/triton/pull/8423
