# `tl.atomic_add` is not correct across threads

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/7402](https://github.com/triton-lang/triton/issues/7402)
- Opened: 2025-07-07T10:05:27Z
- Closed: 2025-07-25T01:38:55Z
- Labels: `bug`

## Reported behavior

### Describe the bug

Example repro: it looks like only thread 0 gets the correct result of the `atomic_add`, while other threads just get a value of 0.


```python
import torch
import triton
import triton.language as tl

@triton.jit
def my_kernel(index_ptr, out_ptr):
    # `write_index` is [1]
    write_index = tl.atomic_add(index_ptr + tl.arange(0, 1), val=1, sem="relaxed")
    # should set `out[1, :]=2`
    tl.store(
        out_ptr
        + write_index[:, None] * 8
        + tl.arange(0, 8)[None, :],
        tl.full([1, 8], 2, dtype=tl.int64)
    )

index = torch.zeros([1], device="cuda", dtype=torch.int32)
index[0] = 1
out = torch.zeros([2, 8], device="cuda", dtype=torch.int64)
my_kernel[(1,)](index, out)
print(out[:, :8])
"""
Result:
tensor([[0, 2, 2, 2, 2, 2, 2, 2],
        [2, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')

Expected:
tensor([[0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 2, 2, 2, 2, 2, 2]], device='cuda:0')
"""
```

### Environment details

Triton: 3.3.1
GPU: H100

## Trigger

A tensor tl.atomic_add has one active element, and its returned index is broadcast and consumed by stores from a wider thread set.

## Relevant entities

atomic RMW participants; returned pre-update value; nonparticipating threads; broadcast consumers; shared scratch storage

## Implemented relation

Every consumer of a broadcasted tensor atomic result must receive the value produced by a participating atomic lane.

## Pre-state

Only the participating lane executes the atomic operation and owns a defined returned value.

## Binding transition

Lowering broadcasts the tensor result to threads that did not participate in the atomic operation.

## Post-state

Nonparticipating consumers retain undefined or zero values instead of the participating lane's returned index.

## Validity obligation

When a tensor atomic result is consumed under a wider layout, lowering supplies each consumer with the corresponding participating result.

## Invalid decision

Atomic lowering returned per-thread values without materializing the cross-thread distribution required by the result layout.

## Physical effect

Store addresses computed by nonparticipating threads use zero or undefined atomic-return values.

## Observed consequence

Only one element is written at the expected output row while the remaining stores target the wrong row.

## Repair site

TritonGPU allocation analysis and NVIDIA/AMD LoadStoreOpToLLVM atomic-result lowering

## Repair

PR 7460 allocates scratch storage when a tensor atomic result requires broadcasting, stores participating results, synchronizes, reloads them for consumers, and adds used-result and unused-result tests.

## Why the repair works

Consumers outside the atomic participant set now read the participating result from synchronized shared storage, while unused results avoid the extra transfer.

## Causal basis

The maintainer identifies undefined values in nonparticipating threads, and the merged patch implements and tests the missing distribution only when the returned value is used.

## Analysis implication

A tensor-atomic check needs the participant mask, result layout, consumer thread set, scratch transfer, and synchronization around result redistribution.

## Fix patches

- [`fixes/triton-7402-pr7460.patch`](fixes/triton-7402-pr7460.patch) — https://github.com/triton-lang/triton/pull/7460
