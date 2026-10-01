# `tl.device_assert` can observe incorrect results from `tl.associative_scan`

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/2298](https://github.com/triton-lang/triton/issues/2298)
- Opened: 2023-09-14T16:36:20Z
- Closed: 2023-09-21T03:02:17Z
- Labels: `bug`

## Reported behavior

In the example below we have 4 warps/128 threads performing a scan over scanning 32 elements so some values are duplicated between multiple threads. I would expect the `tl.device_print` to print a scan value of 1 through 32, repeated 4 times. Instead I see 1 through 128 printed once each and if I run with `TRITON_DEBUG=1` the assert will fail in thread ids 32 through 127.

Note that the output is actually correct, so I guess these incorrect values are thrown away when broadcasting to the output shape. However, this is still problematic because `cumsum` is often used for indexing calculations and in inductor we assert that indices are within bounds.

```python
import triton
import triton.language as tl
import torch

@triton.jit
def fn(out_ptr, in_ptr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xindex = tl.arange(0, XBLOCK)[:, None]
    rindex = tl.arange(0, RBLOCK)[None, :]
    data = tl.load(in_ptr + rindex)
    scan = tl.cumsum(data, 1)
    expected_max = tl.sum(data, 1)
    tl.device_print("scan, sum = ", scan, expected_max)
    tl.device_assert(scan <= expected_max)
    tl.store(out_ptr + xindex * RBLOCK + rindex, scan)

XBLOCK, RBLOCK = 4, 32
in0 = torch.ones(1, RBLOCK, device="cuda")
out0 = torch.empty((XBLOCK, RBLOCK), device="cuda")
fn[(1,)](out0, in0, XBLOCK, RBLOCK, num_warps=XBLOCK)
```

I have confirmed this fails on the latest master (08c16589573621fcb8cd5a9c3b8a0537077f876d).

## Trigger

A 32-element scan runs with four warps, or a smaller scan dimension repeats data across lanes.

## Relevant entities

logical scan elements; physical lanes; repetitive warps; repetitive threads; scan accumulator

## Implemented relation

Each unique logical element contributes once to the prefix scan even when the layout replicates that element across lanes or warps.

## Pre-state

The blocked layout maps one logical scan axis onto a subset of unique lanes and replicates values across the remaining participants.

## Binding transition

Scan lowering derives the scan width from physical threads or warps instead of participants carrying unique data.

## Post-state

Replicas are accumulated as additional logical elements and hold prefix values beyond the tensor's logical extent.

## Validity obligation

Warp and inter-warp scan steps use unique-data lane and warp counts; replicated participants only receive the resulting value.

## Invalid decision

ScanLoweringHelper used total threads per warp and initially incompletely handled repeated warps, so layout replication entered the reduction domain.

## Physical effect

Internal lanes observe prefixes 33 through 128 for a 32-element scan, or repeated 1-through-32 sequences for a 16-element scan.

## Observed consequence

tl.device_assert observes invalid prefixes, and consumers that use the scan for indices can derive out-of-range coordinates even when the final stored tensor is broadcast-correct.

## Repair site

lib/Analysis/Utility.cpp and lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp

## Repair

PR 2330 separates warps with unique data from repetitive warps. PR 2345 adds the unique-data thread count, uses it throughout warp and partial-reduce lowering, and extends the regression to scan sizes 1 and 16.

## Why the repair works

Only lanes and warps representing distinct tensor coordinates participate in the prefix recurrence, so replication no longer changes the logical scan.

## Causal basis

The issue reports the two replication failures; the two linked merged patches repair repeated warps and then repeated threads, and the final regression covers both reported small dimensions.

## Analysis implication

A scan-binding check needs tensor extent, layout size-per-thread, threads and warps per CTA, unique-data participant counts, axis order, and shuffle coordinates.

## Fix patches

- [`fixes/triton-2298-pr2330.patch`](fixes/triton-2298-pr2330.patch) — https://github.com/triton-lang/triton/pull/2330
- [`fixes/triton-2298-pr2345.patch`](fixes/triton-2298-pr2345.patch) — https://github.com/triton-lang/triton/pull/2345
