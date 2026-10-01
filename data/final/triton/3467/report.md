# Reduction Op on MMA Layout produces incorrect results

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/3467](https://github.com/triton-lang/triton/issues/3467)
- Opened: 2024-03-26T12:27:49Z
- Closed: 2024-05-22T14:49:28Z
- Labels: `bug`

## Reported behavior

I am seeing that the MMA Layout for Hopper produces incorrect results on reduction, specifically when warps > 4 and distributed in 2 dimensions ([4,2] rather than [8,1]). I have a couple of minimal reasonable examples to play around with on `test_reduce_layouts` here: https://github.com/vwbaker/triton/commit/d6617967c0bd84a1000319a4bdaee4c9a04c7b7d. If you run this test (`pytest test/unit/language/test_core.py -vvv -k 'test_reduce_layouts[max-float32-reduce1d-0-src_layout8-64-64]'` ) against openai/main today, it fails with incorrect results.

Some context: I talked to @ThomasRaoux about this over slack a while ago, see conversation here: https://triton-lang.slack.com/archives/C01L894JD98/p1707416097320249?thread_ts=1707310247.070569&cid=C01L894JD98, but haven't had time to look more into it since then but am now looking into it again.

I have a fix that stops us from propagating the mma_layout through ReduceOps during the RemoveLayoutConversions pass; is this a fix you would accept or do you have any other guidance here?

## Trigger

A Hopper MMA v3 layout distributes more than four warps across two dimensions and then reduces a tensor encoded in that layout.

## Relevant entities

MMA v3 layout; linear warp ID; two-dimensional warp coordinates; within-warp thread coordinates; reduction participants

## Implemented relation

The layout maps a linear execution ID to separate warp and lane coordinates; Hopper fixes warp order to [0,1] while thread order remains [1,0].

## Pre-state

Eight warps are declared as a two-dimensional [4,2] arrangement over the MMA tile.

## Binding transition

Reduction and related lowering delinearize warp IDs using the layout's generic order.

## Post-state

Warp coordinates follow thread order rather than Hopper's required warp order, so values are assigned to the wrong tile coordinates during reduction.

## Validity obligation

Every lowering that maps warp IDs to tensor coordinates uses the architecture-defined warp order independently of lane order.

## Invalid decision

The layout exposed one getOrder result for both warp and thread coordinate calculations.

## Physical effect

Reduction participants and offsets are associated with incorrect multidimensional warp coordinates.

## Observed consequence

The eight-warp MMA-layout reduction deterministically returns incorrect values, including values outside the expected softmax range.

## Repair site

TritonGPU layout utilities and reduction, scan, offset, and load/store lowering

## Repair

PR 3970 adds getWarpOrder, forces Hopper MMA warp order [0,1], retains the existing thread order, updates consumers, and enables an eight-warp reduction regression.

## Why the repair works

Warp IDs and lane IDs are now delinearized by their distinct layout rules, preserving the declared MMA tile ownership through reduction.

## Causal basis

The failing case requires a two-dimensional eight-warp MMA v3 layout, and the merged patch separates the exact two orders previously conflated and exercises that configuration in the reduction test.

## Analysis implication

A distributed-layout check needs separate warp and lane orders, the warps-per-CTA shape, and each lowering site's coordinate calculation.

## Fix patches

- [`fixes/triton-3467-pr3970.patch`](fixes/triton-3467-pr3970.patch) — https://github.com/triton-lang/triton/pull/3970
