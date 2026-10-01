# Potential bug of memory descriptor of WGMMA.cpp

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/6220](https://github.com/triton-lang/triton/issues/6220)
- Opened: 2025-03-17T21:56:24Z
- Closed: 2025-03-23T14:29:23Z
- Labels: `bug`

## Reported behavior

### Describe the bug

This bug is found when creating the PR: https://github.com/triton-lang/triton/pull/6196

The issue is on Hopper when performs  matrix multiplication:    A x B -> C using  wgmma

A simple bug description/reproducer is that:

Both A, B, C  are float32 data format.
A has the shape  128  X  K ,   float32
B has the shape   K x 128,  float32

For A x B -> C,  suppose k = 32
we can break it down to subtile in ttgir such that:
C = 0
A1 = subview [0, 0]  with shape  128x16
B1 = subview [0, 0]  with shape  16x128
C += wgmma(A1 x B1)
A2 = subview [0, 16]  with shape  128x16
B2 = subview [16, 0]  with shape  16x128
C += wgmma(A2 x B2)

when K = 32, such change will produce correct results.



However, when K  = 64, we break it down into subtiles:

C = 0
A1 = subview [0, 0]  with shape  128x16
B1 = subview [0, 0]  with shape  16x128
C += wgmma(A1 x B1)
A2 = subview [0, 16]  with shape  128x16
B2 = subview [16, 0]  with shape  16x128
C +=  wgmma(A2 x B2)
A3 = subview [0, 32]  with shape  128x16
B3 = subview [32, 0]  with shape  16x128
C +=  wgmma(A3 x B3)
A4 = subview [0, 48]  with shape  128x16
B4 = subview [48, 0]  with shape  16x128
C +=  wgmma(A4 x B4)

It will produce incorrect results.  Change subtitle size 16 to 8 or 32 will also produce incorrect results.


In the above cases, both A and B comes from shared memory, described by the memory descriptor.


There might be potential issue for subview or  WGMMA.cpp when creating memory descriptor for hopper subtilting.


Currently, find the bug exist for databitwidth=32 and  K >=64.




@ThomasRaoux  @lezcano  Wondering if you have any clues for this bug or where the bug could come from?








### Environment details

Hopper 100

## Trigger

A K=64 shared-memory operand is split into multiple K=16 WGMMA subtiles under a swizzled layout.

## Relevant entities

shared-memory descriptor; NVMMASharedEncodingAttr; subview offsets; swizzle tile; WGMMA base address

## Implemented relation

A logical subview coordinate maps to the physical address defined by the descriptor's swizzled shared-memory layout.

## Pre-state

The parent descriptor contains a swizzled 2D tile and accumulated base offsets.

## Binding transition

Subview lowering computes dot(logical_offsets, original_strides) as if the storage were linear strided memory.

## Post-state

Later K subtiles receive base addresses that ignore swizzle-period and tile boundaries.

## Validity obligation

Subview address calculation incorporates layout order, swizzle stride, tile size, element width, and existing descriptor offsets.

## Invalid decision

The old generic memdesc_subview rule treated NVMMASharedEncodingAttr like an unswizzled strided encoding.

## Physical effect

WGMMA loads bytes from the wrong shared-memory tile for later repetitions.

## Observed consequence

K=64 matrix multiplication returns incorrect results after subtiling while K=32 may pass.

## Repair site

lib/Conversion/TritonGPUToLLVM/ViewOpToLLVM.cpp

## Repair

PR 6241 adds NVMMASharedEncodingAttr-specific offset arithmetic based on swizzle stride, tile size, layout order, descriptor offsets, and element bit width, plus a lowering regression.

## Why the repair works

The derived pointer follows the same swizzled coordinate mapping used to store the parent tile.

## Causal basis

Maintainers identify shared-memory subview addressing, and the patch replaces the generic stride formula for the reported encoding.

## Analysis implication

Subview checks need the full encoding and parent descriptor state; shape and nominal strides alone do not determine the physical address.

## Fix patches

- [`fixes/triton-6220-pr6241.patch`](fixes/triton-6220-pr6241.patch) — https://github.com/triton-lang/triton/pull/6241
