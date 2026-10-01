# Dot crashes or produce incorrect results with float16 and N==8 on H100

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/7830](https://github.com/triton-lang/triton/issues/7830)
- Opened: 2025-08-12T12:21:38Z
- Closed: 2025-10-20T10:53:37Z
- Labels: `bug`

## Reported behavior

### Describe the bug

[Commit 19277de](https://github.com/triton-lang/triton/commit/19277de9a45dce943b246cd0875221a3a644c06a) updated min dot sizes to support smaller N==8 tiles. However we've had issues trying to use it with 16-bit floats.

The issue can be reproduced by adding test cases with small `BLOCK_N` to `python/test/unit/language/test_matmul.py`:
```
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES", [..., (128, 8, 16, 4), (64, 8, 16, 4)])
```

These tests fail when float16 is in src or dst. Test failure example:
```
test_simple_matmul[False-False-8-2-128-8-16-4-float16-float16]
...
>       torch.testing.assert_close(ref_out, output, atol=atol, rtol=rtol)
E       AssertionError: Tensor-likes are not close!
E
E       Mismatched elements: 514820 / 524288 (98.2%)
E       Greatest absolute difference: nan at index (384, 130) (up to 0.06 allowed)
E       Greatest relative difference: nan at index (256, 128) (up to 0.06 allowed)

python/test/unit/language/test_matmul.py:162: AssertionError
```

The two test cases `(128, 8, 16, 4), (64, 8, 16, 4)` are chosen to have `BLOCK_M` to match (M==64) and exceed (M==128) the `wgmma.mma_async` shape.

We suspect that missing support for the no-swizzle case in `DotOpMmaV3SmemLoader` and `createDescriptor()` in `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/WGMMA.cpp` is causing the test failures. This is based on the presence of a layout with `swizzlingByteWidth=0` used for the dot RHS:

```
$  MLIR_ENABLE_DUMP=1 pytest --verbose --capture=no python/test/unit/language/test_matmul.py -k 'test_simple_matmul[False-False-4-1-128-8-16-4-float16-float16]'

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>
...
    %accumulator_69 = ttng.warp_group_dot %a_67, %b_68, %2 {isAsync = true} : !ttg.memdesc<128x16xf16, #shared, #smem, mutable, 4x128x16> * !ttg.memdesc<16x8xf16, #shared1, #smem> -> tensor<128x8xf16, #mma> loc(#loc80)

```

We've seen that `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td` sets `swizzlingByteWidth=0` based on the contiguous dimension size, which in our case is 8✕float16=16 bytes. `createDescriptor` incorrectly sets SBO and LBO to 0 when `swizzling==0` and `DotOpMmaV3SmemLoader` similarly sets `elemsPerSwizzlingRow=0` when `swizzlingByteWidth==0`.

### Environment details

Triton: triton-3.4.0+git47334706
GPU: Nvidia H100

## Trigger

A matmul uses BLOCK_N=8, selects WGMMA, and places its right-hand operand in an unswizzled shared-memory layout.

## Relevant entities

logical B tile; N=8 result tile; unswizzled shared memory; WGMMA descriptor; shared-memory offset; mma.sync fallback

## Implemented relation

The WGMMA descriptor and per-instruction offset identify the exact shared-memory submatrix supplying each logical right-hand operand tile.

## Pre-state

The no-swizzle encoding stores a narrow right-hand tile in shared memory with an offset convention different from swizzled layouts.

## Binding transition

The old WGMMA lowering applies swizzle-derived descriptor and offset formulas to the zero-swizzle case while allowing N=8 to select that path.

## Post-state

WGMMA reads a different shared-memory region than the logical B tile.

## Validity obligation

MMA selection admits only tile shapes with a supported layout, and descriptor offsets follow that layout's concrete byte organization.

## Invalid decision

MMA v3/v5 accepted N=8 without a valid no-swizzle descriptor and offset calculation.

## Physical effect

The tensor-core instruction consumes wrong or invalid float16 operand bytes.

## Observed consequence

The reported H100 tests crash or mismatch about 98.2% of output elements, including NaNs.

## Repair site

lib/Analysis/Utility.cpp supportMMA and NVIDIA WGMMA.cpp descriptor/shared-memory loading

## Repair

PR 7981 restricts default MMA v3/v5 selection to supported N and K tiles and introduces no-swizzle descriptor handling; PR 7986 corrects the remaining unswizzled right-hand-tile offset.

## Why the repair works

Default lowering falls back for unsupported N=8 shapes, while the repaired no-swizzle path computes the descriptor and B-tile offset from its actual layout when explicitly exercised.

## Causal basis

The reporter identifies zero-swizzle handling, maintainers link both merged changes, and the reporter confirms the N=8 WGMMA test passes after those changes before separating a remaining XLA integration crash.

## Analysis implication

Tensor-core lowering analysis needs logical tile sizes, MMA generation, element width, shared encoding, swizzle width, descriptor fields, operand role, and byte offsets.

## Fix patches

- [`fixes/triton-7830-pr7981.patch`](fixes/triton-7830-pr7981.patch) — https://github.com/triton-lang/triton/pull/7981
- [`fixes/triton-7830-pr7986.patch`](fixes/triton-7830-pr7986.patch) — https://github.com/triton-lang/triton/pull/7986
