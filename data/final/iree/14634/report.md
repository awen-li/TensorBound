# SPIRV unsigned sub-byte arithmetic correctness issue

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/14634](https://github.com/iree-org/iree/issues/14634)
- Opened: 2023-08-11T00:22:44Z
- Closed: 2023-08-13T04:18:32Z
- Labels: `bug 🐞`, `codegen/spirv`

## Reported behavior

# Problem Description

Starting with the following IR
```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @forward() -> tensor<8x4x2xf32> {
    %cst = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]> : tensor<8x8xi4>
    %expanded_4 = tensor.expand_shape %cst [[0], [1, 2]] : tensor<8x8xi4> into tensor<8x4x2xi4>
    %0 = tensor.empty() : tensor<8x4x2xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_4 : tensor<8x4x2xi4>) outs(%0 : tensor<8x4x2xf32>) {
    ^bb0(%in: i4, %out: f32):
      %6 = arith.extui %in : i4 to i32
      %7 = arith.uitofp %6 : i32 to f32
      linalg.yield %7 : f32
    } -> tensor<8x4x2xf32>
    return %5 : tensor<8x4x2xf32>
  }
}
```
compile + run commands
```
iree-compile packed_only.mlir \
    --iree-stream-resource-index-bits=64 \
    --iree-vm-target-index-bits=64 \
    --iree-input-type=none \
    --iree-hal-target-backends=vulkan-spirv \
    --iree-flow-inline-constants-max-byte-length=0 \
    -o /tmp/i4_packed.vmfb

iree-run-module --device=vulkan --function=forward \
  --module=/tmp/i4_packed.vmfb
```

produces
```
EXEC @forward
result[0]: hal.buffer_view
8x4x2xf32=[[0 1][2 3][4 5][6 7]][[4.29497E+09 4.29497E+09][4.29497E+09 4.29497E+09][4.29497E+09 4.29497E+09][4.29497E+09 4.29497E+09]][[0 1][2 3][4 5][6 7]][[4.29497E+09 4.29497E+09][4.29497E+09 4.29497E+09][4.29497E+09 4.29497E+09][4.29497E+09 4.29497E+09]][[0 1][2 3][4 5][6 7]][[4.29497E+09 4.29497E+09][4.29497E+09 4.29497E+09][4.29497E+09 4.29497E+09][4.29497E+09 4.29497E+09]][[0 1][2 3][4 5][6 7]][[4.29497E+09 4.29497E+09][4.29497E+09 4.29497E+09][4.29497E+09 4.29497E+09][4.29497E+09 4.29497E+09]]
```
# Problem Cause

This is caused by the sub-byte load emulation always treating the loaded value as signed, i.e. `%12` below.
```mlir
      %5 = spirv.AccessChain %__resource_var_0_0__addr[%cst0_i32_0, %4] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32
      %6 = spirv.Load "StorageBuffer" %5 : i32
      %cst8_i32_2 = spirv.Constant 8 : i32
      %cst4_i32 = spirv.Constant 4 : i32
      %7 = spirv.UMod %3, %cst8_i32_2 : i32
      %8 = spirv.IMul %7, %cst4_i32 : i32
      %9 = spirv.ShiftRightArithmetic %6, %8 : i32, i32
      %cst15_i32 = spirv.Constant 15 : i32
      %10 = spirv.BitwiseAnd %9, %cst15_i32 : i32
      %cst28_i32 = spirv.Constant 28 : i32
      %11 = spirv.ShiftLeftLogical %10, %cst28_i32 : i32, i32
      %12 = spirv.ShiftRightArithmetic %11, %cst28_i32 : i32, i32
      %13 = spirv.ConvertUToF %12 : i32 to f32
```
(from https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/MemRefToSPIRV/MemRefToSPIRV.cpp#L527)

## Trigger

An unsigned i4 tensor is extended to i32 and converted to floating point in a Vulkan/SPIR-V dispatch.

## Relevant entities

logical i4 value; wider SPIR-V carrier; arith.extui; mask or shift; floating-point consumer

## Implemented relation

Extension or truncation observes only the logical sub-byte payload, independent of unused bits in the wider carrier type.

## Pre-state

Each i4 element has a four-bit unsigned value even though SPIR-V emulation carries it in a wider integer.

## Binding transition

Arith-to-SPIR-V type emulation reuses the converted carrier without masking or shifting away bits outside the source width.

## Post-state

The consumer receives polluted high bits or the wrong signed extension for values whose top i4 bit is set.

## Validity obligation

Unsigned extension zeroes bits above the logical width; signed extension propagates only the logical sign bit; truncation discards all higher carrier bits.

## Invalid decision

The conversion treated equal converted carrier types as if no semantic extension or truncation remained necessary.

## Physical effect

i4 values 8 through 15 are represented with incorrect wider integer bits before uitofp.

## Observed consequence

The Vulkan result differs from the expected unsigned 0-through-15 sequence.

## Repair site

mlir/lib/Conversion/ArithToSPIRV/ArithToSPIRV.cpp and SPIRVConversion.cpp

## Repair

LLVM commit 2b066501 adds masking and logical/arithmetic shifts for emulated integer extension and truncation. Commit 69a3c9cd permits sub-byte vector conversion. IREE PR 14652 imports them and adds the reported ui4-to-f32 end-to-end regression.

## Why the repair works

The wider SPIR-V value is reconstructed from exactly the logical i4 payload before downstream arithmetic observes it.

## Causal basis

The issue isolates unsigned i4 extension in ConvertToSPIRV; the imported upstream commits repair extension/truncation under type emulation, and the IREE patch adds the exact ui4-to-f32 values as an end-to-end test.

## Analysis implication

A sub-byte lowering check needs logical signedness and width, carrier width, vector shape, extension or truncation kind, mask, and shift semantics.

## Fix patches

- [`fixes/iree-14634-pr14652.patch`](fixes/iree-14634-pr14652.patch) — https://github.com/iree-org/iree/pull/14652
- [`fixes/llvm-14634-2b066501.patch`](fixes/llvm-14634-2b066501.patch) — https://github.com/llvm/llvm-project/commit/2b066501b1bcb21c408310e6cfca31ba02068736
- [`fixes/llvm-14634-69a3c9cd.patch`](fixes/llvm-14634-69a3c9cd.patch) — https://github.com/llvm/llvm-project/commit/69a3c9cddf1919b7caaa625efa1829cd5628bf10
