# Sub-byte type globals produce incorrect results

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/14642](https://github.com/iree-org/iree/issues/14642)
- Opened: 2023-08-11T20:27:43Z
- Closed: 2023-08-12T02:16:36Z
- Labels: `bug 🐞`, `compiler/dialects`

## Reported behavior

```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
ml_program.global public @cst(dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]> : tensor<8x8xi4>) : tensor<8x8xi4>
  func.func @forward() -> tensor<8x4x2xf32> {
    %cst = ml_program.global_load_const @cst : tensor<8x8xi4>
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
Compiled + run through vulkan or cpu will produce zero values
```
EXEC @forward
result[0]: hal.buffer_view
8x4x2xf32=[[0 0][0 0][0 0][0 0]][[0 0][0 0][0 0][0 0]][[0 0][0 0][0 0][0 0]][[0 0][0 0][0 0][0 0]][[0 0][0 0][0 0][0 0]][[0 0][0 0][0 0][0 0]][[0 0][0 0][0 0][0 0]][[0 0][0 0][0 0][0 0]]
```
If the constant is instead an arith.constant it gives correct values.

(compile + run repro)
```
iree-compile packed_global.mlir \
    --iree-stream-resource-index-bits=64 \
    --iree-vm-target-index-bits=64 \
    --iree-input-type=none \
    --iree-hal-target-backends=vulkan-spirv \
    --iree-opt-const-expr-hoisting=false \
    --iree-flow-inline-constants-max-byte-length=0 \
    -o /tmp/i4_packed.vmfb

iree-run-module --device=vulkan --function=forward \
  --module=/tmp/i4_packed.vmfb
```

## Trigger

An i4 global is reshaped and loaded from a dispatch binding with a nonzero byte offset.

## Relevant entities

binding byte offset; i4 element bit width; memref element offset; global tensor; loaded values

## Implemented relation

A byte offset maps to an element offset by multiplying by eight and dividing by the exact element bit width.

## Pre-state

A 64-byte displacement into packed i4 storage denotes element offset 128.

## Binding transition

convertByteOffsetToElementOffset divides by a rounded element byte width of one.

## Post-state

The generated memref uses element offset 64 and addresses the wrong half-byte region.

## Validity obligation

Byte-to-element conversion preserves the physical bit address for sub-byte element types.

## Invalid decision

Bufferization rounded i4 to one byte before converting the offset, losing the packing factor of two.

## Physical effect

Loads and reshapes read packed nibbles from the wrong physical positions.

## Observed consequence

The sub-byte global produces incorrect numeric results on the LLVM CPU path.

## Repair site

compiler/src/iree/compiler/Codegen/Utils/Utils.cpp::convertByteOffsetToElementOffset

## Repair

PR 14647 computes (byte_offset * 8) floor-divided by the exact type bit width, extends bit-width handling to vectors, and checks that 64 bytes becomes memref offset 128 for i4.

## Why the repair works

The logical element index and packed physical bit address remain equal before and after bufferization.

## Causal basis

The patch explicitly fixes #14642, replaces rounded byte width with exact bit arithmetic, and adds an i4 offset regression matching the reported global path.

## Analysis implication

A packed-buffer check needs byte offset, exact storage bit width, vector width, packing factor, and resulting memref element offset.

## Fix patches

- [`fixes/iree-14642-pr14647.patch`](fixes/iree-14642-pr14647.patch) — https://github.com/iree-org/iree/pull/14647
