# [Stream] EmplaceAllocations + ScheduleAllocations swaps resource order for multi-result dispatches

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/14566](https://github.com/iree-org/iree/issues/14566)
- Opened: 2023-08-03T04:38:18Z
- Closed: open at collection time
- Labels: `bug 🐞`, `compiler/dialects`

## Reported behavior

# Problem Reproducer
The following IR
```mlir
#map_in = affine_map<(d0, d1) -> (d0, d1)>
#map_out = affine_map<(d0, d1) -> (d0)>
#map = affine_map<(d0) -> (d0)>
func.func @forward(%arg0: tensor<4x4xf32>) -> (tensor<2x4xf32>) {
  %c0_f32 = arith.constant 0.0 : f32
  %empty_1 = tensor.empty() : tensor<4xf32>
  %empty_2 = tensor.empty() : tensor<4xf32>
  %fill_1 = linalg.fill ins(%c0_f32 : f32) outs(%empty_1 : tensor<4xf32>) -> tensor<4xf32>
  %fill_2 = linalg.fill ins(%c0_f32 : f32) outs(%empty_2 : tensor<4xf32>) -> tensor<4xf32>
  %sum, %root_out = linalg.generic {indexing_maps = [#map_in, #map_out, #map_out], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<4x4xf32>) outs(%fill_1, %fill_2 : tensor<4xf32>, tensor<4xf32>) {
  ^bb0(%in: f32, %out: f32, %out_2: f32):
    %add = arith.addf %in, %out : f32
    %root = math.sqrt %add : f32
    linalg.yield %add, %root : f32, f32
  } -> (tensor<4xf32>, tensor<4xf32>)

  %empty_3 = tensor.empty() : tensor<4xf32>
  %sqrt = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%sum : tensor<4xf32>) outs(%empty_3 : tensor<4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sqrt = math.sqrt %in : f32
    linalg.yield %sqrt : f32
  } -> tensor<4xf32>

  %concat_empty = tensor.empty() : tensor<2x4xf32>
  %inserted_slice_1 = tensor.insert_slice %root_out into %concat_empty[1, 0] [1, 4] [1, 1] : tensor<4xf32> into tensor<2x4xf32>
  %inserted_slice_2 = tensor.insert_slice %sqrt into %inserted_slice_1[0, 0] [1, 4] [1, 1] : tensor<4xf32> into tensor<2x4xf32>

  return %inserted_slice_2 : tensor<2x4xf32>
}
```
Compiled with
```
iree-compile concat.mlir \
    --iree-hal-target-backends=vulkan-spirv \
    -o /tmp/concat.vmfb

iree-run-module --device=vulkan --function=forward --input="4x4xf32=1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0" --module=/tmp/concat.vmfb
```
gives
```
EXEC @forward
result[0]: hal.buffer_view
2x4xf32=[1.77828 2.2581 2.54573 2.75967][10 26 42 58]
```
while the correct result should be the square root of both sets of 4 values, i.e.
```
EXEC @forward
result[0]: hal.buffer_view
2x4xf32=[3.16228 5.09902 6.48074 7.61577][3.16228 5.09902 6.48074 7.61577]
```

# Source of the issue
This happens because after EmplaceAllocations the result resource on the first dispatch (generic) is mapped as a read-write input resource on a slice, e.g.
```mlir
func.func @forward(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c553648160_i32 = arith.constant 553648160 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c4, %c4]) type(%c553648160_i32) encoding(%c1_i32)
  %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<4x4xf32> in !stream.resource<external>{%c64}
  %1 = stream.async.transfer %0 : !stream.resource<external>{%c64} -> !stream.resource<*>{%c64}
  %2:2 = stream.async.dispatch @forward_dispatch_0::@forward_dispatch_0_generic_4x4_f32(%1[%c0 to %c64 for %c64]) : (!stream.resource<*>{%c64}) -> (!stream.resource<*>{%c16}, !stream.resource<*>{%c16})
  %3 = stream.async.dispatch @forward_dispatch_1::@forward_dispatch_1_generic_4_f32(%2#0[%c0 to %c16 for %c16]) : (!stream.resource<*>{%c16}) -> !stream.resource<*>{%c16}
  %4 = stream.async.alloca : !stream.resource<*>{%c32}
  %5 = stream.async.update %2#1, %4[%c16 to %c32] : !stream.resource<*>{%c16} -> %4 as !stream.resource<*>{%c32}
  %6 = stream.async.update %3, %5[%c0 to %c16] : !stream.resource<*>{%c16} -> %5 as !stream.resource<*>{%c32}
  %7 = stream.async.transfer %6 : !stream.resource<*>{%c32} -> !stream.resource<external>{%c32}
  %8 = stream.tensor.export %7 : tensor<2x4xf32> in !stream.resource<external>{%c32} -> !hal.buffer_view
  return %8 : !hal.buffer_view
}
```
becomes
```mlir
func.func @forward(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c553648160_i32 = arith.constant 553648160 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c4, %c4]) type(%c553648160_i32) encoding(%c1_i32)
  %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<4x4xf32> in !stream.resource<external>{%c64}
  %1 = stream.async.transfer %0 : !stream.resource<external>{%c64} -> !stream.resource<*>{%c64}
  %2 = stream.async.alloca : !stream.resource<*>{%c32}
  %3:2 = stream.async.dispatch @forward_dispatch_0::@forward_dispatch_0_generic_4x4_f32(%1[%c0 to %c64 for %c64], %2[%c16 to %c32 for %c16]) : (!stream.resource<*>{%c64}, !stream.resource<*>{%c32}) -> (!stream.resource<*>{%c16}, %2{%c32})
  %4 = stream.async.dispatch @forward_dispatch_1::@forward_dispatch_1_generic_4_f32(%3#0[%c0 to %c16 for %c16], %3#1[%c0 to %c16 for %c16]) : (!stream.resource<*>{%c16}, !stream.resource<*>{%c32}) -> %3#1{%c32}
  %5 = stream.async.transfer %4 : !stream.resource<*>{%c32} -> !stream.resource<external>{%c32}
  %6 = stream.tensor.export %5 : tensor<2x4xf32> in !stream.resource<external>{%c32} -> !hal.buffer_view
  return %6 : !hal.buffer_view
}
```
however after ScheduleAllocations the second result resource is listed before the first, later causing the first and second results to be written to the wrong places.
```mlir
func.func @forward(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c553648160_i32 = arith.constant 553648160 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c4, %c4]) type(%c553648160_i32) encoding(%c1_i32)
  %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<4x4xf32> in !stream.resource<external>{%c64}
  %c0_0 = arith.constant 0 : index
  %1 = stream.resource.alloc uninitialized : !stream.resource<external>{%c32}
  %2:2 = stream.resource.pack slices({
    [1, 2] = %c16
  }) : index
  %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%2#0} => !stream.timepoint
  %3 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg1: !stream.resource<external>{%c64}, %1 as %arg2: !stream.resource<external>{%c32}, %result as %arg3: !stream.resource<transient>{%2#0}) {
    stream.cmd.dispatch @forward_dispatch_0::@forward_dispatch_0_generic_4x4_f32 {
      ro %arg1[%c0 for %c64] : !stream.resource<external>{%c64},
      rw %arg2[%c16 for %c16] : !stream.resource<external>{%c32},
      wo %arg3[%2#1 for %c16] : !stream.resource<transient>{%2#0}
    }
    stream.cmd.dispatch @forward_dispatch_1::@forward_dispatch_1_generic_4_f32 {
      ro %arg3[%2#1 for %c16] : !stream.resource<transient>{%2#0},
      rw %arg2[%c0 for %c16] : !stream.resource<external>{%c32}
    }
  } => !stream.timepoint
  %4 = stream.resource.dealloca await(%3) => %result : !stream.resource<transient>{%2#0} => !stream.timepoint
  %5 = stream.timepoint.join max(%4, %3) => !stream.timepoint
  %6 = stream.timepoint.await %5 => %1 : !stream.resource<external>{%c32}
  %7 = stream.tensor.export %6 : tensor<2x4xf32> in !stream.resource<external>{%c32} -> !hal.buffer_view
  return %7 : !hal.buffer_view
}
```
#14565 attempts to fix this by reassociating read-write bindings with the order of the results, but this collides with other expected orderings of read-write resources.

## Trigger

EmplaceAllocations ties only a later result of a multi-result dispatch to a destination slice before ScheduleAllocations runs.

## Relevant entities

ordered dispatch results; read-write result resources; destination slices; implicit binding order

## Implemented relation

Dispatch result position implicitly determines its associated read-write resource position.

## Pre-state

Two logical results and two destination resources have matching order.

## Binding transition

Sparse emplacement removes one result/resource pair while leaving a later pair emplaced.

## Post-state

ScheduleAllocations presents the remaining resources in an order that no longer matches result positions.

## Validity obligation

Transformations preserve the positional association between every dispatch result and its destination resource.

## Invalid decision

EmplaceAllocations allowed non-prefix sparse emplacement even though the downstream representation relies on implicit ordering.

## Physical effect

The two computed results are written into each other's destination storage.

## Observed consequence

The two-result reproducer returns incorrect tensor values; the reporter confirms correct output after the workaround.

## Repair site

compiler/src/iree/compiler/Dialect/Stream/Transforms/EmplaceAllocations.cpp

## Repair

The pass stops emplacement at the first unemplaceable result, preserving a contiguous result prefix.

## Why the repair works

Prefix-only emplacement prevents removal of an earlier positional binding while a later positional binding remains.

## Causal basis

The patch explicitly names implicit binding ordering and adds a multi-result regression; it supports the reproduced path without claiming a full SSA redesign.

## Analysis implication

A transformation check needs the dispatch result/resource ordering relation when eliminating or tying result storage.

## Fix patches

- [`fixes/iree-14566-fix.patch`](fixes/iree-14566-fix.patch) — https://github.com/iree-org/iree/pull/14581
