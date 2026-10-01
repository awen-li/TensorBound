# OUT_OF_RANGE error in stateful program

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/7988](https://github.com/iree-org/iree/issues/7988)
- Opened: 2021-12-31T23:36:01Z
- Closed: 2022-01-05T00:06:17Z
- Labels: `bug 🐞`

## Reported behavior

Something funny going on with state updates. See attached zip file for compile temps and program trace.

[trivial_kernel_temp.zip](https://github.com/google/iree/files/7797211/trivial_kernel_temp.zip)

To repro:

```
$ iree-run-trace --driver=dylib trivial_kernel_temp/calls/calls.yaml
--- CALL[trivial_kernel.get_params] ---
3x4xf32=[4 4 4 4][4 4 4 4][4 4 4 4]
3x4xf32=[1 1 1 1][1 1 1 1][1 1 1 1]
--- CALL[trivial_kernel.run] ---
3x4xf32=[5 5 5 5][5 5 5 5][5 5 5 5]
--- CALL[trivial_kernel.run] ---
3x4xf32=[15 15 15 15][15 15 15 15][15 15 15 15]
--- CALL[trivial_kernel.get_params] ---
/home/stella/src/iree/iree/hal/buffer.c:293: OUT_OF_RANGE; attempted to access an address off the end of the valid buffer range (offset=64, length=48, buffer byte_length=48); while invoking native function hal.command_buffer.copy_buffer; while calling import;
[ 1]   native hal.command_buffer.copy_buffer:0 -
[ 0] bytecode trivial_kernel.get_params:196 <stdin>:5:10
      at <stdin>:4:3; replaying trace file '/home/stella/scratch/trivial_kernel_temp/calls/calls.yaml'
```

The input MLIR file is in the zip and here for glancing. Nothing looks amiss.

```
module @trivial_kernel  {
  iree_input.global private mutable @_params$0 = dense<4.000000e+00> : tensor<3x4xf32>
  iree_input.global private mutable @_params$1 = dense<1.000000e+00> : tensor<3x4xf32>
  func @get_params() -> (tensor<3x4xf32>, tensor<3x4xf32>) {
    %0 = iree_input.global.load @_params$0 : tensor<3x4xf32>
    %1 = iree_input.global.load @_params$1 : tensor<3x4xf32>
    return %0, %1 : tensor<3x4xf32>, tensor<3x4xf32>
  }
  func @run(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %0 = iree_input.global.load @_params$0 : tensor<3x4xf32>
    %1 = iree_input.global.load @_params$1 : tensor<3x4xf32>
    %2 = call @main(%arg0, %0, %1) : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    iree_input.global.store %2, @_params$0 : tensor<3x4xf32>
    return %2 : tensor<3x4xf32>
  }
  func private @main(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<3x4xf32>
    %1 = mhlo.add %0, %arg2 : tensor<3x4xf32>
    return %1 : tensor<3x4xf32>
  }
  func @set_params(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>) {
    iree_input.global.store %arg0, @_params$0 : tensor<3x4xf32>
    iree_input.global.store %arg1, @_params$1 : tensor<3x4xf32>
    return
  }
}
```

## Trigger

One function stores the same value to two globals while another function stores only the first global.

## Relevant entities

mutable globals; store operations; correlation map; fused resource; storage offsets

## Implemented relation

Globals share physical storage only when their store correlation is symmetric across every analyzed callable block.

## Pre-state

The two globals hold independent tensors and may diverge after a state update.

## Binding transition

The old pass accumulates co-store evidence without intersecting it against each global's own store set.

## Post-state

The pass fuses globals that are updated together on one path but diverge on another.

## Validity obligation

Fusion requires mutual, path-consistent correlation for every member of an equivalence class.

## Invalid decision

One-way co-occurrence was accepted as an equivalence relation.

## Physical effect

Later reads use offsets in a fused allocation whose logical contents no longer match the separate globals.

## Observed consequence

Stateful calls produce variable values and an out-of-range copy at offset 64 in a 48-byte buffer.

## Repair site

iree/compiler/Dialect/Util/Transforms/FuseGlobals.cpp

## Repair

PR 8008 intersects each candidate correlation set with every member's correlation set and clears candidates that are not self-correlated; it adds an asymmetric-store regression.

## Why the repair works

Only globals that are always updated together remain in the same equivalence class.

## Causal basis

The issue's failing IR shows fused divergent globals, while the patch names issue 7988 and adds that exact asymmetric pattern as a test.

## Analysis implication

Storage-fusion analysis needs stores across all functions and paths, not a local pairwise co-occurrence.

## Fix patches

- [`fixes/iree-7988-pr8008.patch`](fixes/iree-7988-pr8008.patch) — https://github.com/iree-org/iree/pull/8008
