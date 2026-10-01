# mmt4d ukernel called with LHS pointer passed also for RHS and OUT

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/12480](https://github.com/iree-org/iree/issues/12480)
- Opened: 2023-03-02T21:29:53Z
- Closed: 2023-03-13T21:48:14Z
- Labels: `bug 🐞`, `codegen/vmvx`

## Reported behavior

Debugging the remaining test failures on #12157, namely https://github.com/openxla/iree/pull/12157#issuecomment-1450736576 :

The problem is that `iree_vmvx_mmt4d` is called with incorrect pointers for RHS and OUT: they are getting the exact same pointer value as LHS.

This reproduces already for the 1x1x1 matmul shape case, so here is the corresponding minimal testcase,  `matmul_i8_1x1x1.mlir`:

```mlir
func.func @matmul_i8(%lhs: tensor<1x1xi8>, %rhs: tensor<1x1xi8>, %acc: tensor<1x1xi32>) -> tensor<1x1xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1x1xi8>, tensor<1x1xi8>) outs(%acc: tensor<1x1xi32>) -> tensor<1x1xi32>
  return %result: tensor<1x1xi32>
}
```

Here is the command line to reproduce the below print-after-all log (this assumes you've applied #12157 and rebuilt `iree-compile`).

```
tools/iree-compile --iree-hal-target-backends=vmvx --iree-vmvx-enable-microkernels --iree-flow-enable-data-tiling ~/matmul_i8_1x1x1.mlir -o /tmp/a.vmfb --mlir-disable-threading --mlir-print-ir-after-all 2>/tmp/log
```

Here is the print-after-all log:
https://gist.githubusercontent.com/bjacob/841774212c53143361f3f234501b3942/raw/d2617932e4f8e88ce6a45554c5a0981e0e3fc464/gistfile1.txt

Some relevant IR:

```
// -----// IR Dump After LLVMCPULowerExecutableTarget (iree-llvmcpu-lower-executable-target) //----- //
hal.executable.variant public @vmvx_bytecode_fb, target = <"vmvx", "vmvx-bytecode-fb", {ukernels = true}> {
  hal.executable.export public @matmul_i8_dispatch_3_matmul_16x16x16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {translation_info = #iree_codegen.translation_info<VMVXDefault>} {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %c1 = arith.constant 1 : index
    %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%arg1]
    %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%arg2]
    hal.return %1, %0, %c1 : index, index, index
  }
  builtin.module {
    func.func private @vmvx.mmt4d.i8i8i32(memref<i8>, index, index, memref<i8>, index, index, memref<i32>, index, index, index, index, index, i32, i32, i32, i32) attributes {vm.import.module = "vmvx"}
    func.func @matmul_i8_dispatch_3_matmul_16x16x16() {
      %c1_i32 = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %c16 = arith.constant 16 : index
      %0:2 = vmvx.query_tile_sizes sizes(%c16, %c16) flags(1048576) -> index, index
      %1 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%0#0]
      %2 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%0#1]
      %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<?x?x?x?xi8>{%1, %2, %0#0, %0#1}
      memref.assume_alignment %3, 64 : memref<?x?x?x?xi8>
      %4:2 = vmvx.query_tile_sizes sizes(%c16, %c16) flags(1114112) -> index, index
      %5 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%4#0]
      %6 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%4#1]
      %7 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c256) flags(ReadOnly) : memref<?x?x?x?xi8>{%5, %6, %4#0, %4#1}
      memref.assume_alignment %7, 64 : memref<?x?x?x?xi8>
      %8:2 = vmvx.query_tile_sizes sizes(%c16, %c16) flags(1179648) -> index, index
      %9 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%8#0]
      %10 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%8#1]
      %11 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c512) : memref<?x?x?x?xi32>{%9, %10, %8#0, %8#1}
      memref.assume_alignment %11, 64 : memref<?x?x?x?xi32>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_count_x = hal.interface.workgroup.count[0] : index
      %workgroup_id_y = hal.interface.workgroup.id[1] : index
      %workgroup_count_y = hal.interface.workgroup.count[1] : index
      %12 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
      %13 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
      scf.for %arg0 = %12 to %1 step %13 {
        %14 = affine.min affine_map<(d0)[s0] -> (-d0 + 16 ceildiv s0, 64)>(%arg0)[%0#0]
        %15 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
        %16 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
        scf.for %arg1 = %15 to %5 step %16 {
          %17 = affine.min affine_map<(d0)[s0] -> (-d0 + 16 ceildiv s0, 64)>(%arg1)[%4#0]
          %subview = memref.subview %3[%arg0, 0, 0, 0] [%14, %2, %0#0, %0#1] [1, 1, 1, 1] : memref<?x?x?x?xi8> to memref<?x?x?x?xi8, strided<[?, ?, ?, 1], offset: ?>>
          %subview_0 = memref.subview %7[%arg1, 0, 0, 0] [%17, %2, %4#0, %0#1] [1, 1, 1, 1] : memref<?x?x?x?xi8> to memref<?x?x?x?xi8, strided<[?, ?, ?, 1], offset: ?>>
          %subview_1 = memref.subview %11[%arg0, %arg1, 0, 0] [%14, %17, %0#0, %4#0] [1, 1, 1, 1] : memref<?x?x?x?xi32> to memref<?x?x?x?xi32, strided<[?, ?, ?, 1], offset: ?>>
          %base_buffer, %offset, %sizes:4, %strides:4 = memref.extract_strided_metadata %subview : memref<?x?x?x?xi8, strided<[?, ?, ?, 1], offset: ?>> -> memref<i8>, index, index, index, index, index, index, index, index, index
          %base_buffer_2, %offset_3, %sizes_4:4, %strides_5:4 = memref.extract_strided_metadata %subview_0 : memref<?x?x?x?xi8, strided<[?, ?, ?, 1], offset: ?>> -> memref<i8>, index, index, index, index, index, index, index, index, index
          %base_buffer_6, %offset_7, %sizes_8:4, %strides_9:4 = memref.extract_strided_metadata %subview_1 : memref<?x?x?x?xi32, strided<[?, ?, ?, 1], offset: ?>> -> memref<i32>, index, index, index, index, index, index, index, index, index
          %18 = arith.index_cast %0#0 : index to i32
          %19 = arith.index_cast %4#0 : index to i32
          %20 = arith.index_cast %0#1 : index to i32
          func.call @vmvx.mmt4d.i8i8i32(%base_buffer, %offset, %strides#0, %base_buffer_2, %offset_3, %strides_5#0, %base_buffer_6, %offset_7, %strides_9#0, %14, %17, %2, %18, %19, %20, %c1_i32) : (memref<i8>, index, index, memref<i8>, index, index, memref<i32>, index, index, index, index, index, i32, i32, i32, i32) -> ()
        }
      }
      return
    }
  }
}
```

There we can see that RHS is in the same binding as LHS but with an offset of 256 (corresponding to the expected 16x16 padded shape) and OUT is in a different binding and with an offset of 512.

So it looks like at least offsets are being ignored, and separately I wonder if bindings are also ignored (as OUT gets the same pointer value in the end, even though it was from a separate binding) or if it's just some kind of aliasing between bindings (as kind of suggested by the offset 512, which is just what it would be if OUT resided in binding 0 after RHS).

## Trigger

The VMVX mmt4d microkernel receives LHS, RHS, and output views whose HAL binding subspans use offsets 0, 256, and 512.

## Relevant entities

HAL interface binding; binding subspan; memref descriptor; extracted base buffer; LHS/RHS/output microkernel arguments

## Implemented relation

A binding number and byte offset jointly identify the physical subspan passed to each microkernel argument.

## Pre-state

LHS, RHS, and output denote distinct physical regions even when two share one binding.

## Binding transition

Code generation converts binding subspans to memrefs and extracts raw base pointers and offsets for the microkernel call.

## Post-state

The lowered descriptors use zero offsets, so the microkernel receives the LHS pointer for RHS and output as well.

## Validity obligation

Lowering a binding subspan preserves both its binding identity and byte offset in the descriptor consumed by the microkernel.

## Invalid decision

The memref representation treated a nonzero binding-subspan offset as if the memref began at offset zero on a backend without pointer-arithmetic lowering.

## Physical effect

Pointer extraction collapses distinct subspans onto the same base address.

## Observed consequence

mmt4d is called with the LHS pointer for RHS and output, causing the matmul tests to fail.

## Repair site

compiler/src/iree/compiler/Codegen/Common/FixupSubspanWithOffsets.cpp and FlattenMemRefSubspanPass.cpp

## Repair

PR 12510 represents nonzero subspan offsets in strided memref metadata, preserves them during flattening, wires the passes into affected backends, and adds offset-sensitive tests.

## Why the repair works

Raw-pointer extraction now combines the binding base with the represented byte offset, keeping the three microkernel arguments on their declared regions.

## Causal basis

The issue traces the wrong pointers to lost subspan offsets, the maintainer reports that the patched representation passes the matmul tests, and the merged patch implements and tests that representation.

## Analysis implication

A lowering check needs binding identity, byte offset, element width, memref layout, and the final pointer/offset arguments at raw-buffer extraction.

## Fix patches

- [`fixes/iree-12480-pr12510.patch`](fixes/iree-12480-pr12510.patch) — https://github.com/iree-org/iree/pull/12510
