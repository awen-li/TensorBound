# [Codegen][AMDGPU Backend] Correctness issue for conv_2d_ngchw_gfchw

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/18798](https://github.com/iree-org/iree/issues/18798)
- Opened: 2024-10-16T18:41:38Z
- Closed: 2025-05-19T23:09:03Z
- Labels: `bug 🐞`, `codegen/llvm`, `codegen/rocm`

## Reported behavior

## Problem Description

The following IR
```mlir
module {
  func.func @test(%arg0: tensor<1x2x8x3x3xi8>, %arg1: tensor<2x1x8x3x3xi8>) -> tensor<1x2x1x1x1xi32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<1x2x1x1x1xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<1x2x1x1x1xi32>) -> tensor<1x2x1x1x1xi32>
    %2 = linalg.conv_2d_ngchw_gfchw_q {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %arg1, %c0_i32, %c0_i32 : tensor<1x2x8x3x3xi8>, tensor<2x1x8x3x3xi8>, i32, i32) outs(%1 : tensor<1x2x1x1x1xi32>) -> tensor<1x2x1x1x1xi32>
    return %2 : tensor<1x2x1x1x1xi32>
  }
}
```

With inputs generated using the following numpy commands
```
import numpy as np

x = np.ones((1, 2, 8, 3, 3), dtype=np.int8)
y = np.array([1, 2, 1], dtype=np.int8)
y = np.broadcast_to(y, (2, 1, 8, 3, 3))

np.save("in1.npy", x)
np.save("in2.npy", y)
```

Produces correct results on gfx1100 and gfx942 using this compile + run command
```
iree-compile dispatch.mlir \
    --iree-hip-target=gfx1100 \
    --iree-hal-target-backends=rocm \
    -o /tmp/dispatch.vmfb
```
```
iree-run-module \
    --module=/tmp/dispatch.vmfb \
    --device=hip \
    --function=test \
    --input=@in1.npy \
    --input=@in2.npy \
    --output=@output.npy
```

and incorrect results when adding `--iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true` on this branch: https://github.com/iree-org/iree/pull/18474
```
Not equal to tolerance rtol=1e-07, atol=0

Mismatched elements: 2 / 2 (100%)
Max absolute difference among violations: 8
Max relative difference among violations: 0.09090909
 ACTUAL: array([[[[[88]]],


        [[[88]]]]], dtype=int32)
 DESIRED: array([[[[[96]]],


        [[[96]]]]], dtype=int32)
```

Changing the llvm optimization level to `None` or `Less` produces correct results when using the above flag: https://github.com/iree-org/iree/blob/c6056d197230161ea1403e88b5b8784d34e071a2/compiler/plugins/target/ROCM/ROCMTarget.cpp#L466

## Investigation

The IR generated immediately before lowering scf to control flow looks like the following:
```mlir
func.func @test_dispatch_0_conv_2d_ngchw_gfchw_q_1x2x1x1x1x8x3x3_i8xi8xi32xi32xi32() {
  %c3 = arith.constant 3 : index
  %cst = arith.constant dense<0> : vector<1xi32>
  %c32 = arith.constant 32 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %thread_id_x = gpu.thread_id  x
  %thread_id_y = gpu.thread_id  y
  %thread_id_z = gpu.thread_id  z
  %0 = arith.muli %thread_id_y, %c32 : index
  %1 = arith.addi %thread_id_x, %0 : index
  %2 = arith.muli %thread_id_z, %c32 : index
  %3 = arith.addi %1, %2 : index
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1x2x8x3x3xi8, strided<[144, 72, 9, 3, 1], offset: ?>, #gpu.address_space<global>>
  memref.assume_alignment %4, 1 : memref<1x2x8x3x3xi8, strided<[144, 72, 9, 3, 1], offset: ?>, #gpu.address_space<global>>
  %5 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<2x1x8x3x3xi8, strided<[72, 72, 9, 3, 1], offset: ?>, #gpu.address_space<global>>
  memref.assume_alignment %5, 1 : memref<2x1x8x3x3xi8, strided<[72, 72, 9, 3, 1], offset: ?>, #gpu.address_space<global>>
  %6 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<1x2x1x1x1xi32, strided<[2, 1, 1, 1, 1], offset: ?>, #gpu.address_space<global>>
  memref.assume_alignment %6, 1 : memref<1x2x1x1x1xi32, strided<[2, 1, 1, 1, 1], offset: ?>, #gpu.address_space<global>>
  scf.for %arg0 = %3 to %c2 step %c32 {
    %7 = vector.extract %cst[0] : i32 from vector<1xi32>
    memref.store %7, %6[%c0, %arg0, %c0, %c0, %c0] : memref<1x2x1x1x1xi32, strided<[2, 1, 1, 1, 1], offset: ?>, #gpu.address_space<global>>
    scf.for %arg1 = %c0 to %c8 step %c1 {
      scf.for %arg2 = %c0 to %c3 step %c1 {
        scf.for %arg3 = %c0 to %c3 step %c1 {
          %8 = memref.load %4[%c0, %arg0, %arg1, %arg2, %arg3] : memref<1x2x8x3x3xi8, strided<[144, 72, 9, 3, 1], offset: ?>, #gpu.address_space<global>>
          %9 = memref.load %5[%arg0, %c0, %arg1, %arg2, %arg3] : memref<2x1x8x3x3xi8, strided<[72, 72, 9, 3, 1], offset: ?>, #gpu.address_space<global>>
          %10 = memref.load %6[%c0, %arg0, %c0, %c0, %c0] : memref<1x2x1x1x1xi32, strided<[2, 1, 1, 1, 1], offset: ?>, #gpu.address_space<global>>
          %11 = arith.extsi %8 : i8 to i32
          %12 = arith.extsi %9 : i8 to i32
          %13 = arith.muli %11, %12 : i32
          %14 = arith.addi %10, %13 : i32
          memref.store %14, %6[%c0, %arg0, %c0, %c0, %c0] : memref<1x2x1x1x1xi32, strided<[2, 1, 1, 1, 1], offset: ?>, #gpu.address_space<global>>
        }
      }
    }
  }
  return
}
```
(workgroup count is `[1, 1 1]`, i.e. single workgroup).

Where it is simply looping over the reduction dims of the `conv_2d` and accumulating. `%8` and `%9` are the loads for the image and filters respectively. In the above sample inputs, `%8` is always `1` (np.ones), while `%9` is broadcasted `[1, 2, 1]` along the inner most dim, so the only index that affects the loaded value is `%arg3`.

Note that switching the input to be `[2, 1, 1]` broadcasted from the inner most dim changes the output to `104` from `88`, and using `[1, 1, 2]` gives correct results, indicating that somehow the load for `%arg3 = 1` somehow got replaced with a duplicate load to the first value. Additionally this only reproduces incorrect results if the input channel dimension (`8` in this example) is >= 7. For smaller input channel dims this produces correct values.

Additionally changing the input values for the image (`%8`) to be broadcasted `[1, 2, 1]` and make the filter (`%9`) uniform gives correct values, indicating that it is specifically the second load in this example that is getting mangled.

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

Tile-and-fuse vectorization lowers the reported integer convolution and LLVM optimization runs at its default level.

## Observed implementation decision

The issue localizes duplicate filter loads to LLVM's AMDGPU LoadStoreVectorizer path.

## Physical effect

The generated code repeats one filter element and omits another during accumulation.

## Observed consequence

Both outputs are 88 instead of 96.

## Reported proposal or workaround

IREE PR 18949 enables SLP vectorization by default, which avoids the failing downstream path; disabling LoadStoreVectorizer or enabling GlobalISel also changes the result.

## Repair trace

The archived change selects a different optimization pipeline but does not identify or repair the invalid LoadStoreVectorizer transformation, and it adds no report-specific regression demonstrating the root repair.

## Archived evidence

- Raw report: `data/raw/github/iree/18798/report.json`
- SHA-256: `f5b100bdce89940507e6dfb58f7bf6892e179a341a67b5966abbcb7dfe402be8`
