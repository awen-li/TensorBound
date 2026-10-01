# [GPU] linalg.conv_2d_nhwc_hwcf result is wrong on cuda backend

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/16175](https://github.com/iree-org/iree/issues/16175)
- Opened: 2024-01-22T14:12:58Z
- Closed: Open at collection
- Labels: `bug 🐞`

## Reported behavior

### What happened?

[GPU] linalg.conv_2d_nhwc_hwcf result is wrong on cuda backend

The answer is wrong when I use iree cuda backend to lower linalg.conv_2d_nhwc_hwcf and run it on V100. Here is an example input mlir cutted from mobilenetv2.
```mlir
module @test{
    func.func @main(%image : tensor<1x225x225x3xf32>, %kernel : tensor<3x3x3x32xf32>) -> tensor<1x112x112x32xf32> {
        %cst_1 = arith.constant 0.000000e+00 : f32
        %0 = tensor.empty() : tensor<1x112x112x32xf32>
        %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
        %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%image, %kernel : tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>) outs(%1 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
        return %2 : tensor<1x112x112x32xf32>
    }
}
```
The result shape is `<1x112x112x32xf32>`. If we use `[n,h,w,c]` as index, when `h%2==0` it's correct and when `h%2==1` it' wrong. The wrong data repeat the first four channel (but this may be an undefined behavior). For example, here are 32 points at `[0,1,0,*]`
```
// wrong, gpu result
[11232. 11259. 11286. 11313. 11232. 11259. 11286. 11313. 11232. 11259. 11286. 11313. 11232. 11259. 11286. 11313. 11232. 11259. 11286. 11313. 11232. 11259. 11286. 11313. 11232. 11259. 11286. 11313. 11232. 11259. 11286. 11313.]

// correct, cpu result
[11232. 11259. 11286. 11313. 11340. 11367. 11394. 11421. 11448. 11475. 11502. 11529. 11556. 11583. 11610. 11637. 11664. 11691. 11718. 11745. 11772. 11799. 11826. 11853. 11880. 11907. 11934. 11961. 11988. 12015. 12042. 12069.]
```

Since this conv_2d comes from the first layer of mobilenetv2, this problem also causes mobilenetv2 to predict wrong.

### Steps to reproduce your issue

This python script can directly reproduce this problem. If all things are correct, gpu and cpu backend should give same result, so the sum of difference should be nearly zero, and the two rows should be nearly same, and the assertion of allclose should pass.

However, I got the sum value 75866110.0, and assertion failed.
```python
import iree.compiler as ireec
import iree.runtime as ireert
import numpy as np

convmlir = r"""
module @test{
    func.func @main(%image : tensor<1x225x225x3xf32>, %kernel : tensor<3x3x3x32xf32>) -> tensor<1x112x112x32xf32> {
        %cst_1 = arith.constant 0.000000e+00 : f32
        %0 = tensor.empty() : tensor<1x112x112x32xf32>
        %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
        %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%image, %kernel : tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>) outs(%1 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
        return %2 : tensor<1x112x112x32xf32>
    }
}
"""

image = np.ones((1, 225, 225, 3), dtype=np.float32)
kernel = np.ones((3, 3, 3, 32), dtype=np.float32)
for i in range(0, 3):
    for j in range(0, 3):
        for k in range(0, 3):
            for c in range(0, 32):
                kernel[i, j, k, c] = i * 288 + j * 96 + k * 32 + c


def gpu_run():
    gpu_compiled = ireec.compile_str(
        convmlir,
        target_backends=["cuda"],
        extra_args=[
            "--iree-hal-cuda-llvm-target-arch=sm_70",
            # "--mlir-print-ir-after=iree-codegen-iree-comprehensive-bufferize",
        ],
    )
    vm_module = ireert.load_vm_flatbuffer(gpu_compiled, driver="cuda")
    f = vm_module.main
    result = f(image, kernel).to_host()
    return result


def cpu_run():
    cpu_compiled = ireec.compile_str(
        convmlir,
        target_backends=["llvm-cpu"],
    )
    vm_module = ireert.load_vm_flatbuffer(cpu_compiled, driver="local-task")
    f = vm_module.main
    result = f(image, kernel).to_host()
    return result


if __name__ == "__main__":
    g = gpu_run()
    c = cpu_run()
    print("Result shapes:", g.shape, c.shape)
    print(np.sum(np.abs(g - c)))
    print(g[0][1][0])
    print(c[0][1][0])
    assert np.allclose(g, c)

```

### What component(s) does this issue relate to?

Compiler

### Version information

e8b2400

### Additional context

The problem may happend at iree-codegen-iree-comprehensive-bufferize pass. After bufferization, tensors are bufferized to memrefs. Here is the dumped mlir.
```mlir
// -----// IR Dump After IREEComprehensiveBufferize (iree-codegen-iree-comprehensive-bufferize) //----- //
module {
  func.func @main_dispatch_0_conv_2d_nhwc_hwcf_1x112x112x32x3x3x3_f32() {
    %c2 = arith.constant 2 : index loc(unknown)
    %cst = arith.constant dense<0.000000e+00> : vector<1x2x2x4xf32> loc(unknown)
    %c3 = arith.constant 3 : index loc(unknown)
    %c1 = arith.constant 1 : index loc(unknown)
    %cst_0 = arith.constant 0.000000e+00 : f32 loc(unknown)
    %c0 = arith.constant 0 : index loc(unknown)
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<1x225x225x3xf32, #hal.descriptor_type<storage_buffer>> loc("<stdin>":3:5)
    memref.assume_alignment %0, 64 : memref<1x225x225x3xf32, #hal.descriptor_type<storage_buffer>> loc("<stdin>":3:5)
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<3x3x3x32xf32, #hal.descriptor_type<storage_buffer>> loc("<stdin>":3:5)
    memref.assume_alignment %1, 64 : memref<3x3x3x32xf32, #hal.descriptor_type<storage_buffer>> loc("<stdin>":3:5)
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<1x112x112x32xf32, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
    memref.assume_alignment %2, 64 : memref<1x112x112x32xf32, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
    %workgroup_id_z = hal.interface.workgroup.id[2] : index loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
    %3 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_z] loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
    %workgroup_id_y = hal.interface.workgroup.id[1] : index loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
    %4 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_y] loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
    %workgroup_id_x = hal.interface.workgroup.id[0] : index loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
    %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x] loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
    %subview = memref.subview %2[0, %3, %4, %5] [1, 4, 4, 32] [1, 1, 1, 1] : memref<1x112x112x32xf32, #hal.descriptor_type<storage_buffer>> to memref<1x4x4x32xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
    %6 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_z] loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
    %7 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_y] loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
    %subview_1 = memref.subview %0[0, %6, %7, 0] [1, 9, 9, 3] [1, 1, 1, 1] : memref<1x225x225x3xf32, #hal.descriptor_type<storage_buffer>> to memref<1x9x9x3xf32, strided<[151875, 675, 3, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
    scf.forall (%arg0, %arg1, %arg2) in (2, 2, 8) {
      %8 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0) loc(callsite("<stdin>":6:14 at "<stdin>":3:5))
      %9 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1) loc(callsite("<stdin>":6:14 at "<stdin>":3:5))
      %10 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2) loc(callsite("<stdin>":6:14 at "<stdin>":3:5))
      %subview_3 = memref.subview %subview[0, %8, %9, %10] [1, 2, 2, 4] [1, 1, 1, 1] : memref<1x4x4x32xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<1x2x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":6:14 at "<stdin>":3:5))
      vector.transfer_write %cst, %subview_3[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x2x2x4xf32>, memref<1x2x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(unknown)
      %subview_4 = memref.subview %subview[0, %8, %9, %10] [1, 2, 2, 4] [1, 1, 1, 1] : memref<1x4x4x32xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<1x2x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":6:14 at "<stdin>":3:5))
      memref.copy %subview_3, %subview_4 : memref<1x2x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<1x2x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":6:14 at "<stdin>":3:5))
    } {mapping = [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>]} loc(callsite("<stdin>":6:14 at "<stdin>":3:5))
    scf.forall (%arg0, %arg1, %arg2) in (2, 2, 8) {
      %8 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg0) loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      %9 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg1) loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      %10 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2) loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0) loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      %12 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1) loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      %subview_3 = memref.subview %subview[0, %11, %12, %10] [1, 2, 2, 4] [1, 1, 1, 1] : memref<1x4x4x32xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<1x2x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      %alloc = memref.alloc() : memref<1x1x2x4xf32, #gpu.address_space<workgroup>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      %subview_4 = memref.subview %subview_3[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : memref<1x2x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<1x1x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      %13 = arith.addi %11, %c1 : index loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      %14 = vector.transfer_read %subview[%c0, %13, %12, %10], %cst_0 {in_bounds = [true, true, true]} : memref<1x4x4x32xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<1x2x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      %15 = vector.transfer_read %subview[%c0, %11, %12, %10], %cst_0 {in_bounds = [true, true, true]} : memref<1x4x4x32xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<1x2x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      %16:2 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %14, %arg5 = %15) -> (vector<1x2x4xf32>, vector<1x2x4xf32>) {
        %17:2 = scf.for %arg6 = %c0 to %c3 step %c1 iter_args(%arg7 = %arg5, %arg8 = %arg4) -> (vector<1x2x4xf32>, vector<1x2x4xf32>) {
          %18 = arith.addi %arg3, %8 : index loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %19 = arith.addi %arg6, %9 : index loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %20 = vector.transfer_read %subview_1[%c0, %18, %19, %c0], %cst_0 {in_bounds = [true, true, true]} : memref<1x9x9x3xf32, strided<[151875, 675, 3, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<1x3x3xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %21 = vector.transfer_read %1[%arg3, %arg6, %c0, %10], %cst_0 {in_bounds = [true, true, true]} : memref<3x3x3x32xf32, #hal.descriptor_type<storage_buffer>>, vector<1x3x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %22 = vector.extract_strided_slice %20 {offsets = [0, 0, 0], sizes = [1, 1, 3], strides = [1, 1, 1]} : vector<1x3x3xf32> to vector<1x1x3xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %23 = vector.extract_strided_slice %20 {offsets = [0, 2, 0], sizes = [1, 1, 3], strides = [1, 1, 1]} : vector<1x3x3xf32> to vector<1x1x3xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %24 = vector.extract %21[0] : vector<3x4xf32> from vector<1x3x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %25 = vector.extract_strided_slice %arg7 {offsets = [0, 0, 0], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<1x2x4xf32> to vector<1x1x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %26 = vector.extract_strided_slice %arg7 {offsets = [0, 1, 0], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<1x2x4xf32> to vector<1x1x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %27 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %24, %25 : vector<1x1x3xf32>, vector<3x4xf32> into vector<1x1x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %28 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %24, %26 : vector<1x1x3xf32>, vector<3x4xf32> into vector<1x1x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %29 = vector.insert_strided_slice %27, %arg7 {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<1x1x4xf32> into vector<1x2x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %30 = vector.insert_strided_slice %28, %29 {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<1x1x4xf32> into vector<1x2x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %31 = arith.addi %arg3, %c2 : index loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %32 = arith.addi %31, %8 : index loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %33 = vector.transfer_read %subview_1[%c0, %32, %19, %c0], %cst_0 {in_bounds = [true, true, true]} : memref<1x9x9x3xf32, strided<[151875, 675, 3, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<1x3x3xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %34 = vector.extract_strided_slice %33 {offsets = [0, 0, 0], sizes = [1, 1, 3], strides = [1, 1, 1]} : vector<1x3x3xf32> to vector<1x1x3xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %35 = vector.extract_strided_slice %33 {offsets = [0, 2, 0], sizes = [1, 1, 3], strides = [1, 1, 1]} : vector<1x3x3xf32> to vector<1x1x3xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %36 = vector.extract_strided_slice %arg8 {offsets = [0, 0, 0], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<1x2x4xf32> to vector<1x1x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %37 = vector.extract_strided_slice %arg8 {offsets = [0, 1, 0], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<1x2x4xf32> to vector<1x1x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %38 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %34, %24, %36 : vector<1x1x3xf32>, vector<3x4xf32> into vector<1x1x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %39 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %35, %24, %37 : vector<1x1x3xf32>, vector<3x4xf32> into vector<1x1x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %40 = vector.insert_strided_slice %38, %arg8 {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<1x1x4xf32> into vector<1x2x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          %41 = vector.insert_strided_slice %39, %40 {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<1x1x4xf32> into vector<1x2x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
          scf.yield %30, %41 : vector<1x2x4xf32>, vector<1x2x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
        } loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
        scf.yield %17#1, %17#0 : vector<1x2x4xf32>, vector<1x2x4xf32> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      } loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      vector.transfer_write %16#1, %subview_4[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x2x4xf32>, memref<1x1x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      vector.transfer_write %16#0, %alloc[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x2x4xf32>, memref<1x1x2x4xf32, #gpu.address_space<workgroup>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      %subview_5 = memref.subview %subview_3[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : memref<1x2x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<1x1x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      memref.copy %subview_4, %subview_5 : memref<1x1x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<1x1x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      %subview_6 = memref.subview %subview_3[0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : memref<1x2x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<1x1x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      gpu.barrier loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      memref.copy %alloc, %subview_6 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<1x1x2x4xf32, #gpu.address_space<workgroup>> to memref<1x1x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      gpu.barrier loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      %subview_7 = memref.subview %subview[0, %11, %12, %10] [1, 2, 2, 4] [1, 1, 1, 1] : memref<1x4x4x32xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<1x2x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
      memref.copy %subview_3, %subview_7 : memref<1x2x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<1x2x2x4xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
    } {mapping = [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>]} loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
    %subview_2 = memref.subview %2[0, %3, %4, %5] [1, 4, 4, 32] [1, 1, 1, 1] : memref<1x112x112x32xf32, #hal.descriptor_type<storage_buffer>> to memref<1x4x4x32xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
    memref.copy %subview, %subview_2 : memref<1x4x4x32xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<1x4x4x32xf32, strided<[401408, 3584, 32, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
    return loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
  } loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
} loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
```

Note this memref.alloc
```mlir
%alloc = memref.alloc() : memref<1x1x2x4xf32, #gpu.address_space<workgroup>> loc(callsite("<stdin>":7:14 at "<stdin>":3:5))
```
It allocs a share memory containing only 8 elements. And every thread will write into all 8 elements, causing parallel write conflicts, the result may be undefined. As the `gpuAllocationFn` in `iree/compiler/Codegen/LLVMGPU/Passes.cpp` shows, all gpu allocation will be share memroy but it's not correct here.

Actually, this allocation should be `#gpu.address_space<private>` to make the result correct or maybe this allocation should never appear at all?

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises iREE CUDA lowering of linalg.conv_2d_nhwc_hwcf with temporary accumulator storage.

## Observed implementation decision

Bufferization allocates a temporary in workgroup memory even though parallel threads use it as private accumulation state.

## Physical effect

Threads concurrently write the same eight shared elements and later consume corrupted accumulators.

## Observed consequence

Alternating convolution rows repeat four channels and disagree with the CPU result.

## Reported proposal or workaround

The report suggests private memory or eliminating the allocation.

## Repair trace

No maintainer-confirmed invalid decision, landed allocation repair, or fixed regression is linked.

## Archived evidence

- Raw report: `data/raw/github/iree/16175/report.json`
- SHA-256: `cc95c0452113326774f13dc2f123edaff264f212a56d5427cb6d7440c01ff1e2`
