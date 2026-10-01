# CUDA_ERROR_MISALIGNED_ADDRESS on Broadcast -> Dot when num_stages > 2

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/7386](https://github.com/triton-lang/triton/issues/7386)
- Opened: 2025-07-03T12:41:21Z
- Closed: open at collection time
- Labels: `bug`

## Reported behavior

### Describe the bug

I have a simple kernel that is doing a broadcast on the LHS of a Dot that is running into a misaligned address when using TMA and the num_stages = 3 or 4. The kernel works fine if I either use no pipelining or num_stages = 2. It would also work with num_stages > 2 without TMA. I provided below the TTIR and IR before going to LLVM.

Let me know if you need additional artifacts or if I can try to modify something for you. Note that the tma_descriptor attribute attached to the arguments can be ignored (I add them to later use it to construct the tensor maps).

### Failing Case (num_stages = 3)
**This is the top-level TTIR**
```
module {
  tt.func @micro_kernel_impl(%arg0: !tt.tensordesc<tensor<16x16xf16>> {tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [256, 512], tile_shape = [16, 16], tile_strides = [1, 1], layout = [1, 0], element_byte_size = 2>}, %arg1: !tt.tensordesc<tensor<16xf16>> {tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [256], tile_shape = [16], tile_strides = [1], layout = [0], element_byte_size = 2>}, %arg2: !tt.tensordesc<tensor<16x16xf16>> {tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [512, 128], tile_shape = [16, 16], tile_strides = [1, 1], layout = [1, 0], element_byte_size = 2>}) {
    %c8_i64 = arith.constant 8 : i64
    %c128_i64 = arith.constant 128 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = scf.for %arg3 = %c0_i64 to %c16_i64 step %c1_i64 iter_args(%arg4 = %cst) -> (tensor<16x16xf32>)  : i64 {
      %12 = arith.muli %1, %c16_i64 : i64
      %13 = arith.addi %12, %arg3 : i64
      %14 = arith.remui %13, %c16_i64 : i64
      %15 = arith.muli %14, %c16_i64 : i64
      %16 = arith.index_castui %15 : i64 to index
      %17 = arith.divui %13, %c128_i64 : i64
      %18 = arith.muli %17, %c16_i64 : i64
      %19 = arith.index_castui %18 : i64 to index
      %20 = arith.index_castui %16 : index to i32
      %21 = arith.index_castui %19 : index to i32
      %22 = tt.descriptor_load %arg0[%20, %21] : !tt.tensordesc<tensor<16x16xf16>> -> tensor<16x16xf16>
      %23 = arith.index_castui %16 : index to i32
      %24 = tt.descriptor_load %arg1[%23] : !tt.tensordesc<tensor<16xf16>> -> tensor<16xf16>
      %25 = tt.expand_dims %24 {axis = 0 : i32} : tensor<16xf16> -> tensor<1x16xf16>
      %26 = tt.broadcast %25 : tensor<1x16xf16> -> tensor<16x16xf16>
      %27 = tt.trans %22 {order = array<i32: 1, 0>} : tensor<16x16xf16> -> tensor<16x16xf16>
      %28 = tt.trans %26 {order = array<i32: 1, 0>} : tensor<16x16xf16> -> tensor<16x16xf16>
      %29 = tt.dot %27, %28, %arg4, inputPrecision = tf32 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %29 : tensor<16x16xf32>
    }
    %3 = arith.truncf %2 : tensor<16x16xf32> to tensor<16x16xf16>
    %4 = arith.divui %1, %c8_i64 : i64
    %5 = arith.muli %4, %c16_i64 : i64
    %6 = arith.index_castui %5 : i64 to index
    %7 = arith.remui %1, %c8_i64 : i64
    %8 = arith.muli %7, %c16_i64 : i64
    %9 = arith.index_castui %8 : i64 to index
    %10 = arith.index_castui %6 : index to i32
    %11 = arith.index_castui %9 : index to i32
    tt.descriptor_store %arg2[%10, %11], %3 : !tt.tensordesc<tensor<16x16xf16>>, tensor<16x16xf16>
    tt.return
  }
}
```

**This is the IR before ConvertTritonGPUToLLVM**
```
// -----// IR Dump Before ConvertTritonGPUToLLVM (convert-triton-gpu-to-llvm) ('builtin.module' operation) //----- //
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 8]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[8, 0]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 2], instrShape = [16, 8]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = true, elementBitWidth = 16, CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.shared = 1120 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 2 : i32} {
  tt.func @micro_kernel_impl(%arg0: !tt.tensordesc<tensor<16x16xf16, #shared>> {tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [256, 512], tile_shape = [16, 16], tile_strides = [1, 1], layout = [1, 0], element_byte_size = 2, swizzle_mode = "32b">}, %arg1: !tt.tensordesc<tensor<16xf16, #shared1>> {tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [256], tile_shape = [16], tile_strides = [1], layout = [0], element_byte_size = 2, swizzle_mode = "none">}, %arg2: !tt.tensordesc<tensor<16x16xf16, #shared>> {tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [512, 128], tile_shape = [16, 16], tile_strides = [1, 1], layout = [1, 0], element_byte_size = 2, swizzle_mode = "32b">}) attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %c2_i64 = arith.constant 2 : i64
    %c14_i64 = arith.constant 14 : i64
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c8_i64 = arith.constant 8 : i64
    %c128_i64 = arith.constant 128 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.muli %1, %c16_i64 : i64
    %3 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x16x16xf16, #shared, #smem, mutable>
    %4 = ttg.local_alloc {allocation.offset = 1024 : i32} : () -> !ttg.memdesc<2x16xf16, #shared1, #smem, mutable>
    %5 = ttg.local_alloc {allocation.offset = 1088 : i32} : () -> !ttg.memdesc<2xi64, #shared2, #smem, mutable>
    %6 = ttg.memdesc_subview %5[%c0_i32] : !ttg.memdesc<2xi64, #shared2, #smem, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    ttng.init_barrier %6, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    %7 = ttg.memdesc_subview %5[%c1_i32] : !ttg.memdesc<2xi64, #shared2, #smem, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    ttng.init_barrier %7, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    %8 = ttg.local_alloc {allocation.offset = 1104 : i32} : () -> !ttg.memdesc<2xi64, #shared2, #smem, mutable>
    %9 = ttg.memdesc_subview %8[%c0_i32] : !ttg.memdesc<2xi64, #shared2, #smem, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    ttng.init_barrier %9, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    %10 = ttg.memdesc_subview %8[%c1_i32] : !ttg.memdesc<2xi64, #shared2, #smem, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    ttng.init_barrier %10, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    %11 = arith.remui %2, %c16_i64 : i64
    %12 = arith.muli %11, %c16_i64 : i64
    %13 = arith.index_castui %12 : i64 to index
    %14 = arith.divui %2, %c128_i64 : i64
    %15 = arith.muli %14, %c16_i64 : i64
    %16 = arith.index_castui %15 : i64 to index
    %17 = arith.index_castui %13 : index to i32
    %18 = arith.index_castui %16 : index to i32
    ttng.barrier_expect %6, 512, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    %19 = ttg.memdesc_subview %3[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 2x16x16>
    %20 = ttng.tensor_desc_to_tma_ptr %arg0 : !tt.tensordesc<tensor<16x16xf16, #shared>> to !tt.ptr<i8>
    ttng.async_tma_copy_global_to_local %20[%17, %18] %19, %6, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 2x16x16>
    ttng.barrier_expect %9, 32, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    %21 = ttg.memdesc_subview %4[%c0_i32, %c0_i32] : !ttg.memdesc<2x16xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16xf16, #shared1, #smem, mutable, 2x16>
    %22 = ttng.tensor_desc_to_tma_ptr %arg1 : !tt.tensordesc<tensor<16xf16, #shared1>> to !tt.ptr<i8>
    ttng.async_tma_copy_global_to_local %22[%17] %21, %9, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2> -> !ttg.memdesc<16xf16, #shared1, #smem, mutable, 2x16>
    %23 = arith.addi %2, %c1_i64 : i64
    %24 = arith.remui %23, %c16_i64 : i64
    %25 = arith.muli %24, %c16_i64 : i64
    %26 = arith.index_castui %25 : i64 to index
    %27 = arith.divui %23, %c128_i64 : i64
    %28 = arith.muli %27, %c16_i64 : i64
    %29 = arith.index_castui %28 : i64 to index
    %30 = arith.index_castui %26 : index to i32
    %31 = arith.index_castui %29 : index to i32
    ttng.barrier_expect %7, 512, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    %32 = ttg.memdesc_subview %3[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 2x16x16>
    ttng.async_tma_copy_global_to_local %20[%30, %31] %32, %7, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 2x16x16>
    ttng.barrier_expect %10, 32, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    %33 = ttg.memdesc_subview %4[%c1_i32, %c0_i32] : !ttg.memdesc<2x16xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16xf16, #shared1, #smem, mutable, 2x16>
    ttng.async_tma_copy_global_to_local %22[%30] %33, %10, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2> -> !ttg.memdesc<16xf16, #shared1, #smem, mutable, 2x16>
    cf.br ^bb1(%c0_i64, %cst, %c1_i32, %c-1_i32, %c0_i32 : i64, tensor<16x16xf32, #mma>, i32, i32, i32)
  ^bb1(%34: i64, %35: tensor<16x16xf32, #mma>, %36: i32, %37: i32, %38: i32):  // 2 preds: ^bb0, ^bb2
    %39 = arith.cmpi slt, %34, %c16_i64 : i64
    cf.cond_br %39, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %40 = arith.cmpi slt, %34, %c14_i64 : i64
    %41 = arith.addi %37, %c1_i32 : i32
    %42 = arith.cmpi sge, %41, %c2_i32 : i32
    %43 = arith.select %42, %c0_i32, %41 : i32
    %44 = arith.xori %38, %c1_i32 : i32
    %45 = arith.select %42, %44, %38 : i32
    %46 = ttg.memdesc_subview %5[%43] : !ttg.memdesc<2xi64, #shared2, #smem, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    ttng.wait_barrier %46, %45 : !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    %47 = ttg.memdesc_subview %3[%43, %c0_i32, %c0_i32] : !ttg.memdesc<2x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 2x16x16>
    %48 = ttg.local_load %47 : !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 2x16x16> -> tensor<16x16xf16, #linear>
    %49 = tt.trans %48 {order = array<i32: 1, 0>} : tensor<16x16xf16, #linear> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %50 = ttg.memdesc_subview %8[%43] : !ttg.memdesc<2xi64, #shared2, #smem, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    ttng.wait_barrier %50, %45 : !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    %51 = ttg.memdesc_subview %4[%43, %c0_i32] : !ttg.memdesc<2x16xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16xf16, #shared1, #smem, mutable, 2x16>
    %52 = ttg.local_load %51 : !ttg.memdesc<16xf16, #shared1, #smem, mutable, 2x16> -> tensor<16xf16, #ttg.slice<{dim = 0, parent = #linear1}>>
    %53 = tt.expand_dims %52 {axis = 0 : i32} : tensor<16xf16, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x16xf16, #linear1>
    %54 = tt.broadcast %53 : tensor<1x16xf16, #linear1> -> tensor<16x16xf16, #linear1>
    %55 = tt.trans %54 {order = array<i32: 1, 0>} : tensor<16x16xf16, #linear1> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %56 = tt.dot %49, %55, %35, inputPrecision = tf32 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
    %57 = arith.addi %36, %c1_i32 : i32
    %58 = arith.cmpi sge, %57, %c2_i32 : i32
    %59 = arith.select %58, %c0_i32, %57 : i32
    %60 = arith.addi %34, %c2_i64 : i64
    %61 = arith.addi %2, %60 : i64
    %62 = arith.remui %61, %c16_i64 : i64
    %63 = arith.muli %62, %c16_i64 : i64
    %64 = arith.index_castui %63 : i64 to index
    %65 = arith.divui %61, %c128_i64 : i64
    %66 = arith.muli %65, %c16_i64 : i64
    %67 = arith.index_castui %66 : i64 to index
    %68 = arith.index_castui %64 : index to i32
    %69 = arith.index_castui %67 : index to i32
    %70 = ttg.memdesc_subview %5[%59] : !ttg.memdesc<2xi64, #shared2, #smem, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    ttng.barrier_expect %70, 512, %40 : !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    %71 = ttg.memdesc_subview %3[%59, %c0_i32, %c0_i32] : !ttg.memdesc<2x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 2x16x16>
    ttng.async_tma_copy_global_to_local %20[%68, %69] %71, %70, %40 : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 2x16x16>
    %72 = ttg.memdesc_subview %8[%59] : !ttg.memdesc<2xi64, #shared2, #smem, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    ttng.barrier_expect %72, 32, %40 : !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    %73 = ttg.memdesc_subview %4[%59, %c0_i32] : !ttg.memdesc<2x16xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16xf16, #shared1, #smem, mutable, 2x16>
    ttng.async_tma_copy_global_to_local %22[%68] %73, %72, %40 : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2> -> !ttg.memdesc<16xf16, #shared1, #smem, mutable, 2x16>
    %74 = arith.addi %34, %c1_i64 : i64
    cf.br ^bb1(%74, %56, %59, %43, %45 : i64, tensor<16x16xf32, #mma>, i32, i32, i32)
  ^bb3:  // pred: ^bb1
    ttng.inval_barrier %9 : !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    ttng.inval_barrier %10 : !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    ttg.local_dealloc %8 : !ttg.memdesc<2xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %6 : !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    ttng.inval_barrier %7 : !ttg.memdesc<1xi64, #shared2, #smem, mutable, 2>
    ttg.local_dealloc %5 : !ttg.memdesc<2xi64, #shared2, #smem, mutable>
    ttg.local_dealloc %4 : !ttg.memdesc<2x16xf16, #shared1, #smem, mutable>
    ttg.local_dealloc %3 : !ttg.memdesc<2x16x16xf16, #shared, #smem, mutable>
    %75 = arith.truncf %35 : tensor<16x16xf32, #mma> to tensor<16x16xf16, #mma>
    %76 = arith.divui %1, %c8_i64 : i64
    %77 = arith.muli %76, %c16_i64 : i64
    %78 = arith.index_castui %77 : i64 to index
    %79 = arith.remui %1, %c8_i64 : i64
    %80 = arith.muli %79, %c16_i64 : i64
    %81 = arith.index_castui %80 : i64 to index
    %82 = arith.index_castui %78 : index to i32
    %83 = arith.index_castui %81 : index to i32
    %84 = ttg.local_alloc %75 {allocation.offset = 0 : i32} : (tensor<16x16xf16, #mma>) -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %85 = ttng.tensor_desc_to_tma_ptr %arg2 : !tt.tensordesc<tensor<16x16xf16, #shared>> to !tt.ptr<i8>
    ttng.async_tma_copy_local_to_global %85[%82, %83] %84 : !tt.ptr<i8>, !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}
```

### Passing Case (num_stages = 2):
**IR before going ConvertTritonGPUToLLVM**
```
// -----// IR Dump Before ConvertTritonGPUToLLVM (convert-triton-gpu-to-llvm) ('builtin.module' operation) //----- //
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 8]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[8, 0]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 2], instrShape = [16, 8]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = true, elementBitWidth = 16, CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.shared = 568 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 2 : i32} {
  tt.func @micro_kernel_impl(%arg0: !tt.tensordesc<tensor<16x16xf16, #shared>> {tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [256, 512], tile_shape = [16, 16], tile_strides = [1, 1], layout = [1, 0], element_byte_size = 2, swizzle_mode = "32b">}, %arg1: !tt.tensordesc<tensor<16xf16, #shared1>> {tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [256], tile_shape = [16], tile_strides = [1], layout = [0], element_byte_size = 2, swizzle_mode = "none">}, %arg2: !tt.tensordesc<tensor<16x16xf16, #shared>> {tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [512, 128], tile_shape = [16, 16], tile_strides = [1, 1], layout = [1, 0], element_byte_size = 2, swizzle_mode = "32b">}) attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %c15_i64 = arith.constant 15 : i64
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c8_i64 = arith.constant 8 : i64
    %c128_i64 = arith.constant 128 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.muli %1, %c16_i64 : i64
    %3 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable>
    %4 = ttg.local_alloc {allocation.offset = 512 : i32} : () -> !ttg.memdesc<1x16xf16, #shared1, #smem, mutable>
    %5 = ttg.local_alloc {allocation.offset = 544 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %6 = ttg.memdesc_subview %5[%c0_i32] : !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %6, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %7 = ttg.local_alloc {allocation.offset = 560 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %8 = ttg.memdesc_subview %7[%c0_i32] : !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %8, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %9 = arith.remui %2, %c16_i64 : i64
    %10 = arith.muli %9, %c16_i64 : i64
    %11 = arith.index_castui %10 : i64 to index
    %12 = arith.divui %2, %c128_i64 : i64
    %13 = arith.muli %12, %c16_i64 : i64
    %14 = arith.index_castui %13 : i64 to index
    %15 = arith.index_castui %11 : index to i32
    %16 = arith.index_castui %14 : index to i32
    ttng.barrier_expect %6, 512, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %17 = ttg.memdesc_subview %3[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 1x16x16>
    %18 = ttng.tensor_desc_to_tma_ptr %arg0 : !tt.tensordesc<tensor<16x16xf16, #shared>> to !tt.ptr<i8>
    ttng.async_tma_copy_global_to_local %18[%15, %16] %17, %6, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 1x16x16>
    ttng.barrier_expect %8, 32, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %19 = ttg.memdesc_subview %4[%c0_i32, %c0_i32] : !ttg.memdesc<1x16xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16xf16, #shared1, #smem, mutable, 1x16>
    %20 = ttng.tensor_desc_to_tma_ptr %arg1 : !tt.tensordesc<tensor<16xf16, #shared1>> to !tt.ptr<i8>
    ttng.async_tma_copy_global_to_local %20[%15] %19, %8, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<16xf16, #shared1, #smem, mutable, 1x16>
    cf.br ^bb1(%c0_i64, %cst, %c0_i32, %c-1_i32, %c0_i32 : i64, tensor<16x16xf32, #mma>, i32, i32, i32)
  ^bb1(%21: i64, %22: tensor<16x16xf32, #mma>, %23: i32, %24: i32, %25: i32):  // 2 preds: ^bb0, ^bb2
    %26 = arith.cmpi slt, %21, %c16_i64 : i64
    cf.cond_br %26, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %27 = arith.cmpi slt, %21, %c15_i64 : i64
    %28 = arith.addi %24, %c1_i32 : i32
    %29 = arith.cmpi sge, %28, %c1_i32 : i32
    %30 = arith.select %29, %c0_i32, %28 : i32
    %31 = arith.xori %25, %c1_i32 : i32
    %32 = arith.select %29, %31, %25 : i32
    %33 = ttg.memdesc_subview %5[%30] : !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.wait_barrier %33, %32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %34 = ttg.memdesc_subview %3[%30, %c0_i32, %c0_i32] : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 1x16x16>
    %35 = ttg.local_load %34 : !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 1x16x16> -> tensor<16x16xf16, #linear>
    %36 = tt.trans %35 {order = array<i32: 1, 0>} : tensor<16x16xf16, #linear> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %37 = ttg.memdesc_subview %7[%30] : !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.wait_barrier %37, %32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %38 = ttg.memdesc_subview %4[%30, %c0_i32] : !ttg.memdesc<1x16xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16xf16, #shared1, #smem, mutable, 1x16>
    %39 = ttg.local_load %38 : !ttg.memdesc<16xf16, #shared1, #smem, mutable, 1x16> -> tensor<16xf16, #ttg.slice<{dim = 0, parent = #linear1}>>
    %40 = tt.expand_dims %39 {axis = 0 : i32} : tensor<16xf16, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x16xf16, #linear1>
    %41 = tt.broadcast %40 : tensor<1x16xf16, #linear1> -> tensor<16x16xf16, #linear1>
    %42 = tt.trans %41 {order = array<i32: 1, 0>} : tensor<16x16xf16, #linear1> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %43 = tt.dot %36, %42, %22, inputPrecision = tf32 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
    %44 = arith.addi %23, %c1_i32 : i32
    %45 = arith.cmpi sge, %44, %c1_i32 : i32
    %46 = arith.select %45, %c0_i32, %44 : i32
    %47 = arith.addi %21, %c1_i64 : i64
    %48 = arith.addi %2, %47 : i64
    %49 = arith.remui %48, %c16_i64 : i64
    %50 = arith.muli %49, %c16_i64 : i64
    %51 = arith.index_castui %50 : i64 to index
    %52 = arith.divui %48, %c128_i64 : i64
    %53 = arith.muli %52, %c16_i64 : i64
    %54 = arith.index_castui %53 : i64 to index
    %55 = arith.index_castui %51 : index to i32
    %56 = arith.index_castui %54 : index to i32
    %57 = ttg.memdesc_subview %5[%46] : !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %57, 512, %27 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %58 = ttg.memdesc_subview %3[%46, %c0_i32, %c0_i32] : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 1x16x16>
    ttng.async_tma_copy_global_to_local %18[%55, %56] %58, %57, %27 : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable, 1x16x16>
    %59 = ttg.memdesc_subview %7[%46] : !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %59, 32, %27 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %60 = ttg.memdesc_subview %4[%46, %c0_i32] : !ttg.memdesc<1x16xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16xf16, #shared1, #smem, mutable, 1x16>
    ttng.async_tma_copy_global_to_local %20[%55] %60, %59, %27 : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<16xf16, #shared1, #smem, mutable, 1x16>
    %61 = arith.addi %21, %c1_i64 : i64
    cf.br ^bb1(%61, %43, %46, %30, %32 : i64, tensor<16x16xf32, #mma>, i32, i32, i32)
  ^bb3:  // pred: ^bb1
    ttng.inval_barrier %8 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttg.local_dealloc %7 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %6 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttg.local_dealloc %5 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttg.local_dealloc %4 : !ttg.memdesc<1x16xf16, #shared1, #smem, mutable>
    ttg.local_dealloc %3 : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable>
    %62 = arith.truncf %22 : tensor<16x16xf32, #mma> to tensor<16x16xf16, #mma>
    %63 = arith.divui %1, %c8_i64 : i64
    %64 = arith.muli %63, %c16_i64 : i64
    %65 = arith.index_castui %64 : i64 to index
    %66 = arith.remui %1, %c8_i64 : i64
    %67 = arith.muli %66, %c16_i64 : i64
    %68 = arith.index_castui %67 : i64 to index
    %69 = arith.index_castui %65 : index to i32
    %70 = arith.index_castui %68 : index to i32
    %71 = ttg.local_alloc %62 {allocation.offset = 0 : i32} : (tensor<16x16xf16, #mma>) -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %72 = ttng.tensor_desc_to_tma_ptr %arg2 : !tt.tensordesc<tensor<16x16xf16, #shared>> to !tt.ptr<i8>
    ttng.async_tma_copy_local_to_global %72[%69, %70] %71 : !tt.ptr<i8>, !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}
```

### Environment details

Triton: ec8cb09329cf25ac241a7dee1eea5a5d94daef8a
GPU: H100

## Trigger

A broadcast operand with a 32- or 64-byte per-stage tile is loaded through TMA in a kernel using more than one pipeline stage.

## Relevant entities

broadcast operand; per-stage tile; multi-buffered shared allocation; pipeline stage; TMA descriptor load; 128-byte alignment

## Implemented relation

Each logical pipeline stage maps to a shared-memory slice whose address must satisfy TMA's 128-byte alignment requirement.

## Pre-state

The base shared allocation is aligned, but one stage occupies fewer than 128 bytes.

## Binding transition

XLA permits TMA and Triton forms a multi-buffer by placing later stages at consecutive per-stage byte offsets.

## Post-state

Stage one or later begins at an address that is not a multiple of 128 bytes.

## Validity obligation

Every stage address passed to a TMA load is 128-byte aligned, not only the base allocation.

## Invalid decision

XLA's CanUseTma selected TMA without combining broadcast consumption, pipeline depth, and per-stage tile-byte alignment.

## Physical effect

A later descriptor load accesses a misaligned shared-memory address.

## Observed consequence

The reported kernel raises CUDA_ERROR_MISALIGNED_ADDRESS for three or four stages; related XLA search cases crash Triton's pipeliner.

## Repair site

xla/backends/gpu/codegen/triton/transforms/triton_xla_extract_insert_to_triton_pass.cc::CanUseTma

## Repair

XLA PR 47198 disables TMA when num_stages > 1, the operand feeds a broadcast, and tile_byte_size is not divisible by 128; tests retain TMA for aligned and unpipelined cases.

## Why the repair works

The affected XLA path uses non-TMA loads for precisely the operands whose later stage offsets would violate TMA alignment.

## Causal basis

Triton maintainers identify the 128-byte stage-offset failure, and the merged XLA patch encodes and tests that exact predicate while labeling itself temporary.

## Analysis implication

A staged-transfer check needs transfer mechanism, consumer shape operation, stage count, element size, tile volume, allocation alignment, and every stage offset.

## Fix patches

- [`fixes/xla-7386-pr47198.patch`](fixes/xla-7386-pr47198.patch) — https://github.com/openxla/xla/pull/47198
