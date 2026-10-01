# [Fusion] error codegen for fusing reduce_max and sub ops into a dispatch region

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/15666](https://github.com/iree-org/iree/issues/15666)
- Opened: 2023-11-22T13:29:53Z
- Closed: Open at collection
- Labels: `bug 🐞`

## Reported behavior

The conditions for fusing reduce_max and sub ops into a dispatch region are error because of missing barrier.

The following is the corresponding input shape and compute(input tensor shape is [3,12,1024,1024], dim = -1):
`
x_max = max(input, dim, keepdim = True)
result = input - x_max
`

If a location in shared memory is written by one thread as result data, another thread reads data at the same location as input. it is error, when the two threads come from different warps.

This error has been verified through a set of data.

Reducemax and sub are not merged, or the default number of threads is limited to 32, the error will be fixed.

After LLVMGPUVectorLowering we end up with an IR like:

`%36`(shared memory) is the result to store `x_max`, and  `%37` will load it.

// -----// IR Dump After LLVMGPUVectorLowering (iree-llvmgpu-vector-lowering) //----- //
func.func @main_dispatch_0_generic_3x12x1024x1024_f32() {
  %c0 = arith.constant 0 : index
  %c-1024 = arith.constant -1024 : index
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  %c-1 = arith.constant -1 : index
  %c256 = arith.constant 256 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c4 = arith.constant 4 : index
  %c1024 = arith.constant 1024 : index
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<3x12x1024x1024xf32, #gpu.address_space<global>>
  memref.assume_alignment %0, 64 : memref<3x12x1024x1024xf32, #gpu.address_space<global>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<3x12x1024x1024xf32, #gpu.address_space<global>>
  memref.assume_alignment %1, 64 : memref<3x12x1024x1024xf32, #gpu.address_space<global>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %2 = arith.cmpi slt, %workgroup_id_x, %c0 : index
  %3 = arith.subi %c-1, %workgroup_id_x : index
  %4 = arith.select %2, %3, %workgroup_id_x : index
  %5 = arith.divsi %4, %c256 : index
  %6 = arith.subi %c-1, %5 : index
  %7 = arith.select %2, %6, %5 : index
  %8 = arith.cmpi slt, %7, %c0 : index
  %9 = arith.subi %c-1, %7 : index
  %10 = arith.select %8, %9, %7 : index
  %11 = arith.divsi %10, %c16 : index
  %12 = arith.subi %c-1, %11 : index
  %13 = arith.select %8, %12, %11 : index
  %14 = arith.cmpi slt, %13, %c0 : index
  %15 = arith.subi %c-1, %13 : index
  %16 = arith.select %14, %15, %13 : index
  %17 = arith.divsi %16, %c12 : index
  %18 = arith.subi %c-1, %17 : index
  %19 = arith.select %14, %18, %17 : index
  %20 = arith.remsi %13, %c12 : index
  %21 = arith.cmpi slt, %20, %c0 : index
  %22 = arith.addi %20, %c12 : index
  %23 = arith.select %21, %22, %20 : index
  %24 = arith.muli %7, %c64 : index
  %25 = arith.muli %13, %c-1024 : index
  %26 = arith.addi %24, %25 : index
  %27 = arith.muli %workgroup_id_x, %c4 : index
  %28 = arith.muli %7, %c-1024 : index
  %29 = arith.addi %27, %28 : index
  %alloc = memref.alloc() : memref<1x1x64xf32, #gpu.address_space<workgroup>>
  %30 = gpu.thread_id  x
  memref.store %cst, %alloc[%c0, %c0, %30] : memref<1x1x64xf32, #gpu.address_space<workgroup>>
  %alloc_0 = memref.alloc() : memref<1x1x64xi64, #gpu.address_space<workgroup>>
  memref.store %c0_i64, %alloc_0[%c0, %c0, %30] : memref<1x1x64xi64, #gpu.address_space<workgroup>>
  scf.for %arg0 = %c0 to %c1024 step %c4 {
    scf.for %arg1 = %c0 to %c4 step %c1 {
      %33 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%26, %30]
      %34 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg0, %arg1]
      %35 = memref.load %0[%19, %23, %33, %34] : memref<3x12x1024x1024xf32, #gpu.address_space<global>>
      %36 = memref.load %alloc[%c0, %c0, %30] : memref<1x1x64xf32, #gpu.address_space<workgroup>>
      %37 = memref.load %alloc_0[%c0, %c0, %30] : memref<1x1x64xi64, #gpu.address_space<workgroup>>
      %38 = arith.addi %arg1, %arg0 : index
      %39 = arith.index_cast %38 : index to i64
      %40 = arith.cmpf oge, %36, %35 : f32
      %41 = arith.select %40, %36, %35 : f32
      %42 = arith.cmpf oeq, %36, %35 : f32
      %43 = arith.minsi %37, %39 : i64
      %44 = arith.select %40, %37, %39 : i64
      %45 = arith.select %42, %43, %44 : i64
      memref.store %41, %alloc[%c0, %c0, %30] : memref<1x1x64xf32, #gpu.address_space<workgroup>>
      memref.store %45, %alloc_0[%c0, %c0, %30] : memref<1x1x64xi64, #gpu.address_space<workgroup>>
    }
  }
  %31 = gpu.thread_id  y
  %32 = arith.muli %31, %c64 : index
  scf.for %arg0 = %30 to %c4 step %c64 {
    scf.for %arg1 = %c0 to %c64 step %c1 {
      %33 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%26, %32, %arg1]
      %34 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%29, %arg0]
      %35 = memref.load %0[%19, %23, %33, %34] : memref<3x12x1024x1024xf32, #gpu.address_space<global>>
      %36 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%32, %arg1]
      %37 = memref.load %alloc[%c0, %c0, %36] : memref<1x1x64xf32, #gpu.address_space<workgroup>>
      %38 = arith.subf %35, %37 : f32
      %39 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%26, %32, %arg1]
      %40 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%29, %arg0]
      memref.store %38, %1[%19, %23, %39, %40] : memref<3x12x1024x1024xf32, #gpu.address_space<global>>
    }
  }
  return
}

### Steps to reproduce your issue

1、through torch.compile and torch mlir to get stablehlo ir
```
class SimpleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, mat):
        result = torch.max(mat, -1, keepdim=True)
        res = mat - result[0]
        return res
device = 'cuda'
optimized_mod = torch.compile(mod, backend="grace")
mat = torch.randn(3, 12, 1024, 1024).to(device)
res = optimized_mod(mat)
```
2、using stablehlo ir as iree input ir and mat as iree input tensor.
3、By printing ir after LLVMGPUVectorLowering or dumping ptx code for further analysis, you will find that barrier is missing.

### What component(s) does this issue relate to?

Compiler

### Version information

_No response_
python -m pip list| grep -E "iree|torch"
iree commit id: bb51f6f1a1b4ee619fb09a7396f449dadb211447
pytorch               2.1.0+cu118
torch-mlir commit id: ff7f8b21dcc842a4f70209a6d255d54c4ef6e39b

### Additional context

_No response_

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises iREE fusion of a reduction and dependent subtraction across multiple GPU warps.

## Observed implementation decision

Lowering emits a workgroup-memory producer and consumer without a barrier between cross-warp writes and reads.

## Physical effect

Consumer warps can read shared-memory reduction values before producer writes become visible.

## Observed consequence

The fused reduce_max/sub computation returns wrong values.

## Reported proposal or workaround

Preventing fusion or restricting execution to one warp avoids the failure.

## Repair trace

The issue has no linked landed barrier insertion, fusion guard, or fixed regression.

## Archived evidence

- Raw report: `data/raw/github/iree/15666/report.json`
- SHA-256: `0df1bc6000264d9ac7554eca4a489647030314f17b2d20ecc6a09916e28766f7`
