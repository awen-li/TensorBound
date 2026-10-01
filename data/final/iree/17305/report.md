# [GPU] Garbage values in output when unused inputs are present

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/17305](https://github.com/iree-org/iree/issues/17305)
- Opened: 2024-05-08T15:24:57Z
- Closed: Open at collection
- Labels: `bug 🐞`

## Reported behavior

### What happened?

I have an mlir input that I compile using `iree-compile` and run via rocm using `iree-run-module` where the number of inputs messes with the output of the program even if it is unused.

I include a reproducer below which takes an unused input (arg0) and writes the constant 1.0 into the output.
When executed, the reproducer produces only garbage values. Commenting in the 4 lines to use arg0 and copy some values into shared memory makes the output work as expected. The input is not used in any other way.

Removing arg0 from the program entirely, such that there are no inputs at all makes the program work as well.
This makes me think that the handling of the output buffer in the dispatch somehow depends on the input buffers and their usage.


### Steps to reproduce your issue

This is my input IR:
```mlir
#translation = #iree_codegen.translation_info<None workgroup_size = [128, 2, 1] subgroup_size = 64>
module {
  stream.executable private @func {
    stream.executable.export public @func workgroups() -> (index, index, index) {
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      stream.return %c2, %c2, %c1 : index, index, index
    }
    builtin.module {
      func.func @func(%arg0: !stream.binding,  %arg2: !stream.binding) attributes {translation_info = #translation} {
        %c19 = arith.constant 19 : index
        %c18 = arith.constant 18 : index
        %c17 = arith.constant 17 : index
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c64 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %c16 = arith.constant 16 : index
        %c0 = arith.constant 0 : index
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y

        // Commenting in these 4 lines makes the program work
        // %alloc_0 = memref.alloc() : memref<4x4xf32, #gpu.address_space<workgroup>>
        // %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<128x256xf32>
        // %2 = vector.load %0[%c0, %c0] : memref<128x256xf32>, vector<4xf32>
        // vector.store %2, %alloc_0[%c0, %c0] : memref<4x4xf32, #gpu.address_space<workgroup>>, vector<4xf32>

        // writing 1.0 to all positions in the output
        %cst = arith.constant dense<1.000000e+00> : vector<1xf32>
        %4 = arith.divsi %thread_id_x, %c64 : index
        %5 = arith.muli %4, %c32 : index
        %6 = arith.muli %workgroup_id_0, %c64 : index
        %7 = arith.remsi %thread_id_x, %c16 : index
        %8 = arith.remsi %thread_id_x, %c64 : index
        %9 = arith.divsi %8, %c16 : index
        %10 = arith.muli %9, %c4 : index
        %11 = arith.muli %thread_id_y, %c32 : index
        %12 = arith.muli %workgroup_id_1, %c64 : index
        %13 = arith.addi %7, %12 : index
        %14 = arith.addi %13, %11 : index
        %15 = arith.addi %14, %c16 : index
        %16 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<128x128xf32>
        %17 = arith.addi %6, %5 : index
        %18 = arith.addi %17, %10 : index
        vector.store %cst, %16[%18, %14] : memref<128x128xf32>, vector<1xf32>
        %19 = arith.addi %18, %c1 : index
        vector.store %cst, %16[%19, %14] : memref<128x128xf32>, vector<1xf32>
        %20 = arith.addi %18, %c2 : index
        vector.store %cst, %16[%20, %14] : memref<128x128xf32>, vector<1xf32>
        %21 = arith.addi %18, %c3 : index
        vector.store %cst, %16[%21, %14] : memref<128x128xf32>, vector<1xf32>
        vector.store %cst, %16[%18, %15] : memref<128x128xf32>, vector<1xf32>
        vector.store %cst, %16[%19, %15] : memref<128x128xf32>, vector<1xf32>
        vector.store %cst, %16[%20, %15] : memref<128x128xf32>, vector<1xf32>
        vector.store %cst, %16[%21, %15] : memref<128x128xf32>, vector<1xf32>
        %22 = arith.addi %18, %c16 : index
        vector.store %cst, %16[%22, %14] : memref<128x128xf32>, vector<1xf32>
        %23 = arith.addi %18, %c17 : index
        vector.store %cst, %16[%23, %14] : memref<128x128xf32>, vector<1xf32>
        %24 = arith.addi %18, %c18 : index
        vector.store %cst, %16[%24, %14] : memref<128x128xf32>, vector<1xf32>
        %25 = arith.addi %18, %c19 : index
        vector.store %cst, %16[%25, %14] : memref<128x128xf32>, vector<1xf32>
        vector.store %cst, %16[%22, %15] : memref<128x128xf32>, vector<1xf32>
        vector.store %cst, %16[%23, %15] : memref<128x128xf32>, vector<1xf32>
        vector.store %cst, %16[%24, %15] : memref<128x128xf32>, vector<1xf32>
        vector.store %cst, %16[%25, %15] : memref<128x128xf32>, vector<1xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<128x256xf32>) -> tensor<128x128xf32> {
    %2 = flow.dispatch @func::@func(%arg0) : (tensor<128x256xf32>) -> tensor<128x128xf32>
    return %2 : tensor<128x128xf32>
  }
}
```

Compiled using:
```
iree-compile example.mlir \
    --iree-hal-target-backends=rocm \
    -o example.vmfb
```

Ran using:
```
iree-run-module \
  --device=rocm://0 \
  --device_allocator=caching \
  --module=example.vmfb \
  --function=isolated_benchmark \
  --input=4xf32 \
```

### What component(s) does this issue relate to?

_No response_

### Version information

Built from commit161be85aeb4b5003a1a19053967c35f2bd00c762

### Additional context

_No response_

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

[GPU] Garbage values in output when unused inputs are present

## Observed implementation decision

LLVMGPU function-signature construction derives arguments from used kernel values instead of the declared pipeline binding layout.

## Physical effect

ROCm lowering omits an unused input from the generated LLVM function even though the pipeline layout and caller retain both buffer bindings, shifting the packed argument layout seen by the output binding.

## Observed consequence

A kernel that writes constant ones produces garbage only when an unused input remains in the dispatch interface.

## Reported proposal or workaround

The discussion isolates the mismatched two-binding layout and one-argument LLVM function and contrasts correct SPIR-V binding behavior; no fix is archived.

## Repair trace

The mismatched dispatch layout and LLVM function signature are shown, but the correcting source change and fixed regression are unavailable.

## Archived evidence

- Raw report: `data/raw/github/iree/17305/report.json`
- SHA-256: `765a05fb69d73b1f773ac841dc1f047c53ea6fe7ab5a887773ef1f18bf6334b6`
