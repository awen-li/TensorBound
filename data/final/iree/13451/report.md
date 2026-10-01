# `--iree-codegen-llvmgpu-enable-transform-dialect-matmul-tensorcore-strategy` miscompiles with mma-sync

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/13451](https://github.com/iree-org/iree/issues/13451)
- Opened: 2023-05-08T14:09:53Z
- Closed: open at collection time
- Labels: `bug 🐞`

## Reported behavior

### What happened?

When compiling for cuda with `--iree-codegen-llvmgpu-enable-transform-dialect-matmul-tensorcore-strategy` and `--td-matmul-strategy-use-mma-sync` I run into miscompiles.

### Steps to reproduce your issue

Using the following IR:
```mlir
module {
  func.func @fill_matmul_static(%arg0: tensor<3452x2044xf32>, %arg1: tensor<2044x1024xf32>) -> tensor<3452x1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<3452x1024xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3452x1024xf32>) -> tensor<3452x1024xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<3452x2044xf32>, tensor<2044x1024xf32>) outs(%1 : tensor<3452x1024xf32>) -> tensor<3452x1024xf32>
    return %2 : tensor<3452x1024xf32>
  }
}
```

Compile with:
```
iree-compile --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 --td-matmul-strategy-use-mma-sync  --iree-codegen-llvmgpu-enable-transform-dialect-matmul-tensorcore-strategy -o <out>.vmfb <in>.mlir
```

Run:
```
iree-run-module --function=fill_matmul_static --device=cuda  --module=<out>.vmfb --input=3452x2044xf32=1 --input=2044x1024xf32=1
```

Results:
```
3452x1024xf32=[2012 2012 2012 2012 2012 2012 ...
```

Expected results:
```
3452x1024xf32=[2044 2044 2044 2044 2044 2044 ...
```

I get the expected results if I do one of these things on the `iree-compile` command line:
- Disable the td strategy all together: `--iree-codegen-llvmgpu-enable-transform-dialect-matmul-tensorcore-strategy=false` (hence, instead of `--iree-codegen-llvmgpu-enable-transform-dialect-matmul-tensorcore-strategy=true`)
- Disable mma sync: `--td-matmul-strategy-use-mma-sync=false` (hence, instead `--td-matmul-strategy-use-mma-sync=true`)
- Disable the async copies: `-td-matmul-strategy-use-async-copies=false`

### What component(s) does this issue relate to?

_No response_

### Version information

_No response_

### Additional context

_No response_

## Trigger

An unaligned matmul leaves one shared-memory operand outside effective multibuffering while pipeline depth exceeds one.

## Relevant entities

global-to-shared copies; shared-memory allocation; ldmatrix; mma.sync; software-pipeline schedule

## Implemented relation

Every shared-memory value consumed by a pipelined tensor-core iteration belongs to the matching multibuffer stage.

## Pre-state

The loop body has one logical shared-memory tile per operand and a correct sequential execution order.

## Binding transition

The strategy constructs a multi-stage schedule even though one ldmatrix source did not acquire a stage dimension.

## Post-state

Loads and asynchronous copies for different K iterations address the same physical shared-memory slot.

## Validity obligation

Pipelining proceeds only when the schedule contains the required copies, waits, barriers, warp operations, and stage-separated buffers.

## Invalid decision

The old strategy allowed a pipeline schedule after multibuffering failed to separate every participating shared-memory operand.

## Physical effect

A later iteration overwrites a tile before the corresponding mma.sync consumes it.

## Observed consequence

The reported matmul returns values short by 16 times pipeline_depth minus one, including 2012 instead of 2044.

## Repair site

compiler/src/iree/compiler/Codegen/Common/GPU/GPUPipelining.cpp and the tensor-core transform strategy

## Repair

PR 13971 makes the mma.sync schedule explicit, rejects incomplete scheduling ingredients, strengthens strategy validation, and adds conservative size and depth guards.

## Why the repair works

A loop either receives a complete stage-consistent schedule or remains unpipelined, so a shared tile is not reused while still live.

## Causal basis

The discussion isolates the unstaged ldmatrix buffer and PR 13971 changes the schedule construction and guards used by that exact strategy.

## Analysis implication

Pipeline validation needs the copy, wait, barrier, warp-operation, buffer-stage, loop-depth, and tile-size facts together.

## Fix patches

- [`fixes/iree-13451-pr13971.patch`](fixes/iree-13451-pr13971.patch) — https://github.com/iree-org/iree/pull/13971
