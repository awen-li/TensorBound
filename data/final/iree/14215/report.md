# [Stream] Buffer views produce stream resources with incorrect offset and size

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/14215](https://github.com/iree-org/iree/issues/14215)
- Opened: 2023-06-24T23:46:46Z
- Closed: 2023-06-27T21:48:32Z
- Labels: `bug 🐞`, `compiler/dialects`

## Reported behavior

### What happened?

Resources constructed from tensors imported from buffer views have incorrect offsets and sizes

```
cat test.mlir
```

```mlir
func.func @test(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  %buffer = hal.buffer_view.buffer<%arg0 : !hal.buffer_view> : !hal.buffer

  %c1_i32 = arith.constant 1 : i32
  %c553648160_i32 = arith.constant 553648160 : i32

  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index

  %lhs_view = hal.buffer_view.create buffer(%buffer : !hal.buffer)[%c0, %c64]
                                shape([%c4, %c4])
                                type(%c553648160_i32)
                                encoding(%c1_i32) : !hal.buffer_view

  %rhs_view = hal.buffer_view.create buffer(%buffer : !hal.buffer)[%c64, %c64]
                                shape([%c4, %c4])
                                type(%c553648160_i32)
                                encoding(%c1_i32) : !hal.buffer_view

  %lhs = hal.tensor.import %lhs_view : !hal.buffer_view -> tensor<4x4xf32>
  %rhs = hal.tensor.import %rhs_view : !hal.buffer_view -> tensor<4x4xf32>

  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4x4xf32>
  %out = linalg.fill ins(%cst : f32)
                     outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>

  %res = linalg.matmul ins(%lhs, %rhs : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>

  %res_view = hal.tensor.export %res : tensor<4x4xf32> -> !hal.buffer_view
  return %res_view : !hal.buffer_view
}
```

```
iree-compile  --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_60 ./test.mlir --compile-to=hal
```

### Problem:

```mlir
func.func @test(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  %buffer = hal.buffer_view.buffer<%arg0 : !hal.buffer_view> : !hal.buffer
  ...
  hal.command_buffer.push_descriptor_set<%cmd : !hal.command_buffer> layout(%_pipeline_layout_0 :
    !hal.pipeline_layout)[%c0] bindings([
      %c0 = (%buffer : !hal.buffer)[%c0, %c64],
      %c1 = (%buffer : !hal.buffer)[%c0, %c64],
      %c2 = (%buffer_1 : !hal.buffer)[%c0, %c64]
    ])

}
```

Offsets inside `%arg0` ignored and both `lhs` and `rhs` arguments point to the same location.

### Proof:

```
iree-run-module --device=cuda --module=/tmp/0 --function=test --input='32xf32=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]'
EXEC @test
result[0]: hal.buffer_view
4x4xf32=[56 62 68 74][152 174 196 218][248 286 324 362][344 398 452 506]
```

```python
lhs = np.arange(16).astype(np.float32).reshape(4,4)
np.dot(lhs, lhs)
array([[ 56.,  62.,  68.,  74.],
       [152., 174., 196., 218.],
       [248., 286., 324., 362.],
       [344., 398., 452., 506.]], dtype=float32)
```

## Trigger

Two tensor imports use buffer views over different nonzero ranges of one backing buffer.

## Relevant entities

HAL buffer; buffer view; source offset; view length; imported tensors; descriptor ranges

## Implemented relation

A buffer extracted from a view denotes the view's byte range, including its source offset.

## Pre-state

The left and right views identify distinct 64-byte ranges in one backing buffer.

## Binding transition

SkipBufferViewBufferOp replaces buffer_view.buffer with the original source buffer without testing the view offset.

## Post-state

Both imported tensors are represented as the same zero-offset range of the backing buffer.

## Validity obligation

The accessor can be replaced by the source buffer only when the constructed view starts at offset zero.

## Invalid decision

The old canonicalization treated common source-buffer identity as sufficient and discarded a nonzero view offset.

## Physical effect

Both dispatch bindings address the first 64 bytes instead of distinct left and right matrix ranges.

## Observed consequence

The matmul computes lhs multiplied by lhs and returns the corresponding wrong tensor values.

## Repair site

compiler/src/iree/compiler/Dialect/HAL/IR/HALOpFolders.cpp::SkipBufferViewBufferOp

## Repair

PR 14232 restricts the fold to a source offset matched by m_Zero and adds a nonzero-offset no-fold regression.

## Why the repair works

A nonzero-offset buffer accessor remains explicit, so later tensor import and descriptor construction preserve the view range.

## Causal basis

The report shows the two offsets collapsing to zero, and the merged patch guards the responsible fold and adds the corresponding nonzero-offset test.

## Analysis implication

View-folding analysis needs backing-buffer identity and byte offset together; identity alone does not establish range equivalence.

## Fix patches

- [`fixes/iree-14215-pr14232.patch`](fixes/iree-14215-pr14232.patch) — https://github.com/iree-org/iree/pull/14232
