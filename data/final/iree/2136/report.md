# The input buffer of reduce_window is always initialed to zero

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/2136](https://github.com/iree-org/iree/issues/2136)
- Opened: 2020-06-08T18:33:03Z
- Closed: 2021-02-22T18:49:49Z
- Labels: `bug 🐞`, `codegen`

## Reported behavior

The buffer should be initialed with initial value that the op carried.

https://github.com/google/iree/blob/0fb27f4d73f6a808e9cffe6265a70ca4b37124f1/iree/compiler/Conversion/HLOToLinalg/HLOToLinalgOnBuffers.cpp#L679-L682

Example test:
```
func @reduce_window_min_4x6xf32() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[ 1.0,  2.0,  3.0,  4.0,  5.0,  6.0],
                                       [ 7.0,  8.0,  9.0, 10.0, 11.0, 12.0],
                                       [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                                       [19.0, 20.0, 21.0, 22.0, 23.0, 24.0]]> : tensor<4x6xf32>
  %1 = iree.unfoldable_constant dense<100.0> : tensor<f32>
  %res = "xla_hlo.reduce_window"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "xla_hlo.minimum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "xla_hlo.return"(%3) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[2, 3]> : tensor<2xi64>,
      window_strides = dense<[2, 3]> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<2x2xf32>
  check.expect_almost_eq_const(%res, dense<[[9.0, 12.0], [21.0, 24.0]]> : tensor<2x2xf32>) : tensor<2x2xf32>
  return
}
```

In this case, all the result values are zero on llvm-ir and vulkan backend, but it should be `2x2xf32=[1 4][13 16]`.

We can look up the initial buffer. If it's a constant, we can extract the value and fill it to the buffer. If it's a memref, we can do a load and fill it to the buffer.

## Trigger

A reduce_window minimum carries scalar initial value 100 over an input containing positive values.

## Relevant entities

reduce_window init operand; result buffer; window elements; reduction combiner; output tensor

## Implemented relation

Each physical result accumulator begins with the initial value carried by the logical reduce_window operation.

## Pre-state

The operation specifies 100 as the identity seed for every output window.

## Binding transition

The old lowering zero-fills the result buffer before the pooling reduction.

## Post-state

Every minimum accumulator begins at zero, which is below all positive input elements.

## Validity obligation

Buffer initialization preserves the reduce_window init operand before window elements are combined.

## Invalid decision

The lowering hard-coded zero as the result-buffer initializer instead of reading the operation's init operand.

## Physical effect

The minimum reduction retains zero for every output window.

## Observed consequence

Both LLVM and Vulkan backends return an all-zero 2x2 tensor instead of values 1, 4, 13, and 16.

## Repair site

iree/compiler/Conversion/HLOToLinalg/HLOToLinalgOnBuffers.cpp::ConvertReduceWindowOp

## Repair

PR 4911 extracts a constant init value or loads a dynamic init buffer, fills the result with that value, and adds constant and argument-backed initialization tests.

## Why the repair works

The physical accumulator state now matches the logical seed before the reduction body consumes any tensor element.

## Causal basis

The issue identifies the hard-coded zero fill, and the linked patch replaces that fill with the exact init operand and adds reduce_window regressions.

## Analysis implication

Reduction lowering checks need the logical init operand, its constant-or-buffer representation, result allocation, fill value, and combiner semantics.

## Fix patches

- [`fixes/iree-2136-pr4911.patch`](fixes/iree-2136-pr4911.patch) — https://github.com/iree-org/iree/pull/4911
