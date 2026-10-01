# Using an operand's `iree.abi.tied` attribute to `MaterializeCopyOnWrite` in a `func::CallOp`

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/13003](https://github.com/iree-org/iree/issues/13003)
- Opened: 2023-04-10T21:06:11Z
- Closed: Open at collection
- Labels: `bug 🐞`

## Reported behavior

### What happened?

`MaterializeCopyOnWrite` ignores the tying of operand to the result and assumes the operand is not modified, passing the updated operand incorrectly downstream, when the original value is required.

In the following example, `Triple.function` gets the updated value of a previous `tensor.empty`, instead of getting a new `tensor.empty`.

```
module @example {
  func.func private @Double.function(tensor<?x?xi32>, tensor<?x?xi32>) -> (tensor<?x?xi32> {iree.abi.tied = 1 : index}) attributes {
    iree.abi.model = "coarse-fences",
    nosideeffects
  }

  func.func private @Triple.function(tensor<?x?xi32>, tensor<?x?xi32>) -> (tensor<?x?xi32> {iree.abi.tied = 1 : index}) attributes {
    iree.abi.model = "coarse-fences",
    nosideeffects
  }

  func.func @main(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {

    %c0 = arith.constant 0: index
    %c1 = arith.constant 1: index
    %dim0 = tensor.dim %arg0, %c0: tensor<?x?xi32>
    %dim1 = tensor.dim %arg0, %c1: tensor<?x?xi32>
    %0 = arith.muli %arg0, %arg0 : tensor<?x?xi32>
    %init = tensor.empty(%dim0, %dim1) : tensor<?x?xi32>
    %p1 = call @Double.function(%0, %init) : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
    %init2 = tensor.empty(%dim0, %dim1) : tensor<?x?xi32>
    %1 = call @Triple.function(%p1, %init2) : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>

    %2 = arith.muli %1, %1 : tensor<?x?xi32>

    return %2 : tensor<?x?xi32>
  }

}
```

### Steps to reproduce your issue
$iree-compile  --iree-execution-model=async-external --iree-hal-target-backends=llvm-cpu samples/custom_module/async/test/example.mlir --compile-to=stream

This will produce a single stream allocation for both the `tensor.empty()`.
```
    %10 = stream.resource.alloc uninitialized : !stream.resource<external>{%3} <--------------- allocation
    stream.timepoint.chain_external %9 => (%fence : !hal.fence)
    %fence_0 = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
    %11 = stream.tensor.export %6 : tensor<?x?xi32>{%0, %1} in !stream.resource<external>{%3} -> !hal.buffer_view
    %12 = stream.tensor.export %10 : tensor<?x?xi32>{%0, %1} in !stream.resource<external>{%3} -> !hal.buffer_view <------------- export
    %13 = call @Double.function(%11, %12, %fence, %fence_0) : (!hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> !hal.buffer_view <--------------- first use
    %14 = hal.buffer_view.dim<%13 : !hal.buffer_view>[0] : index
    %15 = hal.buffer_view.dim<%13 : !hal.buffer_view>[1] : index
    hal.buffer_view.assert<%13 : !hal.buffer_view> message("tensor") shape([%14, %15]) type(%c268435488_i32) encoding(%c1_i32)
    %16 = stream.timepoint.import %fence_0 : (!hal.fence) => !stream.timepoint
    %fence_1 = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
    stream.timepoint.chain_external %16 => (%fence_1 : !hal.fence)
    %fence_2 = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
    %17 = call @Triple.function(%13, %12, %fence_1, %fence_2) : (!hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> !hal.buffer_view <------------------ second use
```

### What component(s) does this issue relate to?

Compiler, Runtime

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

Using an operand's `iree.abi.tied` attribute to `MaterializeCopyOnWrite` in a `func::CallOp`

## Observed implementation decision

The generic func.call lacks a verified tied-operand/result interface, so copy-on-write analysis treats the operand as unmodified and reuses its storage for a later call that requires the original logical value.

## Physical effect

MaterializeCopyOnWrite does not model the call result's tie to an output operand, allowing two tensor.empty values to share one stream allocation across calls that update it.

## Observed consequence

The second external function receives the storage updated by the first function instead of a distinct empty output buffer.

## Reported proposal or workaround

No repository fix is archived; the maintainer documents queue-ordered allocation inside the external function as the supported workaround and explains the missing call semantics.

## Repair trace

The report documents missing tied-call semantics and an allocation workaround, but no implementation repair or fixed regression establishes restoration of that relation.

## Archived evidence

- Raw report: `data/raw/github/iree/13003/report.json`
- SHA-256: `d99a22d8dd7c5a8278b734f6067b500e6e86c7081ccf47e7dc4ab0d2c1e89aa1`
