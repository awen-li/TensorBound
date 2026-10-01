# [LinalgExt] Sort op causes incorrect bufferization

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/24665](https://github.com/iree-org/iree/issues/24665)
- Opened: 2026-07-03T09:58:54Z
- Closed: Open at collection
- Labels: `bug 🐞`

## Reported behavior

### What happened?

Try compiling and running the following example (this could probably be simplified further):
```mlir
// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu -o %t.vmfb
// RUN: iree-run-module --module=%t.vmfb --function=forward --iree-llvmcpu-target-cpu=generic --input='4xf16=[0 1 2 3]' | FileCheck %s
// CHECK: result[0]: hal.buffer_view
// CHECK: 4xf16=0 1 2 3

#id_1d = affine_map<(d0) -> (d0)>

// Let's assume %input = [0, 1, 2, ..., N-1]
func.func @forward(%input: tensor<?xf16>) -> tensor<?xf16> {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64

  %N = tensor.dim %input, %c0 : tensor<?xf16>
  %N_i64 = arith.index_cast %N : index to i64

  // %iota = [0, 1, 2, ..., N-1]
  %iota_init = tensor.empty(%N) : tensor<?xi64>
  %iota = linalg.map outs(%iota_init : tensor<?xi64>) (%out: i64) {
    %i = linalg.index 0 : index
    %i64 = arith.index_cast %i : index to i64
    linalg.yield %i64 : i64
  }

  // %negated = [0, -1, -2, ..., -(N-1)]
  %negated_init = tensor.empty(%N) : tensor<?xi64>
  %negated = linalg.map ins(%iota : tensor<?xi64>) outs(%negated_init : tensor<?xi64>) (%id: i64, %out: i64) {
    %neg_id = arith.subi %c0_i64, %id : i64
    linalg.yield %neg_id : i64
  }

  // %sorted = ([-(N-1), ..., -2, -1, 0], [N-1, N-2, ..., 0])
  %sorted:2 = iree_linalg_ext.sort dimension(0)
    outs(%negated, %iota : tensor<?xi64>, tensor<?xi64>)
  {
  ^bb0(%lhs_key: i64, %rhs_key: i64, %lhs_id: i64, %rhs_id: i64):
    %take_lhs = arith.cmpi sle, %lhs_key, %rhs_key : i64
    iree_linalg_ext.yield %take_lhs : i1
  } -> tensor<?xi64>, tensor<?xi64>

  // %descending = [N-1, N-2, ..., 0]
  %descending_init = tensor.empty(%N) : tensor<?xi64>
  %descending = linalg.max ins(%sorted#0, %sorted#1 : tensor<?xi64>, tensor<?xi64>) outs(%descending_init : tensor<?xi64>) -> tensor<?xi64>

  // %projected = [N-1, N-2, ..., 0]
  %projected_init = tensor.empty(%N) : tensor<?xf16>
  %projected = linalg.generic {
      indexing_maps = [#id_1d, #id_1d],
      iterator_types = ["parallel"]
    } ins(%descending : tensor<?xi64>)
      outs(%projected_init : tensor<?xf16>) {
  ^bb0(%in: i64, %out: f16):
    %index = arith.index_cast %in : i64 to index
    %value = tensor.extract %input[%index] : tensor<?xf16>
    linalg.yield %value : f16
  } -> tensor<?xf16>

  // %expanded = [[N-1], [N-2], ..., [0]]
  %expanded = tensor.expand_shape %descending [[0, 1]]
      output_shape [%N, 1] : tensor<?xi64> into tensor<?x1xi64>

  // %scattered = [N-1, N-2, ..., 0]
  %scattered_init = tensor.empty(%N) : tensor<?xi64>
  %scattered = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
      ins(%iota, %expanded : tensor<?xi64>, tensor<?x1xi64>)
      outs(%scattered_init : tensor<?xi64>) {
  ^bb0(%id: i64, %out: i64):
    iree_linalg_ext.yield %id : i64
  } -> tensor<?xi64>

  // %restored = [0, 1, 2, ..., N-1]
  %restored_init = tensor.empty(%N) : tensor<?xf16>
  %restored = linalg.generic {
      indexing_maps = [#id_1d, #id_1d],
      iterator_types = ["parallel"]
    } ins(%scattered : tensor<?xi64>)
      outs(%restored_init : tensor<?xf16>) {
  ^bb0(%in: i64, %out: f16):
    %index = arith.index_cast %in : i64 to index
    %value = tensor.extract %projected[%index] : tensor<?xf16>
    linalg.yield %value : f16
  } -> tensor<?xf16>

  return %restored : tensor<?xf16>
}
```

### Steps to reproduce your issue

1. Run the lit test above
2. Notice the output doesn't match
```sh
$ build/tools/iree-run-module --module=bug.vmfb --function=forward --input='4xf16=[0 1 2 3]'
EXEC @forward
result[0]: hal.buffer_view
4xf16=3 2 1 0
```

### What component(s) does this issue relate to?

MLIR, Compiler

### Version information

Try the latest `be57650d3723e4b8b44a341bd007fcb02062dd5b`, but the bug has been present for a long time, so it should still show up on basically any other version.

### Additional context

```mlir
%26 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%9) => with(%2 as %arg1: !stream.resource<external>{%1}, %result as %arg2: !stream.resource<external>{%1}, %result_0 as %arg3: !stream.resource<transient>{%8}) {
      stream.cmd.concurrent {
        stream.cmd.dispatch @forward_dispatch_0::@forward_dispatch_0_elementwise_broadcast_D_i64[%0](%11, %13 : i32, i32) {
          wo %arg3[%c0 for %8] : !stream.resource<transient>{%8}
        }
        stream.cmd.dispatch @forward_dispatch_1::@forward_dispatch_1_elementwise_broadcast_D_i64[%0](%15, %17, %11, %13 : i32, i32, i32, i32) {
          wo %arg3[%c0 for %8] : !stream.resource<transient>{%8}
        }
      }
      stream.cmd.dispatch @forward_dispatch_2::@forward_dispatch_2_sort_Dxi64[%0](%15, %17, %11, %13 : i32, i32, i32, i32) {
        rw %arg3[%c0 for %8] : !stream.resource<transient>{%8},
        rw %arg3[%c0 for %8] : !stream.resource<transient>{%8}
      }
      stream.cmd.dispatch @forward_dispatch_3::@forward_dispatch_3_elementwise_D_i64xi64xi64xf16[%0](%15, %17, %19, %21, %23, %25, %11, %13 : i32, i32, i32, i32, i32, i32, i32, i32) {
        ro %arg1[%c0 for %1] : !stream.resource<external>{%1},
        ro %arg3[%c0 for %8] : !stream.resource<transient>{%8},
        wo %arg3[%c0 for %8] : !stream.resource<transient>{%8},
        wo %arg3[%c0 for %8] : !stream.resource<transient>{%8}
      }
      stream.cmd.dispatch @forward_dispatch_4::@forward_dispatch_4_scatter_Dxi64_dispatch_tensor_store[%0](%19, %21, %15, %17, %11, %13 : i32, i32, i32, i32, i32, i32) {
        ro %arg3[%c0 for %8] : !stream.resource<transient>{%8},
        wo %arg3[%c0 for %8] : !stream.resource<transient>{%8}
      }
      stream.cmd.dispatch @forward_dispatch_5::@forward_dispatch_5_elementwise_D_i64xf16[%0](%23, %25, %15, %17, %11, %13 : i32, i32, i32, i32, i32, i32) {
        ro %arg3[%c0 for %8] : !stream.resource<transient>{%8},
        wo %arg2[%c0 for %1] : !stream.resource<external>{%1}
      }
    } => !stream.timepoint
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

[LinalgExt] Sort op causes incorrect bufferization

## Observed implementation decision

The bufferizer assigns the operation result and its outs operand the same resource without inserting the copy needed to preserve the still-live pre-operation value.

## Physical effect

Bufferization performs destructive destination-style sort and scatter updates directly in an input tensor's storage even though the original SSA tensor remains live and is returned separately.

## Observed consequence

Sorting or scattering mutates both the operation result and the separately returned input, causing downstream calculations to consume reordered values and return incorrect tensors.

## Reported proposal or workaround

Two minimized programs contrast the faulty LinalgExt behavior with linalg.generic, and stream IR shows multiple logically distinct values mapped to the same transient range.

## Repair trace

The report shows destructive bufferization and its consequence, but no source-localized repair or fixed regression is available.

## Archived evidence

- Raw report: `data/raw/github/iree/24665/report.json`
- SHA-256: `dee598d06732913659403f340f87361f2c7f025fa2c28b1376eb59205191b328`
