# Tranposing to different layout permutations results in different numerics

- Repository: `openxla/xla`
- Report: [https://github.com/openxla/xla/issues/17276](https://github.com/openxla/xla/issues/17276)
- Opened: 2024-09-17T19:19:23Z
- Closed: open at collection time
- Labels: `bug`

## Reported behavior

Hello, we stumbled upon a numerical issue for below modules while training fp8 quantizated models.
```
ENTRY main {
    %p0 = f8e4m3fn[12288,4096]{0,1} parameter(0)
    %b = f8e4m3fn[4096,12288]{1,0} bitcast(%p0)
    %transpose = f8e4m3fn[12288,4096]{1,0} transpose(%b), dimensions={1,0}
    %p1 = f8e4m3fn[4096,16384]{0,1} parameter(1)
    %p2 = f32[] parameter(2)
    %constant_1 = f32[] constant(1)
    ROOT %cublas-gemm.1.0 = (bf16[12288,16384]{1,0}, s8[33554432]{0}) custom-call(f8e4m3fn[12288,4096]{1,0} %transpose,  f8e4m3fn[4096,16384]{0,1} %p1,f32[] %p2, f32[]%constant_1, f32[]%constant_1, f32[]%constant_1), custom_call_target="__cublas$lt$matmul$f8", backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT","DEFAULT"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","damax_output":false,"selected_algorithm":"6","lhs_stride":"50331648","rhs_stride":"67108864","grad_x":false,"grad_y":false},"force_earliest_schedule":false}
  }
```

```
 ENTRY main {
    %p0 = f8e4m3fn[12288,4096]{0,1} parameter(0)
    %transpose = f8e4m3fn[4096,12288]{0,1} transpose(%p0), dimensions={1,0}
    %p1 = f8e4m3fn[4096,16384]{0,1} parameter(1)
    %p2 = f32[] parameter(2)
    %constant_1 = f32[] constant(1)
    ROOT %cublas-gemm.1.0 = (bf16[12288,16384]{1,0}, s8[33554432]{0}) custom-call(f8e4m3fn[4096,12288]{0,1} %transpose, f8e4m3fn[4096,16384]{0,1} %p1, f32[] %p2, f32[] %constant_1, f32[] %constant_1, f32[] %constant_1), custom_call_target="__cublas$lt$matmul$f8", backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["0"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT","DEFAULT"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","damax_output":false,"selected_algorithm":"7","lhs_stride":"50331648","rhs_stride":"67108864","grad_x":false,"grad_y":false},"force_earliest_schedule":false}
  }
```

This resulted in different numerics and upon checking the cublas runtime thunk - it processed the logical layout correctly and buffer assignment worked exactly the same.

We then had a unit test for testing out tranpose numerics as below
```
ENTRY main {
    %p0 = f8e4m3fn[12288,4096]{0,1} parameter(0)
    %b = f8e4m3fn[4096,12288]{1,0} reshape(%p0)
    %transpose = f8e4m3fn[12288,4096]{1,0} transpose(%b), dimensions={1,0}
    ROOT bitcast = f8e4m3fn[4096,12288]{0,1} reshape(%transpose)
  }
```
```
ENTRY main {
    %p0 = f8e4m3fn[12288,4096]{0,1} parameter(0)
    ROOT %transpose = f8e4m3fn[4096,12288]{0,1} transpose(%p0), dimensions={1,0}
  }
```
The numerical results of them were 99% different with relative errors > 1e-2.


Could you please help us understand why tranpose to different layout permutation would result in numerical difference? Is the default / non-default layout tranpose a known issue or are we making any unintentional assumptions / mistakes?

## Trigger

The GPU FP8 GEMM rewriter transposes an operand whose HLO shape has a non-descending layout.

## Relevant entities

FP8 operand; logical dimensions; minor-to-major layout; inserted transpose; cuBLASLt custom call

## Implemented relation

The transpose permutation and shape layout jointly map logical matrix coordinates to the physical bytes passed to the GEMM custom call.

## Pre-state

The FP8 operand has a valid non-default physical layout.

## Binding transition

TransposeMatrix swaps contracting and noncontracting dimensions while copying the original layout onto the transposed shape.

## Post-state

The generated transpose/custom-call pattern assigns a logical interpretation inconsistent with the operand's physical layout.

## Validity obligation

A GEMM rewrite preserves the logical-to-physical coordinate mapping when it inserts a transpose for any accepted input layout.

## Invalid decision

TransposeMatrix handled non-default layouts with the same direct transpose construction used for descending layouts.

## Physical effect

The GEMM reads FP8 values under a different coordinate permutation from the source tensor.

## Observed consequence

Equivalent transpose formulations produce outputs that differ in about 99 percent of elements with relative errors above 1e-2.

## Repair site

xla/service/gpu/transforms/gemm_rewriter.cc::TransposeMatrix

## Repair

PR 17798 detects non-default layouts, bitcasts to an equivalent descending-layout shape, transposes in that normalized representation, and bitcasts back; it adds an FP8 column-major rewrite test.

## Why the repair works

The two bitcasts preserve physical storage while the transpose operates in a layout whose permutation matches its logical dimension swap.

## Causal basis

The archived patch explicitly names issue 17276 and scopes its repair to the non-descending-layout FP8 GEMM rewrite; the broader open issue is not treated as fully resolved.

## Analysis implication

A layout rewrite check needs logical permutation, minor-to-major order, physical-layout-preserving bitcasts, and the custom-call dimension contract.

## Fix patches

- [`fixes/xla-17276-pr17798.patch`](fixes/xla-17276-pr17798.patch) — https://github.com/openxla/xla/pull/17798
