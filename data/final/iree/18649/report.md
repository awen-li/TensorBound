# [ROCM] TopK e2e test failing on MI250, 7900XTX

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/18649](https://github.com/iree-org/iree/issues/18649)
- Opened: 2024-10-01T14:24:59Z
- Closed: Open at collection
- Labels: `bug 🐞`

## Reported behavior

In a recent PR, the TopK e2e test fails in CI: https://github.com/iree-org/iree/actions/runs/11107992173/job/30867743807?pr=18634

The following test is what fails:
```
func.func @topk_2d_dim1_inverted_max() {
  %input_values = util.unfoldable_constant dense<[[6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]> : tensor<2x6xf32>
  %input_indices = util.unfoldable_constant dense<[[0, 1, 2, 3, 4, 5],[6, 7, 8, 9, 10, 11]]> : tensor<2x6xi32>

  %out_values_empty = tensor.empty() : tensor<2x3xf32>
  %out_indices_empty = tensor.empty() : tensor<2x3xi32>
  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_values = linalg.fill ins(%neg_inf : f32) outs(%out_values_empty : tensor<2x3xf32>) -> tensor<2x3xf32>
  %out_indices = linalg.fill ins(%c0 : i32) outs(%out_indices_empty : tensor<2x3xi32>) -> tensor<2x3xi32>
  %0:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%input_values, %input_indices : tensor<2x6xf32> , tensor<2x6xi32>)
        outs(%out_values, %out_indices : tensor<2x3xf32>, tensor<2x3xi32>) {
        ^bb0(%arg0 : f32, %arg1 : f32):
         %0 = arith.cmpf ogt, %arg0, %arg1 : f32
         iree_linalg_ext.yield %0 : i1
        } -> tensor<2x3xf32>, tensor<2x3xi32>

  check.expect_almost_eq_const(
      %0#0,
      dense<[[6.0, 5.0, 4.0],[12.0, 11.0, 10.0]]> : tensor<2x3xf32>
  ) : tensor<2x3xf32>

  check.expect_eq_const(
      %0#1,
      dense<[[0, 1, 2],[11, 10, 9]]> : tensor<2x3xi32>
  ) : tensor<2x3xi32>

  return
}
```
This can be run by building the `iree-test-deps` target and then running `ctest -R iree/tests/e2e/linalg_ext_ops/check_rocm_hip_top-k.mlir --output-on-failure`. As far as I can tell, this only reproduces on MI250 cards based on the results of the CI. I have not accessed an MI250 machine yet to reproduce this myself, but it won't reproduce on MI300 cards.

The change in the PR causes linalg.fill ops to no longer be constant folded, so the difference in the PR is that `%out_values` and `%out_indices` stay as `linalg.fill` instead of being folded into a splat `arith.constant`.

### Repro Instructions ###
1. get a machine with an MI250 card
2. checkout https://github.com/Max191/iree/tree/const-hoist-linalg-operands-with-topk-test
3. run `cmake --build ../iree-build --target iree-test-deps` from iree directory
4. run `ctest -R iree/tests/e2e/linalg_ext_ops/check_rocm_hip_top-k.mlir --output-on-failure` from build directory

### Additional Clues ###
When comparing the results of compilation for `gfx942` and `gfx90a`, there are very few differences in the llvm ir dumped from `--iree-hal-dump-executable-intermediates-to`. Running a diff on the optimized llvm IR shows that the only difference is the kernel arguments being marked with `inreg` for `gfx942`:
```
diff /tmp/x/gfx90a/module__topk_2d_dim1_inverted_max_dispatch_0_rocm_hsaco_fb.optimized.ll /tmp/x/gfx942/module__topk_2d_dim1_inverted_max_dispatch_0_rocm_hsaco_fb.optimized.ll
6c6
< define amdgpu_kernel void @_topk_2d_dim1_inverted_max_dispatch_0_topk_2x6xf32(ptr addrspace(1) noalias readonly align 16 %0, ptr addrspace(1) noalias readonly align 16 %1, ptr addrspace(1) noalias align 16 %2, ptr addrspace(1) noalias align 16 %3) local_unnamed_addr #0 !reqd_work_group_size !0 {
---
> define amdgpu_kernel void @_topk_2d_dim1_inverted_max_dispatch_0_topk_2x6xf32(ptr addrspace(1) inreg noalias readonly align 16 %0, ptr addrspace(1) inreg noalias readonly align 16 %1, ptr addrspace(1) inreg noalias align 16 %2, ptr addrspace(1) inreg noalias align 16 %3) local_unnamed_addr #0 !reqd_work_group_size !0 {
```

This gist has the resulting rocasm for each target chip: https://gist.github.com/Max191/10e96721ab25c9c14cba1a5cfd3f4db6

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

TopK executes on MI250 or 7900XTX after GPU distribution.

## Observed implementation decision

The discussion suspects that multiple threads may access the same output locations without a valid happens-before relation.

## Physical effect

The test intermittently observes incorrect TopK values or indices.

## Observed consequence

The TopK end-to-end test flakes on two AMD GPU classes.

## Reported proposal or workaround

PR 20327 skips the TopK tests; issue 20358 repairs a demonstrated SortOp distribution problem.

## Repair trace

The issue labels the connection to the separately repaired SortOp failure as a suspicion, and the only report-specific patch skips tests; exact causal equivalence and a TopK repair are not demonstrated.

## Archived evidence

- Raw report: `data/raw/github/iree/18649/report.json`
- SHA-256: `fd16243a86c9a52a097e6867be699f2019b2b942a5df259d942c2b8d98c29cd2`
