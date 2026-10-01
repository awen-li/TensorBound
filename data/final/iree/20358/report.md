# GPU data races as LLVMGPUDistribute has ops distributed to multiple threads performing the same memory accesses

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/20358](https://github.com/iree-org/iree/issues/20358)
- Opened: 2025-03-24T16:46:51Z
- Closed: open at collection time
- Labels: `bug 🐞`, `codegen/rocm`

## Reported behavior

We are seeing many CI flakes with multiple ROCm targets (RDNA3, CDNA3, CDNA2). The common theme seems to be ops that are handled by LLVMGPUDistribute that end up not really being thread-distributed, so that all threads end up performing the exact same memory accesses, racing with each other.

As discussed [on discord](https://discord.com/channels/689900678990135345/1282818085153407038/1353760643668512931) and in https://github.com/iree-org/iree/issues/18649 and https://github.com/iree-org/iree/pull/20327.

Sample failure: https://github.com/iree-org/iree/actions/runs/14038781847/job/39303988986?pr=20355

```
122/176 Test  #733: iree/tests/e2e/stablehlo_ops/check_rocm_hip_stream_sort.mlir .....................................***Failed    0.53 sec
[==========] Running 4 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 4 tests from module
[ RUN      ] module.sort1D
[       OK ] module.sort1D (405 ms)
[ RUN      ] module.sort2D
[       OK ] module.sort2D (26 ms)
[ RUN      ] module.sort3D
iree/runtime/src/iree/modules/check/module.cc:478: Failure
Failed
Expected equality of these values. Contents does not match.
  lhs:
    1x2x4xi32=[[1 2 3 4][1 2 4 4]]
  rhs:
    1x2x4xi32=[[1 2 3 4][1 2 3 4]]
```

Isolating this `sort3D` function as a testcase:

```mlir
func.func @sort3D() {
  %input = util.unfoldable_constant dense<[[[1, 2, 3, 4],
                                            [4, 3, 2, 1]]]> : tensor<1x2x4xi32>

  %sort = "stablehlo.sort"(%input) ( {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):  // no predecessors
    %compare = "stablehlo.compare"(%arg1, %arg2) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%compare) : (tensor<i1>) -> ()
  }) {dimension = 2 : i64, is_stable = false} : (tensor<1x2x4xi32>) -> tensor<1x2x4xi32>

  check.expect_eq_const(%sort, dense<[[[1, 2, 3, 4], [1, 2, 3, 4]]]> : tensor<1x2x4xi32>) : tensor<1x2x4xi32>
  return
}
```

Compiling:

```
tools/iree-compile --output-format=vm-bytecode --mlir-print-op-on-diagnostic=false --iree-hal-target-backends=rocm --iree-input-type=stablehlo --iree-hip-target=gfx942 ~/a.mlir -o  ~/a.vmfb
```

-> print-IR-after-all log: https://gist.github.com/bjacob/1dfc92d50e61865304b9eaf05fcb6a3f
-> target asm:  https://gist.github.com/bjacob/908ace1302fb8dd98965c241156d3145

A workgroup size of 128 is being selected here: https://gist.github.com/bjacob/1dfc92d50e61865304b9eaf05fcb6a3f#file-log-mlir-L11874

But no corresponding thread-distribution of the code is done.

Should probably instead select the smallest possible workgroup size (e.g. subgroup size) and then have a `if (lane_id == 0) {...}` in the code?

@qedawkins suggested it might be a matter of switching to LLVMGPUTileAndFuse: https://discord.com/channels/689900678990135345/1353760643668512931/1353768473167134784

## Trigger

Kernel configuration selects LLVMGPUDistribute and a multi-thread workgroup for LinalgExt SortOp.

## Relevant entities

SortOp; LLVMGPUDistribute pipeline; workgroup threads; shared output memory accesses

## Implemented relation

The selected lowering pipeline binds one logical sort operation to a workgroup execution configuration.

## Pre-state

The logical sort requires a distribution or ownership scheme that gives concurrent threads compatible accesses.

## Binding transition

Kernel configuration assigns the operation to LLVMGPUDistribute with multiple threads.

## Post-state

The operation remains undistributed, so every thread executes the same memory accesses.

## Validity obligation

A multi-thread execution mapping assigns nonconflicting work or introduces synchronization sufficient for shared accesses.

## Invalid decision

The kernel configuration selected LLVMGPUDistribute for SortOp although that path did not distribute the operation.

## Physical effect

Multiple workgroup threads race on the same memory locations.

## Observed consequence

Sort tests intermittently return different and incorrect tensor contents.

## Repair site

compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp and GPU target configuration utilities

## Repair

SortOp is configured for LLVMGPUTileAndFuse with sort-specific tiling and workgroup-count handling instead of LLVMGPUDistribute.

## Why the repair works

The replacement pipeline supplies an explicit tile/distribution mapping for SortOp rather than replicating identical accesses across the workgroup.

## Causal basis

The patch description explicitly attributes Sort races to the old pipeline and moves SortOp to the replacement pipeline; this supports the Sort subcase only.

## Analysis implication

A placement check needs both the workgroup size and evidence that the selected pipeline distributes or serializes the operation's memory accesses.

## Fix patches

- [`fixes/iree-20358-fix.patch`](fixes/iree-20358-fix.patch) — https://github.com/iree-org/iree/pull/20411
