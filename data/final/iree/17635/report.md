# Crash / ASan issues using VmModule.wrap_buffer from Python

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/17635](https://github.com/iree-org/iree/issues/17635)
- Opened: 2024-06-10T22:32:52Z
- Closed: Open at collection
- Labels: `bug 🐞`, `bindings/python`

## Reported behavior

(Lots of red herrings here, see https://github.com/iree-org/iree/issues/17635#issuecomment-2163644331 for latest issue description)

---


Splitting off of https://github.com/llvm/torch-mlir/issues/3433 to discuss IREE-specific details.

I wrote some test cases for https://pytorch.org/docs/stable/generated/torch.Tensor.index_put_.html starting from PyTorch and going through iree-turbine / torch-mlir to IREE: https://gist.github.com/ScottTodd/1e95795e79d17964078217ca98a3a398. Latest results:
```
test_single_value                  | PASS
test_multiple_values               | PASS (then crash)
test_broadcast_value_along_axis    | FAIL
test_broadcast_value_along_indices | FAIL
test_broadcast_values_along_axis   | PASS (then crash)
```

The two "pass then crash" test cases used to pass without crashing and I found some interesting results bisecting through IREE releases.

release | test result
-- | --
iree_compiler==20240514.893 | PASS --> CRASH
iree_compiler==20240508.887 |
iree_compiler==20240507.886 | Error invoking function ...
iree_compiler==20240506.885 |
iree_compiler==20240505.884 |
iree_compiler==20240504.883 |
iree_compiler==20240503.882 |
iree_compiler==20240502.881 |
iree_compiler==20240501.880 | Error invoking function ...
iree_compiler==20240430.879 |
iree_compiler==20240429.878 |
iree_compiler==20240428.877 |
iree_compiler==20240427.876 | Error invoking function ...
iree_compiler==20240426.875 |
iree_compiler==20240425.874 | Error invoking function ...
iree_compiler==20240424.873 | Error invoking function ...
iree_compiler==20240423.872 | PASS --> CRASH
iree_compiler==20240422.871 |
iree_compiler==20240421.870 |
iree_compiler==20240420.869 | PASS --> CRASH
iree_compiler==20240419.868 | PASS --> CRASH
iree_compiler==20240418.867 | PASS
iree_compiler==20240417.866 | PASS
iree_compiler==20240416.865 |
iree_compiler==20240415.864 |
iree_compiler==20240414.863 |
iree_compiler==20240412.861 |
iree_compiler==20240411.860 |
iree_compiler==20240410.859 | PASS

Crash (during shutdown, maybe from writing out of bounds?):
```
======================================== 1 passed in 3.60s ======================================== Exception Code: 0xC0000005
 #0 0x00007ffa6c89a495 PyInit__runtime (D:\dev\projects\SHARK-TestSuite\iree_tests\pytorch\operators\.venv\Lib\site-packages\iree\_runtime_libs\_runtime.cp311-win_amd64.pyd+0x4a495)
 ...
 #5 0x00007ffa6c91bf1c nanobind::python_error::what(void) const (D:\dev\projects\SHARK-TestSuite\iree_tests\pytorch\operators\.venv\Lib\site-packages\iree\_runtime_libs\_runtime.cp311-win_amd64.pyd+0xcbf1c)
```

"Error invoking function" (fixed by https://github.com/iree-org/iree/pull/17339):
```
    def _invoke(self, arg_list, ret_list):
>       self._vm_context.invoke(self._vm_function, arg_list, ret_list)
E       ValueError: Error invoking function: D:\a\iree\iree\c\runtime\src\iree\hal\command_buffer_validation.c:363: INVALID_ARGUMENT; source and target ranges overlap within the same buffer; while invoking native function hal.command_buffer.copy_buffer; while calling import;
E       [ 2]   native hal.command_buffer.copy_buffer:0 -
E       [ 1] bytecode module.main$async:526 d:\dev\projects\SHARK-TestSuite\iree_tests\pytorch\operators\index_put_test.py:88:0
E       [ 0] bytecode module.main:62 d:\dev\projects\SHARK-TestSuite\iree_tests\pytorch\operators\index_put_test.py:88:0
```

## 20240418.867 to 20240419.868, pass --> crash

Between 20240418.867 and 20240419.868, the `test_multiple_values` test case started crashing.

before: https://github.com/iree-org/iree/releases/tag/candidate-20240418.867
after: https://github.com/iree-org/iree/releases/tag/candidate-20240419.868

diff (https://github.com/iree-org/iree/compare/ff624dd8b27029f4d69f884a15ff3af8606b15b9...a2476ceef53f6d4ec0f7a9cf36892647626d324b):
```
λ git log --oneline candidate-20240418.867..candidate-20240419.868
a2476ceef5 [metal] Disable failing semaphore submission test until fixing (#17100)
0e1e6bfe7a Clarify fusion heuristic (#17098)
125f4200d8 [CodeGen] Add a pattern to fold extract_slice consumer into xfer.write. (#17067)
f755b42289 [Codegen] Add folding in createBoundedTileSize for partially dynamic wgSize. (#17089)
a0b4853dc7 Fixes to enable out-of-tree plugin builds. (#17095)
bd1b10626c [CodeGen] Drop encoding for HAL and Flow ops when DT is not supported. (#17081)
886c4169ac [Winograd] Use TilingInterface for all levels of winograd op tiling (#17061)
244195960f Add GPU dialect dependencies to C/Python bindings. (#17090)
ab4babee07 [ConstEval] Add flag to adjust tensor size limit for hoisting (#17064)
eec081c2c6 [LLVMGPU] Fallback if dynamic dim found on vector distribute. (#17085)
f86f21cd80 [python] Expose MLIR python bindings for gpu and transform (#17088)
4778f5f909 Categorize matmul/metvec like generic ops for dispatches (#17084)
f32a87ccd2 [Flow] Move elementwise op fusion and bubble up expand shapes patterns into their own pass. (#17068)
3677fbcce8 [runtime] Add semaphore test where 2 batches wait on a former batch amongst 2 (#17080)
```

### 20240423.872 to 20240424.873 crash --> error invoking function

before: https://github.com/iree-org/iree/releases/tag/candidate-20240423.872
after: https://github.com/iree-org/iree/releases/tag/candidate-20240424.873

diff (https://github.com/iree-org/iree/compare/f5660ee72bd8e71910d00cec726375fb4b4a6de8...59532d30c0b8b3a0cdf261eaa4e69784c9fddeee):
```
λ git log --oneline candidate-20240423.872..candidate-20240424.873
59532d30c0 Extend DecomposeConvolutionToLowerDimOpsPass (#17069)
1b4c76f931 Adding a hoistable attr interface to allow attaching attrs to hoists. (#17139)
729ebc642f [runtime][metal] exclude properly the failing semaphore test (#17151)
30acc53257 [runtime][cts] add test where a device batch signals another and the host (#17138)
94728e34c2 [LinalgExt][Winograd] Add winograd.filter_transform op (#17102)
4bc90e7431 Run pytorch/models tests on AMDGPU with Vulkan. (#17129)
330651efa6 [plugins][ROCM] Fix minor loc source resize bug (#17133)
f7098e3f66 Moving regression suite to azure (#17140)
290d812d96 [runtime][cts] add semaphore test where a batch waits on another and a host signal (#17130)
8736479f5d Prevent stream.async.update folding unless confirmed safe. (#17135)
7192e8c195 [Preprocessing] Add support for general ContractionOps and handle dynamic dimension using pad + expand. (#17123)
d7db353adf Add initial website documentation for ONNX frontend. (#17004)
568bb31780 [runtime][cts] Add test waiting on a semaphore for finite time and fix Vulkan driver (#17126)
655b71a282 Executable library call hooks system, and a sample Linux/CPU event implementation (#15803)
3dde925d11 [VectorDistribution] Add distribution pattern for vector::MultiDimReductionOp (#17076)
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

VmModule.wrap_buffer receives mapped memory owned by a compiler Output object.

## Observed implementation decision

The wrapped module does not establish an ownership edge that keeps Output alive.

## Physical effect

Output finalization frees mapped VM bytecode before module/context teardown reads it.

## Observed consequence

ASan reports a read from freed bytecode memory during teardown.

## Reported proposal or workaround

None recorded.

## Repair trace

PR 17657 changes the test to reproduce the ASan failure but contains no ownership repair or fixed behavior, so the repair relationship is unsupported.

## Archived evidence

- Raw report: `data/raw/github/iree/17635/report.json`
- SHA-256: `46c94f548174b30106af1d4f658940114400f71df5ac72d2a00760252b20e8a5`
