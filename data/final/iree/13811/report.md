# Accuracy Error on A100 with Transform Dialect + mma.sync

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/13811](https://github.com/iree-org/iree/issues/13811)
- Opened: 2023-05-26T20:11:39Z
- Closed: 2023-12-11T20:09:45Z
- Labels: `bug 🐞`, `codegen/nvvm`

## Reported behavior

IREE will produce incorrect output on the following IR (MNK = 514x500x130)
```
!input_tensor_t = tensor<514x130xf32>
!weight_tensor_t = tensor<130x500xf32>
!output_tensor_t = tensor<514x500xf32>
func.func @matmul(%in: !input_tensor_t, %wei: !weight_tensor_t) -> !output_tensor_t {
  %cst_0 = arith.constant 0.0 : f32
  %empty = tensor.empty() : !output_tensor_t
  %out = linalg.fill ins(%cst_0 : f32) outs(%empty : !output_tensor_t) -> !output_tensor_t
  %res = linalg.matmul
     ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
    outs(%out: !output_tensor_t) -> !output_tensor_t
  return %res : !output_tensor_t
}
```
when using unaligned transform + mma.sync
```
iree-compile matmul.mlir --iree-hal-target-backends=cuda \
                         --iree-stream-resource-index-bits=64 \
                         --iree-vm-target-index-bits=64 \
                         --iree-hal-cuda-llvm-target-arch=sm_80 \
                         --iree-codegen-llvmgpu-enable-transform-dialect-jit=true \
                         --iree-codegen-llvmgpu-enable-transform-dialect-matmul-tensorcore-strategy \
                         --td-matmul-strategy-use-mma-sync=true
```

(script used to verify, I only have torch here because numpy's testing function is very slow for ireert.device_arrays)
```python
import torch
torch.manual_seed(0)

module = """
!input_tensor_t = tensor<514x130xf32>
!weight_tensor_t = tensor<130x500xf32>
!output_tensor_t = tensor<514x500xf32>
func.func @matmul(%in: !input_tensor_t, %wei: !weight_tensor_t) -> !output_tensor_t {
  %cst_0 = arith.constant 0.0 : f32
  %empty = tensor.empty() : !output_tensor_t
  %out = linalg.fill ins(%cst_0 : f32) outs(%empty : !output_tensor_t) -> !output_tensor_t
  %res = linalg.matmul
     ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
    outs(%out: !output_tensor_t) -> !output_tensor_t
  return %res : !output_tensor_t
}

"""

example_input = torch.randn(514, 130)
example_weight = torch.randn(130, 500)

import iree.runtime as ireert
from iree.runtime import get_driver, get_device
import iree.compiler as ireec

device = "cuda"
runtime_device = "cuda"

flatbuffer = ireec.compile_str(
    module,
    target_backends=[device],
    extra_args=[
        "--iree-stream-resource-index-bits=64",
        "--iree-vm-target-index-bits=64",
        "--iree-hal-cuda-llvm-target-arch=sm_80",
        "--iree-codegen-llvmgpu-enable-transform-dialect-jit=true",
        "--iree-codegen-llvmgpu-enable-transform-dialect-matmul-tensorcore-strategy",
        "--td-matmul-strategy-use-mma-sync=true",
        ],
)

config = ireert.Config(driver_name=runtime_device)
vm_module = ireert.VmModule.from_flatbuffer(config.vm_instance, flatbuffer)
ctx = ireert.SystemContext(config=config)
ctx.add_vm_module(vm_module)
forward = ctx.modules.module["matmul"]
td_iree_output = forward(example_input.numpy(), example_weight.numpy()).to_host()

flatbuffer = ireec.compile_str(
    module,
    target_backends=[device],
    extra_args=[
        "--iree-stream-resource-index-bits=64",
        "--iree-vm-target-index-bits=64",
        "--iree-hal-cuda-llvm-target-arch=sm_80",
        ],
)

config = ireert.Config(driver_name=runtime_device)
vm_module = ireert.VmModule.from_flatbuffer(config.vm_instance, flatbuffer)
ctx = ireert.SystemContext(config=config)
ctx.add_vm_module(vm_module)
forward = ctx.modules.module["matmul"]
iree_output = forward(example_input.numpy(), example_weight.numpy()).to_host()

td_result = torch.from_numpy(td_iree_output)
iree_result = torch.from_numpy(iree_output)

torch.testing.assert_close(
    td_result, iree_result, rtol=1e-03, atol=4.1e-02
)
print("Success!")
```

which gives
```
Traceback (most recent call last):
  File "/home/quinn/e2e_tests/matmul/pretty_script.py", line 72, in <module>
    torch.testing.assert_close(
  File "/home/quinn/iree/iree_venv/lib/python3.10/site-packages/torch/testing/_comparison.py", line 1511, in assert_close
    raise error_metas[0].to_error(msg)
AssertionError: Tensor-likes are not close!

Mismatched elements: 256393 / 257000 (99.8%)
Greatest absolute difference: 70.04857635498047 at index (278, 173) (up to 0.041 allowed)
Greatest relative difference: 142397.28330606376 at index (251, 133) (up to 0.001 allowed)
```

Note that if I turn off asynchronous copies (`--td-matmul-strategy-use-async-copies=false`) then it gives * almost * correct output but does still differ a bit
```
Traceback (most recent call last):
  File "/home/quinn/e2e_tests/matmul/pretty_script.py", line 71, in <module>
    torch.testing.assert_close(
  File "/home/quinn/iree/iree_venv/lib/python3.10/site-packages/torch/testing/_comparison.py", line 1511, in assert_close
    raise error_metas[0].to_error(msg)
AssertionError: Tensor-likes are not close!

Mismatched elements: 2 / 257000 (0.0%)
Greatest absolute difference: 16.835737228393555 at index (257, 70) (up to 0.041 allowed)
Greatest relative difference: 1.0 at index (257, 66) (up to 0.001 allowed)

```
Importantly I only observe this some of the time indicating that there is some synchronization problem with mma_sync through transform that is exacerbated by asynchronous copies + pipelining (or perhaps two separate issues).

## Duplicate mapping

- Primary report: `github-iree-org-iree-13451`
- Shared cause: `cause-iree-13451-incomplete-multibuffering-still-pipelined`

Maintainers identify report 13811 as another instance of 13451; both use the same mma.sync strategy, fail after pipelining, and are repaired by PR 13971.

The sections below describe the shared cause reconstructed from the primary report. The reported behavior above remains the observation specific to this report.

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

- [`fixes/iree-13811-pr13971.patch`](fixes/iree-13811-pr13971.patch) — https://github.com/iree-org/iree/pull/13971
