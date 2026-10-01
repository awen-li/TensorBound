# Vulkan/CUDA runtime error: failed to wait on timepoint

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/12670](https://github.com/iree-org/iree/issues/12670)
- Opened: 2023-03-17T06:50:28Z
- Closed: 2023-03-29T18:32:59Z
- Labels: `bug 🐞`

## Reported behavior

### What happened?

Unet failed on some AMD rdna2/rdna3 and nvidia A100 devices, with error:
```
<vm>:0: OK; failed to wait on timepoint;
[ 0] bytecode module.forward:131934 [
    <eval_with_key>.9:5042:14,
    <eval_with_key>.9:5039:15,
    <eval_with_key>.9:5038:15,
```
However, all the dumped dispatches ran without any problem.

### Version information

1. Download the latest [Unet model](https://storage.googleapis.com/shark-public/vivian/unet_1_64_512_512_fp16_stable-diffusion-2-1-base_vulkan_torch.mlir)
2. Compile command for Vulkan
```iree-compile --iree-input-type=none --iree-hal-target-backends=vulkan --iree-stream-resource-index-bits=64 --iree-vm-target-index-bits=64 --iree-util-zero-fill-elided-attrs -iree-vulkan-target-triple=rdna2-7900-linux --iree-preprocessing-pass-pipeline='builtin.module(func.func(iree-flow-detach-elementwise-from-named-ops,iree-flow-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=32}))' unet_1_64_512_512_fp16_stable-diffusion-2-1-base_vulkan/unet_1_64_512_512_fp16_stable-diffusion-2-1-base_vulkan_torch.mlir -o unet.vmfb```
3. Benchmark command for Vulkan
```iree-benchmark-module --module=unet.vmfb --function=forward --device=vulkan --input=1x4x64x64xf16 --input=1xf16 --input=2x64x1024xf16 --input=f32=1.0```


### Additional context

Also failed on A100, commands for cuda path are as following:
```iree-compile --iree-input-type=none --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-hal-target-backends=cuda --iree-llvmcpu-target-cpu-features=host --iree-hal-cuda-disable-loop-nounroll-wa --iree-hal-cuda-llvm-target-arch=sm_80 --iree-stream-resource-index-bits=64 --iree-vm-target-index-bits=64 --iree-util-zero-fill-elided-attrs --iree-preprocessing-pass-pipeline='builtin.module(func.func(iree-flow-detach-elementwise-from-named-ops,iree-flow-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=32}))' unet_1_64_512_512_fp16_stable-diffusion-2-1-base_cuda/unet_1_64_512_512_fp16_stable-diffusion-2-1-base_cuda_torch.mlir -o unet.vmfb```

```iree-benchmark-module --module=unet.vmfb --device=cuda --function=forward --input=1x4x64x64xf16 --input=1xf16 --input=2x64x1024xf16 --input=f32=1.0 ```

## Trigger

hal.fence.await pushes a wait frame while a bytecode function is executing synchronously.

## Relevant entities

VM stack; current frame; integer registers; reference registers; cached register pointers; hal.fence.await import

## Implemented relation

Each bytecode register operand resolves through the register arrays of the current VM stack frame.

## Pre-state

The dispatch loop caches pointers to the current frame's integer and reference register arrays.

## Binding transition

The variadic import pushes a wait frame and stack growth relocates the register storage.

## Post-state

The cached register pointers still address the old storage although the current frame and its register arrays have moved.

## Validity obligation

After any call that can grow the VM stack, dispatch refreshes the current frame, register pointers, and program counter before decoding the next instruction.

## Invalid decision

The variadic-import path resumed dispatch without refreshing its cached register pointers.

## Physical effect

The following conditional branch reads a stale integer slot instead of the zero returned by hal.fence.await.

## Observed consequence

A successful fence wait is reported as 'failed to wait on timepoint' on Vulkan and CUDA executions of the UNet model.

## Repair site

runtime/src/iree/vm/bytecode/dispatch.c Call and CallVariadic handlers

## Repair

PR 12809 restores regs_i32, regs_ref, and pc after every bytecode or import call that may have changed the stack.

## Why the repair works

The next instruction now resolves its operands through the relocated current-frame storage rather than a pointer cached before stack growth.

## Causal basis

The report traces a zero import result followed by a nonzero branch read; the merged repair names issue 12670 and refreshes the exact cached register bases after imports.

## Analysis implication

VM analysis needs stack-growth effects, current-frame identity, cached register bases, import-call boundaries, and the next operand read.

## Fix patches

- [`fixes/iree-12670-pr12809.patch`](fixes/iree-12670-pr12809.patch) — https://github.com/iree-org/iree/pull/12809
