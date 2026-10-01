# torch.compile on MPS: rms_norm not invoked

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/150629](https://github.com/pytorch/pytorch/issues/150629)
- Opened: 2025-04-03T18:00:18Z
- Closed: 2025-04-17T04:15:27Z
- Labels: `triaged`, `module: correctness (silent)`, `module: reductions`, `module: mps`, `oncall: pt2`, `module: inductor`

## Reported behavior

### 🐛 Describe the bug

When forcing torch.compile to use the fused rms_norm MPS implementation, it doesn't do it
One way to notice the bug, is to see that the compiled model outputs zeros, instead of the correct values:

```python
import torch
with torch.no_grad():
    x = torch.randn(1024, requires_grad=False, device="mps")
    model = torch.nn.RMSNorm(1024, device="mps")
    y1 = model(x)
    model = torch.compile(model)
    y2 = model(x)
    print(y1)
    print(y2)
```

Outputs something like:
```
tensor([ 1.7966, -0.1217,  0.0942,  ...,  1.1045, -0.0254,  0.0899], device='mps:0')
tensor([0., 0., 0.,  ..., 0., 0., 0.], device='mps:0')
```

The reason for that, is that the generated code looks like this:
```
def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, ), (1, ))
    assert_size_stride(arg1_1, (1024, ), (1, ))
    with torch._ops.contextlib.nullcontext():
        # MPS set device
        buf0 = empty_strided((1024, ), (1, ), device='mps', dtype=torch.float32)
    return (buf0, )
```

Notice there is no call to `aten.rms_norm`

### Versions

PyTorch version: 2.8.0.dev20250330
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: macOS 15.3.2 (arm64)
GCC version: Could not collect
Clang version: 16.0.0 (clang-1600.0.26.6)
CMake version: Could not collect
Libc version: N/A

Python version: 3.12.1 | packaged by Anaconda, Inc. | (main, Jan 19 2024, 09:45:58) [Clang 14.0.6 ] (64-bit runtime)
Python platform: macOS-15.3.2-arm64-arm-64bit
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Apple M1 Pro

Versions of relevant libraries:
[pip3] numpy==1.26.4
[pip3] torch==2.8.0.dev20250330
[pip3] torchaudio==2.6.0.dev20250330
[pip3] torchvision==0.18.0.dev20240223
[conda] numpy 1.26.4 py312h7f4fdc5_0
[conda] numpy-base 1.26.4 py312he047099_0
[conda] torch 2.8.0.dev20250330 pypi_0 pypi
[conda] torchaudio 2.6.0.dev20250330 pypi_0 pypi
[conda] torchvision 0.18.0.dev20240223 py312_cpu pytorch-nightly

cc @kulinseth @albanD @malfet @DenisVieriu97 @jhavukainen @voznesenskym @penguinwu @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @chenyang78 @kadeng @muchulee8 @amjames @chauhang @aakhundov

## Trigger

torch.compile traces RMSNorm on MPS when inference dispatch selects the fused native implementation.

## Relevant entities

rms_norm_symint; MPS native helper; ATen dispatcher; compiled graph; output buffer

## Implemented relation

The selected fused RMSNorm operation must be represented as a traceable graph operation that binds its output buffer to the MPS computation.

## Pre-state

Eager rms_norm_symint calls the fused MPS helper and returns its computed tensor.

## Binding transition

Compilation traces the function while the fused path invokes a native C++ helper that is not registered as an ATen operator.

## Post-state

The graph retains output allocation but has no represented RMSNorm computation to write that output.

## Validity obligation

Every backend operation selected during capture is visible to dispatcher-based tracing or has an equivalent captured decomposition.

## Invalid decision

The fused MPS implementation was called as an unregistered native helper from rms_norm_symint.

## Physical effect

Generated code allocates an MPS output buffer and returns it without launching the RMSNorm kernel.

## Observed consequence

The compiled model returns zeros while eager execution returns nonzero normalized values.

## Repair site

aten/src/ATen/native/layer_norm.cpp, native_functions.yaml, and torch/_inductor/lowering.py

## Repair

PR 150661 registers _rms_norm_fused as an ATen MPS operation, dispatches through it, and makes Inductor fall back on that traceable operator.

## Why the repair works

Capture now records an explicit operator whose fallback invokes the MPS kernel and produces the returned buffer.

## Causal basis

The report shows the missing generated call, and the linked fix exposes only the fused helper through ATen and Inductor fallback.

## Analysis implication

A capture check needs selected backend branch, dispatcher registration, graph node presence, fallback policy, and output write provenance.

## Fix patches

- [`fixes/pytorch-150629-pr150661.patch`](fixes/pytorch-150629-pr150661.patch) — https://github.com/pytorch/pytorch/pull/150661
