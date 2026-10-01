# SDXL vae_decode model hangs and fails to execute at runtime

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/17834](https://github.com/iree-org/iree/issues/17834)
- Opened: 2024-07-09T15:31:08Z
- Closed: 2025-02-19T15:09:52Z
- Labels: `bug 🐞`

## Reported behavior

### What happened?

Compiled vmfb fails to execute at runtime. I see the memory allocated but no actual gpu execution appears to happen while the runtime seems to just hang indefinitely. Similar model has been run with changing just the batch size(both smaller and larger batch sizes) without issue but for some reason batch size 16 has this issue.

### Steps to reproduce your issue

IR for model in question: https://sharkpublic.blob.core.windows.net/sharkpublic/ian/stable_diffusion_xl_base_1_0_bs16_1024x1024_fp16_vae_decode_decomp_.mlir

compiled via this command:
```
./build-release/tools/iree-compile --iree-hal-target-backends=rocm --iree-rocm-target-chip=gfx942 --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-rocm-waves-per-eu=2 --iree-flow-enable-aggressive-fusion --iree-codegen-llvmgpu-use-vector-distribution=true --iree-global-opt-propagate-transposes=true --iree-opt-const-eval=false --iree-opt-outer-dim-concat=true --iree-vm-target-truncate-unsupported-floats --iree-llvmgpu-enable-prefetch=true --iree-opt-data-tiling=false --iree-codegen-gpu-native-math-precision=true --iree-rocm-waves-per-eu=2 --iree-flow-inline-constants-max-byte-length=1 --iree-preprocessing-pass-pipeline="builtin.module(iree-preprocessing-transpose-convolution-pipeline, iree-global-opt-raise-special-ops, util.func(iree-preprocessing-pad-to-intrinsics))" --iree-codegen-transform-dialect-library=/models/stable_diffusion_fp16/turbine_sdxl/attention_and_matmul_spec_mfma.mlir /models/stable_diffusion_fp16/turbine_sdxl/stable_diffusion_xl_base_1_0_bs16_1024x1024_fp16_vae_decode_decomp_.mlir -o vae.vmfb
```

run via:
```
./tools/iree-benchmark-module --module=/models/stable_diffusion_fp16/turbine_sdxl/stable_diffusion_xl_base_1_0_bs16_1024x1024_fp16_vae_decode_decomp_gfx942.vmfb --parameters=model=/models/stable_diffusion_fp16/vae_decode_fp16.safetensors --device=hip --function
=main --input=16x4x128x128xf16
```

### What component(s) does this issue relate to?

Runtime

### Version information

9ffe4735c8ed54a622e40b9a16df37657c0417b4

### Additional context

_No response_

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The batch-size-specific compiled module launches a generated dispatch on gfx942.

## Observed implementation decision

Maintainers suspect an invalid workgroup size derived from the problem shape, while later dispatch isolation also exposes invalid configuration, unavailable kernel images, out-of-range memory access, and oversized allocation.

## Physical effect

The HIP driver rejects a kernel launch and the asynchronous path does not propagate that error to the completion fence.

## Observed consequence

The runtime appears to hang; a separate ROCm path reports hipErrorInvalidConfiguration.

## Reported proposal or workaround

A later check on IREE main no longer reproduces the issue.

## Repair trace

The report contains several candidate failures and only a later non-reproduction statement. No fixing commit, selected root cause, repair site, or report-specific regression identifies which binding failure was corrected.

## Archived evidence

- Raw report: `data/raw/github/iree/17834/report.json`
- SHA-256: `0526d5fc77be15164309c0ac63e17e48d212ecfa6fe35058e08afb8bfcf90341`
