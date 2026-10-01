# torch.compile produces incorrect output

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/155690](https://github.com/pytorch/pytorch/issues/155690)
- Opened: 2025-06-11T12:21:31Z
- Closed: 2025-07-11T21:30:09Z
- Labels: `high priority`, `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `pt2: ubn`

## Reported behavior

### 🐛 Describe the bug

While working with `torch.compile`, I encountered a bug that can be reproduced using the following minimal example. To illustrate this issue, I created a simple toy function called `do_view`. Within the function, I performed two `view` operations on the input tensor without making any in-place modifications. The function then returns the tensor after the `view` operations. However, after compiling the `do_view` function with `torch.compile`, the returned tensor may have completely different values compared to the input tensor. The example is shown below:

```
import torch

@torch.compile(fullgraph=True)
def do_view(input):
    dout, din = input.shape
    input = input.view(-1, 1)
    output = input.view(dout, din)
    return output

inputs = torch.randn(4, 128, 16, device='cuda', dtype=torch.float32)
for i in range(4):
    output = do_view(inputs[i])

inputs = torch.randn(4, 171, 6, device='cuda', dtype=torch.float32)
for i in range(4):
    output = do_view(inputs[i])
    print(torch.equal(output, inputs[i]))
```
Outputs:
```
True
False
True
False
```

The following characteristics are critical for reproducing this issue:
1. **Non-zero** `storage_offset()`: The input to the `do_view` function must be a slice of a tensor (e.g., inputs[i]) so that the input tensor has a non-zero `storage_offset()`.
2. **Alignment of** `data_ptr()`: The shape of the inputs tensor in the example, (4, 171, 6), is essential. It ensures that the data_ptr() values of some input tensors are not 16-byte aligned (https://github.com/pytorch/pytorch/blob/v2.7.0/torch/_inductor/utils.py#L2427). Specifically, for the inputs tensor (of type fp32), the second and fourth tensors in the sliced inputs have data_ptr() values that are not divisible by 16.
3. **Diverse input shapes**: It is necessary to pass at least two inputs with different shapes into the `do_view` function, as demonstrated in the example.

While debugging the torch.compile-related code, I identified that [copy_misaligned_inputs](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/_inductor/utils.py#L2403) clones tensors with data_ptr() values that are not 16-byte aligned. This operation causes the cloned tensor to lose its original `storage_offset()` property. Consequently, when the `AliasOfInputHandler.__call__` function is executed in [`torch/_functorch/_aot_autograd/runtime_wrappers.py`](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/_functorch/_aot_autograd/runtime_wrappers.py#L183), the storage_offset() of the `target_meta_tensor` does not match that of the `aliased_base_tensor`, ultimately leading to the bug.

I hope this explanation provides enough clarity to understand and reproduce the issue. Please let me know if additional details or clarifications are needed. Thank you for your time and assistance!

### Error logs

There were no error logs generated. However, the results returned were incorrect.

### Versions

PyTorch version: 2.7.1+cu126
Is debug build: False
CUDA used to build PyTorch: 12.6
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.1 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Clang version: Could not collect
CMake version: version 3.31.6
Libc version: glibc-2.39

Python version: 3.12.3 (main, Feb  4 2025, 14:48:35) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-5.10.134-16.103.al8.x86_64-x86_64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: 12.8.93
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA L20Y
GPU 1: NVIDIA L20Y
GPU 2: NVIDIA L20Y
GPU 3: NVIDIA L20Y
GPU 4: NVIDIA L20Y
GPU 5: NVIDIA L20Y
GPU 6: NVIDIA L20Y
GPU 7: NVIDIA L20Y

Nvidia driver version: 570.133.20
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_engines_runtime_compiled.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_heuristic.so.9.8.0
/usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9.8.0
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Address sizes:                   52 bits physical, 57 bits virtual
Byte Order:                      Little Endian
CPU(s):                          192
On-line CPU(s) list:             0-191
Vendor ID:                       GenuineIntel
Model name:                      Intel(R) Xeon(R) Platinum 8468V
CPU family:                      6
Model:                           143
Thread(s) per core:              2
Core(s) per socket:              48
Socket(s):                       2
Stepping:                        8
CPU(s) scaling MHz:              76%
CPU max MHz:                     3800.0000
CPU min MHz:                     800.0000
BogoMIPS:                        4800.00
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cat_l2 cdp_l3 invpcid_single intel_ppin cdp_l2 ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect avx_vnni avx512_bf16 wbnoinvd dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req hfi avx512vbmi umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq rdpid bus_lock_detect cldemote movdiri movdir64b enqcmd fsrm uintr md_clear serialize tsxldtrk pconfig arch_lbr amx_bf16 avx512_fp16 amx_tile amx_int8 flush_l1d arch_capabilities
Virtualization:                  VT-x
L1d cache:                       4.5 MiB (96 instances)
L1i cache:                       3 MiB (96 instances)
L2 cache:                        192 MiB (96 instances)
L3 cache:                        195 MiB (2 instances)
NUMA node(s):                    2
NUMA node0 CPU(s):               0-47,96-143
NUMA node1 CPU(s):               48-95,144-191
Vulnerability Itlb multihit:     Not affected
Vulnerability L1tf:              Not affected
Vulnerability Mds:               Not affected
Vulnerability Meltdown:          Not affected
Vulnerability Mmio stale data:   Not affected
Vulnerability Retbleed:          Not affected
Vulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:        Mitigation; Enhanced IBRS, IBPB conditional, RSB filling, PBRSB-eIBRS SW sequence
Vulnerability Srbds:             Not affected
Vulnerability Tsx async abort:   Not affected

Versions of relevant libraries:
[pip3] intel-openmp==2021.4.0
[pip3] mkl==2021.1.1
[pip3] mkl-devel==2021.1.1
[pip3] mkl-include==2021.1.1
[pip3] mypy-extensions==1.0.0
[pip3] numpy==1.26.0
[pip3] nvidia-cublas-cu12==12.6.4.1
[pip3] nvidia-cuda-cupti-cu12==12.6.80
[pip3] nvidia-cuda-nvrtc-cu12==12.6.77
[pip3] nvidia-cuda-runtime-cu12==12.6.77
[pip3] nvidia-cudnn-cu12==9.5.1.17
[pip3] nvidia-cudnn-frontend==1.10.0
[pip3] nvidia-cufft-cu12==11.3.0.4
[pip3] nvidia-curand-cu12==10.3.7.77
[pip3] nvidia-cusolver-cu12==11.7.1.2
[pip3] nvidia-cusparse-cu12==12.5.4.2
[pip3] nvidia-cusparselt-cu12==0.6.3
[pip3] nvidia-nccl-cu12==2.26.2
[pip3] nvidia-nvjitlink-cu12==12.6.85
[pip3] nvidia-nvtx-cu12==12.6.77
[pip3] nvtx==0.2.11
[pip3] onnx==1.17.0
[pip3] optree==0.14.1
[pip3] pynvjitlink==0.3.0
[pip3] pytorch-triton==3.2.0+gitb2684bf3b.nvinternal
[pip3] tbb==2021.13.1
[pip3] torch==2.7.1+cu126
[pip3] torch-geometric==2.6.1
[pip3] torch_tensorrt==2.7.0a0
[pip3] torchao==0.11.0
[pip3] torchprofile==0.0.4
[pip3] torchvision==0.22.1+cu126
[pip3] triton==3.3.1
[conda] Could not collect

cc @ezyang @gchanan @zou3519 @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @chenyang78 @muchulee8 @amjames @aakhundov

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

torch.compile produces incorrect output

## Observed implementation decision

copy_misaligned_inputs substitutes a zero-offset clone without rebasing AliasOfInputHandler's target metadata to the cloned storage.

## Physical effect

Inductor clones misaligned sliced inputs for alignment, losing their nonzero storage offset, while AOTAutograd later reconstructs output aliases using metadata that still reflects the original offset.

## Observed consequence

Two view operations that should preserve values reconstruct aliases at the wrong location and intermittently return data different from the input slice.

## Reported proposal or workaround

The report isolates alignment, shape specialization, and nonzero offset, and identifies the two responsible runtime paths; the reproducer no longer fails on a later nightly.

## Repair trace

The reproducer stops failing on a later nightly, but the fixing commit is unknown; the observed recovery therefore cannot be tied to the diagnosed metadata-rebasing defect.

## Archived evidence

- Raw report: `data/raw/github/pytorch/155690/report.json`
- SHA-256: `429bbe1db3710bff7a9f6e5a6dbeddcefa8608efc0916ca77ae7b04a1b0f706c`
