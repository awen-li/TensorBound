# [inductor] comprehensive_padding silently returns wrong and nondeterministic results for a Conv3d decoder at misaligned spatial strides

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/194281](https://github.com/pytorch/pytorch/issues/194281)
- Opened: 2026-08-20T21:09:29Z
- Closed: 2026-08-21T16:28:52Z
- Labels: `triage review`, `module: correctness (silent)`, `module: padding`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

torch.compile with inductor (default mode, no cudagraphs, static shapes, inference only) returns wrong and nondeterministic output for a small Conv3d decoder, with no warning or error. The failure depends only on spatial shape: the identical graph run at 46x46 and at 46x80 in the same process differs from eager by ~1.6 and ~1e-3 respectively.

Two things make it more than a numerics complaint:

- Not self-consistent — two consecutive calls with the same input tensor return different results. Wrong and nondeterministic reads like uninitialized memory, not a reassociated reduction.
- Not a precision issue — fp32, fp16 and bf16 all fail by ~1.6, and in fp32 the correct configuration matches eager to 6e-4. The failure is ~3 orders of magnitude outside the arithmetic noise.
aot_eager is bit-exact, so tracing and AOT functionalization are fine; this is codegen.

*Trigger*: torch._inductor.config.comprehensive_padding (default True). Setting it False makes the output correct and deterministic; so does a graph break before the last upsample. It tracks the alignment of the first upsampled stage's H*W channel stride — 46x46 → 8464 (132.25 × 64) wrong, 46x80 → 14720 (230.0 × 64) ok.

Repro script is attached.

[repro_inductor_padding_miscompile.py](https://github.com/user-attachments/files/31281353/repro_inductor_padding_miscompile.py)

### Versions

PyTorch version: 2.11.0+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.3 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Clang version: 18.1.3 (1ubuntu1)
CMake version: version 3.28.3
Libc version: glibc-2.39

Python version: 3.12.12 (main, Feb 12 2026, 00:42:14) [Clang 21.1.4 ] (64-bit runtime)
Python platform: Linux-6.6.105+-x86_64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: 13.0.88
CUDA_MODULE_LOADING set to:
GPU models and configuration:
GPU 0: NVIDIA H100 80GB HBM3
GPU 1: NVIDIA H100 80GB HBM3
GPU 2: NVIDIA H100 80GB HBM3
GPU 3: NVIDIA H100 80GB HBM3

Nvidia driver version: 580.95.05
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.14.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.14.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.14.0
/usr/lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so.9.14.0
/usr/lib/x86_64-linux-gnu/libcudnn_engines_runtime_compiled.so.9.14.0
/usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9.14.0
/usr/lib/x86_64-linux-gnu/libcudnn_heuristic.so.9.14.0
/usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9.14.0
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True
Caching allocator config: N/A

CPU:
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           52 bits physical, 57 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  208
On-line CPU(s) list:                     0-207
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz
CPU family:                              6
Model:                                   143
Thread(s) per core:                      2
Core(s) per socket:                      52
Socket(s):                               2
Stepping:                                8
BogoMIPS:                                5399.99
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rtm avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx_vnni avx512_bf16 arat avx512vbmi umip avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid cldemote movdiri movdir64b fsrm md_clear serialize tsxldtrk amx_bf16 avx512_fp16 amx_tile amx_int8 arch_capabilities
Hypervisor vendor:                       KVM
Virtualization type:                     full
L1d cache:                               4.9 MiB (104 instances)
L1i cache:                               3.3 MiB (104 instances)
L2 cache:                                208 MiB (104 instances)
L3 cache:                                210 MiB (2 instances)
NUMA node(s):                            2
NUMA node0 CPU(s):                       0-51,104-155
NUMA node1 CPU(s):                       52-103,156-207
Vulnerability Gather data sampling:      Not affected
Vulnerability Indirect target selection: Mitigation; Aligned branch/return thunks
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; PBRSB-eIBRS SW sequence; BHI SW loop, KVM SW loop
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Not affected

Versions of relevant libraries:
[pip3] ema-pytorch==0.7.7
[pip3] facenet-pytorch==2.5.3
[pip3] guided-filter-pytorch==3.7.5
[pip3] mypy_extensions==1.1.0
[pip3] numpy==2.2.6
[pip3] nvidia-cublas==13.1.0.3
[pip3] nvidia-cuda-cupti==13.0.85
[pip3] nvidia-cuda-nvrtc==13.0.88
[pip3] nvidia-cuda-runtime==13.0.96
[pip3] nvidia-cuda-runtime-cu13==0.0.0a0
[pip3] nvidia-cudnn-cu13==9.24.0.43
[pip3] nvidia-cudnn-frontend==1.26.0
[pip3] nvidia-cufft==12.0.0.61
[pip3] nvidia-curand==10.4.0.35
[pip3] nvidia-cusolver==12.0.4.66
[pip3] nvidia-cusparse==12.6.3.3
[pip3] nvidia-cusparselt-cu13==0.8.0
[pip3] nvidia-nccl-cu13==2.29.7
[pip3] nvidia-nvjitlink==13.0.88
[pip3] nvidia-nvtx==13.0.85
[pip3] nvtx==0.2.15
[pip3] onnx==1.18.0
[pip3] onnx_graphsurgeon==0.5.8
[pip3] onnx-ir==0.2.1
[pip3] onnxruntime-gpu==1.28.0
[pip3] onnxscript==0.6.2
[pip3] open-clip-torch==2.22.0
[pip3] pytorch-lightning==2.6.1
[pip3] pytorch-msssim==1.0.0
[pip3] pytorchvideo==0.1.5
[pip3] rotary-embedding-torch==0.8.9
[pip3] rw_torch_extensions==0.1.8+tkgemmar
[pip3] torch==2.11.0+cu130
[pip3] torch_c_dlpack_ext==0.1.5
[pip3] torch-fidelity==0.3.0
[pip3] torch-stoi==0.2.3
[pip3] torch-tb-profiler==0.4.3
[pip3] torch_tensorrt==2.10.0+cu130
[pip3] torchao==0.17.0+cu130
[pip3] torchaudio==2.11.0+cu130
[pip3] torchcodec==0.11.0+cu130
[pip3] torchdata==0.11.0
[pip3] torchdiffeq==0.2.5
[pip3] torchft-nightly==2026.5.11
[pip3] torchlibrosa==0.1.0
[pip3] torchmetrics==1.9.0
[pip3] torchvision==0.26.0+cu130
[pip3] triton==3.6.0
[pip3] vector-quantize-pytorch==1.20.11
[conda] Could not collect

cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Trigger

A Conv3d decoder creates a noncontiguous reshape/view and comprehensive padding later selects an aligned physical stride for its backing producer.

## Relevant entities

logical reshape; ReinterpretView index map; FlexibleLayout; backing storage; padding decision; producer stride; consumer load

## Implemented relation

A ReinterpretView's logical-to-physical index map is derived from, and remains valid only for, the fixed layout of its backing storage.

## Pre-state

The reshape derives an index map from the backing tensor's current layout while that layout is still flexible.

## Binding transition

handle_unbacked_or_dynamic_reshape calls as_storage_and_layout with freeze=False and constructs a ReinterpretView from the returned strides.

## Post-state

A later layout decision can change the backing storage strides without rebuilding the already-created view index map.

## Validity obligation

Creating a persistent physical index map freezes the backing layout that the map encodes before later padding or reuse decisions run.

## Invalid decision

The reshape path allowed a ReinterpretView to retain an index map derived from a mutable FlexibleLayout.

## Physical effect

The consumer calculates addresses with strides that no longer describe the producer's physical layout and reads padding or uninitialized storage.

## Observed consequence

With comprehensive padding enabled, repeated compiled calls return different wrong values; fp32 differs from eager by about 1.58 and low-precision runs can produce NaNs.

## Repair site

torch/_inductor/ir.py::View.create_reinterpret_view within the noncontiguous reshape path

## Repair

Commit 0ff4b33 removes freeze=False from as_storage_and_layout, freezing the backing layout before deriving the ReinterpretView index map, and adds a nonaligned-width forward-and-backward regression that crosses the padding threshold.

## Why the repair works

Subsequent layout optimization can no longer change the storage strides beneath an existing view, so producer and consumer use the same physical address function.

## Causal basis

A maintainer reproduces the report, identifies a stale producer/consumer layout, confirms that PR 192575 makes the same repro deterministic and correct, and closes the issue as fixed by that landed change.

## Analysis implication

View analysis needs the source layout's mutability state, frozen strides, derived index map, later padding decision, producer layout, and consumer address expression together.

## Fix patches

- [`fixes/pytorch-194281-commit-0ff4b33.patch`](fixes/pytorch-194281-commit-0ff4b33.patch) — https://github.com/pytorch/pytorch/commit/0ff4b33e3d38cb3d3671c9ac20bf9ef32e8bbe15
