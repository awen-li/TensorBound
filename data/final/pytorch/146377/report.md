# FlexAttention compiled backward gives garbage data in certain stride situations for K.grad

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/146377](https://github.com/pytorch/pytorch/issues/146377)
- Opened: 2025-02-04T00:55:26Z
- Closed: 2025-02-09T17:41:37Z
- Labels: `high priority`, `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: dynamic shapes`, `module: inductor`, `module: flex attention`

## Reported behavior

### 🐛 Describe the bug

```python
import torch
import torch.nn.attention.flex_attention
torch.set_default_device("cuda")
print(torch.__version__)
flex_compiled = torch.compile(torch.nn.attention.flex_attention.flex_attention)
for fix_issue in [False, True]:
	for i in range(10):
		torch.manual_seed(0)
		shape = (1, 16, 4096, 64)
		Q = torch.randn(shape, requires_grad=True)
		K = torch.randn(shape, requires_grad=True)
		V = torch.randn(shape, requires_grad=True)
		flex_compiled(Q, K, V) # why does this line have to be here??

		K_sliced = K[:, :, :-128]
		V_sliced = V[:, :, :-128]

		if fix_issue:
			K_sliced = K_sliced.clone()

		flex_compiled(Q, K_sliced, V_sliced).sum().backward()
		print("Q", Q.grad.mean(), "K", K.grad.mean(), "V", V.grad.mean(), K_sliced.is_contiguous())
```

When `K_sliced.is_contiguous()` is true, there is no issue. When `K_sliced.is_contiguous()` is false, the `K.grad` contains garbage data.

See:

```
2.6.0+cu124
Q tensor(0.0002, device='cuda:0') K tensor(4.3783e-07, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(4.9176e-07, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(-1.0043e-06, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(5.5709e-07, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(-2.0452e-06, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(-2.7107e-07, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(-3.6084e-07, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(3.0617e-06, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(1.6151e-08, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(7.5523e-07, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
```

Same on nightly:

```
2.7.0.dev20250203+cu124
Q tensor(0.0002, device='cuda:0') K tensor(-1.8570e-06, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(1.0961e-06, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(7.7451e-06, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(-1.1176e-05, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(-6.7828e-07, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(-1.1892e-05, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(2.9753e-06, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(2.1702e-05, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(9.9964e-06, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(2.9386e-07, device='cuda:0') V tensor(1., device='cuda:0') False
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
Q tensor(0.0002, device='cuda:0') K tensor(3.0923e-11, device='cuda:0') V tensor(1., device='cuda:0') True
```

This is an *extremely* weird issue that was really hard to pin down. At first I thought "uninitialized memory" but that's not fully satisfactory, because then you would expect NaN from time to time, and I've never seen it NaN, and additionally compute-sanitizer insists that's not happening. But at the same time, I found that seemingly unrelated changes would drastically affect the loss, for example, adding an unused allocation like `unused_tensor = torch.zeros_like(flex_out)` could dramatically change the final loss. I was able to reduce all the weird coincidences down to one: in this reproduction, you can see there is this line: `flex_compiled(Q, K, V)`. For some reason, the bug does not occur if you comment that out. (!!!)


### Versions

<details>
<summary>Env</summary>

```
Collecting environment information...
PyTorch version: 2.7.0.dev20250203+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.12 (main, Jan 17 2025, 14:35:34) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-51-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA H100 PCIe
Nvidia driver version: 565.57.01
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        46 bits physical, 57 bits virtual
Byte Order:                           Little Endian
CPU(s):                               26
On-line CPU(s) list:                  0-25
Vendor ID:                            GenuineIntel
Model name:                           Intel(R) Xeon(R) Platinum 8480+
CPU family:                           6
Model:                                143
Thread(s) per core:                   1
Core(s) per socket:                   1
Socket(s):                            26
Stepping:                             8
BogoMIPS:                             4000.00
Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology cpuid tsc_known_freq pni pclmulqdq vmx ssse3 fma cx16 pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch cpuid_fault ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx_vnni avx512_bf16 wbnoinvd arat vnmi avx512vbmi umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid bus_lock_detect cldemote movdiri movdir64b fsrm md_clear serialize tsxldtrk avx512_fp16 arch_capabilities
Virtualization:                       VT-x
Hypervisor vendor:                    KVM
Virtualization type:                  full
L1d cache:                            832 KiB (26 instances)
L1i cache:                            832 KiB (26 instances)
L2 cache:                             104 MiB (26 instances)
L3 cache:                             416 MiB (26 instances)
NUMA node(s):                         1
NUMA node0 CPU(s):                    0-25
Vulnerability Gather data sampling:   Not affected
Vulnerability Itlb multihit:          Not affected
Vulnerability L1tf:                   Not affected
Vulnerability Mds:                    Not affected
Vulnerability Meltdown:               Not affected
Vulnerability Mmio stale data:        Unknown: No mitigations
Vulnerability Reg file data sampling: Not affected
Vulnerability Retbleed:               Not affected
Vulnerability Spec rstack overflow:   Not affected
Vulnerability Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:             Mitigation; Enhanced / Automatic IBRS; IBPB conditional; RSB filling; PBRSB-eIBRS SW sequence; BHI SW loop, KVM SW loop
Vulnerability Srbds:                  Not affected
Vulnerability Tsx async abort:        Mitigation; TSX disabled

Versions of relevant libraries:
[pip3] flake8==4.0.1
[pip3] numpy==1.21.5
[pip3] nvidia-cublas-cu12==12.4.5.8
[pip3] nvidia-cuda-cupti-cu12==12.4.127
[pip3] nvidia-cuda-nvrtc-cu12==12.4.127
[pip3] nvidia-cuda-runtime-cu12==12.4.127
[pip3] nvidia-cudnn-cu12==9.1.0.70
[pip3] nvidia-cufft-cu12==11.2.1.3
[pip3] nvidia-curand-cu12==10.3.5.147
[pip3] nvidia-cusolver-cu12==11.6.1.9
[pip3] nvidia-cusparse-cu12==12.3.1.170
[pip3] nvidia-cusparselt-cu12==0.6.2
[pip3] nvidia-nccl-cu12==2.21.5
[pip3] nvidia-nvjitlink-cu12==12.4.127
[pip3] nvidia-nvtx-cu12==12.4.127
[pip3] optree==0.13.1
[pip3] pytorch-triton==3.2.0+gitb2684bf3
[pip3] torch==2.7.0.dev20250203+cu124
[pip3] triton==3.2.0
[conda] Could not collect
```

</details>

cc @ezyang @gchanan @zou3519 @kadeng @msaroufim @chauhang @penguinwu @bobrenjc93 @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @yf225 @chenyang78 @muchulee8 @amjames @desertfire @aakhundov @Chillee @drisspg @yanboliang @BoyuanFeng @ydwu4 @bdhirsh

## Trigger

FlexAttention backward receives K and V sliced along sequence length, preserving a noncontiguous stride pattern, and compiles with dynamic shapes.

## Relevant entities

key and value views; backward meta outputs; broadcasted grad-key buffer; broadcasted grad-value buffer; stride order; implicit layout conversion; fused epilogue

## Implemented relation

Backward meta outputs and lowering buffers agree on gradient shape and dense physical stride order before the scheduler plans output copies.

## Pre-state

The sliced inputs carry a noncontiguous stride order that the returned K and V gradients need to preserve semantically.

## Binding transition

The old fake/meta function and lowering describe intermediate and returned gradient buffers with different sizes or raw sliced strides, causing an implicit layout-conversion node to be inserted.

## Post-state

The scheduler fuses that conversion into the backward epilogue, where it writes an intermediate and then reads overlapping regions of the same memory.

## Validity obligation

Meta inference, kernel output allocation, batch-head reduction, and returned-gradient layout use compatible shapes and dense nonoverlapping strides with the input's stride order.

## Invalid decision

The backward contract reused raw input strides for differently shaped intermediates and did not distinguish broadcasted gradient storage from final correctly strided outputs.

## Physical effect

The fused epilogue races between stores and loads over overlapping regions, so K-gradient values depend on execution and allocation state.

## Observed consequence

For noncontiguous K slices, repeated compiled runs produce changing garbage-valued K.grad means; cloning K into contiguous storage makes the result stable.

## Repair site

torch/_higher_order_ops/flex_attention.py backward meta and torch/_inductor/kernel/flex_attention.py::flex_attention_backward

## Repair

Commit 69feef5 defines broadcasted K/V gradient shapes explicitly, derives dense nonoverlapping strides that preserve input stride order, copies reduced gradients into correctly strided returned buffers, aligns fake-mode metadata with lowering, and adds strided backward correctness and stride-order tests.

## Why the repair works

Lowering now realizes each intermediate and final gradient in the layout promised by metadata, so no hidden overlapping permutation is fused into the epilogue.

## Causal basis

The landed commit explicitly fixes report 146377, shows the overlapping epilogue, traces it to mismatched backward metadata, changes both meta and lowering layouts, and adds the report's sliced K/V gradient comparison as a regression.

## Analysis implication

Higher-order-kernel analysis needs input view strides, meta shapes and strides, intermediate broadcast dimensions, reduction dimensions, output allocation strides, implicit conversions, and fused memory ranges together.

## Fix patches

- [`fixes/pytorch-146377-commit-69feef5.patch`](fixes/pytorch-146377-commit-69feef5.patch) — https://github.com/pytorch/pytorch/commit/69feef5a94731581ea0a8d8f9db8a286b12a2670
