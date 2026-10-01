# [inductor] torch._foreach_add_([x], [x.flip(0)]) reads x while writing it: silent wrong results (CUDA and CPU)

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/198033](https://github.com/pytorch/pytorch/issues/198033)
- Opened: 2026-09-21T21:06:21Z
- Closed: Open at collection
- Labels: `high priority`, `triaged`, `module: correctness (silent)`, `module: mta`, `oncall: pt2`, `module: inductor`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug

An in-place foreach op whose other operand is computed from the same tensor at other positions gives wrong values under `torch.compile`. Eager is fine, and so is the plain `x.add_(x.flip(0))`.

```python
import torch


def f(x):
    torch._foreach_add_([x], [x.flip(0)])
    return x


device = "cuda" if torch.cuda.is_available() else "cpu"
compiled = torch.compile(f)
for shape in ((8, 8), (1024, 1024)):
    x = torch.rand(*shape, device=device)
    want = f(x.clone())
    print(shape, [int((compiled(x.clone()) != want).sum()) for _ in range(5)])
```

CUDA, 2.14.0+cu130, RTX 4070 (2.9.1 is the same): right at the small shape, wrong at the large one.

```
(8, 8) [0, 0, 0, 0, 0]
(1024, 1024) [496640, 495296, 496560, 495504, 496512]
```

CPU, 2.14.0 and nightly 2.15.0.dev20260921: wrong at any size.

```
(8, 8) [32, 32, 32, 32, 32]
(1024, 1024) [524204, 524184, 524192, 524136, 524244]
```

The kernel gets `x` as input and as output, and reads the mirrored row while it is being written:

```python
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (56 + x0 + ((-8)*x1)), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)

triton_poi_fused_0.run(arg0_1, arg0_1, 64, stream=raw_stream0)
```

Same with `x.roll(1, 0)`, `x.T.clone()` or `x.flip(0) * 2.0` as the other operand, with `_foreach_mul_`, `_foreach_copy_` and `_foreach_addcdiv_`, and with lists of more than one tensor.

`register_foreach_inplace` in `torch/_inductor/lowering.py` lowers the op as the out-of-place one followed by `mutate_to(arg, result, unsafe_alias=True)`. With `unsafe_alias`, `realize_into` skips the copy and gives the pointwise that computes `result` the layout `MutationLayoutSHOULDREMOVE(arg)`, which is only safe when `result` reads `arg` at the position it writes. Would checking the read indices there, and falling back to `unsafe_alias=False` otherwise, be the right fix? I can send it.

### Versions

<details>

```
Collecting environment information...
PyTorch version: 2.14.0+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.4 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Clang version: 18.1.3 (1ubuntu1)
CMake version: version 3.28.3
Libc version: glibc-2.39

Python version: 3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-7.0.0-31-generic-x86_64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to:
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4070
Nvidia driver version: 580.178.04
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_train.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_ops_train.so.8.9.7
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False
Caching allocator config: N/A

CPU:
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           46 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  20
On-line CPU(s) list:                     0-19
Vendor ID:                               GenuineIntel
Model name:                              12th Gen Intel(R) Core(TM) i7-12700
CPU family:                              6
Model:                                   151
Thread(s) per core:                      2
Core(s) per socket:                      12
Socket(s):                               1
Stepping:                                2
CPU(s) scaling MHz:                      74%
CPU max MHz:                             2100.0000
CPU min MHz:                             800.0000
BogoMIPS:                                4224.00
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault cat_l2 cdp_l2 ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdt_a rdseed adx smap clflushopt clwb intel_pt sha_ni xsaveopt xsavec xgetbv1 xsaves split_lock_detect user_shstk avx_vnni dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp hwp_pkg_req hfi vnmi umip pku ospke waitpkg gfni vaes vpclmulqdq rdpid movdiri movdir64b fsrm md_clear serialize pconfig arch_lbr ibt flush_l1d arch_capabilities
Virtualization:                          VT-x
L1d cache:                               512 KiB (12 instances)
L1i cache:                               512 KiB (12 instances)
L2 cache:                                12 MiB (9 instances)
L3 cache:                                25 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-19
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Old microcode:             Vulnerable
Vulnerability Reg file data sampling:    Vulnerable: No microcode
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; PBRSB-eIBRS SW sequence; BHI SW loop, KVM SW loop
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Not affected
Vulnerability Vmscape:                   Mitigation; IBPB before exit to userspace

Versions of relevant libraries:
[pip3] numpy==2.5.3
[pip3] nvidia-cublas==13.1.1.3
[pip3] nvidia-cuda-cupti==13.0.85
[pip3] nvidia-cuda-nvrtc==13.0.88
[pip3] nvidia-cuda-runtime==13.0.96
[pip3] nvidia-cudnn-cu13==9.24.0.43
[pip3] nvidia-cufft==12.0.0.61
[pip3] nvidia-curand==10.4.0.35
[pip3] nvidia-cusolver==12.0.4.66
[pip3] nvidia-cusparse==12.6.3.3
[pip3] nvidia-cusparselt-cu13==0.8.1
[pip3] nvidia-nccl-cu13==2.30.7
[pip3] nvidia-nvjitlink==13.4.92
[pip3] nvidia-nvtx==13.0.85
[pip3] torch==2.14.0
[pip3] triton==3.8.0
[conda] Could not collect
```

</details>


cc @ezyang @gchanan @kadeng @msaroufim @crcrpar @mcarilli @janeyx99 @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises torchInductor in-place foreach lowering when a result reads a transformed view of the same destination.

## Observed implementation decision

The lowering assigns MutationLayout directly to a pointwise result whose reads are not position-identical to its writes.

## Physical effect

The fused kernel reads destination elements while other iterations overwrite them.

## Observed consequence

CPU and CUDA executions silently return wrong values.

## Reported proposal or workaround

PRs 198242 and 198549 propose preserving a copy or preventing realized-buffer storage swinging.

## Repair trace

Both direct repair proposals were unmerged at the study boundary; no landed patch and fixed regression establish the repair.

## Archived evidence

- Raw report: `data/raw/github/pytorch/198033/report.json`
- SHA-256: `c93335c743e5a7b45ec75f270b5e0dd266926f0707d9b463e1d2fbbdc48c8e11`
