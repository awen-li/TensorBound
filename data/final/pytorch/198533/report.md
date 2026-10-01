# [inductor] y.index_put_((mask,), v) with a mask that reads y through a transposed view reads the update's own output (regression)

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/198533](https://github.com/pytorch/pytorch/issues/198533)
- Opened: 2026-09-24T19:47:02Z
- Closed: Open at collection
- Labels: `triage review`, `module: regression`, `module: advanced indexing`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug

```python
import torch


def f(x):
    y = x + x.sum(-2, keepdim=True)
    mask = y.roll(-1, -2) == y.transpose(-1, -2)
    y.index_put_((mask,), torch.tensor(7.0))
    return y


x = (torch.arange(100) % 3).float().view(4, 5, 5)
print((f(x) != torch.compile(f)(x)).sum().item())
```

```
7
```

Wrong on CPU and CUDA (7 of 100 elements here, 21 of 147 at 3 x 7 x 7 on CUDA), silently, and with `accumulate=True` too. On 2.14.0 `y[mask] = 7` does the same (the nightly traces that one another way and gets it right); `y.masked_fill_(mask, 7)` is right. 2.14.0 and the nightlies of 21 and 23 Sep are wrong, 2.9.1 is right.

The post-grad graph is fine (`index_put_(add, [eq], full)`, with `eq` reading `add` rolled and permuted). In the generated code the permuted read loads the buffer the fill is writing: the loop body of `buf1` has `ops.load('buf1', 25*p0 + p1 + 5*p2)`, and the C++ kernel does `transpose_mxn(out_ptr1 + ...)` with `out_ptr1 = buf1`, where it should read `buf0` (`add`).

I think this is `mutate_to`'s fast path. `index_put_as_masked_fill` calls `mutate_to(self, where(mask, value, self))`, and since `add` is not an input the fast path swings its StorageBox to the new value (`changed_data.data = val.data`). `add` was already realized (the roll's `index` needs it), so `y.transpose(-1, -2)` was lowered as a ReinterpretView of that StorageBox, and ReinterpretView's loader looks the buffer name up only when it runs, after lowering, when the name is already the fill's. The roll is only there to get `add` realized before the transpose is lowered.

Skipping the fast path when the target is already realized, and writing the value into it with `realize_into` instead, makes these right on CPU and CUDA. I have it with a test and can open a PR.

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


cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises torchInductor mutate_to handling of a realized tensor read through rolled and transposed views during index_put_.

## Observed implementation decision

The fast path swings a realized StorageBox to the update result; existing ReinterpretView loaders resolve the new buffer name and read the update's output.

## Physical effect

The mask calculation reads values being written by the same index_put operation.

## Observed consequence

CPU and CUDA return silently incorrect filled elements.

## Reported proposal or workaround

PRs 198549 and 198715 propose disabling the fast path for realized targets and related alias repairs.

## Repair trace

The direct proposals were unmerged at the study boundary; no landed repair and fixed regression complete the chain.

## Archived evidence

- Raw report: `data/raw/github/pytorch/198533/report.json`
- SHA-256: `f744dd6e2cb9c6cadf29c220050f78c30f5f928f9b29305ace58260c94b12fe7`
