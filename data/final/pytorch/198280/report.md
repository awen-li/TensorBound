# [inductor] x.copy_(x.transpose(-1, -2) * 1.0) on CUDA: a weak dep pruned before loop reordering lets the copy fuse with a transposed read (silent wrong results, regression)

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/198280](https://github.com/pytorch/pytorch/issues/198280)
- Opened: 2026-09-22T21:40:10Z
- Closed: Open at collection
- Labels: `high priority`, `triage review`, `module: regression`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug

Transposing a batch of matrices in place gives wrong values under `torch.compile` on CUDA at some sizes. Eager is fine, and so is 2.9.1.

```python
import torch


def f(x):
    x.copy_(x.transpose(-1, -2) * 1.0)
    return x


compiled = torch.compile(f)
for shape in ((652, 5, 5), (80000, 5, 5), (64, 256, 256)):
    x = torch.rand(*shape, device="cuda")
    want = f(x.clone())
    print(shape, [int((compiled(x.clone()) != want).sum()) for _ in range(5)])
```

2.14.0+cu130, RTX 4070:

```
(652, 5, 5) [0, 0, 0, 0, 0]
(80000, 5, 5) [2499, 13060, 13165, 13558, 13757]
(64, 256, 256) [0, 0, 0, 0, 0]
```

(the last shape gives 3,000 to 4,000 wrong in other runs). Same with `x[:] = x.transpose(-1, -2).clone()`, `x.copy_(x.transpose(-1, -2).contiguous())` and `x.copy_(x.mT.clone())`. With `torch._inductor.config.loop_ordering_after_fusion = False` it's right, which I think is why 2.9.1 is.

In the pre-fusion IR, op0 computes the transposed value and reads `arg0_1` at the natural index; op1 is the copy back into `arg0_1` (buf1, a mutation of arg0_1), reads buf0 at the transposed index, and has `WeakDep(name='buf0', mutating_buf='buf1')`. `_prune_redundant_deps` drops that weak dep, because op1 also has a MemoryDep on buf0 and `fusable_weak_dep` sees op0 read `arg0_1` at the index op1 writes it. Then `shared_data_after_reordering_loop` reorders op0's loops so that its write of buf0 matches op1's read, and that moves op0's read of `arg0_1` to the transposed index too. Nothing checks the mutation again, and the fused kernel reads `arg0_1` transposed while writing it:

```
triton_poi_fused_copy_copy__mul_0.run(arg0_1, arg0_1, 400000, 5, stream=raw_stream0)
```

Not reordering the loops of a pair where one node mutates a buffer the other reads makes all of these right. Would you rather have that, or the pruned weak deps checked again after a reorder? I can send either.

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


cc @ezyang @gchanan @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises torchInductor loop reordering and fusion of an in-place copy with a transposed read of its destination.

## Observed implementation decision

A weak mutation dependency is pruned before loop reordering, and the reordered producer no longer satisfies the positional-read condition used to prune it.

## Physical effect

The fused kernel reads transposed elements while writing the same storage.

## Observed consequence

Large CUDA inputs produce nondeterministic wrong elements.

## Reported proposal or workaround

PR 198328 proposes preventing loop reordering across the read/mutate pair.

## Repair trace

PR 198328 was not merged, so the regression lacks a landed scheduler repair and fixed test.

## Archived evidence

- Raw report: `data/raw/github/pytorch/198280/report.json`
- SHA-256: `ba2dc5008905fa6c5a521673e82e7ec6a6c1381153f36c79a63b8606f93b96da`
