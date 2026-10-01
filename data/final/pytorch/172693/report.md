# Miscompile: torch.compile inductor dynamic compilation assumes strided tensor is contiguous when reshaping

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/172693](https://github.com/pytorch/pytorch/issues/172693)
- Opened: 2026-01-17T00:13:16Z
- Closed: 2026-02-06T03:28:41Z
- Labels: `high priority`, `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: dynamic shapes`, `module: inductor`

## Reported behavior

### 🐛 Describe the bug

When performing a reshape on a strided tensor, torch.compile with dynamic compilation appears to assume the tensor is contiguous. The key code is:

```python
key = torch.empty_strided((2, 16, 9), (288, 9, 1), device=device).normal_()
reshaped_key = key.reshape(key.shape[0], 8, -1)
reshaped_key + 1
```

Calling contiguous on the key or compiling with dynamic=False avoids the bug. This is significantly reduced from a program where dynamic compilation was triggered automatically, so there was weird behavior around what had been previously compiled in the same context. Hopefully I haven't introduced something invalid while creating the minimal reproducer. I've attached a reproduction script and the log from TORCH_TRACE

[dedicated_log_torch_trace_vw5keozw.log](https://github.com/user-attachments/files/24683689/dedicated_log_torch_trace_vw5keozw.log)
[torch_compile_dynamic_strides_bug.py](https://github.com/user-attachments/files/24683688/torch_compile_dynamic_strides_bug.py)

### Error logs

Huge difference between eager and compiled output (like 10^33)

### Versions

I tested this on various versions. I figured that nightly cpu was the simplest of these:

<details>

```
PyTorch version: 2.11.0.dev20260116+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0
Clang version: 14.0.0-1ubuntu1.1
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.12.12 (main, Jan  9 2026, 19:18:18) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
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
CPU(s):                                  256
On-line CPU(s) list:                     0-255
Vendor ID:                               AuthenticAMD
Model name:                              AMD EPYC 9575F 64-Core Processor
CPU family:                              26
Model:                                   2
Thread(s) per core:                      2
Core(s) per socket:                      64
Socket(s):                               2
Stepping:                                1
Frequency boost:                         enabled
CPU max MHz:                             5008.0068
CPU min MHz:                             1500.0000
BogoMIPS:                                6590.17
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 invpcid_single hw_pstate ssbd mba ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local avx_vnni avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif v_spec_ctrl avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid bus_lock_detect movdiri movdir64b overflow_recov succor smca fsrm avx512_vp2intersect flush_l1d
Virtualization:                          AMD-V
L1d cache:                               6 MiB (128 instances)
L1i cache:                               4 MiB (128 instances)
L2 cache:                                128 MiB (128 instances)
L3 cache:                                512 MiB (16 instances)
NUMA node(s):                            2
NUMA node0 CPU(s):                       0-63,128-191
NUMA node1 CPU(s):                       64-127,192-255
Vulnerability Gather data sampling:      Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; STIBP always-on; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Not affected
Vulnerability Vmscape:                   Not affected

Versions of relevant libraries:
[pip3] numpy==2.3.5
[pip3] torch==2.11.0.dev20260116+cpu
[pip3] torchvision==0.25.0.dev20260116+cpu
[conda] Could not collect
```

</details>

cc @ezyang @gchanan @kadeng @msaroufim @chauhang @penguinwu @bobrenjc93 @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Trigger

A tensor with shape (2, 16, 9) and stride (288, 9, 1) is reshaped to (2, 8, -1) under dynamic compilation.

## Relevant entities

input storage; symbolic input sizes; input strides; reshape view; output strides; logical-to-storage index

## Implemented relation

A reshape view maps every output logical index to the same storage element as eager reshape using strides valid for the input layout.

## Pre-state

The input has a gap between its two leading slices but its adjacent inner dimensions can still be collapsed into a valid view.

## Binding transition

The old dynamic reshape branch treats the view like a contiguous reshape and assigns contiguous output strides without first deriving them from the input's physical strides.

## Post-state

The compiled view indexes the original gapped storage as if the leading slices were adjacent, so output positions after the first slice bind to the wrong bytes.

## Validity obligation

A metadata-only reshape either computes output strides from the actual input size and stride or materializes/reindexes the tensor when no valid view exists.

## Invalid decision

Dynamic lowering selected contiguous output strides solely from the new shape even though the input storage layout was noncontiguous.

## Physical effect

Generated loads use a dense address progression across a physical gap and read unrelated or uninitialized values.

## Observed consequence

The compiled reshape-plus-one output differs from eager execution by values on the order of 10^33; making the input contiguous or disabling dynamic shapes avoids the failure.

## Repair site

torch/_inductor/ir.py::View.create and torch/_inductor/sizevars.py::SizeVarAllocator.to_symints_or_ints

## Repair

Commit b337828 converts symbolic sizes and strides to SymInt, applies _compute_stride to the actual input layout, emits a ReinterpretView only when that calculation succeeds, and otherwise uses the dynamic reindex or contiguous-materialization path. It adds the report's empty_strided reshape as a regression.

## Why the repair works

The compiler now derives the view's physical stride mapping from the same size-and-stride conditions that determine whether eager reshape is a valid view, preserving the storage gap instead of erasing it.

## Causal basis

The issue links pull request 173062 as its fix and closes on landed commit b337828. The commit states that it fixes 172693, identifies the eager/compiled output-stride mismatch, changes View.create to compute a stride from the actual layout, and adds the exact reproducer as test_inductor_view_issue_172693.

## Analysis implication

Analysis of symbolic reshape needs the input size, input stride, storage offset, symbolic guards, chosen view strides, fallback materialization, and generated index map together.

## Fix patches

- [`fixes/pytorch-172693-commit-b337828.patch`](fixes/pytorch-172693-commit-b337828.patch) — https://github.com/pytorch/pytorch/commit/b337828b4b1793b0ec786400b80e7f4cdf0494d4
