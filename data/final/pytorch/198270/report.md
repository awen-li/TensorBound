# [inductor] x.index_put_(...) followed by x.add_(x.flip(0)): the copy back into x is fused with a read of x at other positions (silent wrong results, CUDA and CPU)

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/198270](https://github.com/pytorch/pytorch/issues/198270)
- Opened: 2026-09-22T20:36:21Z
- Closed: open at collection time
- Labels: `high priority`, `triage review`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `module: pt2-dispatcher`, `module: reinplacing`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug

An in-place update that reads the tensor at other positions, after an `index_put_` into the same tensor, gives wrong values under `torch.compile`. Either statement alone is fine.

```python
import torch


def f(x):
    x.index_put_((torch.arange(x.size(0), device=x.device),), x * 2.0)
    x.add_(x.flip(0))
    return x


device = "cuda" if torch.cuda.is_available() else "cpu"
compiled = torch.compile(f)
for shape in ((4, 8), (1024, 2048)):
    x = torch.rand(*shape, device=device)
    want = f(x.clone())
    print(shape, [int((compiled(x.clone()) != want).sum()) for _ in range(5)])
```

CUDA, 2.14.0+cu130, RTX 4070 (2.9.1 is the same): right at the small shape, wrong at the large one.

```
(4, 8) [0, 0, 0, 0, 0]
(1024, 2048) [1017616, 1020608, 1019568, 1020448, 1018848]
```

CPU, nightly 2.15.0.dev20260921: wrong at every size.

```
(4, 8) [16, 16, 16, 16, 16]
(1024, 2048) [1048576, 1048464, 1048576, 1048576, 1048576]
```

Same with `index_copy_`, `index_add_` or `x[idx] = ...` as the first statement, and with `x.copy_(x.flip(0))`, `x[1:] = x[:-1].clone()` or a foreach update as the second. With `x.mul_(2.0)` as the first statement it's right.

After reinplacing, `index_put_` writes `x` directly, and the flip and the add read `x` through the name of the index_put_ output. The copy back into `x` is then fused with them, so one kernel reads `x` at the flipped index while writing it:

```python
triton_poi_fused_add_copy__flip_1.run(arg0_1, arg0_1, 32, ...)
```

I think the check that should stop that is `Scheduler.fusable_weak_dep`. It looks for reads named `mutation_real_name[weak_dep.mutating_buf]` (`arg0_1`), but the reader reads `buf1` (the index_put_ output, whose real name is also `arg0_1`), so it finds no reads, skips the same-index check and allows the fusion. Resolving the read's name too:

```python
if self.mutation_real_name.get(read.name, read.name) == real_name
```

makes all of these right, CPU and CUDA. I'm running the Inductor tests against it and will send it.

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


cc @ezyang @gchanan @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @muchulee8 @amjames @aakhundov @coconutruben @jataylo @bdhirsh @bobrenjc93 @aorenste

## Trigger

x.index_put_(...) is followed by x.add_(x.flip(0)), so the second mutation reads the same storage at reversed positions.

## Relevant entities

base tensor storage; index_put_ output name; later in-place mutation; flipped read; WeakDep; mutation_real_name

## Implemented relation

Every versioned name produced by an in-place mutation continues to identify the same backing storage for dependency checks against later mutations.

## Pre-state

index_put_ produces a new graph name for x while preserving x's physical storage identity.

## Binding transition

fusable_weak_dep counts reads whose graph name equals the later mutation's resolved backing-buffer name.

## Post-state

The flipped read uses the index_put_ output name, so the old equality test finds no relevant read and permits fusion with the later write.

## Validity obligation

Before fusing a reader with a later mutation, read identities are resolved through the mutation chain and differing read/write indices remain ordered across kernels.

## Invalid decision

The dependency check resolved the later mutation to its real buffer but compared each read's unresolved versioned name.

## Physical effect

One kernel reads x at reversed positions while writing x, allowing reads to observe elements already changed by other iterations.

## Observed consequence

The reported program returns widespread silent value mismatches on CPU and CUDA, including about one million differing elements at shape 1024 by 2048.

## Repair site

torch/_inductor/scheduler.py::fusable_weak_dep

## Repair

Commit 8513bdd resolves every candidate read through mutation_real_name before comparing it with the mutated buffer. It adds the report's index_put_-then-flip regression.

## Why the repair works

The dependency check now sees all reads of the backing storage across mutation-version names and rejects fusion when their index map differs from the later write.

## Causal basis

The landed commit states that it fixes 198270, names the missed earlier-mutation read, changes fusable_weak_dep at that comparison, and adds the exact index_put_-then-flip regression.

## Analysis implication

Mutation-fusion analysis needs storage identity across versioned graph names, mutation order, read and write index maps, and the scope of concurrent iterations.

## Fix patches

- [`fixes/pytorch-198270-commit-8513bdd.patch`](fixes/pytorch-198270-commit-8513bdd.patch) — https://github.com/pytorch/pytorch/commit/8513bddb54689f22e4af0cdd71bdebb3989ac9d9
