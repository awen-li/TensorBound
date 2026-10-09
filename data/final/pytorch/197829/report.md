# [inductor] `x[1:] = x[:-1].clone()` compiles to a copy over overlapping memory: silent wrong results on CUDA and CPU

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/197829](https://github.com/pytorch/pytorch/issues/197829)
- Opened: 2026-09-20T18:06:23Z
- Closed: 2026-09-22T00:22:06Z
- Labels: `high priority`, `module: partial aliasing`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug

Shifting the rows of a tensor in place gives wrong values under `torch.compile`. Eager is fine.

```python
import torch

def shift(x):
    x[1:] = x[:-1].clone()   # every row takes the row before it
    return x

compiled = torch.compile(shift)
for device in ("cuda", "cpu"):
    for shape in ((3, 16, 16), (1024, 2048)):
        x = torch.rand(*shape, device=device)
        want = shift(x.clone())
        wrong = [int((compiled(x.clone()) != want).sum()) for _ in range(5)]
        print(device, shape, wrong, "of", want.numel())
```

On 2.14.0+cu130 with an RTX 4070:

```
cuda (3, 16, 16) [0, 0, 0, 0, 0] of 768
cuda (1024, 2048) [0, 39968, 45984, 11328, 26496] of 2097152
cpu (3, 16, 16) [256, 256, 256, 256, 256] of 768
cpu (1024, 2048) [2070528, 2070528, 2070528, 2070528, 2070528] of 2097152
```

On CUDA it is right at the small shape and wrong, differently from run to run, at the large one, so a test at a small shape does not see it. On the CPU backend it is wrong at any size: row 0 ends up in every row. The CPU backend of today's nightly (2.15.0.dev20260920+cpu) is wrong in the same way.

The clone is gone from the generated code, and the kernel gets the same buffer as its input and its output:

```python
@triton.jit
def triton_poi_fused_slice_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (256 + x0), tmp0, xmask)

# in call():
triton_poi_fused_slice_0.run(arg0_1, arg0_1, 512, stream=raw_stream0)
```

Other spellings that are wrong in the same way: `x[1:].copy_(x[:-1].clone())`, `x[1:] = x[:-1] + 0.0`, `x[2:] = x[:-2].clone()`, `t = x[:-1].clone(); x[1:] = t`, and on CUDA also `x[:-1] = x[1:].clone()`. These are right: `x[:, 1:] = x[:, :-1].clone()`, `x.copy_(torch.roll(x, 1, 0))`, and the shift followed by `x[0] = 0`.

I looked for an existing report and did not find one.

### Versions

<details>
<summary>collect_env (the CUDA machine)</summary>

```
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
CPU(s) scaling MHz:                      41%
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

The CPU numbers above are from `torch==2.14.0+cpu` and `torch==2.15.0.dev20260920+cpu` (pip wheels, Python 3.12, Linux x86_64).


cc @ezyang @gchanan @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Trigger

A row shift assigns x[:-1].clone() into x[1:], after a prior pass has removed the clone while the graph still has functional scatter semantics.

## Relevant entities

base storage; source slice; destination slice; removed clone; functional generalized scatter; reinplaced copy_; kernel blocks

## Implemented relation

A mutating copy reads a stable source value for every destination element when its source and destination ranges overlap within one storage.

## Pre-state

After clone removal, both the source slice x[:-1] and the scatter input x refer to the same base storage, while the functional scatter would still clone its input before applying updates.

## Binding transition

The reinplacing pass converts the functional scatter into view(inp).copy_(src) based on later uses of inp but does not test whether src shares inp's storage.

## Post-state

The generated copy reads earlier rows from the same allocation while simultaneously overwriting later, overlapping rows.

## Validity obligation

Scatter reinplacing is allowed only when its source storage is independent of the storage that the mutating decomposition overwrites.

## Invalid decision

The pre-fix in-place safety test considered later uses of `inp` but omitted the comparison between `get_node_storage(inp)` and `get_node_storage(src)`.

## Physical effect

CPU iteration propagates row zero through later rows, while concurrent GPU blocks race between source loads and destination stores.

## Observed consequence

Large CUDA inputs produce run-dependent wrong elements and CPU inputs deterministically repeat row zero, although eager clone semantics shift each row exactly once.

## Repair site

torch/_inductor/fx_passes/reinplace.py::should_reinplace_scatter

## Repair

Commit 0f9fcef obtains the storage identities of the generalized-scatter input and source and declines reinplacing when they match. The functional decomposition then retains its cloned base. It adds the reported row-shift regression on CPU.

## Why the repair works

Keeping the operation functional separates the reads that construct the output from later mutation of the original storage, restoring the snapshot semantics supplied by the source clone.

## Causal basis

The issue closes on commit 0f9fcef, whose message states that it fixes 197829, reconstructs the clone-removal and reinplacing sequence, adds the storage-alias guard at should_reinplace_scatter, and includes the exact row-shift comparison.

## Analysis implication

Mutation optimization analysis needs base-storage identity, view ranges, source/destination overlap, clone elimination history, operation order, and parallel execution together.

## Fix patches

- [`fixes/pytorch-197829-commit-0f9fcef.patch`](fixes/pytorch-197829-commit-0f9fcef.patch) — https://github.com/pytorch/pytorch/commit/0f9fcefa1e62a80e51ad8f7b0f9f0d20c3411b2f
