# [inductor] y = x.clone(); op(y, x) becomes op(x, x) for custom ops and user-defined Triton kernels that mutate x: silent wrong results

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/198031](https://github.com/pytorch/pytorch/issues/198031)
- Opened: 2026-09-21T20:57:24Z
- Closed: Open at collection
- Labels: `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `module: pt2-dispatcher`, `module: reinplacing`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug

Same root cause as #197829 and #156786, a third way in: custom ops and user-defined Triton kernels. When the user passes a clone of the tensor the op mutates, the clone is removed and the op gets the same tensor twice.

```python
import torch


@torch.library.custom_op("mylib::reverse_into", mutates_args=("dst",))
def reverse_into(src: torch.Tensor, dst: torch.Tensor) -> None:
    n = src.shape[0]
    for k in range(n):
        dst[n - 1 - k] = src[k]


def f(x):
    y = x.clone()
    reverse_into(y, x)
    return x * 2.0


x = torch.arange(8.0)
a, b = x.clone(), x.clone()
f(a)
torch.compile(f)(b)
print(a.tolist())
print(b.tolist())
```

```
[7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
[0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0]
```

CPU, 2.14.0 and nightly 2.15.0.dev20260921, with `enable_auto_functionalized_v2` on or off. The generated code calls `torch.ops.mylib.reverse_into.default(arg0_1, arg0_1)`.

With a Triton kernel it is a race between program instances, so it only shows once the tensor is large enough:

```python
import torch
import triton
import triton.language as tl


@triton.jit
def reverse_kernel(src, dst, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst + (n - 1 - offs), tl.load(src + offs, mask=mask), mask=mask)


def f(x):
    y = x.clone()
    n = x.numel()
    reverse_kernel[(triton.cdiv(n, 1024),)](y, x, n, BLOCK=1024)
    return x * 2.0


compiled = torch.compile(f)
for n in (4096, 1 << 21):
    x = torch.rand(n, device="cuda")
    want = f(x.clone())
    print(n, [int((compiled(x.clone()) != want).sum()) for _ in range(5)])
```

2.14.0+cu130, RTX 4070:

```
4096 [0, 0, 0, 0, 0]
2097152 [870624, 851048, 850368, 840736, 862704]
```

```python
reverse_kernel_0.run(arg0_1, arg0_1, 4096, 4, 1, 1, stream=raw_stream0)
```

`remove_noop_ops` drops the clone, which is fine while the graph is functional. `reinplace_and_refine_tensors_to_clone` then decides `x` needs no clone because nothing uses it after the node. It doesn't look at the other arguments of the same node, so `functional_op(src=x, dst=x)` becomes the mutating op on `x`.

I have a patch that keeps the clone of a mutated argument when another argument of the node overlaps what the op writes (for v2, the mutated views rather than the whole base). What it costs: a call the user wrote as `op(x, x)` is no longer done in place, since after `remove_noop_ops` it is the same graph. Would you rather have that, or keep the clone in `remove_noop_ops` when its consumer mutates the source? I can send either.

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


cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo @bdhirsh @bobrenjc93 @aorenste

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

A graph clones x and passes the clone as a read argument to an operation that mutates x.

## Observed implementation decision

Clone removal and reinplacement consider later uses but omit another argument of the same mutating node that reads the storage.

## Physical effect

The compiled operation receives x for both its source and destination, so reads race or follow earlier writes.

## Observed consequence

The custom-op reproducer returns a partially mirrored vector and the Triton reproducer has many mismatched elements.

## Reported proposal or workaround

PR 198260 proposes retaining a clone when another argument reads storage that the node mutates.

## Repair trace

The existing repair for report 197829 covers overlapping scatter copies, not custom-op or user-kernel argument aliasing. Direct PR 198260 was not merged, so no landed change and fixed regression close this report's repair chain.

## Archived evidence

- Raw report: `data/raw/github/pytorch/198031/report.json`
- SHA-256: `9ab19caf89523287c9aded43ad5b26344b880f19bdbe4238343d620b45ff024b`
