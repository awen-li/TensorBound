# [Inductor][aot_autograd][autocast][dynamic_shape] torch.compile with dynamic=True produces incorrect values for fallback ops (e.g., sum) under autocast due to tracing/runtime autocast state mismatch

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/194329](https://github.com/pytorch/pytorch/issues/194329)
- Opened: 2026-08-21T06:14:27Z
- Closed: Open at collection
- Labels: `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: dynamic shapes`, `module: inductor`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

When using `torch.compile` with `dynamic=True` inside an `autocast` region, fallback operators (e.g., `aten.sum`) may produce garbage values (e.g., `-512.0`) during the backward pass. This leads to dtype mismatches between the compiled graph's buffer and the operator's runtime output.

This issue does not occur when `dynamic=False`.

```python
import torch
from torch import nn
from torch._inductor.lowering import make_fallback

make_fallback(torch.ops.aten.sum)

device = "cuda"
torch.manual_seed(1234)

class Repro(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.weight_normed_linear = nn.utils.parametrizations.weight_norm(
            nn.Linear(in_features, out_features=2)
        )
        self.linear = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        return self.linear(self.weight_normed_linear(x))

def f(m, x):
    with torch.amp.autocast(device_type=device, dtype=torch.half):
        loss = m(x).sum()
        loss.backward()
    return loss

in_features = 1025
x = torch.randn(2, in_features, dtype=torch.half, requires_grad=True).to(
    device=device
)
m = Repro(in_features)
m = m.to(device)

f(m, x)

ref_grad_list = [p.grad for p in m.parameters()]

for p in m.parameters():
    p.grad = None

opt_f = torch.compile(f, dynamic=True) # fail
# opt_f = torch.compile(f, dynamic=False) # pass

opt_f(m, x)
act_grad_list = [p.grad for p in m.parameters()]
torch.testing.assert_close(ref_grad_list, act_grad_list)
```
output
```
Traceback (most recent call last):
  File "/root/test.py", line 46, in <module>
    torch.testing.assert_close(ref_grad_list, act_grad_list)
  File "/usr/local/lib/python3.12/dist-packages/torch/testing/_comparison.py", line 1631, in assert_close
    raise error_metas[0].to_error(msg)
AssertionError: Tensor-likes are not close!

Mismatched elements: 2 / 2 (100.0%)
Greatest absolute difference: 511.45361328125 at index (0,) (up to 1e-05 allowed)
Greatest relative difference: 1.09567391872406 at index (1,) (up to 1.3e-06 allowed)

The failure occurred for item [0]
```

### Versions

Collecting environment information...
PyTorch version: 2.13.0+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.4 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Clang version: Could not collect
CMake version: version 4.3.4
Libc version: glibc-2.39

Python version: 3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-6.6.98-40.4.tl4.x86_64-x86_64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: 13.0.88
CUDA_MODULE_LOADING set to:
GPU models and configuration:
GPU 0: NVIDIA B300 SXM6 AC
GPU 1: NVIDIA B300 SXM6 AC

Nvidia driver version: 580.159.03
cuDNN version: Could not collect
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False
Caching allocator config: N/A

CPU:
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           52 bits physical, 57 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  256
On-line CPU(s) list:                     0-255
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Xeon(R) 6768P
CPU family:                              6
Model:                                   173
Thread(s) per core:                      2
Core(s) per socket:                      64
Socket(s):                               2
Stepping:                                1
CPU(s) scaling MHz:                      60%
CPU max MHz:                             3900.0000
CPU min MHz:                             800.0000
BogoMIPS:                                4800.00
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cat_l2 cdp_l3 intel_ppin cdp_l2 ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect avx_vnni avx512_bf16 wbnoinvd dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req vnmi avx512vbmi umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid bus_lock_detect cldemote movdiri movdir64b enqcmd fsrm md_clear serialize tsxldtrk pconfig arch_lbr ibt amx_bf16 avx512_fp16 amx_tile amx_int8 flush_l1d arch_capabilities
Virtualization:                          VT-x
L1d cache:                               6 MiB (128 instances)
L1i cache:                               8 MiB (128 instances)
L2 cache:                                256 MiB (128 instances)
L3 cache:                                672 MiB (2 instances)
NUMA node(s):                            4
NUMA node0 CPU(s):                       0-31,128-159
NUMA node1 CPU(s):                       32-63,160-191
NUMA node2 CPU(s):                       64-95,192-223
NUMA node3 CPU(s):                       96-127,224-255
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
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Vulnerable: eIBRS with unprivileged eBPF
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Not affected

Versions of relevant libraries:
[pip3] numpy==2.5.1
[pip3] nvidia-cublas==13.1.1.3
[pip3] nvidia-cuda-cupti==13.0.85
[pip3] nvidia-cuda-nvrtc==13.0.88
[pip3] nvidia-cuda-runtime==13.0.96
[pip3] nvidia-cudnn-cu13==9.20.0.48
[pip3] nvidia-cufft==12.0.0.61
[pip3] nvidia-curand==10.4.0.35
[pip3] nvidia-cusolver==12.0.4.66
[pip3] nvidia-cusparse==12.6.3.3
[pip3] nvidia-cusparselt-cu13==0.8.1
[pip3] nvidia-nccl-cu13==2.29.7
[pip3] nvidia-nvjitlink==13.2.78
[pip3] nvidia-nvtx==13.0.85
[pip3] optree==0.19.1
[pip3] torch==2.13.0+cu130
[pip3] torchaudio==2.11.0+cu130
[pip3] torchelastic==0.2.2
[pip3] torchvision==0.28.0+cu130
[pip3] triton==3.7.1
[conda] Could not collect


cc @chauhang @penguinwu @ezyang @bobrenjc93 @aditvenk @laithsakka @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises dynamic-shape AOTAutograd/Inductor fallback execution under autocast.

## Observed implementation decision

Tracing and runtime disagree about autocast state and therefore about the dtype of a fallback operator's output buffer.

## Physical effect

The fallback writes values under a dtype contract different from the compiled graph's allocation.

## Observed consequence

Backward gradients contain large incorrect values.

## Reported proposal or workaround

The report proposes disabling runtime autocast or making AOTAutograd represent its state.

## Repair trace

The issue remains open with no implemented repair, maintainer adjudication, or fixed regression.

## Archived evidence

- Raw report: `data/raw/github/pytorch/194329/report.json`
- SHA-256: `7d499589504fe59a1795c047e62a0de899d1e1e1823ec2af136bfcd4765acee2`
