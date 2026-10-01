# [Inductor][Triton][CUDA][sm_120] BF16 autocast fusion of embeddings, Linear, and RMSNorm silently produces wrong results

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/191433](https://github.com/pytorch/pytorch/issues/191433)
- Opened: 2026-07-29T03:20:01Z
- Closed: open at collection time
- Labels: `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

## Description

`torch.compile` silently produces incorrect CUDA results when a BF16-autocast
`Linear` output is added to FP32 embedding outputs and the result feeds an
RMSNorm‑style reduction. The eager result is finite and stable; the compiled
result is finite but shows a maximum absolute error of ≈ 2.2 in the reproducer
below.

The issue reproduces on an RTX 5080 (`sm_120`) and is shape‑dependent. The
provided shape (`batch=96`, `length=8`, `width=192`) reproduces consistently.

Preliminary debugging points to a fusion kernel that writes an FP32 embedding
intermediate into an in‑place buffer, then later reads that buffer while mixing
in a BF16 result, breaking correctness. Details in the Root cause analysis section.

## Reproduction

```python
import platform

import torch
from torch import nn

BATCH = 96
LENGTH = 8
WIDTH = 192


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(4, WIDTH) for _ in range(5))
        self.numeric = nn.Linear(2, WIDTH, bias=False)
        self.weight = nn.Parameter(torch.ones(WIDTH))

    def forward(self, indices, values):
        hidden = self.embeddings[0](indices[:, :, 0])
        for column, embedding in enumerate(self.embeddings[1:], start=1):
            hidden = hidden + embedding(indices[:, :, column])
        hidden = hidden + self.numeric(values)
        scale = hidden.float().square().mean(dim=-1, keepdim=True)
        return hidden * torch.rsqrt(scale + 1e-6).to(hidden.dtype) * self.weight


torch.manual_seed(0)
device = torch.device("cuda")
model = Model().to(device).eval()
indices = torch.arange(BATCH * LENGTH * 5, device=device).reshape(
    BATCH, LENGTH, 5
) % 4
values = torch.arange(
    BATCH * LENGTH * 2, device=device, dtype=torch.float32
).reshape(BATCH, LENGTH, 2) / 100.0
compiled = torch.compile(model, fullgraph=True, dynamic=False)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    expected = model(indices, values)
    actual = compiled(indices, values)
torch.cuda.synchronize()

delta = (expected.float() - actual.float()).abs()
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__} ({torch.version.git_version})")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"different: {(expected != actual).sum().item()}")
print(f"nonfinite: {(~torch.isfinite(actual)).sum().item()}")
print(f"max_delta: {delta.max().item():.9g}")
```

Actual output:

```text
Python: 3.14.6
PyTorch: 2.13.0+cu130 (cf30153c4c131c8164ee7798e5022d810682e2cb)
CUDA: 13.0
GPU: NVIDIA GeForce RTX 5080
different: 37901
nonfinite: 0
max_delta: 2.20142961
```

## Expected behavior

The compiled result may differ by normal BF16 rounding, but it must not exhibit
an absolute error of 2.2. For comparison:

- Removing the autocast `Linear` branch gives a maximum error of `2.38e-7`.
- Converting all model weights to native BF16 before compiling gives a maximum
  error of `0.015625`.

In the original Transformer policy, the default Inductor mode changes 2 to 3 of
128 greedy actions and has a maximum raw‑logit error between 16 and 20.
`reduce‑overhead` has also produced hundreds of non‑finite logits.

## Triage results

- Dynamo eager, AOT eager, the cudagraphs backend, `torch.export`, and
  TorchScript trace do **not** reproduce the large error.
- FP32 Inductor has a maximum error below `3.3e-4` on the full history encoder.
- Native BF16 weights avoid the large error.
- Forcing ATen‑only or Triton‑only GEMM does **not** remove the error.
- Disabling epilogue fusion, layout optimization, pattern matching, or in‑place
  buffers does **not** remove the error.
- Setting `torch._inductor.config.max_fusion_size = 1` reduces the maximum
  error of the minimal reproducer from `56.6` to `2.38e-7`.
- Detailed logging reports no graph breaks or recompilations.

## Root cause analysis

The faulty kernel is named similarly to:

```text
triton_red_fused__to_copy__unsafe_view_add_clamp_embedding_lt_mean_mul_pow_rsqrt_scalar_tensor_select_where_2
```

Its metadata includes `mutated_arg_names=['in_out_ptr0']`. The kernel fuses:

- embedding lookup and addition,
- the RMS reduction (`square → mean → rsqrt`),
- and the final output multiplication.

It first stores an FP32 embedding intermediate into `in_out_ptr0`, then reads
that buffer in a second reduction loop and adds the BF16 `Linear` result. This
mixing prevents the fusion from restoring correct numerical behaviour and is
the likely source of the large error.


### Error logs

_No response_

### Versions

PyTorch version: 2.13.0+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCM used to build PyTorch: N/A

OS: Fedora Linux 44 (Workstation Edition) (x86_64)
GCC version: (GCC) 16.1.1 20260515 (Red Hat 16.1.1-2)
Clang version: 22.1.8 (Fedora 22.1.8-1.fc44)
CMake version: version 4.3.0
Libc version: glibc-2.43

Python version: 3.14.6 | packaged by Anaconda, Inc. | (main, Jun 18 2026, 21:28:09) [GCC 14.3.0] (64-bit runtime)
Python platform: Linux-7.1.3-201.fc44.x86_64-x86_64-with-glibc2.43
Is CUDA available: True
CUDA runtime version: 13.2.86
CUDA_MODULE_LOADING set to:
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 5080
Nvidia driver version: 595.80
cuDNN version: Could not collect
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False
Caching allocator config: N/A

CPU:
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           48 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  32
On-line CPU(s) list:                     0-31
Vendor ID:                               AuthenticAMD
Model name:                              AMD Ryzen 9 9950X 16-Core Processor
CPU family:                              26
Model:                                   68
Thread(s) per core:                      2
Core(s) per socket:                      16
Socket(s):                               1
Stepping:                                0
Frequency boost:                         enabled
CPU(s) scaling MHz:                      74%
CPU max MHz:                             5756.4521
CPU min MHz:                             624.1940
BogoMIPS:                                8583.96
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl xtopology nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpuid_fault cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx_vnni avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid bus_lock_detect movdiri movdir64b overflow_recov succor smca fsrm avx512_vp2intersect flush_l1d amd_lbr_pmc_freeze
Virtualization:                          AMD-V
L1d cache:                               768 KiB (16 instances)
L1i cache:                               512 KiB (16 instances)
L2 cache:                                16 MiB (16 instances)
L3 cache:                                64 MiB (2 instances)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-31
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Old microcode:             Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Mitigation; IBPB on VMEXIT only
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; STIBP always-on; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Not affected
Vulnerability Vmscape:                   Mitigation; IBPB on VMEXIT

Versions of relevant libraries:
[pip3] numpy==2.4.6
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
[pip3] nvidia-nvjitlink==13.3.33
[pip3] nvidia-nvtx==13.0.85
[pip3] onnx==1.22.0
[pip3] onnxconverter-common==1.16.0
[pip3] torch==2.13.0
[pip3] triton==3.7.1
[conda] numpy                         2.4.6            pypi_0              pypi
[conda] nvidia-cublas                 13.1.1.3         pypi_0              pypi
[conda] nvidia-cuda-cupti             13.0.85          pypi_0              pypi
[conda] nvidia-cuda-nvrtc             13.0.88          pypi_0              pypi
[conda] nvidia-cuda-runtime           13.0.96          pypi_0              pypi
[conda] nvidia-cudnn-cu13             9.20.0.48        pypi_0              pypi
[conda] nvidia-cufft                  12.0.0.61        pypi_0              pypi
[conda] nvidia-curand                 10.4.0.35        pypi_0              pypi
[conda] nvidia-cusolver               12.0.4.66        pypi_0              pypi
[conda] nvidia-cusparse               12.6.3.3         pypi_0              pypi
[conda] nvidia-cusparselt-cu13        0.8.1            pypi_0              pypi
[conda] nvidia-nccl-cu13              2.29.7           pypi_0              pypi
[conda] nvidia-nvjitlink              13.3.33          pypi_0              pypi
[conda] nvidia-nvtx                   13.0.85          pypi_0              pypi
[conda] torch                         2.13.0           pypi_0              pypi
[conda] triton                        3.7.1            pypi_0              pypi

cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Duplicate mapping

- Primary report: `github-pytorch-pytorch-190517`
- Shared cause: `cause-pytorch-190517-global-scratch-reload-lacks-cross-warp-barrier`

Report 191433 identifies an in-place intermediate stored before a second RMS reduction loop, shows that disabling fusion removes the large error, and is triaged against report 190517 and PR 190519. The primary report's landed repair inserts the missing barrier at this same invalidated-store reload boundary and adds a matching normalization regression.

The sections below describe the shared cause reconstructed from the primary report. The reported behavior above remains the observation specific to this report.

## Trigger

A fused normalization kernel materializes partial values to a global output/scratch buffer and a later reduction loop reads those values back.

## Relevant entities

producer reduction loop; global scratch buffer; store warp; consumer reduction loop; load warp; invalidated-store set; barrier

## Implemented relation

A load in a later loop observes all earlier writes to the same global scratch range, including writes issued by other warps in the kernel.

## Pre-state

Different warps execute stores for different portions of the materialized intermediate.

## Binding transition

Code generation emits the later reload after the reduction-loop boundary without a cross-warp barrier, even though the buffer is recorded in cse.invalidated_stores.

## Post-state

A consumer warp can reach its load before another warp's corresponding store is visible.

## Validity obligation

A same-kernel global read-after-write edge that crosses warps is ordered by a barrier before the first reload.

## Invalid decision

The load path did not discharge the read-after-write synchronization obligation tracked by invalidated_stores.

## Physical effect

The second loop reads stale or partially written global scratch values and incorporates them into the reduction result.

## Observed consequence

The reported pit_b_224 XPU inference fails accuracy with RMSE around 0.4; the second report observes nondeterministic compiled gradients on AMD GPUs.

## Repair site

torch/_inductor/codegen/triton.py load code generation

## Repair

Commit 62a4f51 emits tl.debug_barrier before a non-CPU load when its buffer name is in cse.invalidated_stores and adds a compiled normalization regression that checks for the barrier and correct output.

## Why the repair works

All warps complete and publish the earlier global stores before any warp performs the tracked reload, restoring the required read-after-write order.

## Causal basis

The landed commit explicitly fixes report 190517, names the cross-warp read-after-write race, adds the barrier at the tracked reload, and reports that the failing model now passes.

## Analysis implication

Fused-kernel analysis needs buffer identity, store and load loop regions, warp mapping, invalidation state, memory space, barrier placement, and reload consumers together.

## Fix patches

- [`fixes/pytorch-191433-commit-62a4f51.patch`](fixes/pytorch-191433-commit-62a4f51.patch) — https://github.com/pytorch/pytorch/commit/62a4f51cf34e32503ae138c8f977b434bb90a325
