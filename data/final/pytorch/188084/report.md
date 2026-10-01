# [BUG] Inductor reinplaces `index_put` on expanded view and silently corrupts gradients

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/188084](https://github.com/pytorch/pytorch/issues/188084)
- Opened: 2026-06-24T17:46:02Z
- Closed: 2026-06-24T18:04:58Z
- Labels: `triage review`, `module: correctness (silent)`, `oncall: pt2`, `module: functorch`, `module: inductor`, `module: pt2-dispatcher`, `module: reinplacing`, `dynamo-functorch`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

## Summary

`torch.compile(torch.func.vmap(torch.func.grad_and_value(f)), fullgraph=True, dynamic=True)` can silently return an incorrect gradient when the backward graph contains functional `aten.index_put(..., accumulate=True)` whose target is an internally overlapping `expand` view.

The forward value is correct. The gradient is wrong by an exact batch-size factor. Reproduced on CPU and A100/CUDA with the same wrong-gradient magnitude.

## Minimal repro

```python
import torch
from torch.func import grad_and_value, vmap

torch.set_default_dtype(torch.float64)

B, D = 4, 4
idx = torch.tensor([0, 2])
w = torch.tensor([1.0, -1.0])
X = torch.randn(B, D)


def f(x):
    stacked = torch.stack([x[..., k] for k in range(D)], dim=-1)
    return (torch.index_select(stacked, -1, idx) * w).sum(-1)


def ref_grad():
    Xr = X.clone().requires_grad_(True)
    vals = torch.stack([f(Xr[i : i + 1]).reshape(()) for i in range(B)])
    return torch.autograd.grad(vals.sum(), Xr)[0]


def row(x):
    return f(x).reshape(())


ref = ref_grad()

for backend in ("eager", "aot_eager", "inductor"):
    torch._dynamo.reset()
    g, v = torch.compile(
        vmap(grad_and_value(row)),
        backend=backend,
        fullgraph=True,
        dynamic=True,
    )(X.clone())
    print(backend, "max grad err:", (g - ref).abs().max().item())
    print("  ref[0]     =", ref[0].tolist())
    print("  compiled[0]=", g[0].tolist())
```

Actual:

```text
eager max grad err: 0.0
aot_eager max grad err: 0.0
inductor max grad err: 3.0
  ref[0]      = [1.0, 0.0, -1.0, 0.0]
  compiled[0] = [4.0, 0.0, -4.0, 0.0]
```

Expected:

```text
inductor max grad err: 0.0
```

Replacing `torch.index_select(stacked, -1, idx)` with `stacked[..., idx]` fixes the issue.

## Root cause [claude/gpt-assisted analysis]

The post-grad graph contains this pattern:

```text
full_1    = aten.full([D], 0)
expand_2  = aten.expand(full_1, [B, D])       # stride (0, 1), internal overlap
index_put = aten.index_put(expand_2, [None, idx], grad, accumulate=True)
```

For `expand_2.meta["val"]`:

```python
torch._debug_has_internal_overlap(expand_2.meta["val"]) == 1
```

Functional `aten.index_put` lowering clones the target first:

```python
# torch/_inductor/lowering.py
@register_lowering(aten.index_put, type_promotion_kind=None)
def index_put(x, indices, values, accumulate=False):
    return index_put_impl_(
        clone(x), indices, values, accumulate, check=True, may_realize=False
    )
```

That clone materializes a real non-overlapping `[B, D]` buffer, preserving functional semantics.

However, `torch/_inductor/fx_passes/reinplace.py` marks `aten.index_put.default` as inplaceable:

```python
inplaceable_ops = {
    aten.index_put.default: InplaceableOp(aten.index_put_.default, 0),
    ...
}
```

The reinplace pass rewrites the functional `index_put` to `index_put_`, so the scatter-add mutates the stride-0 expanded view directly. All vmap rows alias the same storage, and the gradient folds across the batch.

The existing Inductor `can_inplace` logic checks cross-tensor aliasing among tensors sharing storage, but it does not reject a mutated argument with internal self-overlap. The generic FX reinplace pass already has this safety check:

```python
# torch/fx/passes/reinplace.py
if torch._debug_has_internal_overlap(self_meta) == 1:
    self_has_wrong_metadata = True
```

## Proposed fix

Add the same definite-internal-overlap guard to Inductor's reinplace safety check for mutated args:

```python
def can_inplace(node, mutated_arg):
    ...

    if get_node_storage(mutated_arg) is None:
        return False

    mutated_val = mutated_arg.meta.get("val", None)
    if (
        isinstance(mutated_val, torch.Tensor)
        and torch._debug_has_internal_overlap(mutated_val) == 1
    ):
        return False

    shared_view_nodes = storage_to_nodes[get_node_storage(mutated_arg)]
    ...
```

A narrower alternative is to add this as an `extra_check` for `aten.index_put.default` and aten._unsafe_index_put.default` only.

### Error logs

None, bug is silent

### Versions

Collecting environment information...
PyTorch version: 2.12.1+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCM used to build PyTorch: N/A

OS: AlmaLinux 10.1 (Heliotrope Lion) (x86_64)
GCC version: (GCC) 14.3.1 20250617 (Red Hat 14.3.1-2)
Clang version: 21.1.8 (AlmaLinux OS Foundation 21.1.8-1.el10.alma.1)
CMake version: version 4.3.2
Libc version: glibc-2.39

Python version: 3.13.12 | packaged by conda-forge | (main, Feb  5 2026, 05:53:46) [GCC 14.3.0] (64-bit runtime)
Python platform: Linux-6.12.0-124.52.3.el10_1.x86_64-x86_64-with-glibc2.39
Is CUDA available: False
CUDA runtime version: 13.2.78
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: Could not collect
Nvidia driver version: Could not collect
cuDNN version: Could not collect
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True
Caching allocator config: N/A

CPU:
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           45 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  6
On-line CPU(s) list:                     0-5
Vendor ID:                               AuthenticAMD
Model name:                              AMD Ryzen 7 9800X3D 8-Core Processor
CPU family:                              26
Model:                                   68
Thread(s) per core:                      1
Core(s) per socket:                      6
Socket(s):                               1
Stepping:                                0
BogoMIPS:                                9382.37
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext ssbd ibrs ibpb ibrs_enhanced vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves user_shstk avx_vnni avx512_bf16 clzero wbnoinvd arat npt svm_lock nrip_save vmcb_clean flushbyasid decodeassists avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid movdiri movdir64b overflow_recov succor fsrm avx512_vp2intersect flush_l1d
Virtualization:                          AMD-V
Hypervisor vendor:                       VMware
Virtualization type:                     full
L1d cache:                               288 KiB (6 instances)
L1i cache:                               192 KiB (6 instances)
L2 cache:                                6 MiB (6 instances)
L3 cache:                                96 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-5
Vulnerability Gather data sampling:      Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Vulnerable: Safe RET, no microcode
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; STIBP disabled; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Not affected
Vulnerability Vmscape:                   Not affected

Versions of relevant libraries:
[pip3] botorch==0.17.2
[pip3] gpytorch==1.15.2
[pip3] mypy_extensions==1.1.0
[pip3] numpy==2.5.0
[pip3] numpyro==0.21.0
[pip3] nvidia-cublas==13.1.1.3
[pip3] nvidia-cublas-cu12==12.8.4.1
[pip3] nvidia-cuda-cupti==13.0.85
[pip3] nvidia-cuda-cupti-cu12==12.8.90
[pip3] nvidia-cuda-nvrtc==13.0.88
[pip3] nvidia-cuda-nvrtc-cu12==12.8.93
[pip3] nvidia-cuda-runtime==13.0.96
[pip3] nvidia-cuda-runtime-cu12==12.8.90
[pip3] nvidia-cudnn-cu12==9.10.2.21
[pip3] nvidia-cudnn-cu13==9.20.0.48
[pip3] nvidia-cufft==12.0.0.61
[pip3] nvidia-cufft-cu12==11.3.3.83
[pip3] nvidia-curand==10.4.0.35
[pip3] nvidia-curand-cu12==10.3.9.90
[pip3] nvidia-cusolver==12.0.4.66
[pip3] nvidia-cusolver-cu12==11.7.3.90
[pip3] nvidia-cusparse==12.6.3.3
[pip3] nvidia-cusparse-cu12==12.5.8.93
[pip3] nvidia-cusparselt-cu12==0.7.1
[pip3] nvidia-cusparselt-cu13==0.8.1
[pip3] nvidia-nccl-cu12==2.27.5
[pip3] nvidia-nccl-cu13==2.29.7
[pip3] nvidia-nvjitlink==13.0.88
[pip3] nvidia-nvjitlink-cu12==12.8.93
[pip3] nvidia-nvtx==13.0.85
[pip3] nvidia-nvtx-cu12==12.8.90
[pip3] onnxruntime==1.24.2
[pip3] torch==2.12.1
[pip3] torchvision==0.27.1
[pip3] triton==3.7.1
[conda] botorch                               0.17.2           pypi_0                 pypi
[conda] cuda-cudart                           13.2.75          hecca717_0             conda-forge
[conda] cuda-cudart-dev                       13.2.75          hecca717_0             conda-forge
[conda] cuda-cudart-dev_linux-64              13.2.75          h376f20c_0             conda-forge
[conda] cuda-cudart-static                    13.2.75          hecca717_0             conda-forge
[conda] cuda-cudart-static_linux-64           13.2.75          h376f20c_0             conda-forge
[conda] cuda-cudart_linux-64                  13.2.75          h376f20c_0             conda-forge
[conda] gpytorch                              1.15.2           pypi_0                 pypi
[conda] numpy                                 2.5.0            pypi_0                 pypi
[conda] numpyro                               0.21.0           pypi_0                 pypi
[conda] nvidia-cublas                         13.1.1.3         pypi_0                 pypi
[conda] nvidia-cublas-cu12                    12.8.4.1         pypi_0                 pypi
[conda] nvidia-cuda-cupti                     13.0.85          pypi_0                 pypi
[conda] nvidia-cuda-cupti-cu12                12.8.90          pypi_0                 pypi
[conda] nvidia-cuda-nvrtc                     13.0.88          pypi_0                 pypi
[conda] nvidia-cuda-nvrtc-cu12                12.8.93          pypi_0                 pypi
[conda] nvidia-cuda-runtime                   13.0.96          pypi_0                 pypi
[conda] nvidia-cuda-runtime-cu12              12.8.90          pypi_0                 pypi
[conda] nvidia-cudnn-cu12                     9.10.2.21        pypi_0                 pypi
[conda] nvidia-cudnn-cu13                     9.20.0.48        pypi_0                 pypi
[conda] nvidia-cufft                          12.0.0.61        pypi_0                 pypi
[conda] nvidia-cufft-cu12                     11.3.3.83        pypi_0                 pypi
[conda] nvidia-curand                         10.4.0.35        pypi_0                 pypi
[conda] nvidia-curand-cu12                    10.3.9.90        pypi_0                 pypi
[conda] nvidia-cusolver                       12.0.4.66        pypi_0                 pypi
[conda] nvidia-cusolver-cu12                  11.7.3.90        pypi_0                 pypi
[conda] nvidia-cusparse                       12.6.3.3         pypi_0                 pypi
[conda] nvidia-cusparse-cu12                  12.5.8.93        pypi_0                 pypi
[conda] nvidia-cusparselt-cu12                0.7.1            pypi_0                 pypi
[conda] nvidia-cusparselt-cu13                0.8.1            pypi_0                 pypi
[conda] nvidia-nccl-cu12                      2.27.5           pypi_0                 pypi
[conda] nvidia-nccl-cu13                      2.29.7           pypi_0                 pypi
[conda] nvidia-nvjitlink                      13.0.88          pypi_0                 pypi
[conda] nvidia-nvjitlink-cu12                 12.8.93          pypi_0                 pypi
[conda] nvidia-nvtx                           13.0.85          pypi_0                 pypi
[conda] nvidia-nvtx-cu12                      12.8.90          pypi_0                 pypi
[conda] torch                                 2.12.1           pypi_0                 pypi
[conda] torchvision                           0.27.1           pypi_0                 pypi
[conda] triton                                3.7.1            pypi_0                 pypi


cc @chauhang @penguinwu @Chillee @samdow @kshitij12345 @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo @bdhirsh @bobrenjc93 @aorenste

## Trigger

A functional indexed write targets a stride-zero expanded tensor during Inductor reinplacement.

## Relevant entities

functional index operation; expanded view; stride-zero storage; logically distinct rows

## Implemented relation

Expansion maps multiple logical elements or rows to the same physical storage locations.

## Pre-state

The functional operation requires an independent result whose logical writes do not collapse through internal overlap.

## Binding transition

The reinplacement pass rewrites the functional indexed operation to its in-place form on the expanded view.

## Post-state

Distinct logical writes mutate aliased locations in one overlapping storage region.

## Validity obligation

Reinplacement is allowed only when the mutated destination has no definite internal overlap that changes functional semantics.

## Invalid decision

can_inplace checked inter-tensor aliasing but omitted the destination's definite internal-overlap check.

## Physical effect

Writes from different expanded rows fold through the same storage locations.

## Observed consequence

The compiled gradient is wrong by the batch-size factor on CPU and CUDA.

## Repair site

torch/_inductor/fx_passes/reinplace.py

## Repair

can_inplace rejects a mutated argument when _debug_has_internal_overlap returns definite overlap; regression tests cover indexed operations.

## Why the repair works

The pass retains the functional materialization path for expanded tensors, so logically distinct writes receive nonoverlapping storage.

## Causal basis

The added guard is the exact missing condition identified by the report and is exercised by the new expanded-index tests.

## Analysis implication

A reinplacement check needs internal-overlap information, not only alias relations between different tensor values.

## Fix patches

- [`fixes/pytorch-188084-fix.patch`](fixes/pytorch-188084-fix.patch) — https://github.com/pytorch/pytorch/pull/184488
