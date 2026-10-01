# [Inductor] Incorrect epilogue fusion for user-defined Triton kernels when output has layout-changing ops (`.T.contiguous()`)

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/184247](https://github.com/pytorch/pytorch/issues/184247)
- Opened: 2026-05-18T18:21:04Z
- Closed: 2026-06-23T13:50:45Z
- Labels: `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `module: user triton`, `bot-triaged`, `repro:agent:h100:repro_success:reproduced`

## Reported behavior

### 🐛 Describe the bug

During codegen for a fused custom op, we rewrite the `tl.store` to use a new `value` operand that references the variable from the last computation in the epilogue. See: https://github.com/pytorch/pytorch/blob/9a8a62cf0b558e80e46565025a31ebdb5f8ec243/torch/_inductor/codegen/triton.py#L6573-L6582

This assumes the write index of the epilogue equals that of the read. Currently, the write reindex from `contiguous()` is silently substituted away with the user's original store index. 

### Repro

```python
import torch

import triton 
import triton.language as tl

torch._inductor.config.epilogue_fusion_user_defined_triton_kernel = True

@triton.jit
def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(in_ptr0 + offs, mask=mask)
    y = tl.load(in_ptr1 + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def fn(a, b):
    out = torch.empty_like(a)
    grid = (triton.cdiv(a.numel(), 1024),)
    add_kernel[grid](a, b, out, a.numel(), BLOCK_SIZE=1024)
    return out.T.contiguous()

a = torch.randn(32, 32, device="cuda").double()
b = torch.randn(32, 32, device="cuda").double()

fn_compiled = torch.compile(fn)
torch.testing.assert_close(fn_compiled(a, b), fn(a, b))
```

### IR 

See read index != write index:
```
op2: SchedulerNode(ComputedBuffer)
op2.writes = [MemoryDep('buf2', 32*d0 + d1, {d0: 32, d1: 32})]
op2.unmet_dependencies = [MemoryDep('buf1', d0 + 32*d1, {d0: 32, d1: 32})]
op2.met_dependencies = []
op2.min_input_distance = 2
op2.max_input_distance = 2
op2.outputs = [
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float64, size=[32, 32], stride=[32, 1])
    buf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op2.group.device = cuda:0
op2.group.iteration = (1024, 1)
op2.sizes = ([32, 32], [])
buf1_layout = NoneLayout(device=device(type='cuda', index=0), size=[0], stride=[0])
buf2_layout = FixedLayout('cuda:0', torch.float64, size=[32, 32], stride=[32, 1])
class op2_loop_body:
    var_ranges = {p0: 32, p1: 32}
    index0 = p0 + 32*p1
    index1 = 32*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf0', get_index)
        get_index_1 = self.get_index('index1')
        store = ops.store('buf2', get_index_1, load, None)
        return store
```

### Error logs

```
Traceback (most recent call last):
  File "repro.py", line 28, in <module>
    torch.testing.assert_close(fn_compiled(a, b), fn(a, b))
  File "/home/jjvraw/Git/pytorch/torch/testing/_comparison.py", line 1631, in assert_close
    raise error_metas[0].to_error(msg)
AssertionError: Tensor-likes are not close!

Mismatched elements: 992 / 1024 (96.9%)
Greatest absolute difference: 5.866803228855133 at index (1, 3) (up to 1e-07 allowed)
Greatest relative difference: 1150.1576401612313 at index (20, 5) (up to 1e-07 allowed)
```

### Versions

```
PyTorch version: N/A
Is debug build: N/A
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: N/A

OS: Ubuntu 25.04 (x86_64)
GCC version: (Ubuntu 14.2.0-19ubuntu2) 14.2.0
Clang version: 20.1.2 (0ubuntu1)
CMake version: version 3.31.10
Libc version: glibc-2.41

Python version: 3.13.5 | packaged by Anaconda, Inc. | (main, Jun 12 2025, 16:09:02) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.14.0-37-generic-x86_64-with-glibc2.41
Is CUDA available: N/A
CUDA runtime version: 12.8.61
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4090
Nvidia driver version: 580.95.05
cuDNN version: Could not collect
Is XPU available: N/A
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: N/A
Caching allocator config: N/A

CPU:
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           48 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  32
On-line CPU(s) list:                     0-31
Vendor ID:                               AuthenticAMD
Model name:                              AMD Ryzen 9 7950X3D 16-Core Processor
CPU family:                              25
Model:                                   97
Thread(s) per core:                      2
Core(s) per socket:                      16
Socket(s):                               1
Stepping:                                2
Frequency boost:                         enabled
CPU(s) scaling MHz:                      63%
CPU max MHz:                             5763.0000
CPU min MHz:                             545.0000
BogoMIPS:                                8383.77
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl xtopology nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid overflow_recov succor smca fsrm flush_l1d amd_lbr_pmc_freeze
Virtualization:                          AMD-V
L1d cache:                               512 KiB (16 instances)
L1i cache:                               512 KiB (16 instances)
L2 cache:                                16 MiB (16 instances)
L3 cache:                                128 MiB (2 instances)
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
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Mitigation; Safe RET
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; STIBP always-on; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Mitigation; Clear CPU buffers
Vulnerability Tsx async abort:           Not affected
Vulnerability Vmscape:                   Mitigation; IBPB before exit to userspace

Versions of relevant libraries:
[pip3] flake8==7.3.0
[pip3] flake8-bugbear==24.12.12
[pip3] flake8-comprehensions==3.16.0
[pip3] flake8-executable==2.1.3
[pip3] flake8-logging-format==2024.24.12
[pip3] flake8-pyi==25.5.0
[pip3] flake8_simplify==0.22.0
[pip3] intel-cmplr-lib-ur==2025.3.1
[pip3] intel-openmp==2025.3.1
[pip3] mkl-include==2025.3.0
[pip3] mkl-static==2025.3.0
[pip3] mypy==1.16.0
[pip3] mypy_extensions==1.1.0
[pip3] numpy==2.1.0
[pip3] onemkl-license==2025.3.0
[pip3] optree==0.17.0
[pip3] tbb==2022.3.0
[pip3] tbb-devel==2022.3.0
[pip3] tcmlib==1.4.1
[pip3] torch==2.12.0a0+gitaeee308
[pip3] triton==3.7.0
[pip3] umf==1.0.2
[conda] intel-cmplr-lib-ur          2025.3.1             pypi_0              pypi
[conda] intel-openmp                2025.3.1             pypi_0              pypi
[conda] mkl-include                 2025.3.0             pypi_0              pypi
[conda] mkl-static                  2025.3.0             pypi_0              pypi
[conda] numpy                       2.1.0                pypi_0              pypi
[conda] onemkl-license              2025.3.0             pypi_0              pypi
[conda] optree                      0.17.0               pypi_0              pypi
[conda] tbb                         2022.3.0             pypi_0              pypi
[conda] tbb-devel                   2022.3.0             pypi_0              pypi
[conda] tcmlib                      1.4.1                pypi_0              pypi
[conda] torch                       2.12.0a0+gitaeee308  pypi_0              pypi
[conda] triton                      3.7.0                pypi_0              pypi
[conda] umf                         1.0.2                pypi_0              pypi
```

cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo @oulgen @davidberard98

## Trigger

A user-defined Triton kernel writes a matrix in its original layout and is followed by transpose, contiguous, and pointwise operations.

## Relevant entities

user Triton store; kernel output buffer; epilogue read; epilogue write; read index; write index

## Implemented relation

Fusing an epilogue into a user kernel store preserves the epilogue's mapping from each output position to the value read for that position.

## Pre-state

The user kernel stores at its original linear index, while the unfused layout-changing epilogue reads and writes with its own index maps.

## Binding transition

The fusion path substitutes the epilogue value expression into the user kernel store.

## Post-state

The substituted expression may read the transposed position but is emitted at the user kernel's original store position.

## Validity obligation

A substituted epilogue is unary and pointwise over the output buffer, with identical read and write index expressions and iteration sizes.

## Invalid decision

The fusion check required reads to name the output buffer but did not compare the epilogue's read and write indices.

## Physical effect

A value selected through the layout-changing read map is written through a different physical position map.

## Observed consequence

The reported fused transpose-contiguous path returns values that differ from the unfused result.

## Repair site

torch/_inductor/scheduler.py::_can_fuse and torch/_inductor/codegen/triton.py::UserDefinedTritonKernelStoreMode.store

## Repair

Commit 12b766b permits this fusion only for one read and one write of the output with equal index expressions and sizes. The user-kernel store mode also rejects an unexpected epilogue store, and the commit adds the report's transpose-contiguous regression.

## Why the repair works

Equal read and write maps make substitution preserve each output position; layout-changing epilogues remain separate kernels.

## Causal basis

The landed commit states that it fixes 184247, identifies the write-equals-read assumption, changes the corresponding scheduler guard, and adds the reported layout-changing sequence as a regression.

## Analysis implication

User-kernel fusion analysis needs buffer identity, read and write cardinality, index expressions, iteration sizes, and the store-substitution boundary.

## Fix patches

- [`fixes/pytorch-184247-commit-12b766b.patch`](fixes/pytorch-184247-commit-12b766b.patch) — https://github.com/pytorch/pytorch/commit/12b766bf533ff188b4d861184c54b7278ceaca9a
