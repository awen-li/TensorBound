# triton_helpers.bucketize_binary_search does not support strided tensors

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/135873](https://github.com/pytorch/pytorch/issues/135873)
- Opened: 2024-09-12T17:22:31Z
- Closed: 2024-10-04T19:26:07Z
- Labels: `high priority`, `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`

## Reported behavior

### 🐛 Describe the bug

The Triton implementation of bucketizing, used to lower `aten.bucketize` and `aten.searchsorted`, does not support tensors that are strided in the last dimension.  The bucket loading code (and the optional sorter loading code) both assume contiguous memory, leading to incorrect results when passed these tensors.

This only occurs on strides in the last dimension; the code implicitly already processes strides in earlier dimensions.

Assuming simple striding in the last dimension, this should (a) be easy to fix, and (b) be an easy performance win for Triton, which shouldn't have to do the copy that eager mode does.

### Error logs

_No response_

### Minified repro

```python
import torch

fn = torch.bucketize
compiled_fn = torch.compile(fn)

boundaries = torch.linspace(-1.0, 1.0, 33, device="cuda")
vals = torch.randn(16, device="cuda")
success = fn(vals, boundaries) == compiled_fn(vals, boundaries)
assert torch.all(success), "Contiguous check failed!"  # passes

noncontiguous_boundaries = boundaries[::2]
# This next line raises a warning about this exact issue from eager-mode bucketize.
success = fn(vals, noncontiguous_boundaries) == compiled_fn(vals, noncontiguous_boundaries)
assert torch.all(success), "Non-contiguous check failed!"  # fails

# This repro would also work with torch.searchsorted (once the lowering in
# https://github.com/pytorch/pytorch/pull/135701 merges, if the checks for continuity in
# the lowering are removed).
```

### Versions

```
Collecting environment information...
PyTorch version: 2.5.0a0+git3a0dfdb
Is debug build: False
CUDA used to build PyTorch: 12.6
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.2 LTS (x86_64)
GCC version: (conda-forge gcc 13.3.0-1) 13.3.0
Clang version: Could not collect
CMake version: version 3.30.3
Libc version: glibc-2.35

Python version: 3.8.19 | packaged by conda-forge | (default, Mar 20 2024, 12:47:35)  [GCC 12.3.0] (64-bit runtime)
Python platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.10
Is CUDA available: True
CUDA runtime version: 12.6.68
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA GeForce RTX 2060
GPU 1: NVIDIA GeForce RTX 2060

Nvidia driver version: 545.23.08
cuDNN version: Probably one of the following:
/usr/local/cuda-11.7.1/targets/x86_64-linux/lib/libcudnn.so.8
/usr/local/cuda-11.7.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
/usr/local/cuda-11.7.1/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
/usr/local/cuda-11.7.1/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8
/usr/local/cuda-11.7.1/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8
/usr/local/cuda-11.7.1/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8
/usr/local/cuda-11.7.1/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
/usr/local/cuda-12.3.2/targets/x86_64-linux/lib/libcudnn.so.9
/usr/local/cuda-12.3.2/targets/x86_64-linux/lib/libcudnn_adv.so.9
/usr/local/cuda-12.3.2/targets/x86_64-linux/lib/libcudnn_cnn.so.9
/usr/local/cuda-12.3.2/targets/x86_64-linux/lib/libcudnn_engines_precompiled.so.9
/usr/local/cuda-12.3.2/targets/x86_64-linux/lib/libcudnn_engines_runtime_compiled.so.9
/usr/local/cuda-12.3.2/targets/x86_64-linux/lib/libcudnn_graph.so.9
/usr/local/cuda-12.3.2/targets/x86_64-linux/lib/libcudnn_heuristic.so.9
/usr/local/cuda-12.3.2/targets/x86_64-linux/lib/libcudnn_ops.so.9
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                      43 bits physical, 48 bits virtual
Byte Order:                         Little Endian
CPU(s):                             64
On-line CPU(s) list:                0-63
Vendor ID:                          AuthenticAMD
Model name:                         AMD Ryzen Threadripper 3970X 32-Core Processor
CPU family:                         23
Model:                              49
Thread(s) per core:                 2
Core(s) per socket:                 32
Socket(s):                          1
Stepping:                           0
Frequency boost:                    enabled
CPU max MHz:                        3700.0000
CPU min MHz:                        2200.0000
BogoMIPS:                           7400.38
Flags:                              fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif v_spec_ctrl umip rdpid overflow_recov succor smca sme sev sev_es
Virtualization:                     AMD-V
L1d cache:                          1 MiB (32 instances)
L1i cache:                          1 MiB (32 instances)
L2 cache:                           16 MiB (32 instances)
L3 cache:                           128 MiB (8 instances)
NUMA node(s):                       1
NUMA node0 CPU(s):                  0-63
Vulnerability Gather data sampling: Not affected
Vulnerability Itlb multihit:        Not affected
Vulnerability L1tf:                 Not affected
Vulnerability Mds:                  Not affected
Vulnerability Meltdown:             Not affected
Vulnerability Mmio stale data:      Not affected
Vulnerability Retbleed:             Mitigation; untrained return thunk; SMT enabled with STIBP protection
Vulnerability Spec rstack overflow: Mitigation; safe RET
Vulnerability Spec store bypass:    Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:           Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:           Mitigation; Retpolines, IBPB conditional, STIBP always-on, RSB filling, PBRSB-eIBRS Not affected
Vulnerability Srbds:                Not affected
Vulnerability Tsx async abort:      Not affected

Versions of relevant libraries:
[pip3] flake8==6.1.0
[pip3] flake8-bugbear==23.3.23
[pip3] flake8-comprehensions==3.15.0
[pip3] flake8-executable==2.1.3
[pip3] flake8-logging-format==0.9.0
[pip3] flake8-pyi==23.3.1
[pip3] flake8-simplify==0.19.3
[pip3] mypy==1.10.0
[pip3] mypy-extensions==1.0.0
[pip3] numpy==1.24.3
[pip3] optree==0.12.1
[pip3] pytorch-triton==3.1.0+5fe38ffd73
[pip3] torch==2.5.0a0+git3a0dfdb
[pip3] torchaudio==2.5.0a0+97ed7b3
[pip3] torchdata==0.9.0a0+8b6e903
[pip3] torchtext==0.17.0a0+1d4ce73
[conda] libmagma                  2.8.0                h0af6554_0    conda-forge
[conda] libmagma_sparse           2.8.0                h0af6554_0    conda-forge
[conda] magma                     2.8.0                h51420fd_0    conda-forge
[conda] mkl                       2024.2.1           ha957f24_103    conda-forge
[conda] mkl-include               2024.2.1           ha957f24_103    conda-forge
[conda] numpy                     1.24.3                   pypi_0    pypi
[conda] optree                    0.12.1           py38h5a8d4af_0    conda-forge
[conda] pytorch-triton            3.1.0+5fe38ffd73          pypi_0    pypi
[conda] torch                     2.5.0a0+git3a0dfdb           dev_0    <develop>
[conda] torchaudio                2.5.0a0+97ed7b3          pypi_0    pypi
[conda] torchdata                 0.9.0a0+8b6e903          pypi_0    pypi
[conda] torchtext                 0.17.0a0+1d4ce73          pypi_0    pypi
```

cc @ezyang @gchanan @zou3519 @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @yf225 @chenyang78 @muchulee8 @ColinPeppler @amjames @desertfire @davidberard98

## Trigger

bucketize searches a boundary tensor produced by a step slice such as boundaries[::2].

## Relevant entities

logical boundary sequence; boundary storage; last-dimension stride; binary-search midpoint; sorter stride; Triton load address

## Implemented relation

A logical binary-search position is bound to the physical boundary address at boundary_base plus set_offset plus midpoint times the last-dimension stride.

## Pre-state

The boundary view exposes every second element of the backing tensor, so consecutive logical boundaries are separated by a physical stride of two.

## Binding transition

The old Triton helper adds the midpoint directly to the boundary pointer and has no boundary-stride or sorter-stride input.

## Post-state

Logical midpoint k is bound to backing-storage element k instead of backing-storage element 2k for the reported step slice.

## Validity obligation

Every binary-search load applies the lookup tensor's last-dimension stride after selecting the correct boundary set and optional sorter entry.

## Invalid decision

The generated bucketize helper assumed contiguous last-dimension storage for all boundary and sorter tensors.

## Physical effect

The binary search compares values against elements outside the logical strided boundary sequence and moves its low and high bounds using those comparisons.

## Observed consequence

Compiled bucketize returns results that differ from eager execution for boundaries[::2], while the contiguous control passes.

## Repair site

torch/_inductor/runtime/triton_helpers.py::bucketize_binary_search and the bucketize/searchsorted lowering and Triton code-generation path

## Repair

Commit a968576 extends the bucketize operation with boundary-set indices, storage extent, boundary stride, optional sorter stride, and sorter indices; the Triton helper multiplies midpoint indices by those strides and adds multidimensional, strided, sorted, and searchsorted tests.

## Why the repair works

Each logical midpoint now resolves through the view's actual physical step, so the kernel compares against exactly the boundary sequence visible to PyTorch semantics.

## Causal basis

The landed commit message explicitly closes report 135873, adds stride support to ops.bucketize, and changes the failing boundary load from pointer plus midpoint to pointer plus base index plus stride times midpoint.

## Analysis implication

Generated search analysis needs logical sequence length, base set index, backing-storage extent, last-dimension stride, optional sorter mapping, and final load address.

## Fix patches

- [`fixes/pytorch-135873-commit-a968576.patch`](fixes/pytorch-135873-commit-a968576.patch) — https://github.com/pytorch/pytorch/commit/a9685767773157440c162caaf125856e04e2981f
