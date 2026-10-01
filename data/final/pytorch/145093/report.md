# Inductor aten.clone lowering ignores Conjugate and Negative dispatch keys

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/145093](https://github.com/pytorch/pytorch/issues/145093)
- Opened: 2025-01-17T18:29:45Z
- Closed: Open at collection
- Labels: `triaged`, `actionable`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `module: pt2-dispatcher`

## Reported behavior

### 🐛 Describe the bug

In runtime-dispatched `torch`, conjugation and certain forms of negation are lazily evaluated at dispatch. The current lowering for `aten.clone` ignores this. See the minimal reproducer below:

```python
import torch

fn = torch.compile(torch.ops.aten.clone.default)  # this issue does not occur in "eager" or "aot_eager"
u = torch.randn(5, dtype=torch.complex64).conj().imag  # sets Negative dispatch key
assert torch.all(fn(u) == u)  # fails
```

### Error logs

_No response_

### Versions

```
Collecting environment information...
PyTorch version: 2.7.0a0+git6759d9c
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.1 LTS (x86_64)
GCC version: (conda-forge gcc 12.4.0-1) 12.4.0
Clang version: Could not collect
CMake version: version 3.31.2
Libc version: glibc-2.39

Python version: 3.9.21 | packaged by conda-forge | (main, Dec  5 2024, 13:51:40)  [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-6.8.0-51-generic-x86_64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: Quadro RTX 8000
GPU 1: Quadro RTX 8000

Nvidia driver version: 560.35.05
cuDNN version: Probably one of the following:
/usr/local/cuda-12.6.3/targets/x86_64-linux/lib/libcudnn.so.9
/usr/local/cuda-12.6.3/targets/x86_64-linux/lib/libcudnn_adv.so.9
/usr/local/cuda-12.6.3/targets/x86_64-linux/lib/libcudnn_cnn.so.9
/usr/local/cuda-12.6.3/targets/x86_64-linux/lib/libcudnn_engines_precompiled.so.9
/usr/local/cuda-12.6.3/targets/x86_64-linux/lib/libcudnn_engines_runtime_compiled.so.9
/usr/local/cuda-12.6.3/targets/x86_64-linux/lib/libcudnn_graph.so.9
/usr/local/cuda-12.6.3/targets/x86_64-linux/lib/libcudnn_heuristic.so.9
/usr/local/cuda-12.6.3/targets/x86_64-linux/lib/libcudnn_ops.so.9
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        43 bits physical, 48 bits virtual
Byte Order:                           Little Endian
CPU(s):                               48
On-line CPU(s) list:                  0-47
Vendor ID:                            AuthenticAMD
Model name:                           AMD Ryzen Threadripper 2970WX 24-Core Processor
CPU family:                           23
Model:                                8
Thread(s) per core:                   2
Core(s) per socket:                   24
Socket(s):                            1
Stepping:                             2
Frequency boost:                      enabled
CPU(s) scaling MHz:                   77%
CPU max MHz:                          3000.0000
CPU min MHz:                          2200.0000
BogoMIPS:                             5987.89
Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid amd_dcm aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb hw_pstate ssbd ibpb vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt sha_ni xsaveopt xsavec xgetbv1 clzero irperf xsaveerptr arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif overflow_recov succor smca sev sev_es
Virtualization:                       AMD-V
L1d cache:                            768 KiB (24 instances)
L1i cache:                            1.5 MiB (24 instances)
L2 cache:                             12 MiB (24 instances)
L3 cache:                             64 MiB (8 instances)
NUMA node(s):                         4
NUMA node0 CPU(s):                    0-5,24-29
NUMA node1 CPU(s):                    12-17,36-41
NUMA node2 CPU(s):                    6-11,30-35
NUMA node3 CPU(s):                    18-23,42-47
Vulnerability Gather data sampling:   Not affected
Vulnerability Itlb multihit:          Not affected
Vulnerability L1tf:                   Not affected
Vulnerability Mds:                    Not affected
Vulnerability Meltdown:               Not affected
Vulnerability Mmio stale data:        Not affected
Vulnerability Reg file data sampling: Not affected
Vulnerability Retbleed:               Mitigation; untrained return thunk; SMT vulnerable
Vulnerability Spec rstack overflow:   Mitigation; Safe RET
Vulnerability Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:             Mitigation; Retpolines; IBPB conditional; STIBP disabled; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                  Not affected
Vulnerability Tsx async abort:        Not affected

Versions of relevant libraries:
[pip3] bert_pytorch==0.0.1a4
[pip3] flake8==6.1.0
[pip3] flake8-bugbear==23.3.23
[pip3] flake8-comprehensions==3.15.0
[pip3] flake8-executable==2.1.3
[pip3] flake8-logging-format==0.9.0
[pip3] flake8-pyi==23.3.1
[pip3] flake8-simplify==0.19.3
[pip3] functorch==1.14.0a0+b71aa0b
[pip3] mypy==1.13.0
[pip3] mypy-extensions==1.0.0
[pip3] numpy==1.26.4
[pip3] onnx==1.17.0
[pip3] optree==0.13.0
[pip3] pytorch-labs-segment-anything-fast==0.2
[pip3] pytorch-triton==3.2.0+git0d4682f0
[pip3] torch==2.7.0a0+git6759d9c
[pip3] torch_geometric==2.4.0
[pip3] torchao==0.7.0
[pip3] torchaudio==2.6.0a0+b6d4675
[pip3] torchdata==0.11.0a0+227d3d7
[pip3] torchmultimodal==0.1.0b0
[pip3] torchtext==0.17.0a0+1d4ce73
[pip3] torchvision==0.22.0a0+d3beb52
[conda] bert-pytorch              0.0.1a4                   dev_0    <develop>
[conda] cuda-cudart               12.4.127             he02047a_2    conda-forge
[conda] cuda-cudart-dev           12.4.127             he02047a_2    conda-forge
[conda] cuda-cudart-dev_linux-64  12.4.127             h85509e4_2    conda-forge
[conda] cuda-cudart-static        12.4.127             he02047a_2    conda-forge
[conda] cuda-cudart-static_linux-64 12.4.127             h85509e4_2    conda-forge
[conda] cuda-cudart_linux-64      12.4.127             h85509e4_2    conda-forge
[conda] cuda-cupti                12.4.127             he02047a_2    conda-forge
[conda] cuda-cupti-dev            12.4.127             he02047a_2    conda-forge
[conda] cuda-libraries-dev        12.4.1               ha770c72_1    conda-forge
[conda] cuda-nvrtc                12.4.127             he02047a_2    conda-forge
[conda] cuda-nvrtc-dev            12.4.127             he02047a_2    conda-forge
[conda] cuda-nvtx                 12.4.127             he02047a_2    conda-forge
[conda] cuda-nvtx-dev             12.4.127             ha770c72_2    conda-forge
[conda] cuda-opencl               12.4.127             he02047a_1    conda-forge
[conda] cuda-opencl-dev           12.4.127             he02047a_1    conda-forge
[conda] cudnn                     9.3.0.75             h62a6f1c_2    conda-forge
[conda] functorch                 1.14.0a0+b71aa0b          pypi_0    pypi
[conda] libcublas                 12.4.5.8             he02047a_2    conda-forge
[conda] libcublas-dev             12.4.5.8             he02047a_2    conda-forge
[conda] libcufft                  11.2.1.3             he02047a_2    conda-forge
[conda] libcufft-dev              11.2.1.3             he02047a_2    conda-forge
[conda] libcurand                 10.3.5.147           he02047a_2    conda-forge
[conda] libcurand-dev             10.3.5.147           he02047a_2    conda-forge
[conda] libcusolver               11.6.1.9             he02047a_2    conda-forge
[conda] libcusolver-dev           11.6.1.9             he02047a_2    conda-forge
[conda] libcusparse               12.3.1.170           he02047a_2    conda-forge
[conda] libcusparse-dev           12.3.1.170           he02047a_2    conda-forge
[conda] libmagma                  2.8.0                h0af6554_0    conda-forge
[conda] libmagma_sparse           2.8.0                h0af6554_0    conda-forge
[conda] libnvjitlink              12.4.127             he02047a_2    conda-forge
[conda] libnvjitlink-dev          12.4.127             he02047a_2    conda-forge
[conda] magma                     2.8.0                h51420fd_0    conda-forge
[conda] mkl                       2025.0.0           h901ac74_941    conda-forge
[conda] mkl-include               2025.0.0           hf2ce2f3_941    conda-forge
[conda] numpy                     1.26.4                   pypi_0    pypi
[conda] optree                    0.13.0                   pypi_0    pypi
[conda] pytorch-labs-segment-anything-fast 0.2                      pypi_0    pypi
[conda] pytorch-triton            3.2.0+git0d4682f0          pypi_0    pypi
[conda] torch                     2.7.0a0+git6759d9c           dev_0    <develop>
[conda] torch-geometric           2.4.0                    pypi_0    pypi
[conda] torchao                   0.7.0                    pypi_0    pypi
[conda] torchaudio                2.6.0a0+b6d4675          pypi_0    pypi
[conda] torchdata                 0.11.0a0+227d3d7          pypi_0    pypi
[conda] torchfix                  0.4.0                    pypi_0    pypi
[conda] torchmultimodal           0.1.0b0                  pypi_0    pypi
[conda] torchtext                 0.17.0a0+1d4ce73          pypi_0    pypi
[conda] torchvision               0.22.0a0+d3beb52          pypi_0    pypi
```

cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @yf225 @chenyang78 @kadeng @muchulee8 @ColinPeppler @amjames @desertfire @aakhundov @zou3519 @bdhirsh

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises torchInductor clone lowering for tensors carrying lazy conjugate or negative dispatch bits.

## Observed implementation decision

Clone lowering copies storage values without resolving the logical conjugate/negative transformation represented by dispatch metadata.

## Physical effect

The compiled clone represents different logical values from eager clone.

## Observed consequence

The equality assertion fails for a conjugated imaginary view.

## Reported proposal or workaround

Discussion proposes resolving the bits while preserving boundary aliases.

## Repair trace

The issue remains open and PRs 184161, 192315, and 197856 are unmerged or address adjacent behavior; no landed clone repair and regression establish the required dispatch-bit and alias semantics.

## Archived evidence

- Raw report: `data/raw/github/pytorch/145093/report.json`
- SHA-256: `7af1290720b8a11cd18cce0a369ca69618e08689f147db8c0d8773d71049b1b7`
