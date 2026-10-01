# [ROCm][Inductor] `as_strided` on a graph-input view reads back zeros when the rebased offset is negative

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/195823](https://github.com/pytorch/pytorch/issues/195823)
- Opened: 2026-09-03T08:27:06Z
- Closed: Open at collection
- Labels: `high priority`, `module: rocm`, `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug

The following is AI analysis edited for clarity:

> On ROCm, `torch.compile` produces wrong values for `torch.as_strided(x, size, stride, 0)`
when `x` is a graph input that is a view with a non-zero `storage_offset()`.

> Since #184232, Inductor translates explicit storage-relative `as_strided` offsets onto
the graph input's runtime data pointer. That is, `storage_offset - x.storage_offset()`.

> For example, for a view at offset 6 that is `as_strided` back to offset 0, the generated Triton kernel
loads at `in_ptr0 + (-6 + x0)`. So the generated Triton kernel loads in at negative offsets from the pointer it
was handed. On CUDA this is ordinary pointer arithmetic and the result is correct. On
ROCm the elements at negative offsets read back as 0, so the pointer is now in the wrong position.

> This is pre-existing on `main`. #192227 only adds a test that exercises it on GPU (the
test added in #184232 runs on CPU), which is how it surfaced.

### Minimum Reproducing Example

The following is an AI generated minimum reproducing example

```python
import torch

def fn(x):
    return torch.as_strided(x, (12,), (1,), 0) + 1.0

base = torch.arange(12, dtype=torch.float32, device="cuda")
x = base[6:9]  # storage_offset() == 6

print(fn(x))                 # [1., 2., ..., 12.]
print(torch.compile(fn)(x))  # ROCm: [1., 1., 1., 1., 1., 1., 7., ..., 12.]
```

> Equivalent test: `PYTORCH_TEST_WITH_ROCM=1 python test/inductor/test_torchinductor.py GPUTests.test_as_strided_past_input_extent_cuda`

### Observed in CI (linux-jammy-rocm-py3.10-mi350, gfx950)

```
AssertionError: Tensor-likes are not close!
Mismatched elements: 5 / 12 (41.7%)
Greatest absolute difference: 5.0 at index (5,) (up to 1e-05 allowed)
```

> Indices 0-5 (the six loads at offsets -6..-1 from the input pointer) all come back as
0 + 1.0; index 0 happens to match because its expected value is also 1.0. Indices 6-11
(non-negative offsets) are correct.

- https://github.com/pytorch/pytorch/actions/runs/33675357509/job/100405054717
- https://github.com/pytorch/pytorch/actions/runs/33675357509/job/100405054641
- https://github.com/pytorch/pytorch/actions/runs/33675357509/job/100405054556

I don't have access to any AMD hardware, so we don't have a smoking gun on the cause, however my agent suggested this may be a good place to start.

> The Triton AMD backend enables buffer ops by default (`knobs.amd.use_buffer_ops`,
env `AMDGCN_USE_BUFFER_OPS`, default `True` at the CI pin c01b6774b) and specializes
tensor args whose storage is <= 2GB with `tt.pointer_range=32`. AMD buffer loads take an
unsigned 32-bit offset and return 0 for out-of-range offsets, which matches a negative
offset relative to the kernel's base pointer reading back 0. Running the repro with
`AMDGCN_USE_BUFFER_OPS=0` should confirm or rule this out.

### Versions

```
PyTorch version: 2.15.0a0+gitfe09e29
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.4 LTS (aarch64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Clang version: Could not collect
CMake version: version 4.4.2
Libc version: glibc-2.39

Python version: 3.10.20 (main, Jun  2 2026, 22:26:24) [Clang 22.1.3 ] (64-bit runtime)
Python platform: Linux-6.11.0-1016-nvidia-aarch64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: 13.0.88
CUDA_MODULE_LOADING set to:
GPU models and configuration: GPU 0: NVIDIA GB10
Nvidia driver version: 580.95.05
cuDNN version: Could not collect
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False
Caching allocator config: N/A

CPU:
Architecture:                         aarch64
CPU op-mode(s):                       64-bit
Byte Order:                           Little Endian
CPU(s):                               20
On-line CPU(s) list:                  0-19
Vendor ID:                            ARM
Model name:                           Cortex-X925
Model:                                1
Thread(s) per core:                   1
Core(s) per socket:                   10
Socket(s):                            1
Stepping:                             r0p1
CPU(s) scaling MHz:                   101%
CPU max MHz:                          4004.0000
CPU min MHz:                          1378.0000
BogoMIPS:                             2000.00
Flags:                                fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh bti ecv afp wfxt
Model name:                           Cortex-A725
Model:                                1
Thread(s) per core:                   1
Core(s) per socket:                   10
Socket(s):                            1
Stepping:                             r0p1
CPU(s) scaling MHz:                   91%
CPU max MHz:                          2860.0000
CPU min MHz:                          338.0000
BogoMIPS:                             2000.00
Flags:                                fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh bti ecv afp wfxt
L1d cache:                            1.3 MiB (20 instances)
L1i cache:                            1.3 MiB (20 instances)
L2 cache:                             25 MiB (20 instances)
L3 cache:                             24 MiB (2 instances)
NUMA node(s):                         1
NUMA node0 CPU(s):                    0-19
Vulnerability Gather data sampling:   Not affected
Vulnerability Itlb multihit:          Not affected
Vulnerability L1tf:                   Not affected
Vulnerability Mds:                    Not affected
Vulnerability Meltdown:               Not affected
Vulnerability Mmio stale data:        Not affected
Vulnerability Reg file data sampling: Not affected
Vulnerability Retbleed:               Not affected
Vulnerability Spec rstack overflow:   Not affected
Vulnerability Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:             Mitigation; __user pointer sanitization
Vulnerability Spectre v2:             Not affected
Vulnerability Srbds:                  Not affected
Vulnerability Tsx async abort:        Not affected

Versions of relevant libraries:
[pip3] numpy==2.2.6
[pip3] optree==0.19.1
[pip3] torch==2.15.0a0+gitfe09e29
[pip3] triton==3.8.0+gitc01b6774
[conda] Could not collect
```

In addition, my agent also provided:

- PyTorch `main` (observed at 819614e28c8, includes #192227)
- Triton pin c01b6774b1865984607d89d89d3a10833de92037
- ROCm MI350 (gfx950) CI runners

cc @ezyang @gchanan @kadeng @msaroufim @jeffdaily @sunway513 @jithunnair-amd @pruthvistony @ROCmSupport @jataylo @hongxiayang @naromero77amd @pragupta @jerrymannil @xinyazhang @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @muchulee8 @amjames @aakhundov @coconutruben

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises rOCm TorchInductor loads for storage-relative as_strided views rebased to negative offsets from a graph-input pointer.

## Observed implementation decision

Inductor rebases a storage offset onto the view data pointer while HIP Triton selects 32-bit unsigned buffer offsets for the resulting negative index.

## Physical effect

Negative logical offsets are interpreted as out-of-range buffer operations and return zero.

## Observed consequence

The first six elements of the compiled view are wrong.

## Reported proposal or workaround

PR 196115 proposes disabling 32-bit buffer operations when a load index has a negative constant.

## Repair trace

PR 196115 was not merged; the earlier PR 192227 exposes the failure but explicitly does not repair ROCm, so no landed repair chain exists.

## Archived evidence

- Raw report: `data/raw/github/pytorch/195823/report.json`
- SHA-256: `cf0be646bd8f995444febcd71c041ecbe0ac30c8c64e3387a9cee535e6c676bb`
