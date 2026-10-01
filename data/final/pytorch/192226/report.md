# BUG: Out of Bounds Reads with offset/strided views on compiled kernels

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/192226](https://github.com/pytorch/pytorch/issues/192226)
- Opened: 2026-08-05T13:19:33Z
- Closed: 2026-09-04T06:37:08Z
- Labels: `high priority`, `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `module: user triton`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

Sections marked as AI analysis are reviewed and edited by me for clarity.

AI Analysis:
> When a tensor that is a view with a non-zero `storage_offset` is passed as a mutated pointer argument to a raw `triton.jit` kernel, `torch.compile` silently produces an incorrect result. This is due to data being read from beyond the end of a buffer into memory.

My Analysis:
At a high level, strided/offset views of tensors can read out of bounds when used with torch compile in some edge cases.

AI Analysis:
> The over-read is normally invisible because the kernel overwrites the whole region. The edge case is when the kernel skips writing part of its output as a performance optimization. E.G "if this row is masked/ignored, return early".

My Analysis:
The primary edge case is when we are dealing with masked tensors, where not all of the tensor is used as a performance optimization.

AI Analysis:
> This bug was originally found when compiling Liger-Kernel's `LigerFusedLinearJSD`  and `LigerFusedLinearCrossEntropy` functions, where it silently corrupted outputs returning incorrect losses compared to eager execution whenever any label was `ignore_index` was present. Both of these failures can be silent.
I introduced a workaround with Liger at https://github.com/linkedin/Liger-Kernel/pull/1338, however the bug is fundamentally in PyTorch.

My Analysis:
I was using those kernels and ran into this bug. You can make a workaround in Liger, but this is a Pytorch bug.

## Reproduction


With AI assistance, I created a script which depends only on `torch` and `triton`.

AI Analysis:
> The script poisons the caching allocator so the
corruption is identifiable. The usual `compute-sanitizer --tool memcheck` does not flag this, because CUDA mappings are page-granular and a small over-read still lands in mapped memory.

My Analysis:
In other words, when the overruns are small enough, you land in your paged memory pool, which is still technically owned by the process and doesn't immediately fault.

The AI reproduction script is

```python
import torch, triton, triton.language as tl

N, R, SKIP, POISON, DEV = 2048, 4, 1, 12345.0, "cuda"

@triton.jit
def maybe_fill(out_ptr, do_write, n_cols, BLOCK: tl.constexpr):
    """Fill out_ptr[0:n_cols] with 1.0, but only when do_write != 0."""
    if do_write != 0:
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, 1.0, mask=offs < n_cols)

def run():
    base = torch.zeros(R, N, device=DEV, dtype=torch.float32)
    for i in range(R):
        row = base[i : i + 1]                       # view, storage_offset = i * N
        maybe_fill[(1,)](row, 0 if i == SKIP else 1, N, BLOCK=N)
        base[i : i + 1] = row                       # write-back
    return base

def poison_allocator(count=2048):
    junk = [torch.full((N,), POISON, device=DEV, dtype=torch.float32) for _ in range(count)]
    del junk
    torch.cuda.synchronize()

expected = torch.zeros(R, N, device=DEV)
expected[[i for i in range(R) if i != SKIP]] = 1.0

poison_allocator(); eager = run().clone()
cf = torch.compile(run, fullgraph=True); cf()      # compile before poisoning
poison_allocator(); compiled = cf().clone()

print("eager    matches:", torch.equal(eager, expected))
print("compiled matches:", torch.equal(compiled, expected))
print("freed-memory value present:", int((compiled == POISON).sum()), "elements")
```

AI Analysis:
> Row `SKIP` is never written by the kernel, so it must remain zero in both runs.

**Observed:**

```text
eager    matches: True
compiled matches: False
freed-memory value present: 2048 elements
```

> The poison value `12345.0` never exists in live tensors. It is written into memory that was freed before the call. Its presence is direct evidence of an out-of-bounds read.

My Analysis:
The AI generated script perfectly captures the bug that we are trying to resolve.

### Mechanism

AI Analysis
> Three different pieces interact to cause the bug. The first two are individually reasonable. The third breaks an
assumption the second one depends on.

> 1. Functionalizing a mutated Triton pointer arg clones it via
   `torch._prims_common.clone_preserve_strides`.

> 2. To preserve the view's layout, that helper has to clone `storage_offset + span` elements out of the tensor's storage, not just the tensor's own elements:

   ```python
   needed_size = compute_required_storage_length(x.size(), x.stride(), x.storage_offset())
   buffer = torch.as_strided(x, (needed_size,), (1,), 0).clone()
   return torch.as_strided(buffer, x.size(), x.stride(), x.storage_offset())
   ```

>  For `base[1:2]` of a `(4, 2048)` tensor that is `2048 + 2048 = 4096` elements. This is correct in eager, where the base storage genuinely has 8192 elements to read from.

> 3. Inductor's `as_strided` lowering has two paths. Which one it takes depends on whether the incoming tensor is still a view:

>   - `x` is an unrealized view over a larger buffer, the lowering unwraps to that buffer, so the offset is resolved against storage that really exists. This is the case the code was written for, and it is fine.
>   - `x` has already been materialized into its own buffer: the lowering takes that buffer as the storage. Inductor materializes a view when it is the output of a fused computation, which is exactly what the `slice_scatter` chain from the write-back produces.

> In the second case the storage is the small buffer of say 2048 elements, and the lowering builds a `ReinterpretView` describing 4096 elements. Nothing checks that the requested `ReinterpretView` extent fits.

My Analysis:
The key failure is in point 3 above. Because the tensor has already been materialized into a buffer of smaller size, when inductor then reinterprets the tensor, it uses the larger size which goes beyond the capacity of the allocated buffer being used for the smaller view. The rest is the particulars of the windowdressing that allows that failure to occur.

AI Analysis:
> So the invariant that breaks is this: `aten.as_strided`'s `storage_offset` is relative to the whole storage, but Inductor resolves it against whichever buffer happened to get realized. Those are the same thing right up until a view gets materialized into a buffer sized to the view, and then the lowering is describing memory that was never allocated.

My Analysis:
The storage_offset is relative to whole whole storage, reading out of bounds to the buffer used by the view.


### What hides this bug

AI Analysis:

> - `do_write` must be a runtime arg, not `tl.constexpr`. As a constexpr, Triton compiles the store away, the pointer is no longer seen as mutated, so no clone is emitted.
> - The mutated arg must be a view whose required extent exceeds its own numel. A non-zero `storage_offset` or gaps between elements must be present.
> - Some runtime path must skip writing, or the garbage is always overwritten.
> - The bug is is shape-dependent. It reproduces at `N = 2048` but not at `N = 1 << 20` because Inductor makes different realization choices.
> - The view must be realized as its own buffer. A single kernel launch on `base[1:2]` with no surrounding computation does not reproduce the bug. The loop with write-backs creates the `slice_scatter` chain that forces per-view realization.

My Analysis:
This bug only shows up when a confluence of factors occur. This is because some of the cases are already correctly handled by Inductor, the bug only materializes when the garbage memory is used in some way, and when inductor actually picks the specific kernels which are broken at compile time.

### Versions

```text
Collecting environment information...
PyTorch version: 2.14.0a0+git0a2b9b3
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
Flags:                                fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frintsvei8mm svebf16 i8mm bf16 dgh bti ecv afp wfxt
Model name:                           Cortex-A725
Model:                                1
Thread(s) per core:                   1
Core(s) per socket:                   10
Socket(s):                            1
Stepping:                             r0p1
CPU(s) scaling MHz:                   96%
CPU max MHz:                          2860.0000
CPU min MHz:                          338.0000
BogoMIPS:                             2000.00
Flags:                                fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frintsvei8mm svebf16 i8mm bf16 dgh bti ecv afp wfxt
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
[pip3] torch==2.14.0a0+git0a2b9b3
[pip3] triton==3.8.0+git10f6be36
[conda] Could not collect
```

cc @ezyang @gchanan @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @muchulee8 @amjames @aakhundov @coconutruben @jataylo @oulgen @davidberard98

## Duplicate mapping

- Primary report: `github-pytorch-pytorch-192219`
- Shared cause: `cause-pytorch-192219-realized-view-extent`

Both reports identify the same as_strided lowering decision, the same allocation-extent obligation, the same out-of-bounds physical effect, and the same repair in PR 192227.

The sections below describe the shared cause reconstructed from the primary report. The reported behavior above remains the observation specific to this report.

## Trigger

Inductor materializes an offset or strided view as its own buffer and later lowers as_strided using geometry from the original base storage.

## Relevant entities

original base storage; offset/strided view; realized intermediate buffer; ReinterpretView extent

## Implemented relation

as_strided offsets are relative to the original storage, while a realized intermediate owns only its allocated layout extent.

## Pre-state

The view geometry is valid relative to the original larger storage.

## Binding transition

Lowering constructs a ReinterpretView over the smaller realized buffer using the original offset and span.

## Post-state

The described extent exceeds the realized allocation.

## Validity obligation

Every reinterpreted view over a realized buffer remains within that buffer's allocated extent, or retains the base storage that makes the geometry valid.

## Invalid decision

as_strided lowering accepted a storage-relative offset and span without comparing the required extent with the selected realized allocation.

## Physical effect

Generated code performs an unmasked read beyond the realized buffer into freed or unrelated memory.

## Observed consequence

The compiled output contains 2,048 allocator-poison values and differs from eager execution.

## Repair site

torch/_inductor/lowering.py, torch/_higher_order_ops/triton_kernel_wrap.py, and torch/_prims_common/__init__.py

## Repair

The repair preserves or clones the appropriate base for mutated views and rejects or guards ReinterpretView extents that exceed non-input allocations.

## Why the repair works

View geometry is either derived within a retained allocation that contains it or checked against the actual realized allocation before code generation.

## Causal basis

The two reports, patch description, extent guard, and shared regression suite identify one implementation decision and repair relationship.

## Analysis implication

A view-lowering check needs both storage-relative geometry and the allocated extent of the concrete backing buffer selected after realization.

## Fix patches

- [`fixes/pytorch-192219-192226-fix.patch`](fixes/pytorch-192219-192226-fix.patch) — https://github.com/pytorch/pytorch/pull/192227
