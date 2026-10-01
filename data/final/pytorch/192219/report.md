# BUG: Inductor `as_strided` silently reads out of bounds

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/192219](https://github.com/pytorch/pytorch/issues/192219)
- Opened: 2026-08-05T12:44:48Z
- Closed: 2026-08-06T11:29:00Z
- Labels: `triage review`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `module: user triton`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

When a tensor that is a view with a non-zero `storage_offset` is passed as a mutated pointer argument to a raw `triton.jit` kernel, `torch.compile` silently produces an incorrect result. This is due to data being read from beyond the end of a buffer into memory.

The over-read is normally invisible because the kernel overwrites the whole region. The edge case is when the kernel skips writing part of its output as a performance optimization. E.G "if this row is masked/ignored, return early".

This bug was originally found when compiling Liger-Kernel's `LigerFusedLinearJSD`  and `LigerFusedLinearCrossEntropy` functions, where it silently corrupted outputs returning incorrect losses compared to eager execution whenever any label was `ignore_index` was present. Both of these failures can be silent.
I introduced a workaround with Liger at https://github.com/linkedin/Liger-Kernel/pull/1338, however the bug is fundamentally in PyTorch.

## Reproduction

With AI assistance, I created a script which depends only on `torch` and `triton`. The script poisons the caching allocator so the
corruption is identifiable. The usual `compute-sanitizer --tool memcheck` does not flag this, because CUDA mappings are page-granular and a small over-read still lands in mapped memory. In other words, when the overruns are small enough, you land in your paged memory pool, which is still technically owned by the process and doesn't immediately fault.

The reproduction script is

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

Row `SKIP` is never written by the kernel, so it must remain zero in both runs.

**Observed:**

```text
eager    matches: True
compiled matches: False
freed-memory value present: 2048 elements
```

The poison value `12345.0` never exists in live tensors. It is written into memory that was freed before the call. Its presence is direct evidence of an out-of-bounds read.

### Mechanism

Three different pieces interact to cause the bug. The first two are individually reasonable. The third breaks an
assumption the second one depends on.

1. Functionalizing a mutated Triton pointer arg clones it via
   `torch._prims_common.clone_preserve_strides`.

2. To preserve the view's layout, that helper has to clone `storage_offset + span` elements out of the tensor's storage, not just the tensor's own elements:

   ```python
   needed_size = compute_required_storage_length(x.size(), x.stride(), x.storage_offset())
   buffer = torch.as_strided(x, (needed_size,), (1,), 0).clone()
   return torch.as_strided(buffer, x.size(), x.stride(), x.storage_offset())
   ```

   For `base[1:2]` of a `(4, 2048)` tensor that is `2048 + 2048 = 4096` elements. This is correct in eager, where the base storage genuinely has 8192 elements to read from.

3. Inductor's `as_strided` lowering has two paths. Which one it takes depends on whether the incoming tensor is still a view:

   - `x` is an unrealized view over a larger buffer, the lowering unwraps to that buffer, so the offset is resolved against storage that really exists. This is the case the code was written for, and it is fine.
   - `x` has already been materialized into its own buffer: the lowering takes that buffer as the storage. Inductor materializes a view when it is the output of a fused computation, which is exactly what the `slice_scatter` chain from the write-back produces.

   In the second case the storage is the small buffer of say 2048 elements, and the lowering builds a `ReinterpretView` describing 4096 elements. Nothing checks that the requested `ReinterpretView` extent fits.

So the invariant that breaks is this: `aten.as_strided`'s `storage_offset` is relative to the
whole storage, but Inductor resolves it against whichever buffer happened to get realized. Those are the same thing right up until a view gets materialized into a buffer sized to the view, and then the lowering is describing memory that was never allocated.

From `TORCH_LOGS=output_code`:

```python
buf3 = empty_strided_cuda((1, 2048), (2048, 1), torch.float32)   # 2048 elements
buf4 = empty_strided_cuda((4096, ), (1, ), torch.float32)
triton_poi_fused_2.run(buf3, buf4, 4096)      # <-- reads 4096 elements out of buf3
maybe_fill_1.run(reinterpret_tensor(buf4, (1, 2048), (2048, 1), 2048), 0, ...)
```

The offending Inductor generated kernel. Note how the read is unmasked:

```python
def triton_poi_fused_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xmask = tl.full([XBLOCK], True, tl.int1)[:]   # unmasked
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)          # 2048 elements past the end of buf3
    tl.store(out_ptr0 + (x0), tmp0, None)
```

`maybe_fill_1` then receives `buf4 + 2048`, which is precisely the garbage half. Because `do_write == 0` it never writes there, so the garbage survives into the output.

It gets worse further into the loop, since the buffer sizes track `needed_size`: 4096 / 6144 / 8192 for rows 1 / 2 / 3. Each of those still reads out of a 2048-element source, so the last iteration over-reads by 6144 elements.

This is a PyTorch bug, not a Triton one. The out-of-bounds load is in an Inductor-generated kernel: `tl.load(in_ptr0 + x0, None)` with `xnumel = 4096` carries no bounds information and Triton has no way to know `in_ptr0` is only 2048 elements long. The mutation analysis that decides which args are mutated is correct here.

### What hides this bug

- `do_write` must be a runtime arg, not `tl.constexpr`. As a constexpr, Triton compiles the store away, the pointer is no longer seen as mutated, so no clone is emitted.
- The mutated arg must be a view whose required extent exceeds its own numel. A non-zero `storage_offset` or gaps between elements must be present.
- Some runtime path must skip writing, or the garbage is always overwritten.
- The bug is is shape-dependent. It reproduces at `N = 2048` but not at `N = 1 << 20` because Inductor makes different realization choices.
- The view must be realized as its own buffer. A single kernel launch on `base[1:2]` with no surrounding computation does not reproduce the bug. The loop with write-backs creates the `slice_scatter` chain that forces per-view realization.

### Versions

Reproduced on `2.14.0a0+git0a2b9b3` with triton 3.8.0, and on `2.13.0+cu130` / `2.12.1+cu130`
with triton 3.7.1. NVIDIA GB10, `sm_121`, Ubuntu 24.04 aarch64, Python 3.12 / 3.10.

Also reproduces with `fullgraph=False`, with `dynamic=True`, and under `inference_mode` and
`no_grad`.

cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo @oulgen @davidberard98

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
