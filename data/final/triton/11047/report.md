# warpgroup_mma_wait does not properly synchronize across warpgroups

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/11047](https://github.com/triton-lang/triton/issues/11047)
- Opened: 2026-07-25T13:57:35Z
- Closed: 2026-07-28T00:15:41Z
- Labels: `bug`

## Reported behavior

### Describe the bug

warpgroup_mma_wait can be called from code that spans multiple warpgroups, but it does not synchronize as such. This allows for pernicious thread divergence that breaks the Triton programming model, for example in this program:

```python3
import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
import triton.language as tl

D = tl.constexpr(64)
WG = 2  # warpgroups


@gluon.jit
def kernel(a_ptr, b0_ptr, b1_ptr, out_ptr, NBLK: gl.constexpr, WG: gl.constexpr):
    M: gl.constexpr = 64 * WG
    c_layout: gl.constexpr = gl.NVMMADistributedLayout([3, 0], warps_per_cta=[4 * WG, 1], instr_shape=[16, 64, 16])
    smem_b: gl.constexpr = gl.NVMMASharedLayout.get_default_for([D, D], gl.bfloat16)
    smem_a: gl.constexpr = gl.NVMMASharedLayout.get_default_for([M, D], gl.bfloat16)
    blk: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4 * WG, 1], [1, 0])

    a_smem = gl.allocate_shared_memory(gl.bfloat16, [M, D], smem_a)
    b0_smem = gl.allocate_shared_memory(gl.bfloat16, [D, D], smem_b)
    b1_smem = gl.allocate_shared_memory(gl.bfloat16, [D, D], smem_b)

    rows = gl.arange(0, D, layout=gl.SliceLayout(1, blk))
    cols = gl.arange(0, D, layout=gl.SliceLayout(0, blk))
    offs = rows[:, None] * D + cols[None, :]
    aoffs = gl.arange(0, M, layout=gl.SliceLayout(1, blk))[:, None] * D + cols[None, :]

    acc = gl.zeros([M, D], gl.float32, c_layout)
    inflight = hopper.warpgroup_mma_init(acc)

    cp.async_copy_global_to_shared(a_smem, a_ptr + aoffs)
    cp.commit_group()

    for i in tl.range(0, NBLK):
        cp.async_copy_global_to_shared(b0_smem, b0_ptr + i * D * D + offs)
        acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=(inflight, ))
        cp.async_copy_global_to_shared(b1_smem, b1_ptr + i * D * D + offs)
        cp.commit_group()
        cp.wait_group(0)
        hopper.fence_async_shared()
        t = hopper.warpgroup_mma(a_smem, b0_smem, acc, use_acc=True, is_async=True)
        acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=(t, ))
        inflight = hopper.warpgroup_mma(a_smem, b1_smem, acc, use_acc=True, is_async=True)
    acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=(inflight, ))

    out_r = gl.arange(0, M, layout=gl.SliceLayout(1, c_layout))
    out_c = gl.arange(0, D, layout=gl.SliceLayout(0, c_layout))
    pid = tl.program_id(0)
    tl.store(out_ptr + pid.to(tl.int64) * (M * D) + out_r[:, None] * D + out_c[None, :], acc)


if __name__ == "__main__":
    torch.manual_seed(0)
    d, m = D.value, 64 * WG
    NBLK, GRID = 512, 528

    # A is all ones and B0/B1 are integer-valued, so every product and partial sum is
    # exact in fp32:  acc[r,c] = sum_i sum_k (B0[i,k,c] + B1[i,k,c])
    b0 = torch.randint(-3, 4, (NBLK, d, d), device="cuda", dtype=torch.float32)
    b1 = torch.randint(-3, 4, (NBLK, d, d), device="cuda", dtype=torch.float32)
    expected = (b0.sum(dim=(0, 1)) + b1.sum(dim=(0, 1)))[None, :].expand(m, d)
    assert expected.abs().max() < 2**24
    a = torch.ones((m, d), device="cuda", dtype=torch.bfloat16)
    b0b, b1b = b0.to(torch.bfloat16), b1.to(torch.bfloat16)

    for rep in range(5):
        out = torch.empty((GRID, m, d), dtype=torch.float32, device="cuda")
        kernel[(GRID, )](a, b0b, b1b, out, NBLK=NBLK, WG=WG, num_warps=4 * WG)
        torch.cuda.synchronize()
        err = (out - expected[None]).abs()
        print(f"rep {rep}: max_abs_err={err.max().item():6.1f}  "
              f"programs_wrong={int(err.amax(dim=(1, 2)).gt(0).sum())}/{GRID}")
```

Adding a `gl.debug_barrier()` after the wait resolves the issue, but this should really be inserted by the compiler.

### Environment details

Triton: main
GPU: GH200

## Trigger

Eight warps share WGMMA operands and reuse shared memory after warpgroup_mma_wait(0).

## Relevant entities

two warpgroups; async WGMMA; wait_group; CTA barrier; shared-memory operands

## Implemented relation

A nonlocal WGMMA wait makes completion and shared-memory liveness visible across every warpgroup participating in the CTA.

## Pre-state

One or more warpgroups may still read shared operands for an asynchronous dot.

## Binding transition

Lowering emits only nvg.wgmma_wait_group, whose synchronization scope is one warpgroup.

## Post-state

Another warpgroup advances and overwrites shared memory still in use elsewhere.

## Validity obligation

A wait exposed as CTA-wide inserts cross-warpgroup synchronization when num_warps exceeds four.

## Invalid decision

WarpGroupDotWaitOp did not represent whether its scope was local and lowering omitted a CTA barrier for multi-warpgroup use.

## Physical effect

Warpgroups diverge around shared-memory reuse and read different B tiles.

## Observed consequence

The exact integer-valued matmul accumulation becomes intermittently incorrect.

## Repair site

TritonNVIDIAGPUToLLVM DotOpToLLVM.cpp and Membar analysis

## Repair

PR 11056 adds a warpGroupLocal attribute, emits nvvm.barrier after nonlocal waits when num_warps > 4, teaches membar analysis that such waits synchronize, and adds the reported two-warpgroup regression.

## Why the repair works

No warpgroup can reuse shared operands until all participating warpgroups pass the barrier after WGMMA completion.

## Causal basis

The re-applied patch explicitly fixes #11047 and adds the report's multi-warpgroup kernel plus LLVM barrier checks.

## Analysis implication

An async-wait check needs operation scope, participating warps/warpgroups, pending count, shared operands, and following reuse.

## Fix patches

- [`fixes/triton-11047-pr11056.patch`](fixes/triton-11047-pr11056.patch) — https://github.com/triton-lang/triton/pull/11056
