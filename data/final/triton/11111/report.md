# triton seems to not lower a float8 e5m2 transposed store correctly in some conditions

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/11111](https://github.com/triton-lang/triton/issues/11111)
- Opened: 2026-07-30T11:34:01Z
- Closed: open at collection time
- Labels: `bug`

## Reported behavior

### Describe the bug

  A kernel computes a single fp8 (float8e5) value and stores it to two output buffers — one contiguous layout, one transposed layout. When the value is produced through
  tl.maximum/tl.minimum with tl.PropagateNan.ALL, the transposed store is corrupted (garbage bytes decoding as Inf/NaN in e5m2), while the contiguous store of the same value is always
  correct. Rewriting the min/max as the numerically identical tl.where(a>b | a!=a, a, b) makes the corruption disappear.

  The stored value is provably finite (the contiguous output is always clean, and the cast is cvt.rn.satfinite.e5m2x2.f32, which cannot emit Inf/NaN). So this is a store lowering /
  codegen bug, not a bad computed value. It only manifests when the min/max lower to the min.NaN.f32/max.NaN.f32 PTX intrinsics; the setp+selp form is fine. The exact corruption
  pattern varies with XBLOCK/num_warps (e.g. 72 NaN + 24 Inf, or 3072 Inf, or 96 NaN), which points at register allocation / store vectorization rather than a semantic issue.

Claude distilled this repro for me: https://gist.github.com/vkuzo/c7410babadd904c93beb3fd3638ffc75

Repro code:

```python
import torch, triton, triton.language as tl


@triton.jit
def k_prop(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, XBLOCK: tl.constexpr):
    xnumel = 768
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768 * r0_1), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (0)).to(tl.float32)
    tmp7 = tl.broadcast_to(tmp6, [1, 1])
    tmp5 = tmp0.to(tl.float32)
    tmp8 = tmp7.to(tl.float64)
    tmp9 = tl.full([1, 1], 1e-12, tl.float64)
    tmp10 = tl.maximum(tmp8, tmp9, tl.PropagateNan.ALL)
    tmp11 = tl.full([1, 1], 1.0, tl.float32)
    tmp12 = (tmp11 / tmp10)
    tmp13 = tl.full([1, 1], 57344.0, tl.float64)
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp5 * tmp15
    tmp17 = tl.full([1, 1], -57344.0, tl.float32)
    tmp18 = tl.maximum(tmp16, tmp17, tl.PropagateNan.ALL)
    tmp19 = tl.full([1, 1], 57344.0, tl.float32)
    tmp20 = tl.minimum(tmp18, tmp19, tl.PropagateNan.ALL)
    tmp21 = tmp20.to(tl.float8e5)
    tl.store(out_ptr1 + (r0_1 + 16 * x0), tmp21, r0_mask & xmask)
    tl.store(out_ptr2 + (x0 + 768 * r0_1), tmp21, r0_mask & xmask)


@triton.jit
def k_plain(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, XBLOCK: tl.constexpr):
    # identical, but pre-#186933 tl.where-based min/max (numerically equivalent)
    xnumel = 768
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768 * r0_1), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (0)).to(tl.float32)
    tmp7 = tl.broadcast_to(tmp6, [1, 1])
    tmp5 = tmp0.to(tl.float32)
    tmp8 = tmp7.to(tl.float64)
    tmp9 = tl.full([1, 1], 1e-12, tl.float64)
    tmp10 = tl.where((tmp8 > tmp9) | (tmp8 != tmp8), tmp8, tmp9)   # == maximum
    tmp11 = tl.full([1, 1], 1.0, tl.float32)
    tmp12 = (tmp11 / tmp10)
    tmp13 = tl.full([1, 1], 57344.0, tl.float64)
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp5 * tmp15
    tmp17 = tl.full([1, 1], -57344.0, tl.float32)
    tmp18 = tl.where((tmp16 > tmp17) | (tmp16 != tmp16), tmp16, tmp17)  # == maximum
    tmp19 = tl.full([1, 1], 57344.0, tl.float32)
    tmp20 = tl.where((tmp18 < tmp19) | (tmp18 != tmp18), tmp18, tmp19)  # == minimum
    tmp21 = tmp20.to(tl.float8e5)
    tl.store(out_ptr1 + (r0_1 + 16 * x0), tmp21, r0_mask & xmask)
    tl.store(out_ptr2 + (x0 + 768 * r0_1), tmp21, r0_mask & xmask)


def nf(t):
    tf = t.float()
    return int(torch.isnan(tf).sum()), int(torch.isinf(tf).sum())


def run(kern, in0, in1, XBLOCK, num_warps):
    out0 = torch.zeros(768, device="cuda", dtype=torch.bfloat16)
    out1 = torch.zeros((768, 16), device="cuda", dtype=torch.float8_e5m2)  # transposed
    out2 = torch.zeros((16, 768), device="cuda", dtype=torch.float8_e5m2)  # contiguous
    grid = (triton.cdiv(768, XBLOCK),)
    kern[grid](in0, in1, out0, out1, out2, XBLOCK, num_warps=num_warps)
    torch.cuda.synchronize()
    return out1, out2


def main():
    print(f"torch  {torch.__version__}")
    print(f"triton {triton.__version__}")
    print(f"device {torch.cuda.get_device_name()}\n")

    # Synthetic finite inputs matching the observed values (grad_output amax=1,
    # amax-scale=1). Every produced e5m2 value is exactly 57344 (finite).
    in0 = torch.ones((16, 768), device="cuda", dtype=torch.bfloat16)
    in1 = torch.ones((), device="cuda", dtype=torch.bfloat16)
    print(f"inputs: in0 nan/inf={nf(in0)} absmax={in0.float().abs().amax():.4g} ; "
          f"in1 nan/inf={nf(in1)} val={in1.float().item():.4g}\n")

    print(f"{'variant':10s} {'XBLOCK':>6} {'warps':>5}  "
          f"{'out1(transposed)':>18} {'out2(contiguous)':>18}")
    for name, kern in [("PROP.ALL", k_prop), ("tl.where", k_plain)]:
        for XBLOCK in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024):
            for nw in (1, 2, 4, 8):
                o1, o2 = run(kern, in0, in1, XBLOCK, nw)
                n1 = nf(o1); n2 = nf(o2)
                flag = "  <-- MISMATCH" if (n1 != (0, 0) and n2 == (0, 0)) else ""
                if n1 != (0, 0) or n2 != (0, 0) or flag:
                    print(f"{name:10s} {XBLOCK:>6} {nw:>5}  "
                          f"nan/inf={str(n1):>12} nan/inf={str(n2):>12}{flag}")
        print()


if __name__ == "__main__":
    main()
```

Output of running the repro:

```code
(pt_nightly) dev@gpu-dev-67601812:~/ao (main)$ python /tmp/triton_store_bug_repro.py
torch  2.14.0.dev20260728+cu130
triton 3.8.0
device NVIDIA H100 80GB HBM3

inputs: in0 nan/inf=(0, 0) absmax=1 ; in1 nan/inf=(0, 0) val=1

variant    XBLOCK warps    out1(transposed)   out2(contiguous)
PROP.ALL        8     1  nan/inf=   (0, 3072) nan/inf=      (0, 0)  <-- MISMATCH
PROP.ALL       16     1  nan/inf=   (0, 3072) nan/inf=      (0, 0)  <-- MISMATCH
PROP.ALL       16     2  nan/inf=   (0, 3072) nan/inf=      (0, 0)  <-- MISMATCH
PROP.ALL       32     1  nan/inf=   (0, 3072) nan/inf=      (0, 0)  <-- MISMATCH
PROP.ALL       32     2  nan/inf=    (72, 24) nan/inf=      (0, 0)  <-- MISMATCH
PROP.ALL       32     4  nan/inf=    (72, 24) nan/inf=      (0, 0)  <-- MISMATCH
PROP.ALL       64     2  nan/inf=   (0, 1536) nan/inf=      (0, 0)  <-- MISMATCH
PROP.ALL       64     4  nan/inf=  (36, 1548) nan/inf=      (0, 0)  <-- MISMATCH
PROP.ALL       64     8  nan/inf=   (0, 3072) nan/inf=      (0, 0)  <-- MISMATCH
PROP.ALL      128     1  nan/inf=      (7, 2) nan/inf=      (0, 0)  <-- MISMATCH
PROP.ALL      128     2  nan/inf=    (0, 384) nan/inf=      (0, 0)  <-- MISMATCH
PROP.ALL      128     4  nan/inf=   (0, 1536) nan/inf=      (0, 0)  <-- MISMATCH
PROP.ALL      256     4  nan/inf=   (0, 2688) nan/inf=      (0, 0)  <-- MISMATCH
PROP.ALL     1024     4  nan/inf=     (96, 0) nan/inf=      (0, 0)  <-- MISMATCH
PROP.ALL     1024     8  nan/inf=     (96, 0) nan/inf=      (0, 0)  <-- MISMATCH
```

### Environment details

Environment
  - triton 3.8.0
  - torch 2.14.0.dev20260728+cu126
  - GPU: NVIDIA H100 80GB HBM3 (sm_90)

## Trigger

A Triton kernel computes an FP8 value through NaN-propagating minimum/maximum and stores it through both contiguous and transposed vectorized layouts on Hopper with bundled ptxas 12.9.

## Relevant entities

finite FP8 value; min.NaN/max.NaN PTX; layout conversion; contiguous store; transposed one-byte store; ptxas version

## Implemented relation

Both physical stores of one SSA value must preserve that value's bytes through register allocation and layout conversion.

## Pre-state

The computed value is finite and the contiguous store exposes the expected byte.

## Binding transition

ptxas allocates registers for the min.NaN/max.NaN code shape feeding a nontrivial transposed FP8 store.

## Post-state

The transposed store reads foreign register bytes that were not produced by the kernel, while the contiguous store remains correct.

## Validity obligation

Backend register allocation and layout lowering preserve SSA value identity for every store layout and element type.

## Invalid decision

The ptxas 12.9 compilation path assigns the transposed vectorized one-byte store a corrupted value under the min.NaN/max.NaN instruction shape.

## Physical effect

Foreign bytes such as 0x60 or 0xFC reach the transposed output.

## Observed consequence

The transposed FP8 output contains NaN or Inf even though the identical contiguous output is finite; ptxas 13.3 does not reproduce it.

## Repair site

torchao/float8/_inductor_patch.py and float8_linear_utils.py (scoped downstream workaround)

## Repair

TorchAO PR 4652 routes minimum/maximum through a numerically equivalent tl.where-based form before float8 compilation, avoiding min.NaN/max.NaN in the affected product path.

## Why the repair works

The workaround preserves NaN semantics while removing the PTX instruction shape that triggers corrupted register allocation before the transposed store.

## Causal basis

The issue rules out numerical and barrier explanations and reports the toolchain-version contrast; the merged downstream patch removes only the triggering code shape and explicitly scopes itself as a workaround.

## Analysis implication

A code-generation check needs value provenance, element width, source and destination layouts, emitted PTX opcode form, target architecture, and assembler version.

## Fix patches

- [`fixes/torchao-11111-pr4652.patch`](fixes/torchao-11111-pr4652.patch) — https://github.com/pytorch/ao/pull/4652
