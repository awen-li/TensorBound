# [BUG] sm_103 (B300): `tl.dot` chunked-recurrence kernel is non-deterministic with `num_warps∈{4,8}, num_stages∈{2,3}` — reproduces on Triton 3.7.0, i.e. NOT fixed by #9615 / #9871's fix

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/10590](https://github.com/triton-lang/triton/issues/10590)
- Opened: 2026-06-12T19:44:23Z
- Closed: 2026-06-15T15:09:07Z
- Labels: `bug`, `needs reproducer`

## Reported behavior

### Describe the bug

## Summary

A Triton kernel that carries an fp32 accumulator state across a sequential `for` loop of `tl.dot` updates (the standard "chunked linear-attention recurrence" pattern) produces **bitwise-different outputs on every invocation with identical inputs** on NVIDIA B300 (sm_103). Failing configs (each launched directly, no autotuner): `num_warps=4` with `num_stages=2` (BV=32 and BV=64), `num_warps=4` with `num_stages=3` (BV=32), and `num_warps=8` with `num_stages∈{2,3}` (BV=32). All `num_warps=2` and `num_warps=16` configs are bitwise deterministic. The kernel has no cross-CTA communication (each program writes a disjoint output slice), so this is a compiler/scheduling race, not a kernel logic error.

**Verified on both Triton 3.6.0 (torch 2.10/2.11 stack) and Triton 3.7.0 (torch 2.12 stack) — identical failing configs on both.** Since 3.7.0 includes PR #9615 (the fix that closed #9871 for sm_100), either that fix does not cover sm_103, or this is a related-but-distinct race.

This matches the failure class of #9871 (sm_100), but on sm_103 the envelope is **wider** than the formula verified there (`num_warps=4 AND BV<=32 AND num_stages∈{2,3} AND grid>=160`):

| | #9871 (sm_100, GB200/B200) | This report (sm_103, B300) |
|---|---|---|
| `BV=32, w4, s2/s3` | FAIL | FAIL |
| `BV=64, w4, s2` | **pass** (BV≤32 only) | **FAIL** |
| `BV=32, w8, s2/s3` | **pass** (verified formula: w4 only) | **FAIL** |
| Loop bound | dynamic (memory-loaded) required | **uniform runtime kernel arg fails** (`do_not_specialize`, no memory-loaded bound) |
| Grid threshold | ≥ ~160 CTAs | FAIL observed from 128–192 CTAs (timing-dependent; reliable at ≥ 256) |
| Status after #9615 (Triton 3.7.0) | fixed (regression test PR #10075) | **still failing (verified)** |

## Environment

| Component | Version |
|---|---|
| GPU | NVIDIA B300 SXM6 AC — sm_103 / capability (10, 3), 148 SMs |
| Driver | 580.126.09 (CUDA 13.0) |
| triton | **3.6.0** (torch 2.10.0+cu130) and **3.7.0** (torch 2.12.0+cu130) — both reproduce |
| dtype | bf16 inputs, fp32 accumulators |

Not reproducible on H200 (sm_90) — the identical downstream training pipeline is deterministic there.

## Reproduction

Self-contained script (also attached as `triton_sm103_repro.py`): the kernel mirrors flash-linear-attention's `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (K=V=128, BT=64, two `64×BV` fp32 state accumulators, per-head scalar gate). It launches each config directly with `num_warps`/`num_stages` and counts distinct bitwise output hashes over 10 identical calls.

Observed output on B300, Triton 3.6.0:

```text
sm=(10, 3) dev=NVIDIA B300 SXM6 AC triton=3.6.0 torch=2.10.0+cu130
HV=32 T=67856 K=128 V=128 BT=64 NT=1061

--- config matrix (B=4, 10 identical calls each; distinct>1 = nondeterministic) ---
  grid= 512 BV32_w2_s1: distinct=1/10
  grid= 512 BV32_w2_s2: distinct=1/10
  grid= 512 BV32_w2_s3: distinct=1/10
  grid= 512 BV32_w2_s4: distinct=1/10
  grid= 512 BV32_w4_s1: distinct=1/10
  grid= 512 BV32_w4_s2: distinct=10/10 <-- NONDETERMINISTIC
  grid= 512 BV32_w4_s3: distinct=10/10 <-- NONDETERMINISTIC
  grid= 512 BV32_w4_s4: distinct=1/10
  grid= 512 BV32_w8_s1: distinct=1/10
  grid= 512 BV32_w8_s2: distinct=10/10 <-- NONDETERMINISTIC
  grid= 512 BV32_w8_s3: distinct=10/10 <-- NONDETERMINISTIC
  grid= 512 BV32_w8_s4: distinct=1/10
  grid= 256 BV64_w2_s1: distinct=1/10
  grid= 256 BV64_w2_s2: distinct=1/10
  grid= 256 BV64_w2_s3: distinct=1/10
  grid= 256 BV64_w2_s4: distinct=1/10
  grid= 256 BV64_w4_s1: distinct=1/10
  grid= 256 BV64_w4_s2: distinct=10/10 <-- NONDETERMINISTIC
  grid= 256 BV64_w4_s3: distinct=1/10
  grid= 256 BV64_w4_s4: distinct=1/10
  grid= 256 BV64_w8_s1: distinct=1/10
  grid= 256 BV64_w8_s2: distinct=1/10
  grid= 256 BV64_w8_s3: distinct=1/10
  grid= 256 BV64_w8_s4: distinct=1/10

--- grid-size sweep for failing configs ---
  BV32_w4_s2: grid=128:ok grid=256:FAIL grid=384:FAIL grid=512:FAIL grid=768:FAIL
  BV32_w4_s3: grid=128:ok grid=256:FAIL grid=384:FAIL grid=512:FAIL grid=768:FAIL
  BV64_w4_s2: grid=64:ok  grid=128:ok  grid=192:FAIL grid=256:FAIL grid=384:FAIL
  BV32_w8_s2: grid=128:ok grid=256:FAIL grid=384:FAIL grid=512:FAIL grid=768:FAIL

FAILING CONFIGS: ['BV32_w4_s2', 'BV32_w4_s3', 'BV32_w8_s2', 'BV32_w8_s3', 'BV64_w4_s2']
```

Identical failing configs on **Triton 3.7.0** (torch 2.12.0+cu130, fresh venv, same machine):

```text
sm=(10, 3) dev=NVIDIA B300 SXM6 AC triton=3.7.0 torch=2.12.0+cu130
  grid= 512 BV32_w4_s2: distinct=10/10 <-- NONDETERMINISTIC
  grid= 512 BV32_w4_s3: distinct=10/10 <-- NONDETERMINISTIC
  grid= 512 BV32_w8_s2: distinct=10/10 <-- NONDETERMINISTIC
  grid= 512 BV32_w8_s3: distinct=10/10 <-- NONDETERMINISTIC
  grid= 256 BV64_w4_s2: distinct=10/10 <-- NONDETERMINISTIC
FAILING CONFIGS: ['BV32_w4_s2', 'BV32_w4_s3', 'BV32_w8_s2', 'BV32_w8_s3', 'BV64_w4_s2']
```

Notes:

- Failures begin once the grid exceeds roughly one wave on this part (148 SMs), consistent with #9871's occupancy observation; the exact threshold is timing-dependent (BV32_w4_s3 failed at 128 in one sweep and passed in another).
- **The race is not limited to `num_warps=4`:** `BV=32, num_warps=8, num_stages∈{2,3}` is also non-deterministic (reproduced both with the attached standalone script above and with the original FLA kernel, 25/25 distinct outputs there), while all `num_warps=2` and all `num_warps=16` configs were deterministic in our sweeps.
- The loop bound here is a **uniform runtime scalar** (`T` passed with `do_not_specialize`, `NT = tl.cdiv(T, BT)`); no memory-loaded bound is needed on sm_103, unlike the sm_100 findings in #9871.
- Differences are not tiny ULP noise consolidated at one location: in the originating real workload, ~0.05% of output elements differed per call with max abs diff ~0.045 (bf16 outputs in [-3, 3]).

## Real-world impact

This is not synthetic: the failing configs are inside the autotune space of `flash-linear-attention`'s `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, and on B300 the autotuner **selects** `BV=64, num_warps=4, num_stages=2` as best. Consequences observed in production training of Qwen3.5-MoE (hybrid GatedDeltaNet/MoE, DeepSpeed ZeRO-3, bf16):

- every linear-attention forward on B300 is silently non-reproducible; and
- under non-reentrant activation checkpointing, the forward-vs-recompute divergence propagates to the MoE router, flips ~0.4% of top-k expert assignments, changes data-dependent tensor shapes in Liger-Kernel's fused-MoE routing metadata, and crashes with `torch.utils.checkpoint.CheckpointError` (field report: [linkedin/Liger-Kernel#1255](https://github.com/linkedin/Liger-Kernel/issues/1255); kernel-owner report: fla-org/flash-linear-attention <link to FLA issue>).

All current torch releases are exposed on sm_103: torch 2.10.0 and 2.11.0 pin Triton 3.6.0, and torch 2.12.0 (latest stable) pins Triton 3.7.0 — we verified the failure on both Triton versions.

## Asks

1. **PR #9615 does not fix sm_103** — we verified the repro fails identically on Triton 3.7.0, which contains it. Please reopen #9871 or treat this as the sm_103 tracking issue. (Open question for maintainers: does the #10075 regression test for #9871 pass on sm_103 hardware? Our data suggests the sm_103 failure mode is broader — `BV=64` and uniform loop bounds are affected — so it may be a related but distinct missing-fence/pipelining case.)
2. If feasible, a check on Triton **main** (post-3.7.0) on B300/GB300 hardware with the attached script would tell whether any later change already covers it — we could not test main on this machine.
3. Guidance for released stacks: torch 2.10/2.11 (Triton 3.6.0) and torch 2.12 (Triton 3.7.0) are all affected on sm_103. We are mitigating by restricting affected kernels to `num_warps=2`, which is deterministic in all our testing (~10% slower on the affected kernel); confirmation that this is a sound workaround (vs. coincidental) would be valuable.

## Repro script

See attached `triton_sm103_repro.py` (self-contained, ~150 lines: kernel + config matrix + grid sweep, prints the tables above).

```python
"""Standalone repro: tl.dot chunked-recurrence non-determinism on Blackwell sm_103 (B300).

The kernel mirrors flash-linear-attention's chunk_gated_delta_rule_fwd_kernel_h_blockdim64
(K=V=128, BT=64, per-head scalar gate, uniform runtime loop bound). Each (i_v, i_nh)
program owns a disjoint slice of the outputs and the chunk loop is sequential
in-program, so any run-to-run output difference with identical inputs is a
compiler/scheduling race, not a kernel logic error.

Observed on NVIDIA B300 SXM6 AC (sm_103, 148 SMs), Triton 3.6.0 AND 3.7.0:
  - num_warps=4 + num_stages=2 (BV=32 and BV=64), num_warps=4 + num_stages=3 (BV=32),
    and num_warps=8 + num_stages={2,3} (BV=32) produce bitwise-different outputs on
    every call once grid >= ~128-256 CTAs.
  - All num_warps=2 (and num_warps=16) configs deterministic.
    Related: triton-lang/triton#9871 (sm_100), fixed by PR #9615 — that fix is in
    Triton 3.7.0 but does NOT fix these sm_103 cases.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def fwd_h_repro(
    k, v, w, v_new, g, h,
    T,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    bos = i_n * T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    b_h2 = tl.zeros([64, BV], dtype=tl.float32)

    h += (boh * HV + i_h).to(tl.int64) * K * V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * HV + i_h).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    v_new += (bos * HV + i_h).to(tl.int64) * V

    for i_t in range(NT):
        i_t64 = i_t.to(tl.int64)
        p_h1 = tl.make_block_ptr(h + i_t64 * HV * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        p_h2 = tl.make_block_ptr(h + i_t64 * HV * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v += tl.dot(b_w, b_h2.to(b_w.dtype))

        p_v = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v
        p_vn = tl.make_block_ptr(v_new, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        m_t = (i_t * BT + tl.arange(0, BT)) < T
        b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
        p_g = tl.make_block_ptr(g + (bos * HV + i_h).to(tl.int64), (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
        b_v = b_v * tl.where(m_t, tl.exp2(b_g_last - b_g), 0)[:, None]
        b_gl = tl.exp2(b_g_last)
        b_h1 *= b_gl
        b_h2 *= b_gl

        b_v = b_v.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(k, (K, T), (1, HV * K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        p_k = tl.make_block_ptr(k, (K, T), (1, HV * K), (64, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h2 += tl.dot(b_k, b_v)


def bithash(t: torch.Tensor, max_elems: int = 16_777_216) -> int:
    """Position-weighted bitwise hash (subsampled for big tensors)."""
    v = t.detach().contiguous().view(torch.int16).flatten()
    if v.numel() > max_elems:
        v = v[::(v.numel() + max_elems - 1) // max_elems]
    v = v.to(torch.int64)
    wgt = torch.arange(v.numel(), device=v.device, dtype=torch.int64) % 8191 + 1
    return int((v * wgt).sum().item())


def make_inputs(B: int, T: int, HV: int, K: int, V: int):
    torch.manual_seed(0)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, HV, K, device="cuda", dtype=torch.bfloat16), dim=-1
    ).to(torch.bfloat16).contiguous()
    w = (torch.randn(B, T, HV, K, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    v = torch.randn(B, T, HV, V, device="cuda", dtype=torch.bfloat16).contiguous()
    g = (-torch.rand(B, T, HV, device="cuda", dtype=torch.float32) * 2.0).contiguous()
    return k, w, v, g


def distinct_over_calls(B, T, HV, K, V, BT, BV, num_warps, num_stages, calls=10) -> int:
    k, w, v, g = make_inputs(B, T, HV, K, V)
    NT = triton.cdiv(T, BT)
    grid = (triton.cdiv(V, BV), B * HV)
    hashes = set()
    for _ in range(calls):
        h = k.new_empty(B, NT, HV, K, V)
        v_new = torch.empty_like(v)
        fwd_h_repro[grid](k, v, w, v_new, g, h, T, HV=HV, K=K, V=V, BT=BT, BV=BV,
                          num_warps=num_warps, num_stages=num_stages)
        hashes.add((bithash(h), bithash(v_new)))
        del h, v_new
    return len(hashes)


def main() -> None:
    HV, T, K, V, BT = 32, 67856, 128, 128, 64
    print(f"sm={torch.cuda.get_device_capability()} dev={torch.cuda.get_device_name()} "
          f"triton={triton.__version__} torch={torch.__version__}")
    print(f"HV={HV} T={T} K={K} V={V} BT={BT} NT={triton.cdiv(T, BT)}")

    print("\n--- config matrix (B=4, 10 identical calls each; distinct>1 = nondeterministic) ---")
    fails = []
    for BV in (32, 64):
        for nw in (2, 4, 8):
            for ns in (1, 2, 3, 4):
                d = distinct_over_calls(4, T, HV, K, V, BT, BV, nw, ns)
                grid_size = triton.cdiv(V, BV) * 4 * HV
                tag = f"BV{BV}_w{nw}_s{ns}"
                if d > 1:
                    fails.append(tag)
                print(f"  grid={grid_size:4d} {tag}: distinct={d}/10 {'<-- NONDETERMINISTIC' if d > 1 else ''}")

    print("\n--- grid-size sweep for failing configs ---")
    for BV, nw, ns in ((32, 4, 2), (32, 4, 3), (64, 4, 2), (32, 8, 2)):
        line = f"  BV{BV}_w{nw}_s{ns}: "
        for B in (1, 2, 3, 4, 6):
            d = distinct_over_calls(B, T, HV, K, V, BT, BV, nw, ns)
            grid_size = triton.cdiv(V, BV) * B * HV
            line += f"grid={grid_size}:{'FAIL' if d > 1 else 'ok'} "
        print(line)

    print("\nFAILING CONFIGS:", fails if fails else "none")


if __name__ == "__main__":
    main()
```

### Environment details

| Component | Version |
|---|---|
| GPU | NVIDIA B300 SXM6 AC — sm_103 / capability (10, 3), 148 SMs |
| Driver | 580.126.09 (CUDA 13.0) |
| triton | **3.6.0** (torch 2.10.0+cu130) and **3.7.0** (torch 2.12.0+cu130) — both reproduce |
| dtype | bf16 inputs, fp32 accumulators |

Not reproducible on H200 (sm_90) — the identical downstream training pipeline is deterministic there.

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

A grid larger than roughly one device wave runs recurrence kernels with four or eight warps and two or three pipeline stages.

## Observed implementation decision

The report suspects a missing fence or pipelining dependency but does not identify the transformed operation or generated ordering edge responsible on sm_103.

## Physical effect

Concurrent program instances observe nondeterministic recurrence state.

## Observed consequence

Repeated calls yield distinct outputs and can alter downstream expert routing during training.

## Reported proposal or workaround

Two-warps configurations remain deterministic; the reporter confirms the full configuration matrix succeeds on Triton main.

## Repair trace

Neither maintainers nor the reporter identify which main-branch commit fixed the behavior. Without the responsible source delta, generated-code contrast, and matching regression, the fixed-version run cannot establish the repair relationship.

## Archived evidence

- Raw report: `data/raw/github/triton/10590/report.json`
- SHA-256: `1e43cdc524d44a5f3a0f0dcc815015e79ef63a6f7401671a1705c02004690623`
