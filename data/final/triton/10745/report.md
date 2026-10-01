# [BUG][AMD] fp8 e4m3fnuz/e5m2fnuz NaN (byte 0x80) silently decodes to fp16/bf16 -0.0 instead of NaN on non-CDNA3 GPUs

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/10745](https://github.com/triton-lang/triton/issues/10745)
- Opened: 2026-06-29T03:23:05Z
- Closed: 2026-07-12T02:09:21Z
- Labels: `bug`

## Reported behavior

## Describe the bug

On AMD GPUs **without** native FNUZ-fp8 hardware conversion (everything except CDNA3 / gfx942 — i.e. CDNA1 gfx908, CDNA2 gfx90a, CDNA4 gfx950, and RDNA), upcasting an `float8_e4m3fnuz` or `float8_e5m2fnuz` value whose bit pattern is `0x80` produces a finite `-0.0` instead of NaN. `0x80` is the *only* NaN encoding in these FNUZ formats (the formats have no -0/Inf; the -0 slot is repurposed as the single NaN), so a NaN is silently turned into a finite signed zero with no error — corrupting any downstream NaN-propagation or NaN-masking.

The conversion is done in software on these arches (the HW `cvt` path is gated to CDNA3 only — [`hasFnuzFp8HW`, ConvertFpCastOpToLLVM.cpp#L50](https://github.com/triton-lang/triton/blob/285de7a16f2d4585c6fdecad4167c6e7ee5cb869/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp#L50)), and **three** software decoders mishandle `0x80`. The `→ fp16` decoders produce fp16 `-0.0` (`0x8000`) instead of NaN (`0x7e00`); the `→ bf16` decoder produces bf16 `-0.0` (`0x8000`) instead of NaN (`0x7fc0`):

**`e4m3fnuz` — the NaN guard is present but then overwritten.** In [`Fp8E4M3fnuzToFp16OneValue`](https://github.com/triton-lang/triton/blob/285de7a16f2d4585c6fdecad4167c6e7ee5cb869/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp#L949-L999), the NaN guard maps `0x80 → 0x7E00` at [L974](https://github.com/triton-lang/triton/blob/285de7a16f2d4585c6fdecad4167c6e7ee5cb869/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp#L974), but the denorm/zero LUT loop at [L989-L992](https://github.com/triton-lang/triton/blob/285de7a16f2d4585c6fdecad4167c6e7ee5cb869/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp#L989-L992) runs *after* it and selects on `vAbs = v8 & 0x7F`; for `0x80`, `vAbs == 0` matches `denormsAndZeroLut[0] = 0x0000` and overwrites the NaN, and the trailing sign re-apply at [L995](https://github.com/triton-lang/triton/blob/285de7a16f2d4585c6fdecad4167c6e7ee5cb869/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp#L995) gives `0x8000`.

**`e5m2fnuz → fp16` — no NaN guard at all.** [`Fp8E5M2fnuzToFp16OneValue`](https://github.com/triton-lang/triton/blob/285de7a16f2d4585c6fdecad4167c6e7ee5cb869/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp#L1124-L1158) has no `0x80` case; with `exp == 0` it takes the `e_is_zero` branch `o0 = sign | (m >> 1) = 0x8000`.

**`e4m3fnuz → bf16` — a separate decoder, also no NaN guard.** The bf16 upcast does not reuse the fp16 path; it has its own bit-trick decoder [`Fp8E4M3fnuzToBf16SW`](https://github.com/triton-lang/triton/blob/285de7a16f2d4585c6fdecad4167c6e7ee5cb869/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp#L1257-L1306) (selected at [L1232](https://github.com/triton-lang/triton/blob/285de7a16f2d4585c6fdecad4167c6e7ee5cb869/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp#L1232)). For `0x80` the magnitude is masked off ([`absHalfwordMask`, L1280](https://github.com/triton-lang/triton/blob/285de7a16f2d4585c6fdecad4167c6e7ee5cb869/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp#L1280)) so the value becomes `0`, then the sign bit is OR'd back ([L1299-L1300](https://github.com/triton-lang/triton/blob/285de7a16f2d4585c6fdecad4167c6e7ee5cb869/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp#L1299-L1300)), giving bf16 `0x8000` (`-0.0`) instead of NaN (`0x7fc0`).

The remaining two fnuz upcasts route through the above and so share the defect (and the fix) transitively: `e5m2fnuz → bf16` calls `Fp8E5M2fnuzToFp16SW`, and both `→ fp32` go fp8 → fp16 → fp32. So all six `{e4m3fnuz, e5m2fnuz} × {fp16, bf16, fp32}` upcasts silently lose the NaN.

Minimal kernel that triggers it (run on a non-CDNA3 AMD GPU — gfx908/gfx90a/gfx950/RDNA — so the SW decode path is selected):

```python
import torch
import triton
import triton.language as tl

@triton.jit
def cvt(x_ptr, y_ptr, N: tl.constexpr):
    off = tl.arange(0, N)
    x = tl.load(x_ptr + off)          # tl.float8e4b8 == e4m3fnuz (use float8e5b16 for e5m2fnuz)
    tl.store(y_ptr + off, x.to(tl.float16))

# 0x80 is the e4m3fnuz NaN encoding
x = torch.tensor([0x80], dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fnuz)
y = torch.empty(1, dtype=torch.float16, device="cuda")
cvt[(1,)](x, y, N=1)
print(y.item(), y.view(torch.int16).item())   # expected: nan / 0x7e00 ; actual: -0.0 / 0x8000
```

Expected: `nan` / `0x7e00` — matching the reference `torch.tensor([0x80],dtype=uint8).view(torch.float8_e4m3fnuz).to(torch.float16)`. From the SW decode path it instead produces `-0.0` / `0x8000`; same for `float8_e5m2fnuz`.

Compiling that kernel through the AMD backend at `arch=gfx90a` (host-side codegen, no GPU needed) emits exactly the guard-then-clobber sequence — for the NaN byte `0x80`, `%17` sets the correct NaN `0x7E00` (=32256), then the LUT overwrites it and the sign is re-applied to give `0x8000`:

```llvm
%8  = ... raw.ptr.buffer.load.i8 ...        ; the fp8 byte v
%11 = and i16 %10, -32768                   ; sign (0x8000)
%16 = icmp eq i8 %8, -128                    ; v == 0x80 (the FNUZ NaN code)
%17 = select i1 %16, i16 32256, i16 %14      ; NaN guard -> 0x7E00   (L974)
%18 = icmp eq i8 %15, 0                       ; vAbs == 0 ?           (vAbs = v & 0x7F)
%19 = select i1 %18, i16 0, i16 %17           ; LUT[0] overwrites the NaN with 0  (L989-992)
...                                           ; %21..%33: LUT[1..7], none match for 0x80
%34 = or i16 %33, %11                         ; re-apply sign -> 0x8000 = fp16 -0.0  (L995)
```

(The decode is pure integer + a `bitcast`, so the byte result is target-independent: this LLVM IR is what gfx90a/gfx908/gfx950/RDNA all lower to. I confirmed the emitted IR above and cross-checked all 256 codes against the torch reference, where only `0x80` diverges; I did not run it on a physical AMD GPU.)

**Expected behavior.** Byte `0x80` should decode to a fp16 NaN (`0x7e00`), matching the format's single NaN encoding and the reference decoders (PyTorch's FNUZ dtypes, and the compiler's own intent — the guard at L974 is meant to produce exactly this). The compiler should not turn a NaN into a finite -0.0.

**Suggested fix.**
- `e4m3fnuz → fp16`: apply the `0x80 → 0x7E00` NaN guard *after* both the LUT loop and the sign re-apply (i.e. make it the last write), so the LUT cannot overwrite it. (Moving it just after the LUT but before the sign-or yields `0xfe00`, which is still a NaN but carries the sign bit; making it the final write gives `0x7e00` exactly.)
- `e5m2fnuz → fp16`: add the same `v8 == 0x80 → 0x7E00` guard as the final step.
- `e4m3fnuz → bf16`: add a per-lane `0x80 → 0x7FC0` (bf16 NaN) guard on the two unpacked output lanes.

The LUT-based denorm decode for all non-NaN codes is unaffected. Verified by lowering all six combos through the AMD backend at `arch=gfx90a` (host-side, no GPU): post-fix each emits the per-format NaN guard (`0x7E00` for fp16/fp32, `0x7FC0` for bf16) and the pre-fix build does not; cross-checked against the `torch.float8_e4m3fnuz` / `torch.float8_e5m2fnuz` upcast reference (`0x80 → 0x7e00` fp16 / `0x7fc0` bf16 / NaN fp32). An end-to-end test (`test_typeconvert_upcast_fnuz_nan`) is added so CI exercises it on the AMD runners.

I searched the open and closed tracker (fnuz / e4m3fnuz / e5m2fnuz / fp8 NaN / fp8 signed-zero / "NaN becomes zero" and related) and found no existing report of this defect. The same buggy logic is present in the released tags v3.6.0, v3.7.0, and v3.7.1 (then in `ElementwiseOpToLLVM.cpp`; the functions were moved verbatim into `ConvertFpCastOpToLLVM.cpp` by the NFC refactor in #10441) and on current `main`.

---

## Environment details

```
Triton: source lines cited on main @ 285de7a16f2d4585c6fdecad4167c6e7ee5cb869
        (also present in v3.6.0/v3.7.0/v3.7.1); the gfx90a LLVM-IR dump above was
        produced with the v3.7.1 build (the AMD SW decode body is byte-identical).
GPU:    AMD non-CDNA3 (gfx908 / gfx90a / gfx950 / RDNA) takes the software FNUZ->fp16 path.
        CDNA3 (gfx942) uses the HW cvt path and is not affected.
```

## Trigger

The source byte is 0x80, the sole NaN encoding of an FNUZ FP8 format.

## Relevant entities

FP8 source byte; software decoder; zero/denormal path; sign propagation; FP16/BF16 output bits

## Implemented relation

The logical FNUZ NaN value remains NaN when its physical encoding changes to FP16 or BF16.

## Pre-state

0x80 denotes NaN because FNUZ has no negative zero.

## Binding transition

Decoder arithmetic handles the magnitude as zero and then propagates the sign bit, or an earlier NaN select is overwritten.

## Post-state

The destination bits are 0x8000, which denotes finite negative zero.

## Validity obligation

The format-specific 0x80 case overrides generic zero and sign decoding in every software conversion path.

## Invalid decision

Three software decoders either omitted the 0x80 case or applied it before later operations that clobbered the NaN result.

## Physical effect

A special logical value silently becomes a finite value during representation conversion.

## Observed consequence

NaN propagation and masking observe -0.0 on non-CDNA3 AMD targets.

## Repair site

third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp

## Repair

PR 10747 places an explicit 0x80-to-NaN select after zero/sign decoding for e4m3fnuz and adds the missing e5m2fnuz and BF16 cases with conversion tests.

## Why the repair works

The final format-specific select maps the sole FNUZ NaN encoding to a destination NaN regardless of the generic bit path.

## Causal basis

The issue enumerates the three decoders, and the merged patch adds explicit tests and guards for their exact 0x80 behavior.

## Analysis implication

Numeric-conversion checks need source format semantics, raw bits, target format, hardware/software path, and the ordering of exceptional-value guards.

## Fix patches

- [`fixes/triton-10745-pr10747.patch`](fixes/triton-10745-pr10747.patch) — https://github.com/triton-lang/triton/pull/10747
