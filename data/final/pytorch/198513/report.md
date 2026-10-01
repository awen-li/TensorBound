# flex_attention returns incorrect output under Inductor when seq_len < block size

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/198513](https://github.com/pytorch/pytorch/issues/198513)
- Opened: 2026-09-24T16:37:25Z
- Closed: 2026-09-26T03:55:33Z
- Labels: `triage review`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `module: flex attention`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug

`flex_attention` returns incorrect output under Inductor when `seq_len` is below the 128 mask block size and a pointwise op feeds q/k.

Tested the following masks. The values below show the difference from the corresponding implementation with SDPA.

| `mask_mod` @ seq_len 32, Inductor | vs same-mask SDPA |
|---|---|
| **no-op (`lambda b,h,q,kv: kv >= 0`)** | **3.518e-01** |
| causal | 8.979e-01 |
| anti-causal (`kv >= q`) | 8.080e-01 |
| sliding-window causal | 8.979e-01 |
| block-diagonal | 5.997e-01 |

Interestingly, even with the trivial mask, we still see a difference, so it's not a matter of the mask being ignored. On all of these tests, we found that `flex_attention` matches sdpa when using eager and `aot_eager`, and we only see this discrepancy when using Inductor.

This issue was authored with the assistance of an AI coding assistant; the analysis and reproductions were reviewed and verified by the submitter.

### Minimal reproduction

```python
"""flex_attention returns incorrect output under Inductor when seq_len < 128.

A plain causal BlockMask is compared against F.scaled_dot_product_attention
with is_causal=True. They must agree. Under Inductor they do not, when
seq_len is below the 128 mask block size AND a pointwise op feeds q/k --
with q/k passed straight from the projection, every length is exact.

The `pointwise` column is the control: same shapes, same mask, only the
producer of q/k differs.

Pass/fail is torch.testing.assert_close at its default tolerances (fp32:
rtol=1.3e-6, atol=1e-5); the printed column is the max absolute difference,
for information only. RNG equivalence is structural rather than configured --
`got` and `want` are computed from the same weights and the same input, and
the model contains no random ops.

    python flex_bug.py     # exits non-zero if any configuration mismatches
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

DEV = torch.device("cuda")
B, H, D = 2, 4, 32
DIM = H * D


class CausalAttention(nn.Module):
    def __init__(self, pointwise: bool):
        super().__init__()
        self.pointwise = pointwise
        self.qkv = nn.Linear(DIM, 3 * DIM, bias=False)
        self.register_buffer("scale", torch.rand(D) + 0.5)

    def qkv_of(self, x):
        b, s, _ = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, H, D).permute(2, 0, 3, 1, 4)
        if self.pointwise:
            q, k = q * self.scale, k * self.scale
        return q, k, v

    def forward(self, x, block_mask):
        q, k, v = self.qkv_of(x)
        return flex_attention(q, k, v, block_mask=block_mask, scale=D**-0.5)


def causal_block_mask(seq_len):
    def mask_mod(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=DEV)


def check(seq_len, pointwise, backend):
    torch.manual_seed(0)
    model = CausalAttention(pointwise).to(DEV).eval()
    gen = torch.Generator(device=DEV).manual_seed(0)
    x = torch.randn(B, seq_len, DIM, device=DEV, generator=gen)

    torch._dynamo.reset()
    run = model if backend == "eager" else torch.compile(model, backend=backend)
    with torch.no_grad():
        got = run(x, causal_block_mask(seq_len))
        q, k, v = model.qkv_of(x)
        want = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=D**-0.5)

    err = (got - want).abs().max().item()
    try:
        torch.testing.assert_close(got, want)
        mismatch = False
    except AssertionError:
        mismatch = True

    del model
    torch.cuda.empty_cache()
    return err, mismatch


print(f"torch {torch.__version__}   {torch.cuda.get_device_name()}")
print(f"\n{'seq_len':>8} {'pointwise q/k':>14} {'backend':>11} "
      f"{'vs is_causal SDPA':>18}")
failures = 0
for seq_len in (32, 64, 128, 256):
    for pointwise in (True, False):
        for backend in ("eager", "aot_eager", "inductor"):
            err, bad = check(seq_len, pointwise, backend)
            failures += bad
            print(f"{seq_len:>8} {str(pointwise):>14} {backend:>11} "
                  f"{err:>18.3e}   {'MISMATCH' if bad else 'ok'}")

print(f"\n{failures} mismatching configuration(s)")
sys.exit(1 if failures else 0)
```

### Ablation (H200, torch 2.11.0+cu128)

Ablating `backend` (eager / aot_eager / inductor) against the presence of the pointwise op on q/k, across `seq_len`. Only inductor + pointwise + `seq_len < 128` fails.

```
torch 2.11.0+cu128   NVIDIA H200

 seq_len  pointwise q/k     backend  vs is_causal SDPA
      32           True       eager          3.576e-07   ok
      32           True   aot_eager          3.576e-07   ok
      32           True    inductor          8.979e-01   MISMATCH
      32          False       eager          4.172e-07   ok
      32          False   aot_eager          4.172e-07   ok
      32          False    inductor          3.576e-07   ok
      64           True       eager          3.576e-07   ok
      64           True   aot_eager          3.576e-07   ok
      64           True    inductor          9.591e-01   MISMATCH
      64          False       eager          3.576e-07   ok
      64          False   aot_eager          3.576e-07   ok
      64          False    inductor          3.576e-07   ok
     128           True       eager          3.576e-07   ok
     128           True   aot_eager          3.576e-07   ok
     128           True    inductor          2.980e-07   ok
     128          False       eager          3.576e-07   ok
     128          False   aot_eager          3.576e-07   ok
     128          False    inductor          2.980e-07   ok
     256           True       eager          3.576e-07   ok
     256           True   aot_eager          3.576e-07   ok
     256           True    inductor          2.980e-07   ok
     256          False       eager          3.576e-07   ok
     256          False   aot_eager          3.576e-07   ok
     256          False    inductor          3.576e-07   ok

2 mismatching configuration(s)
```

### AI-generated analysis

The original experiments we were running were in the context of transformers using RoPE embeddings, but upon finding this bug, we explored a bit further to find out if there were specific conditions needed to cause them. The list below is an AI summary that describes the situations in which we saw the incorrect output.

```
- **Inductor codegen.** `eager` and `aot_eager` are exact on the identical model, so tracing, functionalization and AOTAutograd are not involved.
- **Requires a fused pointwise producer on q/k.** The `pointwise` column is an in-table control: same shapes, same mask, only the q/k producer differs. One broadcast multiply is enough. (We first hit this via RoPE; RoPE is incidental, and its `cat` is not required.)
- **Strictly below the block size, not "not a multiple of" it.** Leaks at 32/64/96/120/127; exact at 128/129/130/160/200/256/300/384/512/1000/2048. 160 and 200 carry a partial trailing block and are fine.
- **Not GPU-specific.** Identical boundary and drift on A100 (108 SMs), H100 (132) and H200, in fp32 and bf16 alike.
- **Not fixed by upgrading.** Bit-identical drift on torch 2.11.0+cu128, 2.13.0+cu129 and 2.14.0+cu130 — deterministic codegen behaviour rather than a race.
- **Specific to FlexAttention.** The same masking expressed as a dense boolean mask through `scaled_dot_product_attention` is exact at every length, eager and compiled.

One suggestive detail: an earlier probe of ours appeared clean because the compiled function *returned* q/k alongside the output, forcing them to be materialized. That is consistent with a prologue-fusion fault.
```

### tlparse / TORCH_TRACE

[torch_trace.tar.gz](https://github.com/user-attachments/files/32618434/torch_trace.tar.gz)

This is the full `TORCH_TRACE` directory from one run of the reproducer above (5.7 MB uncompressed), generated with:

```sh
TORCH_TRACE=./torch_trace python flex_bug.py
```

### Not a duplicate of

- **#159247** (flex decoding / paged attention, `Q_S=1`, "KV length not a multiple of block size") — non-multiples above 128 are exact here, so "not a multiple" is not the condition; this is full attention in fp32.
- **#163588** (bfloat16 numerical instability with masks, closed as iterative-softmax precision) — this reproduces in fp32


### Error logs

None. `torch.compile` reports success and returns a tensor of the expected dtype and shape whose values are wrong.

### Versions

```
Collecting environment information...
PyTorch version: 2.11.0+cu128
Is debug build: False
CUDA used to build PyTorch: 12.8
ROCm SDK used to build PyTorch: N/A
HIP used to build PyTorch: N/A

OS: Rocky Linux release 8.10 (Green Obsidian) (x86_64)
GCC version: (GCC) 8.5.0 20210514 (Red Hat 8.5.0-28)
Clang version: 21.1.8 (RESF 21.1.8-1.module+el8.10.0+40180+8e26bdb3)
CMake version: Could not collect
Libc version: glibc-2.28

Python version: 3.13.5 (main, Jul  1 2025, 18:37:36) [Clang 20.1.4 ] (64-bit runtime)
Python platform: Linux-4.18.0-553.155.1.el8_10.x86_64-x86_64-with-glibc2.28
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to:
GPU models and configuration: GPU 0: NVIDIA H200
Nvidia driver version: Could not collect
cuDNN version: Could not collect
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True
Caching allocator config: N/A

CPU:
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              64
On-line CPU(s) list: 0-63
Thread(s) per core:  1
Core(s) per socket:  32
Socket(s):           2
NUMA node(s):        2
Vendor ID:           AuthenticAMD
CPU family:          25
Model:               17
Model name:          AMD EPYC 9374F 32-Core Processor
Stepping:            1
CPU MHz:             4299.986
CPU max MHz:         4304.9312
CPU min MHz:         1500.0000
BogoMIPS:            7700.34
Virtualization:      AMD-V
L1d cache:           32K
L1i cache:           32K
L2 cache:            1024K
L3 cache:            32768K
NUMA node0 CPU(s):   0-31
NUMA node1 CPU(s):   32-63
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc cpb cat_l3 cdp_l3 invpcid_single hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local avx512_bf16 clzero irperf xsaveerptr wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d

Versions of relevant libraries:
[pip3] Could not collect
[conda] No relevant packages
```



cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo @Chillee @drisspg @yanboliang @BoyuanFeng @liangel-02 @howardzhang-cv

## Trigger

A sequence shorter than 128 uses the decoding kernel, and a pointwise scale produces q from a [B,S,H,D] view before FlexAttention lowering.

## Relevant entities

query pointwise buffer; FlexibleLayout; provisional query strides; GQA as_strided view; scheduler layout freeze; decode kernel offsets

## Implemented relation

Every stride baked into the decoding GQA view describes the final physical layout chosen for the realized query buffer.

## Pre-state

The pointwise query is realized but still has a flexible layout whose provisional stride order differs from the fill order inherited from its permuted input.

## Binding transition

The old decoding lowering reads query.get_stride() and embeds those provisional values in a fixed as_strided GQA view before deciding the query buffer's layout.

## Post-state

The scheduler later freezes the underlying query with a different stride order, while the decoding view and kernel retain the earlier stride constants.

## Validity obligation

Lowering freezes a flexible producer's physical layout before exporting its strides into any fixed view or kernel interface.

## Invalid decision

Stride metadata was consumed as a stable physical contract although its owning buffer could still change layout during scheduling.

## Physical effect

The decode kernel maps logical q head and sequence coordinates to wrong physical offsets in the realized pointwise buffer.

## Observed consequence

For sequence lengths 32 and 64 with a pointwise q/k producer, compiled FlexAttention differs from the matching SDPA result by about 0.90 to 0.96; direct q/k inputs and lengths at least 128 are correct.

## Repair site

torch/_inductor/kernel/flex/flex_decoding.py::create_flex_decoding_kernel

## Repair

Commit 17b697f freezes the realized query through as_storage_and_layout before reading its strides and constructing the GQA view. It adds a short-query pointwise-producer regression for MHA and GQA configurations.

## Why the repair works

The view and kernel now capture strides from the same final layout that the scheduler uses for query storage, so logical query coordinates and physical loads remain aligned.

## Causal basis

The issue closes on commit 17b697f, whose message states that it fixes 198513. The patch identifies the provisional-versus-final query stride mismatch, freezes the layout immediately before the stride read, and adds the report's short-query pointwise test.

## Analysis implication

Kernel-interface analysis needs layout mutability, realization state, stride-read time, view construction, scheduler freeze order, and generated address constants together.

## Fix patches

- [`fixes/pytorch-198513-commit-17b697f.patch`](fixes/pytorch-198513-commit-17b697f.patch) — https://github.com/pytorch/pytorch/commit/17b697f26c25932d8987352dbd9a6c2484c7e03a
