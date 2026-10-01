# [FlexAttention] FLASH backend silently wrong: wrapped gather index misclassified as lane-uniform in vectorized score_mod groups

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/188878](https://github.com/pytorch/pytorch/issues/188878)
- Opened: 2026-07-03T04:02:42Z
- Closed: 2026-07-03T04:05:48Z
- Labels: `module: cuda`, `triaged`, `module: vectorization`, `module: correctness (silent)`, `module: inductor`, `module: flex attention`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug
ima close and fold into  https://github.com/pytorch/pytorch/pull/188876

Agent log:

The flex_attention FLASH (FA4/CuTeDSL) backend **silently computes wrong results** (~7/8 of elements) for score_mods that combine a relative-position table gather with any other vectorizable captured load — e.g. the natural ALiBi-style pattern:

```python
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

device, dtype = "cuda", torch.bfloat16
B, H, S, D = 2, 8, 1024, 64
rel_bias = torch.randn(2 * S, device=device)
head_scale = torch.randn(H, device=device)
offset = S - 1

def score_mod(score, b, h, q_idx, kv_idx):
    return score + rel_bias[q_idx - kv_idx + offset] * head_scale[h]

bm = create_block_mask(lambda b, h, q, kv: q >= kv, None, None, S, S,
                       device=device, BLOCK_SIZE=(256, 128))
q, k, v = (torch.randn(B, H, S, D, device=device, dtype=dtype) for _ in range(3))
out = torch.compile(flex_attention)(
    q, k, v, score_mod=score_mod, block_mask=bm, kernel_options={"BACKEND": "FLASH"}
)
# ~92% of elements mismatch vs TRITON backend / eager reference
```

No error is raised; TRITON backend and eager both produce correct results.

### Root cause

The `head_scale[h]` lane-uniform load enables score_mod vectorization (`__vec_size__ = 8`). The `rel_bias` gather index flows through the negative-index wrap in `ModificationWrapperCuteDSL.indirect_indexing`, which returns the wrapped CSE symbol **without registering its semantic index expression** (the non-wrap path does register it). Lane analysis then sees an unknown symbol, which defaults to lane-uniform, and the emitted code loads only lane 0's index and broadcasts it across the group — visible directly in the generated kernel:

```python
tmp6 = cute.where(tmp5 < 0, tmp5 + cutlass.Int32(2048), tmp5)  # per-lane index!
tmp7 = ssa_to_fragment(tmp6, cutlass.Int32)
tmp9 = (in_ptr8[tmp7[0]])              # only lane 0's index is used
for load_idx in cutlass.range(...):
    tmp8[load_idx] = tmp9              # broadcast to all 8 lanes
```

Exactly 7/8 of scores get lane 0's bias → the observed ~92% mismatch. It is also "fast" (issues 1/8 of the loads), so benchmarks without a correctness check look great.

### Fix direction

Register the pre-wrap semantic expression for the wrapped symbol so lane analysis sees the true lane structure, and track wrap-tainted symbols: they keep their per-lane classification but must never classify as *contiguous* (the wrap remaps negative lanes individually, so pre-wrap contiguity does not survive). Per-lane gather loops must also index lane-uniform fragments at `[0]`.

Fix + regression test (fails 91.7% mismatched before, passes after; flex-flash + aux-vectorization suites green, 266 passed on B200) ready on a branch, sibling of the #188871 fixes.

Found while implementing the Sandwich score_mod for attention-gym (meta-pytorch/attention-gym#216); related dtype bugs in the same codegen area: #188871.

### Versions

torch 2.14.0a0+git06b761f (source build 2026-07-02), CUDA 13.3, B200 (SM100), flash-attn CuTeDSL.

cc @ptrblck @msaroufim @eqy @tinglvv @nWEIdia @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo @Chillee @yanboliang @BoyuanFeng @liangel-02 @howardzhang-cv @drisspg

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

A relative-position gather and a lane-uniform captured load enable vectorization while negative-index wrapping creates an unregistered index symbol.

## Observed implementation decision

Lane analysis defaults the unknown wrapped index to lane-uniform and emits one lane's gathered value for the vector group.

## Physical effect

The generated kernel loads the table at lane zero's address and broadcasts that value to eight distinct logical lanes.

## Observed consequence

The FLASH backend silently disagrees with eager and Triton results for most output elements.

## Reported proposal or workaround

PR 188879 proposes semantic-index registration and conservative gather classification; a downstream attention-gym change avoids the affected code path.

## Repair trace

PyTorch PR 188879 was closed without merge, and the merged downstream attention-gym change is not a repair to the reported PyTorch code generator. No landed PyTorch patch and fixed regression establish the repair relationship.

## Archived evidence

- Raw report: `data/raw/github/pytorch/188878/report.json`
- SHA-256: `d9ef6c30e92bf49c44d826cd3d314851be493027d3236ac58c3186130bf9f78a`
