# FlexAttention FLASH ignores sparse BlockMask metadata when mask_mod is trivial

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/188322](https://github.com/pytorch/pytorch/issues/188322)
- Opened: 2026-06-27T05:13:27Z
- Closed: 2026-06-30T04:27:16Z
- Labels: `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `module: flex attention`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

#### Human Note

I had my agent generate this small issue for a real bug I found and will fix :)


### Agent report
`torch.nn.attention.flex_attention` with `kernel_options={"BACKEND": "FLASH"}` appears to mishandle a sparse `BlockMask` when sparsity is represented only by `kv_*` / `full_kv_*` metadata and the `mask_mod` is trivial/noop.

The same `BlockMask`:

- has the expected `to_dense()` representation,
- produces correct output with `BACKEND="TRITON"`, and
- produces correct output with `BACKEND="FLASH"` if I attach an equivalent nontrivial tile-level `mask_mod`.

This suggests the Flex FLASH lowering/CuTe backend may be deciding whether to honor sparse block metadata from whether the `mask_mod` graph is trivial, rather than from the presence of sparse `BlockMask` metadata.

### Repro

```python
from functools import partial
import math

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention


def run(name, fn, q, k, v, block_mask, ref):
    try:
        out = fn(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()
        diff = (out.float() - ref.float()).abs()
        ok = torch.allclose(out, ref, atol=5e-2, rtol=5e-2)
        print(f"{name}: ok={ok} max={diff.max().item():.6f} mean={diff.mean().item():.6f}")
    except Exception as exc:
        print(f"{name}: ERROR {type(exc).__name__}: {str(exc).splitlines()[0]}")


def main():
    print("torch", torch.__version__)
    print("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability())

    torch.manual_seed(0)
    batch, heads, q_blocks, kv_blocks, block_size, head_dim, top_k = 1, 2, 4, 4, 256, 128, 2
    q = torch.randn(batch, heads, q_blocks * block_size, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, heads, kv_blocks * block_size, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, heads, kv_blocks * block_size, head_dim, device="cuda", dtype=torch.bfloat16)

    selected = torch.tensor(
        [[[[0, 2], [1, 3], [0, 3], [2, 3]], [[0, 2], [1, 3], [0, 3], [2, 3]]]],
        device="cuda",
        dtype=torch.int32,
    )
    num_blocks = torch.full(selected.shape[:-1], top_k, device="cuda", dtype=torch.int32)
    zero_num_blocks = torch.zeros_like(num_blocks)
    zero_indices = torch.zeros(*selected.shape[:-1], kv_blocks, device="cuda", dtype=torch.int32)
    full_indices = torch.zeros_like(zero_indices)
    full_indices[..., :top_k] = selected

    block_map = torch.zeros(batch, heads, q_blocks, kv_blocks, device="cuda", dtype=torch.bool)
    block_map.scatter_(-1, selected.long(), True)
    dense_mask = block_map.repeat_interleave(block_size, -2).repeat_interleave(block_size, -1)
    ref = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(head_dim)).masked_fill(~dense_mask, -float("inf")), dim=-1) @ v

    full_metadata_only = BlockMask.from_kv_blocks(
        kv_num_blocks=zero_num_blocks,
        kv_indices=zero_indices,
        full_kv_num_blocks=num_blocks,
        full_kv_indices=full_indices,
        BLOCK_SIZE=block_size,
        seq_lengths=(q_blocks * block_size, kv_blocks * block_size),
    )

    partial_metadata_only = BlockMask.from_kv_blocks(
        kv_num_blocks=num_blocks,
        kv_indices=full_indices,
        BLOCK_SIZE=block_size,
        seq_lengths=(q_blocks * block_size, kv_blocks * block_size),
    )

    def mask_mod(b, h, q_idx, kv_idx):
        return block_map[b, h, q_idx // block_size, kv_idx // block_size]

    full_metadata_with_mask_mod = BlockMask.from_kv_blocks(
        kv_num_blocks=zero_num_blocks,
        kv_indices=zero_indices,
        full_kv_num_blocks=num_blocks,
        full_kv_indices=full_indices,
        BLOCK_SIZE=block_size,
        seq_lengths=(q_blocks * block_size, kv_blocks * block_size),
        mask_mod=mask_mod,
    )

    triton = torch.compile(partial(flex_attention, kernel_options={"BACKEND": "TRITON"}), dynamic=False)
    flash = torch.compile(partial(flex_attention, kernel_options={"BACKEND": "FLASH"}), dynamic=False)

    print("dense block mask")
    print(full_metadata_only.to_dense().cpu())
    run("TRITON full_kv metadata only", triton, q, k, v, full_metadata_only, ref)
    run("FLASH full_kv metadata only", flash, q, k, v, full_metadata_only, ref)
    run("TRITON partial kv metadata only", triton, q, k, v, partial_metadata_only, ref)
    run("FLASH partial kv metadata only", flash, q, k, v, partial_metadata_only, ref)
    run("TRITON full_kv metadata + mask_mod", triton, q, k, v, full_metadata_with_mask_mod, ref)
    run("FLASH full_kv metadata + mask_mod", flash, q, k, v, full_metadata_with_mask_mod, ref)


if __name__ == "__main__":
    main()
```

### Output

```text
torch 2.14.0.dev20260625+cu132
device NVIDIA GB200 (10, 0)
dense block mask
tensor([[[[1, 0, 1, 0],
          [0, 1, 0, 1],
          [1, 0, 0, 1],
          [0, 0, 1, 1]],

         [[1, 0, 1, 0],
          [0, 1, 0, 1],
          [1, 0, 0, 1],
          [0, 0, 1, 1]]]], dtype=torch.int32)
TRITON full_kv metadata only: ok=True max=0.005859 mean=0.000308
FLASH full_kv metadata only: ok=False max=0.258789 mean=0.040552
TRITON partial kv metadata only: ok=True max=0.005859 mean=0.000308
FLASH partial kv metadata only: ok=False max=0.258789 mean=0.040552
TRITON full_kv metadata + mask_mod: ok=True max=0.005859 mean=0.000308
FLASH full_kv metadata + mask_mod: ok=True max=0.005859 mean=0.000309
```

### Expected behavior

`BACKEND="FLASH"` should honor sparse `BlockMask` metadata (`kv_*` and/or `full_kv_*`) even when the `mask_mod` graph is trivial/noop. It should match the dense masked reference and the `BACKEND="TRITON"` result.

### Additional context

In `torch/_inductor/kernel/flex/flex_flash_attention.py`, the FLASH lowering computes:

```python
mask_graph_is_trivial = is_trivial_mask_graph(mask_graph.graph_module)
needs_block_mask = not mask_graph_is_trivial
has_full_blocks = full_kv_num_blocks is not None
```

and then passes `NEEDS_BLOCK_MASK=needs_block_mask` to the CuTeDSL template. In the failing case, sparse full-block metadata is present (`has_full_blocks=True`) but `needs_block_mask=False` because the mask graph is trivial. Attaching an equivalent nontrivial `mask_mod` makes `needs_block_mask=True`, and the result becomes correct.

So the failure may be that Flex FLASH uses the mask graph to decide whether block-sparse metadata is needed, rather than treating sparse `BlockMask` metadata as sufficient to require block-mask handling.

### Versions

- PyTorch: `2.14.0.dev20260625+cu132`
- CUDA runtime/toolkit on the test machine: CUDA 13.x
- GPU: NVIDIA GB200, capability `(10, 0)`
- dtype: `torch.bfloat16`


cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo @Chillee @yanboliang @BoyuanFeng @liangel-02 @howardzhang-cv

## Trigger

A BlockMask expresses selected KV blocks only through kv_* or full_kv_* metadata while mask_mod is the no-op predicate.

## Relevant entities

BlockMask metadata; mask_mod graph; NEEDS_BLOCK_MASK flag; HAS_MASK_MOD flag; BlockSparseTensorsTorch; FA4 attention tiles

## Implemented relation

The FLASH kernel visits the KV blocks named by a real BlockMask independently of whether an elementwise mask_mod is present.

## Pre-state

The BlockMask's kv/full_kv counts and indices describe a sparse set of KV tiles, and the mask_mod graph imposes no additional elementwise filtering.

## Binding transition

The old lowering derives NEEDS_BLOCK_MASK solely from whether mask_mod is nontrivial, so a trivial graph clears the flag even when sparse block metadata is populated.

## Post-state

The template omits BlockSparseTensorsTorch and FA4 computes dense attention over KV blocks that the logical BlockMask excludes.

## Validity obligation

Block-sparse traversal is selected from the presence of a real BlockMask, while mask_mod generation is controlled separately by the predicate graph.

## Invalid decision

The lowering conflated two independent facts: whether tile metadata restricts traversal and whether a per-element predicate must execute.

## Physical effect

FLASH loads and accumulates K/V tiles outside the selected sparse block set.

## Observed consequence

Metadata-only FLASH results differ from the dense masked reference by up to 0.258789, while TRITON and FLASH with an equivalent nontrivial mask_mod match the reference.

## Repair site

torch/_inductor/kernel/flex/flex_flash_attention.py and torch/_inductor/kernel/flex/templates/flash_attention.py.jinja

## Repair

Commit 164a4b3 skips block-sparse tensors only for FlexAttention's synthetic dense no-op BlockMask, separates HAS_MASK_MOD from NEEDS_BLOCK_MASK, supplies zero full-block tensors when only partial metadata exists, and adds metadata-only FLASH regressions.

## Why the repair works

Every real BlockMask now reaches FA4 as block-sparse traversal metadata even when no elementwise predicate is needed, so visited tiles match the logical sparse relation represented by kv_* and full_kv_*.

## Causal basis

The issue identifies NEEDS_BLOCK_MASK as the failing decision and closes on commit 164a4b3. The landed commit states that it fixes 188322, documents the dense-attention consequence, separates block metadata from mask_mod, and adds the report's metadata-only FLASH comparison as a regression.

## Analysis implication

Attention-lowering analysis needs block counts and indices, block sizes, synthetic-no-op identity, predicate-graph triviality, template flags, constructed backend metadata, and visited tile coordinates together.

## Fix patches

- [`fixes/pytorch-188322-commit-164a4b3.patch`](fixes/pytorch-188322-commit-164a4b3.patch) — https://github.com/pytorch/pytorch/commit/164a4b3bb104e32c7f6b3455f49e18dfa090a7b4
