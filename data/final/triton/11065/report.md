# inline_asm_elementwise emits does not correctly handle replicated elements

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/11065](https://github.com/triton-lang/triton/issues/11065)
- Opened: 2026-07-27T23:24:13Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Describe the bug

tl.inline_asm_elementwise runs once per replicated element, rather than once per logical element. Since Triton does not expose replication (intentionally), this makes it impossible to properly use the API for operations that have side effects. Here is an example:

```python3
import torch
import triton
import triton.language as tl

N = 32  # fewer elements than the 128 threads (num_warps=4) -> layout replicates 4x

@triton.jit
def k(addr, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.inline_asm_elementwise(
        "red.global.add.u32 [$1], 1; mov.u32 $0, 0;",  # +1 to cell, dummy result
        "=r,l", [addr + offs.to(tl.int64) * 4],
        dtype=tl.int32, is_pure=False, pack=1,
    )

cells = torch.zeros(N, dtype=torch.int32, device="cuda")
k[(1,)](cells, N=N, num_warps=4)

assert (cells == 1).all()
```

### Environment details

Triton: main
GPU: GH200

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises triton inline_asm_elementwise lowering for impure assembly over layouts with replicated physical elements.

## Observed implementation decision

The lowering invokes side-effecting assembly once per physical replica instead of once per logical element.

## Physical effect

Replicated lanes perform duplicate writes to the same logical address.

## Observed consequence

Each output cell is incremented multiple times.

## Reported proposal or workaround

PR 11089 proposes predicating replicated side effects.

## Repair trace

PR 11089 was not merged, and the open discussion has no alternative landed semantics or regression.

## Archived evidence

- Raw report: `data/raw/github/triton/11065/report.json`
- SHA-256: `f839a194f408a318c48746d8cb19b0a20bfc5ba0fb14d0ea9a8da44932b6e1c8`
