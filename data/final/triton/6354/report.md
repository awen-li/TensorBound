# Loading from TMA descriptor hangs

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/6354](https://github.com/triton-lang/triton/issues/6354)
- Opened: 2025-03-31T23:18:25Z
- Closed: 2025-06-05T16:02:21Z
- Labels: `bug`, `question`

## Reported behavior

### Describe the bug

I haven't looked into why this is happening yet but I've been able to reduce it. If you run the following code it will hang:

```python3
#!/usr/bin/env python3

import torch
import triton


autotune_config = [triton.Config({"BLOCK_SIZE": block_size}) for block_size in map(lambda n: 2**n, range(7, 10))]


@triton.autotune(configs=autotune_config, key=[])
@triton.jit
def test(
    matrix,
    n,
    BLOCK_SIZE: triton.language.constexpr,
):
    data = triton.language.make_tensor_descriptor(
        matrix, block_shape=(BLOCK_SIZE, 16), shape=(n, n), strides=(n, 1)
    ).load((0, 0))

    # Make sure the load happens and isn't optimized out
    triton.language.store(matrix + triton.language.zeros_like(data), data)


if __name__ == "__main__":
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))

    n = 4096

    matrix = torch.zeros(n, n, dtype=torch.int, device="cuda")

    test[(1,)](matrix, n)
```

I see this happen reliably when the autotuner hits BLOCK_SIZE=512, and in the debugger this happens on the very first kernel too. It's stuck at the very end of the kernel, here:

```
   0x0000000327796140 <+2368>:	SYNCS.PHASECHK.TRANS64.TRYWAIT P0[UR8+0x8000],RZ
=> 0x0000000327796150 <+2384>:	@!P0 BRA 0x327796140
```

### Environment details

Triton: built from source at https://github.com/triton-lang/triton/commit/a39389aac290cb5764ba9a47a93f8af8c7197916
GPU: GH200

## Trigger

Autotuning selects BLOCK_SIZE=512 for a descriptor load shaped [512,16] on GH200.

## Relevant entities

logical tensor block; TMA descriptor; hardware block dimension; TMA message; expected byte count; mbarrier

## Implemented relation

The set of TMA messages covers the complete logical block, and barrier completion expects exactly the bytes those messages request.

## Pre-state

The kernel requests a 512-by-16 block and records the full logical byte count for completion.

## Binding transition

The old lowering encodes the 512-element dimension in one TMA message although the hardware descriptor limit is 256.

## Post-state

Hardware truncates the message dimension while the mbarrier still waits for the full 512-element transfer.

## Validity obligation

Each hardware TMA block dimension stays within 256 elements, with larger logical blocks partitioned into enough messages to cover every element and byte.

## Invalid decision

Descriptor lowering mapped a logical dimension larger than 256 directly to one TMA block dimension.

## Physical effect

The issued transfer accounts for fewer bytes than the barrier's expected transaction count.

## Observed consequence

The kernel spins indefinitely in the TMA barrier wait and stalls the autotuner.

## Repair site

TMA block-shape construction and NVIDIA GPU load/store lowering

## Repair

PR 6776 derives a hardware-sized TMA block shape, creates a message-to-offset layout, and emits multiple messages for larger logical dimensions.

## Why the repair works

Every message respects the 256-element limit while their offsets jointly cover the logical tile, so completed transfer bytes match the barrier expectation.

## Causal basis

A maintainer identifies silent hardware truncation and the full-byte barrier wait; the merged repair is explicitly titled and described as splitting dimensions above 256.

## Analysis implication

Asynchronous-copy analysis needs logical tile shape, hardware descriptor limits, message count and offsets, packing, transferred bytes, and barrier expectation.

## Fix patches

- [`fixes/triton-6354-pr6776.patch`](fixes/triton-6354-pr6776.patch) — https://github.com/triton-lang/triton/pull/6776
