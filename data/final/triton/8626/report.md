# Multicta matmul miscompiles

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/8626](https://github.com/triton-lang/triton/issues/8626)
- Opened: 2025-11-03T15:37:19Z
- Closed: 2025-11-07T19:01:24Z
- Labels: `bug`

## Reported behavior

### Describe the bug

Run `test_simple_matmul`  on a B200 with the following parameters
```python
@pytest.mark.parametrize("dtype_src_str", ["float16"])
@pytest.mark.parametrize("dtype_dst_str", ["float32"])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES", [(128, 64, 64, 1)])
@pytest.mark.parametrize("NUM_CTAS", [2])
@pytest.mark.parametrize("NUM_WARPS", [4])
@pytest.mark.parametrize("EPILOGUE_SUBTILE", [False])
@pytest.mark.parametrize("LAYOUT_16x256", [False])
def test_simple_matmul(dtype_src_str, dtype_dst_str, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, NUM_WARPS, NUM_CTAS, device,
                       EPILOGUE_SUBTILE, LAYOUT_16x256, monkeypatch):
```
and you get
```
code/triton/python/test/unit/language/test_matmul.py:162: in test_simple_matmul
    torch.testing.assert_close(ref_out, output, atol=atol, rtol=rtol)
E   AssertionError: Tensor-likes are not close!
E
E   Mismatched elements: 458663 / 524288 (87.5%)
E   Greatest absolute difference: 107.48089599609375 at index (355, 197) (up to 0.001 allowed)
E   Greatest relative difference: inf at index (0, 32) (up to 0.001 allowed)
```

The same test with `BLOCK_N=128` passes (it does not lower to 2cta)

## Trigger

A Blackwell matmul with BLOCK_M=128, BLOCK_N=64, four warps, and two CTAs lowers to tcgen05 two-CTA mode.

## Relevant entities

two cooperating CTAs; tcgen05 MMA; tensor-memory accumulator; row and column layout bases; distributed result

## Implemented relation

The tensor-memory encoding maps register, lane, warp, and block coordinates from both CTAs to one logical accumulator tile.

## Pre-state

Two CTAs jointly compute a 128-row tile, giving each CTA the architecture's M=64 tensor-memory view.

## Binding transition

Linear-layout conversion derives accumulator and load/store coordinates from TensorMemoryEncodingAttr.

## Post-state

The encoding lacks a distinct two-CTA mode and constructs incorrect row, column, warp, or block bases for the M=64-per-CTA layout.

## Validity obligation

Every tensor-memory coordinate used to store or load a two-CTA MMA accumulator maps to the architecture-defined owner and logical output element.

## Invalid decision

TensorMemoryEncodingAttr represented CTA splits without recording whether tcgen05 uses cta_group::2, so the M=64 layout followed the wrong conversion.

## Physical effect

Accumulator fragments are written to or read from tensor-memory coordinates belonging to different logical output elements.

## Observed consequence

The two-CTA matmul mismatches 458,663 of 524,288 output elements, while the non-two-CTA BLOCK_N=128 control passes.

## Repair site

TritonGPU tensor-memory encoding and linear-layout conversion, tcgen05 verification, and Blackwell lowering

## Repair

PR 8644 adds an explicit twoCTAs attribute, constructs the special M=64 two-CTA layout, propagates and verifies the mode, and extends two-CTA matmul/layout tests.

## Why the repair works

Coordinate conversion can now select the architecture-specific M=64 two-CTA mapping and reject operands whose split or tensor-memory mode is inconsistent.

## Causal basis

The report isolates two-CTA lowering, and the linked patch states that the M=64 two-CTA layout was wrong and changes the corresponding tensor-memory coordinate bases and mode representation.

## Analysis implication

A cooperative-MMA check needs CTA-group mode, CTA split, per-CTA block shape, tensor-memory bases, and load/store coordinate ownership.

## Fix patches

- [`fixes/triton-8626-pr8644.patch`](fixes/triton-8626-pr8644.patch) — https://github.com/triton-lang/triton/pull/8644
