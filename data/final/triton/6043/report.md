# Triton incorrectly interprets uint8 indices as int8 in pointer arithmetic operations

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/6043](https://github.com/triton-lang/triton/issues/6043)
- Opened: 2025-02-27T14:39:26Z
- Closed: 2025-03-13T16:45:52Z
- Labels: `bug`

## Reported behavior

### Bug Description
When using `uint8` indices for pointer arithmetic, the indices are incorrectly interpreted as `int8` (negative above 127).

### Minimal Reproducible Example
Here `uint8` indices in the range [0, 255] are used in loading.

```python
import triton
import triton.language as tl
import torch


@triton.jit
def pointer_uint8_bug_kernel(
    out_ptr,
    out_wrong_ptr,
    src_ptr,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    start_idx = tl.program_id(0) * BLOCK_SIZE

    src_indices = tl.cast(tl.arange(0, 256), tl.uint8)
    tl.static_assert(src_indices.dtype is tl.uint8)

    # Incorrect: src_indices interpreted as int8 instead of uint8
    out_wrong = tl.load(src_ptr + src_indices, mask=src_indices < BLOCK_SIZE)

    out_correct = tl.load(
        src_ptr + tl.cast(src_indices, tl.uint32), mask=src_indices < BLOCK_SIZE
    )

    # Store results
    out_indices = start_idx + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + out_indices, out_correct, mask=out_indices < numel)
    tl.store(out_wrong_ptr + out_indices, out_wrong, mask=out_indices < numel)


def test_pointer_uint8_bug():
    numel = 192
    BLOCK_SIZE = 256
    BEFORE_SRC_VAL = 11

    data = torch.arange(-128, numel, dtype=torch.float32, device="cuda")
    data[:128] = BEFORE_SRC_VAL
    src = data[128:]  # values 0 to numel

    out = torch.empty(numel, dtype=torch.float32, device="cuda")
    out_wrong = torch.empty_like(out)

    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    pointer_uint8_bug_kernel[grid](
        out_ptr=out,
        out_wrong_ptr=out_wrong,
        src_ptr=src,
        numel=numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    assert torch.all(out == src)
    assert torch.all(out_wrong[:128] == src[:128])  # correct up to index 128
    assert torch.all(out_wrong[128:] == BEFORE_SRC_VAL)  # incorrect after index 128, memory before of src
    print(out_wrong)


test_pointer_uint8_bug()
```

### Expected Behavior
- Indices 0-255 should access memory locations at offsets 0-255 from the source pointer.
- The loaded arrays `out_wrong` and `out` should be equal.

### Actual Behavior
- Indices 0-127 work correctly.
- Indices 128-255 are interpreted as negative numbers (-128 to -1).
- This causes reading from wrong memory locations before the source pointer.

### Workaround
A workaround is to convert the `uint8` indices to some integer type with more bits:

```python
out_correct = tl.load(src_ptr + tl.cast(src_indices, tl.uint32), mask=src_indices < BLOCK_SIZE)
```

### Comments
I encountered this bug when implementing dequantization.

### Environment details

Triton 3.2.0
CUDA 12.7
NVIDIA RTX A4500
PyTorch 2.6.0+cu124

## Trigger

A uint8 tensor containing offsets 128 through 255 is added to a source pointer before tl.load.

## Relevant entities

uint8 logical index; signless TTIR integer; tt.addptr offset; 64-bit pointer arithmetic; source buffer; loaded element

## Implemented relation

An unsigned logical index denotes the same nonnegative element offset when widened for pointer arithmetic.

## Pre-state

The frontend type records uint8, but its TTIR value uses the signless i8 representation.

## Binding transition

The old pointer-add path passes the narrow signless value directly to tt.addptr, whose widening interprets the high bit as a sign bit.

## Post-state

Offsets 128 through 255 become -128 through -1 in address computation.

## Validity obligation

Pointer lowering widens an unsigned offset with zero extension before any operation that otherwise assigns signed meaning to the signless bits.

## Invalid decision

semantic.add discarded the source unsignedness at the tt.addptr boundary.

## Physical effect

Loads address elements before the source pointer rather than elements 128 through 255 after it.

## Observed consequence

The first 128 results are correct, while later loads return the sentinel values stored before the source slice.

## Repair site

python/triton/language/semantic.py::add pointer-plus-integer lowering

## Repair

Commit b5af392 zero-extends unsigned offsets narrower than 64 bits to i64 before creating tt.addptr and adds a frontend IR assertion for the extension.

## Why the repair works

Zero extension maps every uint8 bit pattern to its 0-through-255 value, so addptr receives the intended nonnegative element offset.

## Causal basis

The report's uint8 boundary behavior matches the pre-existing addptr bug exactly; the maintainer states top of tree is fixed, and the landed patch zero-extends the same unsigned narrow offsets before addptr.

## Analysis implication

Pointer analysis needs source-language signedness, integer width, TTIR signless representation, widening operation, pointer element scale, and the resulting byte range.

## Fix patches

- [`fixes/triton-6043-commit-b5af392.patch`](fixes/triton-6043-commit-b5af392.patch) — https://github.com/triton-lang/triton/commit/b5af3924c0063ae48ee08e63acc27fdb52f9c1ec
