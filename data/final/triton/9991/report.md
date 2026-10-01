# tl.store(i1, i32) implicitly casts int32 as int8 lead unexpected behavior

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/9991](https://github.com/triton-lang/triton/issues/9991)
- Opened: 2026-04-10T15:03:14Z
- Closed: 2026-09-18T23:05:19Z
- Labels: `bug`

## Reported behavior

### Describe the bug

+ Description: There is a discrepancy between the Triton documentation and the actual compiler behavior regarding the data type precision for triton.languange.sum.

According to the [official documentation](https://triton-lang.org/main/python-api/generated/triton.language.sum.html#triton.language.sum), this operation is supposed to handle/process data as int32. However, upon testing the runtime behavior, it appears the compiler is actually lowering the bool as `tl.int8` as shown in https://github.com/triton-lang/triton/blob/main/python/triton/language/semantic.py#L1029.

+ Expected Behavior: The summation of bool type in triton is equivalent as `torch.sum(x, dtype=torch.bool)`

+ Actual Behavior: The summation of boolean Tensor with 256 `True` return `False`

```python
import triton
import triton.language as tl
import time
import numpy as np

import torch
import numpy as np

@triton.jit
def triton_sum_2D_dim0_dim1(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    idx = mblk_idx[None, :] + lblk_idx[:, None] * M
    x = tl.load(in_ptr0 + idx)
    ret = tl.sum(x, 1)
    ret = tl.sum(ret, 0)
    tl.store(out_ptr0, ret)

shape = [128, 2]
x0 = torch.ones(shape).to(torch.bool).cuda()
ans_int8 = torch.sum(x0.to(torch.int8), (0, 1), dtype=torch.int8).to(torch.bool)
ans_bool = torch.sum(x0, (0, 1), dtype=torch.bool)
ans = torch.sum(x0, (0, 1))
output_shape = shape[2:]
output = torch.zeros(output_shape, dtype=torch.bool).cuda()

triton_sum_2D_dim0_dim1[(1,)](x0, output, *shape, debug=True)
print(f'torch sum int8, bool, not specified: {ans_int8}, {ans_bool}, {ans}')
print(f'triton : {output}')
# torch sum int8, bool, not specified: False, True, 256
# triton : False
```

### Environment details

+ NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9
+ Tesla V100-PCIE-16GB
+ triton Version: 3.4.0
+ torch Version: 2.10.0+cu129

## Trigger

A bool reduction produces int32 value 256 and tl.store writes it to a bool output.

## Relevant entities

int32 reduction; bool pointer; int8 storage; semantic cast; stored byte

## Implemented relation

A store to logical bool converts the source by value != 0 before representing the boolean in byte-addressable storage.

## Pre-state

The reduction value is nonzero and therefore logically true.

## Binding transition

store rewrites pointer<int1> to pointer<int8> before coercing the value.

## Post-state

The cast sees int8 and truncates 256 to low byte zero instead of performing a truth conversion.

## Validity obligation

Logical bool conversion precedes storage-width adaptation.

## Invalid decision

TritonSemantic.store erased the bool destination identity before casting its source value.

## Physical effect

The generated IR uses arith.trunci i32 to i8 and stores zero.

## Observed consequence

Summing 256 true values stores False.

## Repair site

python/triton/language/semantic.py::TritonSemantic.store

## Repair

PR 11853 casts the value to int1 before adapting the bool pointer/value to int8 storage and adds an integer-to-bool frontend regression.

## Why the repair works

Every nonzero integer becomes int1 true before any byte truncation can discard high bits.

## Causal basis

The issue's TTIR isolates truncation at store, and the merged patch names #9991 and changes precisely that cast order.

## Analysis implication

A conversion check needs logical destination dtype, storage dtype, source dtype, cast order, and conversion semantics.

## Fix patches

- [`fixes/triton-9991-pr11853.patch`](fixes/triton-9991-pr11853.patch) — https://github.com/triton-lang/triton/pull/11853
