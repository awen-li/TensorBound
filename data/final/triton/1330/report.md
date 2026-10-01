# Misaligned address in triton kernel

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/1330](https://github.com/triton-lang/triton/issues/1330)
- Opened: 2023-03-13T18:18:48Z
- Closed: 2023-03-15T02:26:20Z
- Labels: `bug`

## Reported behavior

Runnable repro, `divisible_by_16` annotations are correct, but incorrect vectorization happens because `1185*<ind>` produces misaligned write addess:
```

from ctypes import c_void_p, c_long
import torch
import math
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_ngimel/be/cbex5j2tvoimdqzshqql46pnbswttydbzm74cckgd76af5csakyg.py
triton_fused_1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (1185*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')




async_compile.wait(globals())
del async_compile

def call(args):
    arg1_1, arg5_1, arg6_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf6 = empty_strided((32, 1185, 1), (1185, 1, 1), device='cuda', dtype=torch.float32)
        stream0 = get_cuda_stream(0)
        buf1 = as_strided(buf6, (32, 32, 1), (1185, 1, 1), 64)  # alias
        print("ARGS", arg1_1.data_ptr() % 16, buf1.data_ptr() % 16)
        triton_fused_1.run(arg1_1, buf1, 1024, grid=grid(1024), stream=stream0)
        del arg1_1
        del arg6_1
        del buf1
        return (arg5_1, )


def benchmark_compiled_module():
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg1_1 = rand_strided((32, 32, 1), (32, 1, 32), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, 1185), (1185, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([arg1_1, arg5_1, arg6_1]))


if __name__ == "__main__":
    import argparse
    from torch._inductor.utils import benchmark_all_kernels

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-kernels", "-k", action="store_true", help="Whether to benchmark each individual kernels")
    parser.add_argument("--benchmark-all-configs", "-c", action="store_true", help="Whether to benchmark each individual config for a kernel")
    args = parser.parse_args()

    if args.benchmark_kernels:
        benchmark_all_kernels('None', args.benchmark_all_configs)
    else:
        benchmark_compiled_module()
```
See https://github.com/pytorch/pytorch/issues/96628

## Trigger

A kernel derives x1 with integer division and forms a store address x0 + 1185*x1 into a view whose row stride is not 16-byte aligned.

## Relevant entities

index tensor; integer division; axis divisibility; row stride; pointer expression; vectorized store

## Implemented relation

The divisibility attached to a computed index is a sound lower bound on the alignment of every address represented by that index.

## Pre-state

The input index has known divisibility, but integer division changes the value pattern and the final odd row stride can produce misaligned addresses.

## Binding transition

The old DivOp visitor divides the left operand's divisibility by a constant divisor and propagates the quotient as alignment information.

## Post-state

The address expression is labeled more divisible than its concrete values guarantee.

## Validity obligation

Vector width is selected only from divisibility facts preserved by the complete arithmetic expression.

## Invalid decision

Division analysis inferred nontrivial divisibility for constant divisors other than one.

## Physical effect

Code generation emits a vector store at an address not aligned for that vector width.

## Observed consequence

The generated CUDA kernel fails with a misaligned-address error.

## Repair site

lib/Analysis/AxisInfo.cpp::DivOpAxisInfoVisitor

## Repair

PR 1341 makes division conservatively return divisibility one except for zero and division by one, and updates alignment-analysis tests.

## Why the repair works

The questionable division no longer supplies a false alignment proof, so vectorization cannot rely on the overstated address divisibility.

## Causal basis

The report attributes the failure to incorrect vectorization of the 1185-stride address, and the linked merged change removes the division rule that supplied that alignment.

## Analysis implication

Memory-vectorization analysis needs arithmetic transfer functions, concrete strides and offsets, pointer-base alignment, and the selected access width.

## Fix patches

- [`fixes/triton-1330-pr1341.patch`](fixes/triton-1330-pr1341.patch) — https://github.com/triton-lang/triton/pull/1341
