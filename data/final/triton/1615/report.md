# Incorrect behavior from mutation

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/1615](https://github.com/triton-lang/triton/issues/1615)
- Opened: 2023-05-04T03:00:10Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

This one seems pretty serious unless I am missing something.

Repro:
```py
import torch
from torch import empty_strided
from torch._dynamo.testing import rand_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

triton_red_fused_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp7 = tmp0 + 1

    # If you comment out this line it works
    tl.store(in_out_ptr0 + (tl.broadcast_to(x0, [XBLOCK, 1])), tmp7, None)

    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tl.device_assert((((0 <= tmp7) & (tmp7 < 2050))) | ~rmask, "index out of bounds: 0 <= tmp7 < 2050")
        tmp8 = tl.load(in_ptr0 + (r1 + (768*tmp7)), rmask, eviction_policy='evict_last', other=0)
        _tmp9 = tl.where(rmask, _tmp9 + tmp8, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp9, None)
''')

async_compile.wait(globals())
del async_compile


def call():
    stream0 = get_cuda_stream(0)
    arg0_1 = rand_strided((2050, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    buf7 = torch.arange(1, 2049, dtype=torch.int64, device="cuda").unsqueeze(0)
    buf8 = empty_strided((1, 2048, 1), (2048, 1, 2048), device='cuda', dtype=torch.float32)
    triton_red_fused_2.run(buf7, arg0_1, buf8, 2048, 768, grid=grid(2048), stream=stream0)


if __name__ == "__main__":
    for _ in range(32):
        call()
        torch.cuda.synchronize()
```

Output
```py
Traceback (most recent call last):
  File "/home/jansel/pytorch/output_code.py", line 68, in <module>
    call()
  File "/home/jansel/pytorch/output_code.py", line 63, in call
    triton_red_fused_2.run(buf7, arg0_1, buf8, 2048, 768, grid=grid(2048), stream=stream0)
  File "/home/jansel/pytorch/torch/_inductor/triton_heuristics.py", line 325, in run
    self.autotune_to_one_config(*args, grid=grid)
  File "/home/jansel/pytorch/torch/_inductor/triton_heuristics.py", line 250, in autotune_to_one_config
    timings = self.benchmark_all_configs(*args, **kwargs)
  File "/home/jansel/pytorch/torch/_dynamo/utils.py", line 177, in time_wrapper
    r = func(*args, **kwargs)
  File "/home/jansel/pytorch/torch/_inductor/triton_heuristics.py", line 226, in benchmark_all_configs
    timings = {
  File "/home/jansel/pytorch/torch/_inductor/triton_heuristics.py", line 227, in <dictcomp>
    launcher: self.bench(launcher, *cloned_args, **kwargs)
  File "/home/jansel/pytorch/torch/_inductor/triton_heuristics.py", line 205, in bench
    return do_bench(kernel_call, rep=40, fast_flush=True)
  File "/home/jansel/pytorch/torch/_inductor/utils.py", line 63, in do_bench
    return triton_do_bench(*args, **kwargs)[0]
  File "/home/jansel/conda/envs/pytorch/lib/python3.10/site-packages/triton/testing.py", line 50, in do_bench
    fn()
  File "/home/jansel/pytorch/torch/_inductor/triton_heuristics.py", line 199, in kernel_call
    launcher(
  File "<string>", line 6, in launcher
RuntimeError: Triton Error [CUDA]: device-side assert triggered
```

However, with this change:
```patch
diff --git a/output_code.py b/output_code.py
index c442f64be0f..af433aa7e2c 100644
--- a/output_code.py
+++ b/output_code.py
@@ -37,7 +37,7 @@ def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexp
     tmp7 = tmp0 + 1

     # If you comment out this line it works
-    tl.store(in_out_ptr0 + (tl.broadcast_to(x0, [XBLOCK, 1])), tmp7, None)
+    # tl.store(in_out_ptr0 + (tl.broadcast_to(x0, [XBLOCK, 1])), tmp7, None)

     _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
```
It runs fine.

It seems almost like the mutation of `in_out_ptr0` is corrupting the value of `tmp7`.

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

RBLOCK spans multiple warps while a loaded value is broadcast across the reduction dimension and the source address is overwritten.

## Observed implementation decision

The generated program assumes a cross-warp read-before-write order without installing a fence.

## Physical effect

Some threads can load after another thread stores the incremented value.

## Observed consequence

The device assertion intermittently observes a nonzero pre-mutation value.

## Reported proposal or workaround

tl.debug_barrier is suggested manually; Triton-CPU PR 89 only lowers that operation to a CPU no-op.

## Repair trace

The discussion states that a proper fix requires alias analysis or an explicit memory model, and the linked CPU patch does not repair CUDA ordering or add a report-specific fixed regression.

## Archived evidence

- Raw report: `data/raw/github/triton/1615/report.json`
- SHA-256: `ce42ede4401ee8ff8c7476f724ded81f26c9297eed34aa0107b71f7fe0adb84b`
