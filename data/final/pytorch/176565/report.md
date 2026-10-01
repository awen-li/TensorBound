# [Inductor] Mismatch between eager and compiled outputs when converting fp32 → bf16 → fp32

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/176565](https://github.com/pytorch/pytorch/issues/176565)
- Opened: 2026-03-05T05:35:35Z
- Closed: 2026-03-10T08:45:50Z
- Labels: `triage review`, `module: bfloat16`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

### Issue Description
When compiling a function that performs a round-trip type conversion (fp32 → bfloat16 → fp32) using `torch.compile()`, the output differs from the eager mode execution.

### Reproduction Code
```python
import torch

class CastModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        bf16_t = x.to(torch.bfloat16)
        fp32_casted_t = bf16_t.to(torch.float32)
        return fp32_casted_t

def func(compile):
    torch.manual_seed(1234)
    x = torch.randn(4, 4).cuda()
    cast_module = CastModule().cuda()
    if compile:
        fp32_casted_t = torch.compile(cast_module)(x)
    else:
        fp32_casted_t = cast_module(x)
    return fp32_casted_t

def compare():
    torch._inductor.config.force_disable_caches = True
    eager_out = func(compile=False)
    compiled_out = func(compile=True)
    assert torch.equal(eager_out, compiled_out), f"Mismatch between eager and compiled outputs: {eager_out} vs {compiled_out}"

if __name__ == "__main__":
    compare()
```
The assertion fails with a mismatch between eager and compiled outputs.

### Generated Triton Kernel
I tried to disable the registered replace pattern `pointless_convert` in https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/joint_graph.py#L761, and get the following generated triton kernel:
```python
@triton_heuristics.pointwise(
    size_hints={'x': 16},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '03ABE04230B1E328310427D5399BB7600AAE91F41BEED9AB62FE9C801110B6A5', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)  # It should be tmp0.to(tl.bfloat16) to make it align with eager run?
    tmp2 = tmp1.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp2, xmask)
```
The generated kernel completely skips the fp32 → bf16 → fp32 conversion chain. It directly loads fp32 values and stores them back as fp32, which is inconsistent with eager mode.

While the Inductor's optimization might produce numerically better results, we have a use case that requires strict bitwise equivalence with eager mode. Is there a way to disable this specific optimization or enforce that type casts are preserved in the generated code?


cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo @yanboliang

### Versions

torch 2.9.1

## Trigger

Compiled code converts float32 input to bfloat16 and immediately back to float32 while eager-compatible precision-cast emulation is requested.

## Relevant entities

float32 input; convert_element_type; bfloat16 logical value; compute-type representation; rounding boundary; float32 output

## Implemented relation

The logical bfloat16 intermediate is rounded to bfloat16 precision before a later float32 consumer observes it.

## Pre-state

Input values carry full float32 precision and the first conversion requests a lower-precision bfloat16 value.

## Binding transition

The old convert_element_type lowering calls to_dtype with compute types enabled even under precision-cast emulation, allowing the low-precision logical intermediate to remain represented as float32 inside a fused kernel.

## Post-state

The following cast to float32 sees an already-float32 compute value, so the round trip becomes a direct float32 load and store without bfloat16 rounding.

## Validity obligation

When eager-compatible cast emulation is enabled and either endpoint is float16 or bfloat16, lowering performs the low-precision conversion at the logical cast boundary before restoring the compute representation.

## Invalid decision

The lowering propagated the logical result dtype but omitted the physical rounding action required by that dtype's eager semantics.

## Physical effect

Bits that eager execution discards at the bfloat16 boundary survive the fused compiled expression.

## Observed consequence

The compiled round-trip output is not bitwise equal to eager execution; the generated Triton kernel converts the loaded value only to float32 and stores it unchanged.

## Repair site

torch/_inductor/lowering.py::to_dtype and _convert_element_type

## Repair

Commit 8aa6538 lets to_dtype disable compute-type substitution, selects that mode for float16/bfloat16 conversions under emulate_precision_casts, then restores a compute-type value after the real low-precision conversion. It adds the report's dynamic CUDA round trip as a regression.

## Why the repair works

The fused expression now contains an explicit low-precision conversion at the same program point as eager execution, preserving the required rounding even though later arithmetic uses the backend compute type.

## Causal basis

The linked pull request says it fixes 176565 and the issue closes on landed commit 8aa6538. The patch routes convert_element_type through non-compute-type casts under emulation and adds the exact float32-bfloat16-float32 equality check.

## Analysis implication

Precision analysis needs logical dtype, physical compute dtype, cast-emulation mode, fusion boundaries, explicit rounding operations, and consumer dtype together.

## Fix patches

- [`fixes/pytorch-176565-commit-8aa6538.patch`](fixes/pytorch-176565-commit-8aa6538.patch) — https://github.com/pytorch/pytorch/commit/8aa653869efa970fd81185c2a8dfb8f42f7235c0
