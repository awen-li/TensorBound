# [inductor] `x.to(torch.bfloat16).float()` precision cast silently elided during fusion

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/179561](https://github.com/pytorch/pytorch/issues/179561)
- Opened: 2026-04-07T10:36:26Z
- Closed: 2026-04-17T03:41:27Z
- Labels: `high priority`, `module: bfloat16`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

## [inductor] `x.to(torch.bfloat16).float()` precision cast silently elided during fusion

### Summary

When a computation graph contains an explicit `fp32 → bfloat16 → fp32` round-trip cast (e.g. `x.to(torch.bfloat16).float()`), Inductor treats the round-trip as a no-op and eliminates both casts during graph lowering. This produces silently incorrect results because the `bfloat16` cast intentionally truncates the mantissa from 23 bits to 7 bits.

The same issue also affects `fp16` round-trips (`x.half().float()`).

### Reproducer

```python
import torch
import torch._dynamo

torch.manual_seed(0)
x = torch.randn(4, 32, 32, device='cuda')
w = torch.randn(32, 32, device='cuda')

def fn(x, w):
    x = torch.matmul(x, w)
    x = x.to(torch.bfloat16).float()  # intentional precision truncation
    x = x * torch.sigmoid(x)
    return x.sum(dim=1)

# Eager (correct)
eager_out = fn(x, w)

# Inductor (incorrect — cast is elided)
torch._dynamo.reset()
compiled_fn = torch.compile(fn, backend='inductor')
inductor_out = compiled_fn(x.clone(), w.clone())

# aot_eager (correct — preserves cast)
torch._dynamo.reset()
aot_fn = torch.compile(fn, backend='aot_eager')
aot_out = aot_fn(x.clone(), w.clone())

print('eager vs inductor: ', (eager_out - inductor_out).abs().max().item())
# => 0.109 (WRONG — should be ~0)
print('eager vs aot_eager:', (eager_out - aot_out).abs().max().item())
# => 0.0 (correct)

# Proof that inductor output matches "no cast at all":
def fn_no_cast(x, w):
    x = torch.matmul(x, w)
    # bf16 cast removed
    x = x * torch.sigmoid(x)
    return x.sum(dim=1)

no_cast_out = fn_no_cast(x, w)
print('inductor vs fn_without_cast:', (inductor_out - no_cast_out).abs().max().item())
# => 1.5e-5 (inductor behaves as if cast never existed)
```

### Evidence from generated Triton code

Setting `TORCH_LOGS=output_code` confirms the cast is removed. The generated Triton kernel header is:

```
# Topologically Sorted Source Nodes: [sigmoid, x_2, sum_1]
# Original ATen: [aten.sigmoid, aten.mul, aten.sum]
```

The `aten._to_copy` nodes for `bfloat16` and `float32` are absent from the fused kernel. The Triton kernel directly loads the `fp32` matmul output and applies `sigmoid → mul → sum` without any intermediate dtype conversion.

### Impact

This affects any user code that uses explicit dtype casts for precision control, which is common in:
- **Mixed-precision training**: `x.to(torch.bfloat16).float()` is used to simulate low-precision forward passes
- **Quantization-aware training**: explicit cast round-trips model quantization noise
- **Numerical stability testing**: intentional precision reduction to test robustness

The diff scales with tensor values — larger matmul outputs (from larger hidden dimensions) produce larger errors:

| Shape | max_diff (bf16) | max_diff (fp16) |
|-------|----------------|----------------|
| (4, 32, 32) | 0.109 | 0.018 |
| (8, 69, 53) | 0.264 | 0.025 |
| (3, 257, 129) | 0.704 | — |

### Additional notes

- **100% reproducible** across all random seeds, all shapes, all config presets (default, aggressive_fusion, no_fusion, freezing)
- **Not affected by** `torch._inductor.config.emulate_precision_casts` — setting it to `True` does not fix the issue
- **`aot_eager` is correct** — the cast is preserved in the AOT Autograd graph, and only eliminated during Inductor lowering/codegen
- The `fp16` round-trip (`x.half().float()`) has the same problem but with smaller magnitude


### Versions


- PyTorch: 2.6.0+cu124
- Triton: 3.2.0
- CUDA: 12.4
- GPU: Tesla T4
- Python: 3.10.19
- OS: Linux 5.4.0-42-generic x86_64


cc @ezyang @gchanan @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Trigger

A fused graph narrows a float32 computation to bfloat16 or float16 and then widens it back to float32 before further arithmetic.

## Relevant entities

source dtype; first cast dtype; second cast dtype; pointless_convert pattern; precision/range; fused consumer

## Implemented relation

A cast pair is collapsed only when the first cast preserves every value representable by the source dtype.

## Pre-state

A matmul produces float32 values, and an explicit low-precision cast is intended to discard mantissa bits before later sigmoid, multiply, and reduction operations.

## Binding transition

The old pointless_convert rewrite accepts any pair drawn from float16, bfloat16, float32, and float64 and replaces both casts with a direct conversion to the final dtype.

## Post-state

The narrowing boundary disappears from the joint graph, so lowering and code generation receive only the float32 consumer path.

## Validity obligation

When precision-cast emulation is enabled, elimination requires the first cast to preserve both precision and representable range of its source dtype.

## Invalid decision

The rewrite classified a downcast-then-upcast pair by dtype membership rather than by whether the first conversion loses information.

## Physical effect

The fused kernel retains float32 mantissa bits that eager execution discards at the bfloat16 or float16 boundary.

## Observed consequence

The compiled bfloat16 example differs from eager by up to 0.109 and closely matches a version with the cast removed; generated code contains no conversion nodes.

## Repair site

torch/_inductor/fx_passes/joint_graph.py::pointless_convert

## Repair

Commit 24c273c adds a finfo-based lossless-widening test and blocks pointless_convert when the first cast loses precision or range under emulate_precision_casts. It adds bfloat16 and float16 round-trip regressions that check both output equality and emitted low-precision conversion code.

## Why the repair works

The optimizer now retains every user-visible narrowing transition while still collapsing cast pairs whose first step can represent all source values exactly.

## Causal basis

The issue closes on commit 24c273c, whose message states that it fixes 179561. The patch changes the exact pointless_convert rule identified by the report and adds the reported matmul-cast-consumer structure for both low-precision dtypes.

## Analysis implication

Graph-rewrite analysis needs the original source dtype, both conversion dtypes, precision and range relations, configuration mode, and downstream fused uses together.

## Fix patches

- [`fixes/pytorch-179561-commit-24c273c.patch`](fixes/pytorch-179561-commit-24c273c.patch) — https://github.com/pytorch/pytorch/commit/24c273c46fab7f60c049783b89daffe10d619910
