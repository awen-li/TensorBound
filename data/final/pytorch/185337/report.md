# [inductor] CPU codegen does not preserve explicit float32 to float16 to float32 cast semantics

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/185337](https://github.com/pytorch/pytorch/issues/185337)
- Opened: 2026-05-27T08:03:29Z
- Closed: 2026-06-02T23:47:05Z
- Labels: `module: cpu`, `triaged`, `module: half`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

`x.to(torch.float16).to(torch.float32)` should preserve the intermediate fp16 rounding. In eager mode, the first cast to fp16 rounds the values, and the second cast returns those rounded values back to float32.

However, with `torch.compile(..., backend="inductor")` on CPU, the compiled result appears to skip or eliminate the intermediate fp16 rounding. The compiled output is closer to the original float32 input than to the eager result.

This is a silent numerical mismatch. It looks related to the existing fp16 cast preservation / CPU precision-cast family, but this repro only uses an explicit cast pair and a simple reduction.

Expected behavior: compiled output should match eager output after the explicit `float32 -> float16 -> float32` conversion.

Actual behavior: compiled output differs from eager output.

```python
import torch

def fn(x):
    y = x.to(dtype=torch.float16).to(dtype=torch.float32)
    return y, y.sum(dim=1)

x = torch.arange(20, dtype=torch.float32).reshape(5, 4).t() / 7.0

compiled_fn = torch.compile(fn, backend="inductor", dynamic=False)

eager_y, eager_sum = fn(x)
compiled_y, compiled_sum = compiled_fn(x)

print("eager_y:",eager_y)
print("compiled_y:",compiled_y)
print("max_abs_y:", (eager_y - compiled_y).abs().max())

print("eager_sum:",eager_sum)
print("compiled_sum:",compiled_sum)
print("max_abs_sum", (eager_sum - compiled_sum).abs().max())

```

Output:
```
eager_y: tensor([[0.0000, 0.5713, 1.1426, 1.7139, 2.2852],
        [0.1428, 0.7144, 1.2861, 1.8574, 2.4277],
        [0.2856, 0.8569, 1.4287, 2.0000, 2.5723],
        [0.4285, 1.0000, 1.5713, 2.1426, 2.7148]])
compiled_y: tensor([[0.0000, 0.5714, 1.1429, 1.7143, 2.2857],
        [0.1429, 0.7143, 1.2857, 1.8571, 2.4286],
        [0.2857, 0.8571, 1.4286, 2.0000, 2.5714],
        [0.4286, 1.0000, 1.5714, 2.1429, 2.7143]])
max_abs_y: tensor(0.0008)
eager_sum: tensor([5.7129, 6.4285, 7.1436, 7.8572])
compiled_sum: tensor([5.7143, 6.4286, 7.1429, 7.8571])
max_abs_sum tensor(0.0014)
```

### Versions

`PyTorch version: 2.10.0+cpu`

cc @jgong5 @mingfeima @XiaobingSuper @sanchitintel @ashokei @jingxu10 @aditew01 @chauhang @penguinwu @voznesenskym @EikanWang @Guobing-Chen @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Trigger

A CPU-compiled function narrows float32 values to float16 and immediately widens them to float32 before returning and reducing them.

## Relevant entities

float32 source; float16 CSE value; reverse-conversion cache; in-register cast; buffer store; float32 consumer

## Implemented relation

The inverse conversion of a low-precision value can reuse its float32 source only after a buffer store and reload that has physically applied low-precision rounding.

## Pre-state

The explicit narrowing cast creates an in-register float16 value whose rounding is observable when it is widened again.

## Binding transition

CPU to_dtype records the original float32 value as the cached inverse of the newly generated float16 CSE value.

## Post-state

The following float16-to-float32 conversion reuses the pre-rounding float32 source instead of converting the rounded float16 value.

## Validity obligation

A reverse low-precision conversion is cached as lossless only when a physical low-precision store has established that the cached source and reloaded value represent the same rounded data.

## Invalid decision

The CPU code generator populated the reverse conversion cache when it formed an in-register low-precision cast rather than when it stored that value.

## Physical effect

Float32 mantissa bits that should be discarded at the float16 boundary remain available to the widening cast.

## Observed consequence

The compiled CPU output stays closer to the original float32 input and differs from eager execution's explicitly rounded values.

## Repair site

torch/_inductor/codegen/cpp.py::to_dtype and round_to_int

## Repair

Commit 33cd17c disables the reverse low-precision-to-float32 cache under emulate_precision_casts in both CPU vector and scalar code paths and adds the report's explicit round-trip regression. The existing store/reload optimization remains available outside emulation.

## Why the repair works

The widening conversion now consumes the actual low-precision CSE value in emulation mode, so the explicit narrowing boundary performs the same rounding as eager execution.

## Causal basis

The landed commit states that it fixes 185337, identifies reverse-cache population at to_dtype as the cause, changes that exact CPU code-generation condition, and adds the reported cast pair and reduction as a regression.

## Analysis implication

Cast analysis needs logical dtype, CPU compute representation, CSE provenance, emulation mode, and whether a low-precision value crossed a physical store boundary.

## Fix patches

- [`fixes/pytorch-185337-commit-33cd17c.patch`](fixes/pytorch-185337-commit-33cd17c.patch) — https://github.com/pytorch/pytorch/commit/33cd17c81ae46a8ebf3822d3da238b414aac0f94
