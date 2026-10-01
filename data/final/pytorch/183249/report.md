# `torch.compile` produces wrong results for `fractional_max_pool2d` when `_random_samples` is not provided

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/183249](https://github.com/pytorch/pytorch/issues/183249)
- Opened: 2026-05-11T03:58:36Z
- Closed: Open at collection
- Labels: `triaged`, `module: random`, `has workaround`, `module: correctness (silent)`, `module: pooling`, `oncall: pt2`, `module: inductor`, `bot-triaged`, `repro:agent:h100:repro_success:reproduced`

## Reported behavior

### 🐛 Describe the bug


## 🐛 Describe the bug

`torch.compile(backend='inductor')` produces incorrect output for `torch.nn.functional.fractional_max_pool2d` when the internal random sampling path is used (i.e., `_random_samples` is not explicitly passed). The compiled version selects **different pooling indices** than eager mode, returning wrong values.

This is **not** a numerical precision issue — the compiled kernel picks entirely different source elements. With `_random_samples` explicitly provided, the results match exactly.

## Reproducer

```python
import torch
import torch._inductor
torch._inductor.config.force_disable_caches = True

x = torch.randn(1, 3, 16, 16, device='cuda')

# Eager
torch.manual_seed(42); torch.cuda.manual_seed(42)
eager_out, eager_idx = torch.nn.functional.fractional_max_pool2d(
    x, 3, output_size=(4, 4), return_indices=True)

# Compiled
torch._dynamo.reset()
torch.manual_seed(42); torch.cuda.manual_seed(42)
comp_out, comp_idx = torch.compile(
    lambda x: torch.nn.functional.fractional_max_pool2d(
        x, 3, output_size=(4, 4), return_indices=True),
    backend='inductor'
)(x)

print(f"Index mismatches: {(eager_idx != comp_idx).sum().item()}/{eager_idx.numel()}")
print(f"Value max_diff: {(eager_out - comp_out).abs().max().item():.4f}")
# Index mismatches: 6/48
# Value max_diff: 1.2810
```

**Workaround**: passing `_random_samples` explicitly produces correct results:

```python
samples = torch.rand(1, 3, 2, device='cuda')
# Both eager and compiled produce identical output when _random_samples is given
eager = torch.nn.functional.fractional_max_pool2d(x, 3, output_size=(4, 4), _random_samples=samples)[0]
compiled = torch.compile(
    lambda x: torch.nn.functional.fractional_max_pool2d(
        x, 3, output_size=(4, 4), _random_samples=samples)[0],
    backend='inductor'
)(x)
print((eager - compiled).abs().max().item())  # 0.0
```

## Observations

| Condition | Result |
|-----------|--------|
| Without `_random_samples` (default) | **Wrong**: 6/48 indices differ, `max_diff=1.28` |
| With explicit `_random_samples` | Correct: `max_diff=0.0` |
| Eager same-seed deterministic | Yes |
| Compiled same-seed deterministic | Yes |
| Eager == Compiled (same seed) | **No** — different indices selected |
| Multiple calls to compiled fn | Each call uses random state (not baked-in) |

The compiled function does consume random state across calls (different seeds produce different outputs), but the random state is consumed differently than in eager mode, causing the pooling regions to diverge.

## Expected behavior

`torch.compile` should produce the same output as eager mode for the same random state. `fractional_max_pool2d` without `_random_samples` should be functionally equivalent between eager and compiled execution.



### Versions

## Environment

- PyTorch: 2.13.0.dev20260501+cu126
- GPU: Tesla T4
- CUDA: 12.6
- OS: Linux



cc @pbelevich @mikaylagawarecki @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises torchInductor fallback execution of fractional_max_pool2d when random samples are generated internally.

## Observed implementation decision

The compiled path consumes a different RNG/sample relation than eager execution despite resetting the same seed.

## Physical effect

Pooling windows select different source indices.

## Observed consequence

Compiled values and indices disagree with eager results.

## Reported proposal or workaround

Supplying _random_samples explicitly restores agreement; PR 183628 proposed an RNG correction.

## Repair trace

PR 183628 was closed without merge and the issue remains open, so the record has no landed correction or fixed regression for the internal sampling path.

## Archived evidence

- Raw report: `data/raw/github/pytorch/183249/report.json`
- SHA-256: `c5bb1c301a8d128b96a2e2378d2ed7f0ac97ad3cef2041b521726920e721536c`
