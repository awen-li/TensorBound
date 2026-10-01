# Inductor should respect rand op order in eager function

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/181136](https://github.com/pytorch/pytorch/issues/181136)
- Opened: 2026-04-22T17:17:52Z
- Closed: 2026-05-01T23:58:31Z
- Labels: `triaged`, `module: random`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

The following code fails with compile. In eager code, rng order is `a -> b -> c`. In compiled code, rng order is `b -> c -> a`.

```python
import torch

def fn():
    a = torch.randn(2, 4, 4, 4, device="cuda")
    b = torch.randn(2, 3, 4, 4, device="cuda")
    c = torch.randn(4, 3, 3, 3, device="cuda")
    y = torch.nn.functional.conv2d(b, c, padding=1)
    z = a + y
    return z

f_c = torch.compile(fn)

torch.manual_seed(42)
eager_out = fn()

torch.manual_seed(42)
eager_out1 = fn()

torch.manual_seed(42)
compile_out = f_c()

torch.testing.assert_close(eager_out, eager_out1)   # PASS
torch.testing.assert_close(compile_out, eager_out)    # FAIL
```

cc @pbelevich @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Trigger

A function creates three independent randn tensors in source order a, b, c, while only b and c feed a convolution before a is added.

## Relevant entities

FX random nodes; nondeterministic_seeded tag; program order; scheduler dependencies; fallback kernels; global CUDA RNG state

## Implemented relation

Fallback random operations that share global CUDA RNG state consume that state in the same order as their operations appear in the eager program.

## Pre-state

The FX graph lists the three randn nodes in eager order, and each fallback call advances one shared RNG state.

## Binding transition

The scheduler orders independent nodes from data dependencies alone and moves the convolution's b and c generators before a.

## Post-state

The same seed is partitioned among tensor shapes in the order b, c, a instead of a, b, c.

## Validity obligation

Stateful seeded operations have explicit control dependencies that preserve their program order even when no tensor data dependency connects them.

## Invalid decision

Seeded nondeterministic fallback nodes were modeled as freely reorderable independent computations.

## Physical effect

Each tensor receives random values from a different segment of the global CUDA RNG stream than eager execution assigns to it.

## Observed consequence

After resetting the seed to 42, eager runs agree with each other but the compiled output differs because Inductor executes the random calls in b, c, a order.

## Repair site

torch/_inductor/fx_passes/post_grad.py::_chain_random_ops_for_ordering and control-dependency lowering

## Repair

Commit 0aaf395 identifies FX nodes tagged nondeterministic_seeded, chains adjacent nodes through preserve_node_ordering when fallback_random is enabled, supports dependency values with multiple IR leaves, and adds the reported randn/conv2d and tuple-output dropout regressions.

## Why the repair works

The scheduler can still reorder ordinary independent work, but every fallback random call waits for its predecessor, so global RNG consumption matches eager source order.

## Causal basis

The landed commit explicitly fixes report 181136 for fallback_random=True, names the b/c/a scheduler reordering, and adds the report's randn and convolution structure as a regression. The separate fused-RNG path is not included in this cause.

## Analysis implication

Randomness analysis needs RNG-state identity, operation tag, source sequence, scheduler dependencies, output structure, fallback mode, and execution order together.

## Fix patches

- [`fixes/pytorch-181136-commit-0aaf395.patch`](fixes/pytorch-181136-commit-0aaf395.patch) — https://github.com/pytorch/pytorch/commit/0aaf3956514cc5079e62eeb8026d8d860e7c0b72
