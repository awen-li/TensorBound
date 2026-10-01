# [TritonGPU] OptimizeThreadLocality miscompiles loop accumulator with an additional in-loop use

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/11836](https://github.com/triton-lang/triton/issues/11836)
- Opened: 2026-09-17T16:45:17Z
- Closed: 2026-09-21T16:55:37Z
- Labels: `bug`

## Reported behavior

### Describe the bug

`TritonGPUOptimizeThreadLocality` can silently miscompile a reduction accumulator when the loop-carried accumulator is also used elsewhere inside the loop.

In the reproducer below, `acc` is stored before each update, so `OUT` should contain the running partial sums:

```text
0, 256, 512, 768
```

Instead, every snapshot is `0`, while the final accumulator is still correct.

## Reproducer

```python
import torch
import triton
import triton.language as tl


@triton.jit
def kernel(X, OUT, FIN, M: tl.constexpr, K: tl.constexpr, STEPS: tl.constexpr):
    m = tl.arange(0, M)
    k = tl.arange(0, K)
    acc = tl.zeros((M,), dtype=tl.float32)

    for i in tl.range(0, STEPS):
        x = tl.load(X + i * M * K + m[:, None] * K + k[None, :])
        r = tl.sum(x, axis=1)

        tl.store(OUT + i * M + m, acc)
        acc = acc + r

    tl.store(FIN + m, acc)


M, K, STEPS = 16, 256, 4

x = torch.ones((STEPS, M, K), device="cuda")
out = torch.empty((STEPS, M), device="cuda")
fin = torch.empty((M,), device="cuda")

kernel[(1,)](x, out, fin, M, K, STEPS, num_warps=4)
torch.cuda.synchronize()

expected_out = (
    torch.arange(STEPS, device="cuda", dtype=torch.float32)[:, None]
    .expand(STEPS, M)
    * K
)
expected_fin = torch.full((M,), STEPS * K, device="cuda", dtype=torch.float32)

print("out[:, 0]:", out[:, 0].tolist())
print("expected :", expected_out[:, 0].tolist())
print("final correct:", torch.equal(fin, expected_fin))
print("snapshots correct:", torch.equal(out, expected_out))
```

Log:

```text
out[:, 0]: [0.0, 0.0, 0.0, 0.0]
expected : [0.0, 256.0, 512.0, 768.0]
final correct: True
snapshots correct: False
```

## Analysis
(By Codex,sorry for wrong)
The issue appears to come from: lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp
The reduction optimization recognizes the pattern: acc = acc + tl.sum(...) and changes it into thread-local accumulation followed by a post-loop reduction.
The matching logic checks that the reduction and its update have one use, but does not check whether the loop-carried accumulator block argument has additional users.
During the rewrite, `createYield` changes the original accumulator to an identity-carried value:
```cpp
yieldValues[oldAccumBlockArgNum - 1] =
    loop.getBody()->getArgument(oldAccumBlockArgNum);
```
The actual partial accumulation is moved to a newly added loop argument.
As a result, additional users of the original accumulator inside the loop, such as the `tl.store` above, continue reading the identity-carried initial value (`0`) rather than the running accumulator. The post-loop result remains correct because it uses the newly introduced accumulator.
A conservative fix may be to skip this optimization when the original accumulator block argument has uses other than the recognized update operation.

### Environment details

Triton: 3.8.0+git6cdc9d5
GPU architecture: sm89

## Trigger

A reduction loop stores the running accumulator inside the loop before updating it.

## Relevant entities

loop-carried accumulator; thread-local partial accumulator; in-loop snapshot store; post-loop reduction

## Implemented relation

Every in-loop use of the accumulator reads the logical running value carried by the original loop iteration argument.

## Pre-state

The original loop-carried accumulator is updated each iteration and is visible to the snapshot store.

## Binding transition

OptimizeThreadLocality re-homes accumulation into a thread-local partial and merges it only after the loop.

## Post-state

The copied in-loop store reads the partial or identity-carried value, while the final post-loop accumulator remains correct.

## Validity obligation

An accumulator rewrite preserves the running value observed by every in-loop consumer, not only the final yielded result.

## Invalid decision

The collector checked uses of the reduction and update but did not reject an accumulator region argument with additional in-loop users.

## Physical effect

Snapshot stores read zero instead of the accumulated values from prior iterations.

## Observed consequence

The snapshots are all zero instead of 0, 256, 512, and 768, although the final result is correct.

## Repair site

lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp

## Repair

PR 11842 requires the accumulator region iteration argument to have exactly one use before applying the rewrite and adds a negative test with an in-loop snapshot.

## Why the repair works

Loops with observable in-loop accumulator reads retain the original loop-carried representation, so those reads continue to observe the running value.

## Causal basis

The issue and patch identify the same omitted use check; the one-line guard and negative regression preserve the path whose in-loop observation was miscompiled.

## Analysis implication

A loop-rewrite check needs the complete use set of each carried value and the point at which a replacement partial is merged into the logical accumulator.

## Fix patches

- [`fixes/triton-11836-pr11842.patch`](fixes/triton-11836-pr11842.patch) — https://github.com/triton-lang/triton/pull/11842
