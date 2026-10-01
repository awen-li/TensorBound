# torch.compile (Inductor): eager != compiled for torch.put with accumulate when input/index/source alias a computed intermediate

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/198567](https://github.com/pytorch/pytorch/issues/198567)
- Opened: 2026-09-24T23:26:58Z
- Closed: open at collection time
- Labels: `triage review`, `module: partial aliasing`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `topic: fuzzer`, `module: pt2-dispatcher`, `module: reinplacing`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug

### Description

`torch.compile` produces a different integer result than eager for `torch.put(r, r, r, accumulate=True)` when `r` is a **computed intermediate** and `input`/`index`/`source` are the same tensor. Both eager and compiled are deterministic across repeated runs; they simply disagree.

**I'm aware this is an unusual aliasing pattern** (input == index == source). I'm reporting it because (a) it's a deterministic eager-vs-compiled divergence, and (b) the operation appears well-defined — with `accumulate=True`, `torch.put` reads the original `index`/`source` values and accumulates into a copy of `input`, so the result should be order-independent and match eager. If aliasing `input` with `index`/`source` is considered unsupported, please close — but flagging in case the compiled path is dropping a required copy.

### Minimal reproducer (CPU)

```python
import torch

base = (torch.arange(48) % 9).reshape(3, 16)     # deterministic int64 tensor

def model(x):
    r = x + 1                                     # r is a compiled intermediate
    return torch.put(r, r, r, accumulate=True)    # input == index == source == r

eager = model(base)
torch._dynamo.reset()
compiled = torch.compile(model)(base)

print("eager == compiled:", torch.equal(eager, compiled))   # False
d = (eager.flatten() != compiled.flatten()).nonzero().flatten()
for i in d.tolist():
    print(f"  pos {i}: eager={eager.flatten()[i].item()} compiled={compiled.flatten()[i].item()}")
# e.g. pos 9: eager=46 compiled=54
```

### Result

```
eager == compiled: False
  pos 9: eager=46 compiled=54
```

Both are stable across repeated runs (eager==eager, compiled==compiled), so this is a genuine divergence, not nondeterminism.

### Ablation (isolates the trigger)

The divergence appears **only** when all three of `input`/`index`/`source` are the same *computed intermediate*:

| case | result |
| --- | --- |
| `put(r, r, r)` with `r = x + 1` (aliased intermediate) | ❌ eager ≠ compiled |
| `put(r.clone(), r, r)` / non-aliased `put(inp, idx, src)` | ✅ match |
| `r` as a graph **input** (leaf) instead of intermediate | ✅ match |
| any single op in isolation | ✅ match |

This points at functionalization/aliasing of `torch.put` when its inputs alias a fused intermediate. Found via fuzzing.

### Versions

PyTorch version: 2.14.0+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.4 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Clang version: 18.1.3 (1ubuntu1)
CMake version: version 3.28.3
Libc version: glibc-2.39

Python version: 3.12.14 | packaged by Anaconda, Inc. (64-bit runtime)
Python platform: Linux-6.8.0-136-generic-x86_64-with-glibc2.39
Is CUDA available: False
CUDA runtime version: No CUDA
GPU models and configuration: No CUDA
Is XPU available: False
Is XNNPACK available: False

CPU:
Model name: Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz

Versions of relevant libraries:
[pip3] numpy==2.5.3
[pip3] torch==2.14.0+cpu
[conda] blas                      1.0                         mkl
[conda] mkl                       2023.1.0         h213fc3f_46344
[conda] numpy                     2.5.3                    pypi_0

cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo @bdhirsh @bobrenjc93 @aorenste

## Trigger

torch.put(r, r, r, accumulate=True) uses one computed intermediate as destination input, flattened indices, and source values.

## Relevant entities

computed intermediate storage; index_put input; indices; values; functional update; reinplaced index_put_; accumulation

## Implemented relation

An in-place indexed update reads stable index and value operands while it mutates the destination storage.

## Pre-state

Functional index_put conceptually builds its result without changing the intermediate that supplies the destination, indices, and values.

## Binding transition

The reinplacing pass observes no later use of the input and substitutes index_put_ without checking whether indices or values are views of the same storage.

## Post-state

The mutating kernel reads subsequent indices and values from a buffer whose earlier positions it has already updated.

## Validity obligation

A functional indexed update becomes in-place only when every index and value operand is storage-independent from the destination input.

## Invalid decision

Liveness of the destination was treated as sufficient for reinplacing, omitting storage dependencies carried by other operands of the same operation.

## Physical effect

Accumulated writes change data that later iterations interpret as indices or source values, altering the update set and contributions.

## Observed consequence

The reported CPU program is deterministic but differs from eager, including position 9 producing 54 instead of 46; CUDA can reject the overlap.

## Repair site

torch/_inductor/fx_passes/reinplace.py::should_reinplace_index_put and inplaceable_ops

## Repair

Commit dd806b6 adds a storage-alias check over all index_put indices, values, and keyword operands and uses it for index_put and unsafe_index_put reinplacing. Aliased cases remain functional; independent operands retain the in-place optimization. It adds the exact torch.put regression and graph-level controls.

## Why the repair works

Keeping the aliased case functional preserves the pre-update snapshot from which indices and values are read, so accumulation matches eager semantics.

## Causal basis

The landed commit states that it fixes 198567, describes the same input-index-value alias pattern, adds the operand-storage guard at the reinplacing decision, and includes the report's computed-intermediate torch.put case as a regression.

## Analysis implication

Indexed-mutation analysis needs storage identity for the destination, every index and value operand, view ancestry, functional-to-in-place conversion, and intra-kernel read/write order together.

## Fix patches

- [`fixes/pytorch-198567-commit-dd806b6.patch`](fixes/pytorch-198567-commit-dd806b6.patch) — https://github.com/pytorch/pytorch/commit/dd806b6d43def42db89610cad528cb382603e3da
