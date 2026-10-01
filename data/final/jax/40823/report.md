# lax.cond can mark an array as replicated when it is not, giving rise to silent errors with shard map

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/40823](https://github.com/jax-ml/jax/issues/40823)
- Opened: 2026-09-20T15:37:06Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

## Examples
This issue popped up when dealing with https://github.com/patrick-kidger/equinox/pull/1262. I will explain it with a contrived gradient example below with 2 CPU devices

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P

mesh = Mesh(np.array(jax.devices()), ("i",))

@jax.shard_map(
    mesh=mesh, in_specs=P(), out_specs=P("i"), check_vma=True
)
def f(x):
    return lax.cond(
        lax.axis_index("i") == 0,
        lambda x: 2 * x,
        lambda x: 3 * x,
        x,
    )[None]

print(f(2.))
print(jax.jit(jax.grad(lambda x: f(x).sum()))(2.))
```

The correct expected gradient is 5, but we end up getting 4.

For a less contrived example where not having that information is a setback, I reproduce my example in the comment here https://github.com/patrick-kidger/equinox/pull/1262

```python
x = [0, 1, 2, 3]   # device 1 has [0, 1] , device 2 has [2, 3], edited
condition = x[0] < 1.5
pred = lax.cond(condition, lambda: True, lambda: False) # lax.cond loses information that the predicate is varying but the array is actually # device 1 has [True, True], device 2 has [False False]

result = eqx.error_if(x, pred, "negative", on_error="nan") # attempted sharding assumes predicate is not varying i.e distributed, therefore it will try to forego pmax and can only give wrong result without comms
```

## Timeline of relevant issues/PRs
I'll only keep the illustrative ones, that are relevant to changes in behavior/have unexpected behavior (I filtered the list I got from chatgpt)

1. https://github.com/jax-ml/jax/issues/24418 Issue similar to one of my examples, where we get an exception raise because of the lack of replication rules
2. https://github.com/jax-ml/jax/pull/24877 PR adding replication rules for cond
3. https://github.com/jax-ml/jax/commit/76271d638ad94f3df854054640f3b35161ee5be4 Here the relevant code is

```python
  b0_vma = [o.vma for o in branches[0].out_avals]
  for branch in branches[1:]:
    b_vma = [o.vma for o in branch.out_avals]
    if b0_vma != b_vma:
      raise Exception("The branches of cond produced mismatched varying manual "
                      f"axes. Got {b0_vma} and {b_vma}. Please open an issue "
                      "at https://github.com/jax-ml/jax/issues, and as a "
                      "temporary workaround pass the check_rep=False argument "
                      "to shard_map")
```

Note that the code only checks the branches of the axes. If the predicate itself is differently satisfied on different processors, it will lead to a varying output.

The original replicating code was later deleted in https://github.com/jax-ml/jax/pull/27929


## fix
imo if `pred` is varying in cond(pred, true_fun, false_fun) then the output should be varying as well, and should not raise an error



### System info (python version, jaxlib version, accelerator, etc.)

jax:    0.11.2
jaxlib: 0.11.2
numpy:  2.2.6
python: 3.12.13 | packaged by conda-forge | (main, Mar  5 2026, 16:50:00) [GCC 14.3.0]
device info: cpu-2, 2 local devices"
process_count: 1
platform: uname_result(system='Linux', node='praharsh-work', release='6.14.0-37-generic', version='#37-Ubuntu SMP PREEMPT_DYNAMIC Fri Nov 14 22:10:32 UTC 2025', machine='x86_64')
JAX_PLATFORMS=cpu
JAX_NUM_CPU_DEVICES=2

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises shard_map variance analysis for lax.cond selectors.

## Observed implementation decision

A varying predicate produces a boolean result classified as replicated.

## Physical effect

Downstream shard-map code treats per-shard state as one shared value.

## Observed consequence

Execution can silently produce wrong distributed results.

## Reported proposal or workaround

PR 40833 proposes a variance guard.

## Repair trace

PR 40833 closed without merge, leaving no landed repair or fixed regression.

## Archived evidence

- Raw report: `data/raw/github/jax/40823/report.json`
- SHA-256: `1dbe20c57a9982f9ac263abdeab233f8ee5535316b4529f715feba091b806023`
