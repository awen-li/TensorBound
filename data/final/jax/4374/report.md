# id_print example from docs broken; crash on GPU

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/4374](https://github.com/jax-ml/jax/issues/4374)
- Opened: 2020-09-22T07:38:35Z
- Closed: 2021-03-18T08:44:27Z
- Labels: `bug`

## Reported behavior

Dear jax team,

I am having problems using `jax.experimental.host_callback.id_print`. I looked at [the docs](https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html) and tried the example there, but it gives me either an error or crashes completely if altered slightly.

The crash happens only on GPU. On CPU, `power3_2` prints correctly.

Am I using `id_print` incorrectly or are the docs not up-to-date? Or am I missing something?

Repro:

```python
import jax
import jax.numpy as jnp
from jax.experimental.host_callback import id_print

def power3(x):
   y = x * x
   _, y = id_print(x, y, what="x,x^2")
   return y * x

jax.vmap(power3)(jnp.arange(3.))
# TypeError: id_print() takes 1 positional argument but 2 were given


def power3_2(x):
   y = x * x
   y = id_print(y, what="x^2")
   return y * x

jax.vmap(power3_2)(jnp.arange(3.))
"""
Works on CPU. Crashed on GPU with:

2020-09-22 09:26:56.242431: F external/org_tensorflow/tensorflow/compiler/xla/service/gpu/outfeed_thunk.cc:49]
Check failed: ShapeUtil::Compatible(hlo_instruction_->operand(0)->shape(), outfeed_buffers->shape())
XLA program outfeed request of shape (f32[3]) did not match the runtime's outfeed buffer of shape u32[2]
Aborted
"""
```

I am using jax and compiling jaxlib from HEAD on CUDA 10.0/450.66.

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

A callback is vectorized or jitted and executed on a GPU through the outfeed implementation.

## Observed implementation decision

The GPU runtime consumes an outfeed buffer associated with a different request shape, which maintainers attribute to bad computation interleaving.

## Physical effect

The compiled f32 callback request is matched to a u32 token-shaped runtime buffer.

## Observed consequence

XLA aborts on its outfeed shape-compatibility check.

## Reported proposal or workaround

Restricting visibility to one GPU made the later reproducer execute, while maintainers continued to track the multi-GPU problem separately.

## Repair trace

The issue closes after a one-GPU observation and links to the still-unrepaired multi-GPU report, not to a source correction. No exact XLA/JAX repair or fixed regression establishes the repaired interleaving relation.

## Archived evidence

- Raw report: `data/raw/github/jax/4374/report.json`
- SHA-256: `1534feecb821ca48922da44d6098664f79a834f2698cbafbd48afd557ee7c3c5`
