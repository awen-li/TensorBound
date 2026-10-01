# vmap(jnp.asarray)(numpy_array) does not return a JAX array

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/25745](https://github.com/jax-ml/jax/issues/25745)
- Opened: 2025-01-07T06:38:24Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

I was surprised to discover that `vmap` does not always convert NumPy arrays into JAX arrays. Instead, it sometimes (rarely) will return NumPy arrays when given NumPy arrays as inputs:
```python
import jax
import jax.numpy as jnp
import numpy as np

numpy_array = np.arange(3)
print(type(jax.vmap(lambda x: x)(numpy_array)))  # <class 'numpy.ndarray'>
print(type(jax.vmap(jnp.asarray)(numpy_array)))  # <class 'numpy.ndarray'>
print(type(jax.vmap(jnp.array)(numpy_array)))  # <class 'jaxlib.xla_extension.ArrayImpl'>
```

The exceptions appear to be cases where the wrapped function is evaluated to the "identity", as least as far as JAXprs are concerned:
```python
print(jax.make_jaxpr(jax.vmap(jnp.asarray))(jax_array))
# { lambda ; a:i32[3]. let  in (a,) }
```

However, `jnp.asarray()` is also explicitly converting inputs into a JAX arrays, so I found this doubly surprsing.

My real-use is a custom pytree type where the constructor (but not `tree_unflatten`) always converts inputs into JAX arrays, which otherwise seems perfectly well behaved and is one of the [suggested patterns](https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization) in the JAX docs. `vmap` returns an otherwise impossible to create value when applied to this "identity function":
```python
@jax.tree_util.register_pytree_node_class
class Wrapper:
  def __init__(self, value):
    self.value = jnp.asarray(value)

  def tree_flatten(self):
    return [self.value], None

  @classmethod
  def tree_unflatten(cls, _, leaves):
    result = super().__new__(cls)
    [result.value] = leaves
    return result

  def __repr__(self):
    return f"Wrapped({self.value!r})"


print(type(jax.vmap(Wrapper)(np.arange(3)).value))
# <class 'numpy.ndarray'>
```

### System info (python version, jaxlib version, accelerator, etc.)

jax:    0.4.38
jaxlib: 0.4.38
numpy:  1.26.4
python: 3.10.12 (main, Nov  6 2024, 20:22:13) [GCC 11.4.0]
device info: cpu-1, 1 local devices"
process_count: 1
platform: uname_result(system='Linux', node='a953559aba37', release='6.1.85+', version='#1 SMP PREEMPT_DYNAMIC Thu Jun 27 21:05:47 UTC 2024', machine='x86_64')

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises vmap of jnp.asarray over NumPy inputs.

## Observed implementation decision

Batching allows a NumPy value to pass through the simplest asarray path without establishing JAX array ownership.

## Physical effect

The mapped output remains a NumPy representation rather than a JAX array.

## Observed consequence

The API returns the wrong array kind.

## Reported proposal or workaround

PR 25835 proposes conversion; other merged arraylike cleanups target different call sites.

## Repair trace

The direct PR was not merged and no landed change is shown to fix this vmap path.

## Archived evidence

- Raw report: `data/raw/github/jax/25745/report.json`
- SHA-256: `ea419e8aecf9ec5609dbfab2a1db8fefbad2e1bd96dad096cfcb7986cdc97b8d`
