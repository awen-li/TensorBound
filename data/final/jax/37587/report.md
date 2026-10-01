# Potential issue with shard_map + ordered IO callbacks that throw exceptions

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/37587](https://github.com/jax-ml/jax/issues/37587)
- Opened: 2026-05-12T17:08:11Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

Suppose I have a Python function that sometimes raising an exception, and I make ordered IO callbacks to that function.

If I run a jitted computation that does an ordered IO callback inside of a shard_map, and the callback raises some exception E, then I observe that all future ordered IO callbacks to any callback function will immediately fail, raising the same exception E instead of running.

That is:
```python
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

jax.config.update("jax_num_cpu_devices", 4)
jax.config.update('jax_traceback_filtering', 'off')

def add1(x):
    print('hello')
    x = np.array(x)
    if (x == 5).any():
        raise ValueError('No fives.')
    return x + 1

@jax.jit
def f(x):
    return jax.experimental.io_callback(
        add1,
        x,
        x,
        ordered=True)

def run_f(x):
    try:
        print(f'f({x}):', f(x))
    except Exception as e:
        print(f'exception while running f({x}):', e)

sharded_f = jax.jit(jax.shard_map(f, in_specs=P('i'), out_specs=P('i')))

def run_sharded_f(x):
    x = jax.reshard(x, P('i'))
    try:
        print(f'sharded_f({x}):', sharded_f(x))
    except Exception as e:
        print(f'exception while running sharded_f({x}):', e)

mesh = jax.make_mesh((4,), ('i',))

run_f(jnp.array(5))  # raises

run_f(jnp.array(6))  # succeeds

with jax.set_mesh(mesh):
      run_sharded_f(jnp.array([3, 4, 5, 6]))  # raises

with jax.set_mesh(mesh):
      run_sharded_f(jnp.array([1, 1, 1, 1]))  # raises `ValueError: No fives.` and does not print

run_f(jnp.array(7))  # raises `ValueError: No fives.` and does not print

def another_callback(x):
  return

@jax.jit
def g(x):
  print('hello 2')
  return jax.experimental.io_callback(another_callback, (), x, ordered=True)

g(jnp.zeros((4, 3)))  # raises `ValueError: No fives.` and does not print
```

[colab repro](https://colab.research.google.com/drive/1YoqJPiJJxE9VGy5fFLBha90WgrgIEbhm?usp=sharing)

Is this expected behavior?





### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.10.0
jaxlib: 0.10.0
numpy:  2.0.2
python: 3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0]
device info: cpu-4, 4 local devices"
process_count: 1
platform: uname_result(system='Linux', node='e2bb59a7faf4', release='6.6.122+', version='#1 SMP Thu Apr 30 18:17:14 UTC 2026', machine='x86_64')
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises ordered IO-callback token state after an exception inside shard_map.

## Observed implementation decision

An exceptional callback leaves the ordered token associated with incomplete prior execution.

## Physical effect

Later callbacks reuse leaked token state across invocations.

## Observed consequence

Subsequent shard_map executions fail or remain incorrectly ordered.

## Reported proposal or workaround

PR 37973 proposes clearing token state after exceptions.

## Repair trace

PR 37973 was not merged, so no landed token-lifecycle repair exists.

## Archived evidence

- Raw report: `data/raw/github/jax/37587/report.json`
- SHA-256: `49f894bda99856f17db5acbad45393005bfa9b7aac71de473d6d4349cdc7ad9d`
