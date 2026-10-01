# JAX getting stuck with asynchronous dispatching (CPU backend)

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/25861](https://github.com/jax-ml/jax/issues/25861)
- Opened: 2025-01-13T18:38:53Z
- Closed: 2025-01-29T14:53:35Z
- Labels: `bug`

## Reported behavior

### Description

In a larger software project, I noticed that JAX occasionally gets stuck in computations when asynchronous dispatching is enabled for the CPU. Finding a short reproducer is not so easy, but I came up with the script below (which doesn't do any useful computation but should run through).  On my machine, but also on the Google Colab CPU runtime, the script almost always hangs.
**Note:** When running in Colab or IPython it appears as if the script would run through, but adding an additional input cell below, such as `print(post_mean)` shows that the execution got stuck.

Disabling asynchronous dispatch with `jax.config.update('jax_cpu_enable_async_dispatch', False)` makes the script run to the end.

```
import jax
from jax import numpy as jnp
from jax import random

# jax.config.update('jax_cpu_enable_async_dispatch', False)

seed = 42
key = random.PRNGKey(seed)
key, subkey = random.split(key)

def cg_pretty_print_it(
    name,
    i,
    maxiter=None,
):
    if maxiter is not None and i == maxiter:
        i_str = f"maxiter: ({i})"
    else:
        i_str = str(i)
    msg = f"{name}: Iteration {i_str}"
    print(msg)

def static_cg(pos):
    from jax.debug import callback
    from jax.lax import while_loop

    name = 'loop'
    maxiter = 100

    def pp(arg):
        cg_pretty_print_it(name, **arg)

    def continue_condition(v):
        return v["info"] < -1

    def cg_single_step(v):
        info = v["info"]
        pos, i = v["pos"], v["iteration"]
        i += 1

        pos = 1 + pos
        info = jnp.where(pos[0,0] > 500, 0, info)
        if name is not None:
            printable_state = {
                "i": i,
                "maxiter": maxiter,
            }
            callback(pp, printable_state)
        ret = {
            "info": info,
            "pos": pos,
            "iteration": i,
        }
        return ret

    val = {
        "info": jnp.array(-2, dtype=int),
        "pos": pos,
        "iteration": jnp.array(0),
    }
    val = while_loop(continue_condition, cg_single_step, val)

    info = val["info"]
    return val["pos"], info


vmap_static_cg = jax.vmap(static_cg, in_axes = 0)

n_samples = 30
inputs = 10*random.normal(subkey, shape=(n_samples, 128, 128))

smpls, smpls_info = vmap_static_cg(inputs)
# smpls, smpls_info = jax.block_until_ready(vmap_static_cg(inputs)) # works

print('done with vmap function')


def mean(forest):
    from functools import reduce

    norm = 1.0 / len(forest)
    add = lambda a,b: a+b
    m = norm * reduce(add, forest)
    return m

post_mean = mean(tuple(s for s in smpls))
```

Interestingly, the script will also run if you wrap the `vmap_static_cg` call in a `jax.block_until_ready( )`.

### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.4.38
jaxlib: 0.4.38
numpy:  2.2.1
python: 3.12.8 | packaged by conda-forge | (main, Dec  5 2024, 14:24:40) [GCC 13.3.0]
device info: cpu-1, 1 local devices"
process_count: 1
platform: uname_result(system='Linux', node='tux', release='6.12.9-arch1-1', version='#1 SMP PREEMPT_DYNAMIC Fri, 10 Jan 2025 00:39:41 +0000', machine='x86_64')
```

## Trigger

A CPU executable invokes a host callback whose body dispatches another JAX CPU computation while outer dispatch is asynchronous.

## Relevant entities

outer CPU executable; host callback; inner CPU computation; XLA intra-op thread pool; execution mode

## Implemented relation

The callback must complete before the callback-bearing executable can complete, while the callback body may itself require CPU execution resources.

## Pre-state

The outer executable occupies CPU execution resources and enters a host callback.

## Binding transition

The callback body submits an inner JAX CPU computation through the asynchronous dispatch path.

## Post-state

Available execution resources are occupied by callbacks waiting for inner computations that cannot be scheduled.

## Validity obligation

A callback-bearing CPU execution leaves a schedulable path for reentrant work required by that callback.

## Invalid decision

PjRtLoadedExecutable::Execute retained asynchronous CPU dispatch even when all_loaded_host_callbacks_ was nonempty.

## Physical effect

The intra-op thread pool can be filled by callback invocations whose inner computations wait for the same pool.

## Observed consequence

The reported vmap/while-loop computation hangs unless asynchronous CPU dispatch is disabled or the result is forced ready earlier.

## Repair site

xla/python/pjrt_ifrt/pjrt_executable.cc, with regression coverage in tests/debugging_primitives_test.py

## Repair

XLA PR 21980 selects synchronous execution for executables with loaded host callbacks; JAX PR 26160 adds the reported reentrant callback regression.

## Why the repair works

Synchronous outer dispatch prevents callback-bearing executions from consuming the asynchronous resources needed by callback-dispatched inner work.

## Causal basis

The issue discussion identifies reentrant callback dispatch, and the linked XLA change gates the execution mode on callback presence while the JAX test preserves the reported deadlock pattern.

## Analysis implication

A dispatch check needs backend, execution mode, callback presence, callback reentrancy, and execution-resource dependencies.

## Fix patches

- [`fixes/xla-25861-pr21980.patch`](fixes/xla-25861-pr21980.patch) — https://github.com/openxla/xla/pull/21980
- [`fixes/jax-25861-pr26160.patch`](fixes/jax-25861-pr26160.patch) — https://github.com/jax-ml/jax/pull/26160
