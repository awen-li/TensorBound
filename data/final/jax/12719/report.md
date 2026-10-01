# Memory leak when closing over large constants in compiled jax.lax.cond statement

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/12719](https://github.com/jax-ml/jax/issues/12719)
- Opened: 2022-10-10T16:11:44Z
- Closed: 2024-07-24T00:20:10Z
- Labels: `bug`

## Reported behavior

### Description

I was running into an issue that I could condense to an elementary example. It seems to me as if closing over large constants in lax.cond creates a memory leak.

I've reported a similar issue concerning lax.fori_loop in #11448 which was solved in PR #11461. However, my followup post in the original issue got no attention so I decided to close the original issue since the originally reported issue was fixed and to open this one. I hope that is ok.

I'm aware that closing over large constant is not recommended and instead they should be passed as arguments. Nonetheless, this leaking behaviour is unexpected. Maybe it can be -- similar to #11448 --  easily fixed taking weak references to true_fun and false_fun.

## MRE

The side-effect-free function leak creates in ints body a large constant and later compiles a function which closes over the large constant within a lax.cond statement. The compiled function called. When leaving the function leak, all references to the large constant are destroyed, including the closure and the compiled closure and the memory should be freed. However, it is not.

```py
import numpy as np
import jax
import psutil


def print_mem_usage() -> None:
    print(
        f"Resident memory: {psutil.Process().memory_info().rss // (1024 * 1024)} MB"
    )

def print_live_buffers() -> None:
    lbs = jax.lib.xla_bridge.get_backend().live_buffers()
    print(f"Number live buffers: {len(lbs)}")
    for idx, lb in enumerate(lbs):
        print(f"\tlbs[{idx}]: shape {lb.shape} at 0x{lb.unsafe_buffer_pointer():x}")


def print_num_live_executables() -> None:
    les = jax.lib.xla_bridge.get_backend().live_executables()
    print(f"Number live executables: {len(les)}")



def leak():
    # ~ 1 GB of float32s
    size = 1024 * 1024 * 1024 // 4
    data = jax.device_put(np.empty((size, ), dtype=np.float32) + 1)

    def g():
        return jax.lax.cond(
            True,
            lambda: data[0],
            lambda: data[1],
        )


    jg = jax.jit(g)

    _ = jg().block_until_ready()

    del g, jg, data, _


def run() -> None:
    for iter in range(5):
        print(f"iter: {iter} ================================================================")
        print_mem_usage()
        _ = leak()
        print_mem_usage()
        print_live_buffers()
        print_num_live_executables()


if __name__ == "__main__":
    run()
```

<details><summary>Output</summary>

```
$ python leak3.py
iter: 0 ================================================================
Resident memory: 116 MB
Resident memory: 1168 MB
Number live buffers: 1
	lbs[0]: shape (268435456,) at 0x293698000
Number live executables: 0
iter: 1 ================================================================
Resident memory: 1168 MB
Resident memory: 2192 MB
Number live buffers: 2
	lbs[0]: shape (268435456,) at 0x2d3698000
	lbs[1]: shape (268435456,) at 0x293698000
Number live executables: 0
iter: 2 ================================================================
Resident memory: 2192 MB
Resident memory: 3216 MB
Number live buffers: 3
	lbs[0]: shape (268435456,) at 0x313698000
	lbs[1]: shape (268435456,) at 0x2d3698000
	lbs[2]: shape (268435456,) at 0x293698000
Number live executables: 0
iter: 3 ================================================================
Resident memory: 3216 MB
Resident memory: 4242 MB
Number live buffers: 4
	lbs[0]: shape (268435456,) at 0x353698000
	lbs[1]: shape (268435456,) at 0x313698000
	lbs[2]: shape (268435456,) at 0x2d3698000
	lbs[3]: shape (268435456,) at 0x293698000
Number live executables: 0
iter: 4 ================================================================
Resident memory: 4242 MB
Resident memory: 5266 MB
Number live buffers: 5
	lbs[0]: shape (268435456,) at 0x393698000
	lbs[1]: shape (268435456,) at 0x353698000
	lbs[2]: shape (268435456,) at 0x313698000
	lbs[3]: shape (268435456,) at 0x2d3698000
	lbs[4]: shape (268435456,) at 0x293698000
Number live executables: 0
```

</details>

### What jax/jaxlib version are you using?

jax v0.3.21, jaxlib v0.3.20

### Which accelerator(s) are you using?

CPU

### Additional system info

Python 3.10.5 on a Mac with M1Pro/Ultimate (MacOS 12.6)

## Trigger

A short-lived jitted cond closes over a new device array on each invocation and then all user references are deleted.

## Relevant entities

branch callables; closed-over array; device buffer; cond jaxpr; trace cache entry

## Implemented relation

A tracing cache preserves reusable jaxpr structure without extending the lifetime of closed-over tensor constants after their callables die.

## Pre-state

The branch closures are the remaining user-level owners of the newly allocated device array.

## Binding transition

_initial_style_jaxprs_with_common_consts stores the callable sequence in a strong @cache entry.

## Post-state

The cache retains the closures and their device array after the compiled function and explicit array reference are deleted.

## Validity obligation

Compilation memoization does not create a strong lifetime edge from a global cache to short-lived branch callables and their tensor constants.

## Invalid decision

The cond tracing helper cached the branch callables strongly instead of caching only weakly keyed jaxpr-padding work.

## Physical effect

One otherwise unreachable device buffer remains live after each invocation.

## Observed consequence

Resident memory grows by about one GiB per iteration and the live-buffer count rises from one to five while no executable remains live.

## Repair site

jax/_src/lax/control_flow/common.py::_initial_style_jaxprs_with_common_consts

## Repair

PR 22610 removes the strong outer cache and adds a weak-reference LRU cache around _pad_jaxpr_constvars; its regression checks that live buffers return to zero.

## Why the repair works

Reusable padding results remain cached through weakly held jaxpr keys without retaining the branch closures or their closed-over arrays.

## Causal basis

A maintainer confirms that removing the named cache eliminates the leak, and the merged patch replaces that cache and adds the issue-specific live-buffer regression.

## Analysis implication

Cache-lifetime analysis needs key reference strength, closure captures, constant ownership, and the lifetime of resulting device buffers.

## Fix patches

- [`fixes/jax-12719-pr22610.patch`](fixes/jax-12719-pr22610.patch) — https://github.com/jax-ml/jax/pull/22610
