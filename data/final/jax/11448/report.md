# Memory leak in compiled loops closing over large constants

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/11448](https://github.com/jax-ml/jax/issues/11448)
- Opened: 2022-07-11T17:01:42Z
- Closed: 2022-10-10T16:13:16Z
- Labels: `bug`

## Reported behavior

I was running into an issue that I could condense to an elementary example. It seems to me as if closing over large constants in lax loops creates a memory leak while unrolling the loop avoids the issue. A disclaimer: my knowledge of how JAX is limited. Thus, the issue described may not be a memory leak but a misunderstanding on my part of how JAX or XLA buffers compiled computations.

In the MRE, each of the side-effect-free functions  `leak` and `no_leak` create in their body a large constant and later compile function which closes over the large constant. The version which includes a lax loop closing over the constant seems to leak memory.

I'm not sure whether this issue is related to #282. However, since the function without a lax loop doesn't seem to leak memory, I think it is different. From my understanding of @jakevdp's [comment](https://github.com/google/jax/issues/7930#issuecomment-995003572) on caching of compiled functions, I'd expect that the cache is cleared and memory should be released once `g`, `jg` and `data` go out of scope or once the references are deleted manually.

However, and here I'm not sure if I use `jaxlib.xla_extension.Client` correctly, it seems that using the lax loop creates a live buffer that doesn't get dropped when both functions, the compiled function and the original function, go out of scope (see output below). I also tried to avoid the jit-compilation of the lax loop (see function `leak_no_compile`). This seems to create similar behavior of the program with the difference that, additionally to the live buffer, also a live executable can be found via the `backend.live_executables()` after the function went out of scope.


## MRE

contents of leak.py
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
        return jax.lax.fori_loop(0, 1, lambda _, __: data[0], 0.0)

    jg = jax.jit(g)

    _ = jg().block_until_ready()

    del g, jg, data, _


def no_leak():
    # ~ 1 GB of float32s
    size = 1024 * 1024 * 1024 // 4
    data = jax.device_put(np.empty((size, ), dtype=np.float32) + 1)

    def g():
        val = None
        for _ in range(1):
            val = data[0]
        return val

    jg = jax.jit(g)

    _ = jg().block_until_ready()

    del g, jg, data, _


def leak_no_compile():
    # ~ 1 GB of float32s
    size = 1024 * 1024 * 1024 // 4
    data = jax.device_put(np.empty((size, ), dtype=np.float32) + 1)

    def g():
        return jax.lax.fori_loop(0, 1, lambda _, __: data[0], 0.0)

    _ = g().block_until_ready()

    del g, data, _


def run() -> None:
    for iter in range(3):
        print(f"iter: {iter} ================================================================")
        print_mem_usage()
        _ = leak()  # <----- The function call is replaced with no_leak and leak_no_compile in the output below
        print_mem_usage()
        print_live_buffers()
        print_num_live_executables()


if __name__ == "__main__":
    run()
```

<details>
  <summary>Output with `leak()`</summary>

```
$ python leak.py
iter: 0 ================================================================
Resident memory: 114 MB
Resident memory: 1170 MB
Number live buffers: 1
	lbs[0]: shape (268435456,) at 0x288dc4000
Number live executables: 0
iter: 1 ================================================================
Resident memory: 1170 MB
Resident memory: 2196 MB
Number live buffers: 2
	lbs[0]: shape (268435456,) at 0x2c8dc4000
	lbs[1]: shape (268435456,) at 0x288dc4000
Number live executables: 0
iter: 2 ================================================================
Resident memory: 2196 MB
Resident memory: 3222 MB
Number live buffers: 3
	lbs[0]: shape (268435456,) at 0x308dc4000
	lbs[1]: shape (268435456,) at 0x2c8dc4000
	lbs[2]: shape (268435456,) at 0x288dc4000
Number live executables: 0
```
</details>

<details>
  <summary>Output with `no_leak()`</summary>

```
$ python leak.py
iter: 0 ================================================================
Resident memory: 116 MB
Resident memory: 1172 MB
Number live buffers: 0
Number live executables: 0
iter: 1 ================================================================
Resident memory: 1172 MB
Resident memory: 1174 MB
Number live buffers: 0
Number live executables: 0
iter: 2 ================================================================
Resident memory: 1174 MB
Resident memory: 1176 MB
Number live buffers: 0
Number live executables: 0
```
</details>


<details>
  <summary>Output with `leak_no_compile()`</summary>

```
$ python leak.py
iter: 0 ================================================================
Resident memory: 116 MB
Resident memory: 1173 MB
Number live buffers: 1
	lbs[0]: shape (268435456,) at 0x298000000
Number live executables: 1
iter: 1 ================================================================
Resident memory: 1173 MB
Resident memory: 2198 MB
Number live buffers: 2
	lbs[0]: shape (268435456,) at 0x2d8000000
	lbs[1]: shape (268435456,) at 0x298000000
Number live executables: 2
iter: 2 ================================================================
Resident memory: 2198 MB
Resident memory: 3223 MB
Number live buffers: 3
	lbs[0]: shape (268435456,) at 0x318000000
	lbs[1]: shape (268435456,) at 0x2d8000000
	lbs[2]: shape (268435456,) at 0x298000000
Number live executables: 3
```

</details>

## System

I'm using jax == 0.3.14 and jaxlib == 0.3.14 on python 3.10.5 on a Mac with M1Pro/Ultimate.

## Trigger

Repeatedly created jitted loop bodies close over distinct large DeviceArrays and then go out of user scope.

## Relevant entities

loop body function; Python closure; DeviceArray constant; initial-style jaxpr cache; compiled executable

## Implemented relation

A compilation cache entry does not outlive the user-visible function whose closure supplies its constant buffers.

## Pre-state

The temporary loop body and its captured array become otherwise unreachable after one iteration of the test.

## Binding transition

The cache uses the function object as a strongly held key and also keeps the body callable in cached wrappers.

## Post-state

The cache retains the closure, its DeviceArray, and the corresponding device buffer and executable.

## Validity obligation

Cache identity may depend on a function without becoming a new owner that extends the function's captured tensor lifetime.

## Invalid decision

The old control-flow caches used strong-reference memoization for function-keyed jaxpr construction.

## Physical effect

Each call leaves a large live device buffer reachable through the cache.

## Observed consequence

Resident memory and live executable/buffer counts grow by about one gigabyte per iteration in the reproducer.

## Repair site

jax/_src/lax/control_flow/common.py and loops.py

## Repair

PR 11461 replaces strong caches with weakref_lru_cache and passes weak references to loop body functions through cached helpers.

## Why the repair works

When the caller releases the function, the cache key and wrapper no longer keep its closure or captured buffers alive.

## Causal basis

The maintainer identifies _initial_style_jaxpr as the retaining cache, and the patch changes its ownership semantics at the named sites.

## Analysis implication

Cache-lifetime analysis needs key ownership, closure captures, weak versus strong references, cached value references, and device-buffer reachability.

## Fix patches

- [`fixes/jax-11448-pr11461.patch`](fixes/jax-11448-pr11461.patch) — https://github.com/jax-ml/jax/pull/11461
