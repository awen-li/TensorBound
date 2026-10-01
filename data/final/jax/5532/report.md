# experimental.enable_x64 fails for random.uniform

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/5532](https://github.com/jax-ml/jax/issues/5532)
- Opened: 2021-01-27T18:48:43Z
- Closed: 2021-03-01T20:33:19Z
- Labels: `bug`

## Reported behavior

Short repro:
```python
from jax import random
from jax.experimental import disable_x64, enable_x64

f = lambda: random.uniform(random.PRNGKey(0), (10,), 'float64', -1, 1)
with disable_x64():
    f()
with enable_x64():
    f()
    f()  # <--- fails
```
Failure:
```pytb
Traceback (most recent call last):
  File "tmp.py", line 11, in <module>
    f()  # <--- fails
  File "tmp.py", line 4, in <lambda>
    f = lambda: random.uniform(random.PRNGKey(0), (10,), 'float64', -1, 1)
  File "/Users/vanderplas/github/google/jax/jax/_src/random.py", line 365, in uniform
    return _uniform(key, shape, dtype, minval, maxval)  # type: ignore
jax._src.traceback_util.FilteredStackTrace: RuntimeError: Invalid argument: Argument does not match host shape or layout of computation parameter 1: want s64[], got s32[]: while running replica 0 and partition 0 of a replicated computation (other replicas may have failed as well).

The stack trace above excludes JAX-internal frames.
The following is the original exception that occurred, unmodified.

--------------------

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "tmp.py", line 11, in <module>
    f()  # <--- fails
  File "tmp.py", line 4, in <lambda>
    f = lambda: random.uniform(random.PRNGKey(0), (10,), 'float64', -1, 1)
  File "/Users/vanderplas/github/google/jax/jax/_src/random.py", line 365, in uniform
    return _uniform(key, shape, dtype, minval, maxval)  # type: ignore
  File "/Users/vanderplas/github/google/jax/jax/_src/traceback_util.py", line 139, in reraise_with_filtered_traceback
    return fun(*args, **kwargs)
  File "/Users/vanderplas/github/google/jax/jax/api.py", line 401, in f_jitted
    return cpp_jitted_f(context, *args, **kwargs)
RuntimeError: Invalid argument: Argument does not match host shape or layout of computation parameter 1: want s64[], got s32[]: while running replica 0 and partition 0 of a replicated computation (other replicas may have failed as well).
```

## Trigger

The same jitted random function is called under different enable_x64 contexts.

## Relevant entities

thread-local x64 state; C++ jit cache key; compiled parameter dtype; runtime argument dtype

## Implemented relation

The x64 mode used to canonicalize arguments matches the mode used to select or compile the cached executable on every call.

## Pre-state

A call under enable_x64 requires 64-bit scalars and an x64-specialized executable.

## Binding transition

The Python context changes a local value, but the C++ jit reads state captured or initialized by an earlier call.

## Post-state

A cached executable expects s64 while the call prepares s32, or vice versa.

## Validity obligation

Dynamic thread-local configuration participates consistently in canonicalization, cache identity, and C++ dispatch.

## Invalid decision

The C++ jit treated jax_enable_x64 as effectively session-global instead of reading the calling thread's current context.

## Physical effect

Executable parameter layout/dtype and the supplied host scalar disagree.

## Observed consequence

random.uniform fails with 'want s64[], got s32[]' after context changes.

## Repair site

jax/config.py, jax/experimental/x64_context.py, and the jax_jit thread-local state bridge

## Repair

PR 5604 makes Python x64 state thread-local and keys caches with it; PR 5851 moves the authoritative context state into the C++ jit bridge and adds alternating-context regressions.

## Why the repair works

The current call's mode drives both cached executable selection and dtype canonicalization across the Python/C++ boundary.

## Causal basis

The issue localizes the failure to C++ jit state, and the closing patch explicitly moves that state to C++ and names #5532.

## Analysis implication

A configuration-sensitive compilation check needs thread identity, dynamic flag value, cache key, canonicalized dtype, and executable signature.

## Fix patches

- [`fixes/jax-5532-pr5604.patch`](fixes/jax-5532-pr5604.patch) — https://github.com/jax-ml/jax/pull/5604
- [`fixes/jax-5532-pr5851.patch`](fixes/jax-5532-pr5851.patch) — https://github.com/jax-ml/jax/pull/5851
