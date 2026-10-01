# Crash with jit of ordered io_callback under shard_map and then no shard_map on CPU

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/25671](https://github.com/jax-ml/jax/issues/25671)
- Opened: 2024-12-23T17:09:42Z
- Closed: 2025-01-16T19:24:27Z
- Labels: `bug`

## Reported behavior

### Description

On my M1 Mac, with jax==0.4.39.dev20241223 and jaxlib==0.4.39.dev20241223 and Python 3.12.6, the following code:
```python
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

mesh = jax.make_mesh((4,), 'i')

def empty_callback(x):
  return

def _f(x, y):
  jax.experimental.io_callback(empty_callback, (), x, ordered=True)
  return x + y[..., jnp.newaxis]

f = jax.jit(shard_map(
    _f, mesh, in_specs=(P(None, 'i'), P(None)), out_specs=P(None, 'i')))

print(f(jnp.zeros((2, 16)), jnp.ones(2)))

print(jax.jit(_f)(jnp.zeros((2, 16)), jnp.ones(2)))
```
crashes when running the last line, with:
```
F external/xla/xla/pjrt/pjrt_client.h:1128] Check failed: on_device_shape().has_layout()
Abort trap: 6
```

When I update `test_empty_io_callback_under_shard_map` in `tests/pjit_test.py` at HEAD to match this code -- see https://github.com/jax-ml/jax/pull/25670 -- the test fails in GitHub's CI and on my Mac.  For example, see https://github.com/jax-ml/jax/actions/runs/12470134880/job/34804668225?pr=25670 :
```
Fatal Python error: Aborted

Current thread 0x00007a4bd1b93000 (most recent call first):
  File "/__w/jax/jax/jax/_src/array.py", line 644 in _value
  File "/__w/jax/jax/jax/_src/profiler.py", line 333 in wrapper
  File "/__w/jax/jax/jax/_src/array.py", line 1113 in shard_sharded_device_array_slow_path
  File "/__w/jax/jax/jax/_src/array.py", line 1170 in _array_shard_arg
  File "/__w/jax/jax/jax/_src/array.py", line 1209 in _token_shard_arg
  File "/__w/jax/jax/jax/_src/interpreters/pxla.py", line 135 in shard_args
  File "/__w/jax/jax/jax/_src/profiler.py", line 333 in wrapper
  File "/__w/jax/jax/jax/_src/dispatch.py", line 510 in _batched_device_put_impl
  File "/__w/jax/jax/jax/_src/core.py", line 941 in process_primitive
  File "/__w/jax/jax/jax/_src/core.py", line 468 in bind_with_trace
  File "/__w/jax/jax/jax/_src/core.py", line 463 in bind
  File "/__w/jax/jax/jax/_src/api.py", line 2294 in device_put
  File "/__w/jax/jax/jax/_src/dispatch.py", line 136 in get_token_input
  File "/__w/jax/jax/jax/_src/interpreters/pxla.py", line 1249 in <listcomp>
  File "/__w/jax/jax/jax/_src/interpreters/pxla.py", line 1248 in _add_tokens_to_inputs
  File "/__w/jax/jax/jax/_src/interpreters/pxla.py", line 1288 in __call__
  File "/__w/jax/jax/jax/_src/profiler.py", line 333 in wrapper
  File "/__w/jax/jax/jax/_src/pjit.py", line 1688 in _pjit_call_impl_python
  File "/__w/jax/jax/jax/_src/pjit.py", line 196 in _python_pjit_helper
```

The code fails with the same error if the last two lines are switched -- i.e., if the not-shard-mapped `_f` is run before the shard-mapped `f`.




### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.4.39.dev20241223
jaxlib: 0.4.39.dev20241223
numpy:  2.2.1
python: 3.12.6 (main, Sep  9 2024, 21:36:32) [Clang 18.1.8 ]
device info: cpu-1, 1 local devices"
process_count: 1
platform: uname_result(system='Darwin', node='jburnim-macbookpro2.roam.internal', release='23.6.0', version='Darwin Kernel Version 23.6.0: Thu Sep 12 23:35:29 PDT 2024; root:xnu-10063.141.1.701.1~1/RELEASE_ARM64_T6000', machine='arm64')
```

## Trigger

An ordered io_callback is executed through shard_map and then through a single-device jitted function, forcing its sequencing token between different shardings.

## Relevant entities

ordered-effect token; multi-device sharding; single-device sharding; underlying token buffer; subsequent callback

## Implemented relation

The token orders a later effect after completion of the earlier callback even when the next executable uses a different sharding.

## Pre-state

The token represents completion of an ordered callback under one sharding.

## Binding transition

The token shard-argument handler attempts to reshard its underlying buffer through the ordinary array device-put path.

## Post-state

The transfer expects an on-device array layout that the token buffer does not have.

## Validity obligation

A token handoff across shardings preserves effect completion and constructs a representation valid for the destination executable.

## Invalid decision

_token_shard_arg delegated token buffers to _array_shard_arg as if they were ordinary arrays with device layouts.

## Physical effect

The array resharding path queries a missing on-device layout on the token buffer.

## Observed consequence

The second jitted ordered callback aborts at pjrt_client.h with on_device_shape().has_layout().

## Repair site

jax/_src/array.py token shard-argument handling

## Repair

PR 25930 waits for each incoming token, creates an empty Boolean value on the requested target layout, and adds the shard-map-to-single-device regression.

## Why the repair works

Waiting provides the effects barrier, and the newly placed empty value supplies a valid destination-sharded token representation without treating the old token as transferable array data.

## Causal basis

The failing stack enters _token_shard_arg and then ordinary array resharding, while the linked patch replaces that path with an explicit wait and target placement and adds the reported transition as a test.

## Analysis implication

An ordered-effect check needs token provenance, source and destination shardings, completion before handoff, and the target token representation.

## Fix patches

- [`fixes/jax-25671-pr25930.patch`](fixes/jax-25671-pr25930.patch) — https://github.com/jax-ml/jax/pull/25930
