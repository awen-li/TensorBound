# JAX/TF2 on multi-GPU silently corrupted computations

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/3802](https://github.com/jax-ml/jax/issues/3802)
- Opened: 2020-07-20T15:44:29Z
- Closed: 2021-01-14T14:36:47Z
- Labels: `bug`, `question`

## Reported behavior

When using 1 or 2 GPUs there's no problem. When using 3 or more GPUs the issue manifests itself resulting in corrupted computations. This was tested on 4 V100 GPUs. There's a workaround (see commented code).

Code for reproduction:
```python
# The bug only shows for physical GPUs, and only when there 3 or more GPUs.
# It silently corrupts the results with no warning or error message.

import jax
import jax.numpy as jn
import numpy as np
import tensorflow as tf

# The following is a workaround from prevents the issue:
# tf.config.experimental.set_visible_devices([], "GPU")

reproduce_bug = True
noop = jax.pmap(lambda x: x)
n = jax.device_count()


def check(delta):
    v = noop(jn.ones((n, 1)) * delta).sum()
    assert v == n * delta, (v, n * delta)


check(1)
if reproduce_bug:
    data = tf.data.Dataset.from_tensor_slices(dict(x=np.random.uniform(-1, 1, size=(16, 1))))

check(2)
# Output is
# Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#  File "<stdin>", line 3, in check
# AssertionError: (DeviceArray(7., dtype=float32), 8)
```

## Trigger

TensorFlow and JAX alternate GPU work on one Python thread while each shared library contains its own StreamExecutor thread-local CUDA-context cache.

## Relevant entities

TensorFlow StreamExecutor; JAX StreamExecutor; thread-local context cache; actual CUDA current context; ScopedActivateContext; GPU allocation and kernel

## Implemented relation

At every outermost StreamExecutor activation, the actual CUDA current context matches the GpuContext required by the entering library.

## Pre-state

JAX's private thread-local cache records its last context after an earlier JAX scope completes.

## Binding transition

TensorFlow changes the thread's CUDA context through its separate StreamExecutor copy, leaving JAX's private cache unchanged.

## Post-state

The actual CUDA context belongs to TensorFlow while JAX's cache still claims that the requested JAX context is current.

## Validity obligation

An outermost library boundary reasserts the requested CUDA context; cached equality is used only within nested scopes controlled by the same StreamExecutor instance.

## Invalid decision

ScopedActivateContext trusted its private cached context at depth zero, where arbitrary foreign GPU code could have changed the process-level CUDA state.

## Physical effect

JAX submits a kernel or buffer operation under a CUDA context different from the context that owns its device pointers.

## Observed consequence

The multi-GPU run executes on the wrong GPU and can report an invalid global read or a corrupted aggregate after TensorFlow GPU activity.

## Repair site

tensorflow/stream_executor/cuda/cuda_driver.cc::ScopedActivateContext

## Repair

Commit 1e336d3 makes every depth-zero activation call cuCtxSetCurrent for the requested context and refresh the thread-local state; it retains cache-based switching for nested scopes and adds a test that changes the context externally between two activations.

## Why the repair works

Foreign code can no longer make an outermost StreamExecutor activation rely on stale private state, while nested activations still restore and reuse contexts under one cache's control.

## Causal basis

The issue discussion derives the two-private-cache failure and references commit 1e336d3. The commit states that it fixes JAX #3802 when incorporated, changes the exact outermost activation decision, and tests an external context switch between scopes.

## Analysis implication

Cross-library GPU analysis needs the actual thread context, each library's private cache, activation depth, buffer-owning context, and kernel submission context together.

## Fix patches

- [`fixes/jax-3802-tensorflow-1e336d3.patch`](fixes/jax-3802-tensorflow-1e336d3.patch) — https://github.com/tensorflow/tensorflow/commit/1e336d3ef0706792772b605a1fed0135f5cc5cfe
