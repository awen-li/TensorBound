# When lowering with repeated inputs the compiled function fails only after a second call.

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/13918](https://github.com/jax-ml/jax/issues/13918)
- Opened: 2023-01-09T06:15:08Z
- Closed: 2024-02-21T00:14:31Z
- Labels: `bug`

## Reported behavior

### Description

I was incorrectly setting my training loop and compiling ahead of time, I used by mistake a reference to the same pytree twice and got errors in an unexpected way when calling the compiled function. I managed to reduce to the following:

```python
In [1]: import jax
In [2]: import jaxlib
In [3]: import jax.numpy as jnp
In [4]: jax.__version__
Out[4]: '0.4.1'
In [5]: jaxlib.__version__
Out[5]: '0.4.1'
In [6]: x = jnp.array(0)
In [7]: y = x
In [8]: def f(x, y):
   ...:     return x
   ...:
In [9]: f_ = jax.jit(f).lower(x, y).compile()
In [10]: f_(x, y)
Out[10]: Array(0, dtype=int32, weak_type=True)
In [11]: f_(x, y)
2023-01-09 06:52:44.191879: E external/org_tensorflow/tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.cc:2163] Execution of replica 0 failed: INVALID_ARGUMENT: Executable expected 1 arguments but got 2
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Input In [11], in <cell line: 1>()
----> 1 f_(x, y)

File ~/miniconda3/lib/python3.9/site-packages/jax/_src/stages.py:504, in Compiled.__call__(self, *args, **kwargs)
    502 def __call__(self, *args, **kwargs):
    503   if self._cpp_call is not None:
--> 504     return self._cpp_call(*args, **kwargs)
    506   outs, _ = Compiled.call(self._params, *args, **kwargs)
    507   return outs

ValueError: INVALID_ARGUMENT: Executable expected 1 arguments but got 2
```
I was originally getting a similar error message but saying `buffers` instead of `arguments`.

I solved my problem before detecting the repeated reference by using `keep_unused=True`, but nothing was telling me that the function was removing an argument, and the function failed only at the second call, so, it was particularly hard to catch.

Here is an example in Colab: https://colab.research.google.com/gist/jorgeecardona/6c2c32a26e0176317b4ec204aa7bb07f/bug-example.ipynb#scrollTo=bC0Ceqwetddh

I would expect either a clear message of which variables are removed when lowering, and to fail on the first call to the compiled function (unless I am missing an extra step here, I don't see a clear reason why the function fails only at the second call)

### What jax/jaxlib version are you using?

0.4.1

### Which accelerator(s) are you using?

GPU

### Additional system info

python 3.9.12 in conda, linux 6, debian sid

### NVIDIA GPU info

```
Mon Jan  9 07:05:52 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.108.03   Driver Version: 510.108.03   CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Quadro T2000        On   | 00000000:01:00.0 Off |                  N/A |
| N/A   49C    P8     5W /  N/A |   3588MiB /  4096MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      8849      G   /usr/lib/xorg/Xorg                  4MiB |
|    0   N/A  N/A     61953      G   /usr/lib/xorg/Xorg                  4MiB |
|    0   N/A  N/A    144074      C   .../miniconda3/bin/python3.9     3575MiB |
+-----------------------------------------------------------------------------+
```

## Trigger

A lowered function receives the same array as x and y, returns only x, and is invoked twice after compilation; y is absent from the executable signature.

## Relevant entities

flattened Python arguments; unused-argument elimination; kept_var_idx; kept_var_bitvec; Python cache-miss call; C++ pjit fast path; compiled executable parameters

## Implemented relation

The logical call signature maps each flattened argument position to either one retained executable parameter or a removed unused argument, and every dispatch path applies the same map.

## Pre-state

The logical call has two flattened arguments, while lowering retains x and removes unused y so that the executable accepts one parameter.

## Binding transition

The first invocation ran through the Python cache-miss handler and then cached C++ fast-path metadata that omitted the retained-argument bit vector.

## Post-state

The cached call knew the two-argument Python signature and the one-parameter executable but lacked the map selecting x from the flattened inputs.

## Validity obligation

A cached dispatch preserves the lowering-time retained-argument map and filters logical inputs to exactly the executable's parameter sequence before execution.

## Invalid decision

The old fast-path metadata treated the Python argument list as if every position survived lowering, even when the executable had pruned an unused input.

## Physical effect

The second invocation submitted two argument buffers to an executable compiled for one parameter.

## Observed consequence

The first call returned normally, while the second failed with `Executable expected 1 arguments but got 2`.

## Repair site

jax/experimental/pjit.py::_cpp_pjit.cache_miss; jax/interpreters/pxla.py::MeshExecutable.create_cpp_call and _MeshExecutableFastpathData

## Repair

Commit 3b35e981 makes cache-miss helpers retain args_flat, derives kept_var_bitvec from the executable's kept_var_idx, stores that bit vector in _MeshExecutableFastpathData, and requires the compatible C++ pjit interface. Its tests cover both pruned and preserved unused arguments.

## Why the repair works

The C++ fast path receives the same positional selection computed during lowering, so later calls pass only retained inputs unless keep_unused explicitly preserves all of them.

## Causal basis

JAX 0.4.1 does not contain commit 3b35e981, while 0.4.2 does. The commit adds the missing retained-argument bit vector and a regression that passes the same unused array in five argument positions, invokes the function a second time specifically to enter C++ dispatch, and verifies the pruned executable keeps only the used position. The issue exhibits the same first-call/second-call split and executable arity mismatch; later reproduction confirms the original program succeeds on 0.4.23 and 0.4.24.

## Analysis implication

Compiled-call analysis needs the logical argument tree, flattened positions, dead-argument decision, retained-position map, cached-call state, and physical executable signature together.

## Fix patches

- [`fixes/jax-13918-commit-3b35e98.patch`](fixes/jax-13918-commit-3b35e98.patch) — https://github.com/jax-ml/jax/commit/3b35e9811dac9db7364481fe0806f57af1403f1d
