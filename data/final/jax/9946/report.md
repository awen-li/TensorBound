# Different outputs on subsequent calls of irfft2 with the same input on GPU with double precision

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/9946](https://github.com/jax-ml/jax/issues/9946)
- Opened: 2022-03-17T21:40:17Z
- Closed: 2022-04-04T21:38:57Z
- Labels: `bug`

## Reported behavior

Hello,

I found a weird issue when calling irfft2 on a complex array on GPU (e.g. v100 but not restricted to it) with double precision. It looks like calling irfft2(x) changes the value of x for subsequent irfft2() or numpy.real() calls, while calling x on its own shows the initial x (as expected), see below. I'm running JAX 0.3.2 with CUDA/11.1.1-GCC-10.2.0  cuDNN/8.0.5.39-CUDA-11.1.1.

Thanks!



```
Python 3.10.2 | packaged by conda-forge | (main, Jan 28 2022, 16:48:22) [GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import jax.numpy as jnp
>>> from jax.config import config
>>> config.update("jax_enable_x64", True)
>>> x = jnp.array([[1.0,2.0],[3.0,4.0]],dtype=jnp.float64)*(1+1j)
>>> x
DeviceArray([[1.+1.j, 2.+2.j],
             [3.+3.j, 4.+4.j]], dtype=complex128)
>>> jnp.fft.irfft2(x)
DeviceArray([[ 2.5, -0.5],
             [-1. ,  0. ]], dtype=float64)
>>> jnp.real(x)
DeviceArray([[ 4.,  6.],
             [-2., -2.]], dtype=float64)
>>> x
DeviceArray([[1.+1.j, 2.+2.j],
             [3.+3.j, 4.+4.j]], dtype=complex128)
>>>
>>>
>>> jnp.fft.irfft2(x)
DeviceArray([[ 1.5, -0.5],
             [ 3.5, -0.5]], dtype=float64)
>>>
>>> jnp.real(x)
DeviceArray([[2., 4.],
             [6., 8.]], dtype=float64)
>>> x
DeviceArray([[1.+1.j, 2.+2.j],
             [3.+3.j, 4.+4.j]], dtype=complex128)
>>>
```

## Trigger

jnp.fft.irfft2 is called repeatedly on the same complex128 GPU array.

## Relevant entities

complex128 input buffer; float64 output buffer; CuFFT C2R execution; scratch copy; subsequent call

## Implemented relation

An out-of-place FFT may read its input without changing bytes that remain bound to the JAX array after the call.

## Pre-state

The JAX array owns a valid complex128 input buffer used by multiple operations.

## Binding transition

CUDAFft executes a complex-to-real CuFFT without the protective scratch copy used for the analogous single-precision operation.

## Post-state

CuFFT overwrites input storage although JAX still exposes that storage as the unchanged source array.

## Validity obligation

Backends with destructive C2R behavior copy every affected input precision before invoking CuFFT.

## Invalid decision

The scratch-copy guard matched complex<float>-to-float but omitted complex<double>-to-double.

## Physical effect

The first inverse FFT mutates the complex128 device buffer used by later inverse FFT and real-part reads.

## Observed consequence

Successive irfft2 calls with the same JAX input produce different outputs.

## Repair site

tensorflow/stream_executor/cuda/cuda_fft.cc::CUDAFft::DoFftInternal

## Repair

TensorFlow/XLA commit d502c624 extends the existing non-aliasing scratch-copy condition to complex<double> and double. JAX PR 10133 records the fixed jaxlib version and tests repeated float32 and float64 IRFFTs.

## Why the repair works

CuFFT operates on a private scratch copy, so its destructive behavior cannot change the buffer still owned by the logical JAX input.

## Causal basis

The maintainer identifies the known CuFFT C2R mutation and missing double-precision workaround; the cross-referenced upstream commit changes that exact guard, and the JAX regression repeats the operation on one input.

## Analysis implication

An external-library call check needs library mutation semantics, input/output alias identity, element precision, scratch-copy policy, and post-call input liveness.

## Fix patches

- [`fixes/jax-9946-pr10133.patch`](fixes/jax-9946-pr10133.patch) — https://github.com/jax-ml/jax/pull/10133
- [`fixes/tensorflow-9946-d502c624.patch`](fixes/tensorflow-9946-d502c624.patch) — https://github.com/tensorflow/tensorflow/commit/d502c62416293ff8a09b9f5bce4f5527fa9568eb
