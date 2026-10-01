# `host_callback.call` fails on multi-gpu machine

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/5577](https://github.com/jax-ml/jax/issues/5577)
- Opened: 2021-02-01T02:05:35Z
- Closed: 2024-09-13T09:36:45Z
- Labels: `bug`, `P1 (soon)`, `needs info`, `NVIDIA GPU`

## Reported behavior

If I run the following code:

```
from jax.experimental import host_callback
import numpy as np
from jax import pmap, jit, partial, ShapeDtypeStruct


def host_fn(x):
    return x


x = np.ones(4, dtype=np.float32)
host_callback.call(host_fn, x, result_shape=x)
```

on a 2-gpu machine then it crashes with the error message
```
2021-01-31 17:53:40.778121: F external/org_tensorflow/tensorflow/compiler/xla/service/gpu/outfeed_thunk.cc:56] Check failed: ShapeUtil::Compatible(config_.input_shape, outfeed_buffers->shape()) XLA program outfeed request of shape (f32[4]) did not match the runtime's outfeed buffer of shape u32[2]
```

If I run with one GPU (by setting `CUDA_VISIBLE_DEVICES=0`) it finishes with no errors. Is there something I've missed in the documentation for host_callback about how it should be used on multi-device setups?

I ran both with the full debug information
`CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=outfeed_receiver=3,host_callback=3,outfeed_receiver_py=3,outfeed_thunk=3,xfeed_manager=3 python test_2.py --verbosity=2 2> test_output_one_gpu.txt` if that's helpful.
[test_output_one_gpu.txt](https://github.com/google/jax/files/5901442/test_output_one_gpu.txt)
[test_output_two_gpu.txt](https://github.com/google/jax/files/5901445/test_output_two_gpu.txt)

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

A host_callback.call executes in a process exposing two GPUs.

## Observed implementation decision

The XLA GPU outfeed request and runtime outfeed buffer are associated with incompatible shapes or device queues.

## Physical effect

The callback transfer consumes an outfeed buffer with a different shape than the compiled request.

## Observed consequence

XLA aborts with a shape-compatibility check failure on a multi-GPU machine.

## Reported proposal or workaround

Maintainers recommend pure_callback or debug.callback and describe a future CustomCall implementation; host_callback was later removed.

## Repair trace

The issue discussion states that the GPU CustomCall mechanism had landed but was not yet used by host_callback, and the issue later closed without confirmation. The inspected history does not provide a host_callback repair and matching fixed regression for this multi-GPU path.

## Archived evidence

- Raw report: `data/raw/github/jax/5577/report.json`
- SHA-256: `8c4c7c5879aa8e002bfb22c14d62b3e540950cac790a7d40549d2b4f52d33b9b`
