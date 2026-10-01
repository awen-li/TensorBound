# Sticking an array to it's own device does not work

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/2225](https://github.com/jax-ml/jax/issues/2225)
- Opened: 2020-02-14T03:45:58Z
- Closed: 2022-08-16T17:58:11Z
- Labels: `bug`, `P0 (urgent)`, `NVIDIA GPU`

## Reported behavior

Here I create two arrays and stick one of them to a different GPU:
```python
import jax

jax.devices()
[GpuDevice(id=0), GpuDevice(id=1)]

a = jax.random.normal(jax.random.PRNGKey(1), (2, 3))
b = jax.random.normal(jax.random.PRNGKey(2), (2, 3))
(a+b).device_buffer.device()
GpuDevice(id=0)

b = jax.device_put(b, jax.devices()[1]); b.device_buffer.device()
GpuDevice(id=1)

(b+a).device_buffer.device()
GpuDevice(id=1)
```

Now let's stick `a` to it's current GPU:
```python
a = jax.device_put(a, jax.devices()[0]); a.device_buffer.device()
GpuDevice(id=0)

(b+a).device_buffer.device()  # Should get an error here!
GpuDevice(id=1)
```
Only when I move it around, I get the device placement error:
```python
a = jax.device_put(a, jax.devices()[1])
a = jax.device_put(a, jax.devices()[0])
(b+a).device_buffer.device()
Traceback (most recent call last):
  File "/opt/intellij-ue-beta/plugins/python/helpers/pydev/_pydevd_bundle/pydevd_exec2.py", line 3, in Exec
    exec(exp, global_vars, local_vars)
  File "<input>", line 1, in <module>
  File "/usr/local/google/_blaze_romann/513cef43ffae8d7478c0c7058e6a84e4/execroot/google3/blaze-out/haswell-cuda101-py3-opt/bin/third_party/py/neural_tangents/predict_test_gpu_x32.runfiles/google3/third_party/py/jax/numpy/lax_numpy.py", line 3372, in deferring_binary_op
    return binary_op(self, other)
  File "/usr/local/google/_blaze_romann/513cef43ffae8d7478c0c7058e6a84e4/execroot/google3/blaze-out/haswell-cuda101-py3-opt/bin/third_party/py/neural_tangents/predict_test_gpu_x32.runfiles/google3/third_party/py/jax/numpy/lax_numpy.py", line 404, in fn
    return lax_fn(x1, x2) if x1.dtype != bool_ else bool_lax_fn(x1, x2)
  File "/usr/local/google/_blaze_romann/513cef43ffae8d7478c0c7058e6a84e4/execroot/google3/blaze-out/haswell-cuda101-py3-opt/bin/third_party/py/neural_tangents/predict_test_gpu_x32.runfiles/google3/third_party/py/jax/lax/lax.py", line 274, in add
    return add_p.bind(x, y)
  File "/usr/local/google/_blaze_romann/513cef43ffae8d7478c0c7058e6a84e4/execroot/google3/blaze-out/haswell-cuda101-py3-opt/bin/third_party/py/neural_tangents/predict_test_gpu_x32.runfiles/google3/third_party/py/jax/core.py", line 179, in bind
    return self.impl(*args, **kwargs)
  File "/usr/local/google/_blaze_romann/513cef43ffae8d7478c0c7058e6a84e4/execroot/google3/blaze-out/haswell-cuda101-py3-opt/bin/third_party/py/neural_tangents/predict_test_gpu_x32.runfiles/google3/third_party/py/jax/interpreters/xla.py", line 158, in apply_primitive
    compiled_fun = xla_primitive_callable(prim, *map(arg_spec, args), **params)
  File "/usr/local/google/_blaze_romann/513cef43ffae8d7478c0c7058e6a84e4/execroot/google3/blaze-out/haswell-cuda101-py3-opt/bin/third_party/py/neural_tangents/predict_test_gpu_x32.runfiles/google3/third_party/py/jax/interpreters/xla.py", line 164, in xla_primitive_callable
    device = _device_from_arg_devices(arg_devices)
  File "/usr/local/google/_blaze_romann/513cef43ffae8d7478c0c7058e6a84e4/execroot/google3/blaze-out/haswell-cuda101-py3-opt/bin/third_party/py/neural_tangents/predict_test_gpu_x32.runfiles/google3/third_party/py/jax/interpreters/xla.py", line 206, in _device_from_arg_devices
    raise ValueError(msg.format(", ".join(map(str, devices))))
ValueError: primitive arguments must be colocated on the same device, got gpu:1, gpu:0
```

## Trigger

An array resides uncommitted on GPU 0, the program explicitly applies device_put(array, GPU 0), then combines it with an array committed to GPU 1.

## Relevant entities

DeviceArray; physical device buffer; logical _device commitment; device_put target; primitive colocation check

## Implemented relation

An explicit device_put binds the result logically to its requested device even when its buffer already resides there.

## Pre-state

The array's buffer is physically on the default GPU 0, but its _device field is None, so later computations may move it.

## Binding transition

The old same-device branch compares only device_buffer.device() with the requested device and returns the original array unchanged.

## Post-state

The buffer remains on GPU 0 and the array remains logically uncommitted despite the explicit GPU 0 placement request.

## Validity obligation

A physical no-copy decision preserves the bytes while still applying the requested logical commitment transition.

## Invalid decision

_copy_device_array_to_device treated matching physical locations as proof that no state change was required and skipped the _device update.

## Physical effect

When combined with a GPU 1 committed array, the supposedly GPU 0 committed operand is treated as movable and follows the GPU 1 operand.

## Observed consequence

The mixed-device addition executes on GPU 1 instead of raising the expected colocation error for operands committed to GPU 0 and GPU 1.

## Repair site

jax/interpreters/xla.py::_copy_device_array_to_device

## Repair

Commit d315564 distinguishes physical residency from commitment: when the buffer already resides on the requested device but x._device differs, it reuses the buffer and constructs the result with the requested device commitment. It also re-enables the same-device commitment assertion.

## Why the repair works

The no-copy optimization now changes only physical transfer behavior; explicit placement still produces a DeviceArray whose _device field records the target, so mixed committed operands reach the colocation check.

## Causal basis

A maintainer verifies the report's expected behavior after the placement work around PR 2882. The immediately following landed commit d315564 states the report's exact same-device device_put failure, changes that branch, and restores the corresponding commitment test.

## Analysis implication

Placement analysis needs both physical buffer location and logical commitment, plus the requested target and later colocation decision.

## Fix patches

- [`fixes/jax-2225-commit-d315564.patch`](fixes/jax-2225-commit-d315564.patch) — https://github.com/jax-ml/jax/commit/d315564ebf9097625d23dac7ead018253cfb3bee
