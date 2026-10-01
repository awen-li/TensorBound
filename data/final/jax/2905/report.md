# Ops involving only CPU tensors run on device, not CPU

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/2905](https://github.com/jax-ml/jax/issues/2905)
- Opened: 2020-04-30T18:58:46Z
- Closed: 2020-05-01T00:21:11Z
- Labels: `bug`

## Reported behavior

In the code below running jax op that has CPU backed inputs runs on device, (which for my use case OOMs):
```
def CpuArray(numpy_array):
  return jax.device_put(numpy_array, device=jax.devices(backend='cpu')[0])

data = CpuArray(np.ones([100, 4], dtype=np.float32))
slice_indexes = CpuArray(np.zeros([100], dtype=np.int32))
# Summing the data runs on TPU.
summed_data = jnp.sum(data[slice_indexes])
# So does slicing it.
sliced_data = data[100]

print("data device", data.device_buffer.device())
print("sliced indexes", slice_indexes.device_buffer.device())
print("sliced data", sliced_data.device_buffer.device())
print("summed data", summed_data.device_buffer.device())
```

Prints:

```
data device cpu:0
sliced indexes device cpu:0
Both output arrays are on TPU:
sliced data device TPU_0(host=0,(0,0,0,0))
summed data device TPU_0(host=0,(0,0,0,0))
```

## Trigger

Indexing or reducing a CPU-backed DeviceArray while the process default backend is GPU or TPU.

## Relevant entities

DeviceArray; source device; optional requested device; default backend; device buffer

## Implemented relation

An unspecified placement request preserves the device of an existing DeviceArray.

## Pre-state

The input and index buffers reside on CPU by explicit user placement.

## Binding transition

The old device_put path interprets device=None as the default device and copies the existing buffer there.

## Post-state

A CPU-only operation's intermediate and result are bound to GPU or TPU.

## Validity obligation

device=None means no placement change for an already committed device array.

## Invalid decision

_copy_device_array_to_device compared the source with a default target even when the caller supplied no target.

## Physical effect

The buffer crosses backend through host memory and executes on the default accelerator.

## Observed consequence

CPU slicing and summation run on TPU/GPU and can exhaust accelerator memory.

## Repair site

jax/interpreters/xla.py::_copy_device_array_to_device

## Repair

PR 2907 returns the input unchanged when device is None, performs copies only for explicit targets, and adds multi-backend indexing and sum regressions.

## Why the repair works

Implicit normalization no longer overrides the established source-device binding.

## Causal basis

The issue identifies asarray/device_put, and the patch changes device=None semantics with the reported cross-backend tests.

## Analysis implication

Placement analysis needs to distinguish an absent target from an explicitly selected default target.

## Fix patches

- [`fixes/jax-2905-pr2907.patch`](fixes/jax-2905-pr2907.patch) — https://github.com/jax-ml/jax/pull/2907
