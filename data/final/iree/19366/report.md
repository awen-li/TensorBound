# PJRT plugin fails to execute in JAX while multiple GPUs are provided

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/19366](https://github.com/iree-org/iree/issues/19366)
- Opened: 2024-12-04T03:53:15Z
- Closed: 2024-12-06T22:51:15Z
- Labels: `bug 🐞`

## Reported behavior

### What happened?

As mentioned in https://github.com/iree-org/iree/pull/19279#issuecomment-2506481128 and https://github.com/iree-org/iree/pull/19279#issuecomment-2516087713, PJRT plugin will abort on `execute_sharded` in JAX if more than one GPU is available.

```log
+ python test/test_add.py
WARNING:jax._src.xla_bridge:Platform 'iree_vulkan' is experimental and not all JAX functionality may be correctly supported!
[IREE-PJRT] DEBUG: Using IREE compiler binary: /home/esaimana/actions-runner-2/_work/iree/iree/.venv/lib/python3.11/site-packages/iree/compiler/_mlir_libs/libIREECompiler.so
[IREE-PJRT] DEBUG: Compiler Version: 3.1.0.dev+da45650503b955e[35](https://github.com/iree-org/iree/actions/runs/12031928938/job/33543044860?pr=19279#step:7:36)a45ea8e34b5f10b30a2f912 @ da45650503b955e35a45ea8e34b5f10b30a2f912 (API version 1.4)
[IREE-PJRT] DEBUG: Partitioner was not enabled. The partitioner can be enabled by setting the 'PARTITIONER_LIB_PATH' config var ('IREE_PJRT_PARTITIONER_LIB_PATH' env var)
[IREE-PJRT] DEBUG: Vulkan driver created
Traceback (most recent call last):
  File "/home/esaimana/actions-runner-2/_work/iree/iree/integrations/pjrt/test/test_add.py", line 7, in <module>
    a = jax.numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9])
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/esaimana/actions-runner-2/_work/iree/iree/.venv/lib/python3.11/site-packages/jax/_src/numpy/lax_numpy.py", line 5592, in asarray
    return array(a, dtype=dtype, copy=bool(copy), order=order, device=device)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/esaimana/actions-runner-2/_work/iree/iree/.venv/lib/python3.11/site-packages/jax/_src/numpy/lax_numpy.py", line 5426, in array
    out_array: Array = lax_internal._convert_element_type(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/esaimana/actions-runner-2/_work/iree/iree/.venv/lib/python3.11/site-packages/jax/_src/lax/lax.py", line 587, in _convert_element_type
    return convert_element_type_p.bind(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/esaimana/actions-runner-2/_work/iree/iree/.venv/lib/python3.11/site-packages/jax/_src/lax/lax.py", line 2981, in _convert_element_type_bind
    operand = core.Primitive.bind(convert_element_type_p, operand,
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/esaimana/actions-runner-2/_work/iree/iree/.venv/lib/python3.11/site-packages/jax/_src/core.py", line 4[38](https://github.com/iree-org/iree/actions/runs/12031928938/job/33543044860?pr=19279#step:7:39), in bind
    return self.bind_with_trace(find_top_trace(args), args, params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/esaimana/actions-runner-2/_work/iree/iree/.venv/lib/python3.11/site-packages/jax/_src/core.py", line 442, in bind_with_trace
    out = trace.process_primitive(self, map(trace.full_raise, args), params)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/esaimana/actions-runner-2/_work/iree/iree/.venv/lib/python3.11/site-packages/jax/_src/core.py", line 955, in process_primitive
    return primitive.impl(*tracers, **params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/esaimana/actions-runner-2/_work/iree/iree/.venv/lib/python3.11/site-packages/jax/_src/dispatch.py", line 91, in apply_primitive
    outs = fun(*args)
           ^^^^^^^^^^
jaxlib.xla_extension.XlaRuntimeError: INVALID_ARGUMENT: Expected args to execute_sharded_on_local_devices to have 4 shards, got: [1]
```

### Steps to reproduce your issue

1. Fetch #19279 to local
2. Make sure that multiple GPUs are available
3. Build vulkan PJRT plugin ([README](https://github.com/iree-org/iree/blob/main/integrations/pjrt/README.md)) (or cuda/rocm, if multiple nv/amd GPUs are available)
4. Run integrations/pjrt/test/test_add.py via `JAX_PLATFORMS=iree_vulkan`
5. See error

### What component(s) does this issue relate to?

_No response_

### Version information

_No response_

### Additional context

_No response_

## Trigger

JAX compiles a single-shard computation while the PJRT client exposes four addressable GPUs.

## Relevant entities

serialized compile options; device assignment; addressable devices; loaded executable; argument shards

## Implemented relation

A loaded executable's device list matches the device assignment selected for that compilation.

## Pre-state

The compile options assign the computation to one device although the client exposes several devices.

## Binding transition

The old plugin ignores the serialized compile options and constructs the loaded executable with every addressable device.

## Post-state

The executable reports four devices for a computation and argument that each have one shard.

## Validity obligation

Compilation preserves the selected device assignment in the loaded executable's execution-device list.

## Invalid decision

PJRT_Client_Compile discarded its compile-options device assignment and unconditionally attached addressable_devices_.

## Physical effect

JAX derives a four-shard execution requirement from an executable compiled for one assigned device.

## Observed consequence

execute_sharded_on_local_devices rejects one argument shard because it expects four.

## Repair site

integrations/pjrt/src/iree_pjrt/common/api_impl.cc::ClientInstance::Compile

## Repair

PR 19369 parses CompileOptionsProto, reads executable_build_options.device_assignment, selects those addressable devices, and passes that list to LoadedExecutableInstance.

## Why the repair works

The executable exposes the same device set used by the compilation request, so JAX's shard count matches the supplied argument shards.

## Causal basis

The report's four-versus-one shard error matches the old all-addressable-device construction, while the merged patch carries the compile assignment into that constructor.

## Analysis implication

Multi-device checking needs the client device set, compile-time assignment, executable device list, and argument-shard cardinality.

## Fix patches

- [`fixes/iree-19366-pr19369.patch`](fixes/iree-19366-pr19369.patch) — https://github.com/iree-org/iree/pull/19369
