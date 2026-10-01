# [vulkan] Failed: Some allocations were not freed before destruction of this memory block

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/2948](https://github.com/iree-org/iree/issues/2948)
- Opened: 2020-08-19T22:13:23Z
- Closed: 2020-08-27T04:31:38Z
- Labels: `bug 🐞`, `runtime`, `hal/vulkan`

## Reported behavior

Hello, I'm facing the error "Check failed: m_pMetadata->IsEmpty() && "Some allocations were not freed before destruction of this memory block!" while running simple.mlir on Samsung S9 (QP1A.190711.020.G960FXXSBETH1)

```
$ build-android/host/bin/iree-translate \
    -iree-mlir-to-vm-bytecode-module \
    -iree-hal-target-backends=vulkan-spirv \
    iree/tools/test/simple.mlir \
    -o /tmp/simple-vulkan.vmfb
```

```
starlte:/data/local/tmp $ ./iree-benchmark-module -driver=vulkan -input_file=simple-vulkan.vmfb -entry_function=abs -inputs="i32=2"
2020-08-20 00:37:18
Running ./iree-benchmark-module
Run on (8 X 1794 MHz CPU s)
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
I ../iree/tools/vm_util.cc:227] Creating driver and device for 'vulkan'...
I ../iree/hal/vulkan/vulkan_driver.cc:298] Created Vulkan Device: Mali-G72
F ../third_party/vulkan_memory_allocator/src/vk_mem_alloc.h:11175] Check failed: m_pMetadata->IsEmpty() && "Some allocations were not freed before destruction of this memory block!"
Aborted
```
@antiagainst can you please share, have you seen this issue?

## Trigger

Benchmark iteration completes while the input VM list still owns HAL buffers allocated from the session device.

## Relevant entities

benchmark inputs; VM references; HAL buffers; session; Vulkan allocator

## Implemented relation

All input-buffer references are released before the session and its device allocator are destroyed.

## Pre-state

The persistent input list retains buffers across benchmark iterations.

## Binding transition

The old benchmark teardown closes the session while the input list remains live in the surrounding scope.

## Post-state

The Vulkan allocator is destroyed with at least one allocation still referenced by the inputs.

## Validity obligation

Objects owning session-allocated buffers release those references before session teardown.

## Invalid decision

iree-benchmark-module relied on scope destruction order that kept inputs alive beyond the session.

## Physical effect

Vulkan Memory Allocator observes a nonempty allocation block during destruction.

## Observed consequence

The benchmark aborts with 'Some allocations were not freed before destruction of this memory block.'

## Repair site

iree/tools/benchmark_module_main.cc benchmark teardown

## Repair

PR 3004 explicitly resets inputs before closing the session and adds the reported Vulkan benchmark command as a regression, later marked for the separate SwiftShader failure.

## Why the repair works

Resetting the VM list drops the remaining buffer references while their allocator and device are still valid.

## Causal basis

Iteration capacity tracks the in-flight fence limit, the maintainer identifies retained inputs, and the linked fix adds inputs.reset immediately before session close.

## Analysis implication

Session-lifetime analysis needs transitive VM reference ownership, not only explicit device and session handles.

## Fix patches

- [`fixes/iree-2948-pr3004.patch`](fixes/iree-2948-pr3004.patch) — https://github.com/iree-org/iree/pull/3004
