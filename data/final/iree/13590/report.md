# Memory leak from fences in `main` function during `--iree-execution-model=async-external` when linking a custom module with `iree-run-module` dynamically

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/13590](https://github.com/iree-org/iree/issues/13590)
- Opened: 2023-05-12T12:56:12Z
- Closed: 2024-11-05T03:10:01Z
- Labels: `bug 🐞`

## Reported behavior

### What happened?

When compiling with ASAN and running https://github.com/openxla/iree/blob/main/samples/custom_module/dynamic/test/example.mlir,  on `llvm-cpu` with attributes for both the imports set to `iree.abi.model = "sync"`, I get memory leak errors.


Output:
```
==1635943==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 48 byte(s) in 1 object(s) allocated from:
    #0 0x366522 in calloc (/home/muralivi/projects/iree-build/tools/iree-run-module+0x366522)
    #1 0x399407 in iree_allocator_system_ctl /home/muralivi/projects/iree/runtime/src/iree/base/allocator.c:86:17
    #2 0x398d6d in iree_allocator_malloc /home/muralivi/projects/iree/runtime/src/iree/base/allocator.c:27:10
    #3 0x3bb7f9 in iree_hal_fence_create_at /home/muralivi/projects/iree/runtime/src/iree/hal/fence.c:37:3
    #4 0x4f289e in iree_tooling_append_async_fence_inputs /home/muralivi/projects/iree/runtime/src/iree/tooling/vm_util.c:301:26
    #5 0x3ba8dc in iree_tooling_run_module_with_data /home/muralivi/projects/iree/runtime/src/iree/tooling/run_module.c:195:9
    #6 0x3ba0cc in iree_tooling_run_module_from_flags /home/muralivi/projects/iree/runtime/src/iree/tooling/run_module.c:321:10
    #7 0x398ae6 in main /home/muralivi/projects/iree/tools/iree-run-module-main.c:43:14
    #8 0x7f0c7d4e2d09 in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x23d09)

Indirect leak of 88 byte(s) in 1 object(s) allocated from:
    #0 0x366522 in calloc (/home/muralivi/projects/iree-build/tools/iree-run-module+0x366522)
    #1 0x399407 in iree_allocator_system_ctl /home/muralivi/projects/iree/runtime/src/iree/base/allocator.c:86:17
    #2 0x398d6d in iree_allocator_malloc /home/muralivi/projects/iree/runtime/src/iree/base/allocator.c:27:10
    #3 0x437144 in iree_hal_task_semaphore_create /home/muralivi/projects/iree/runtime/src/iree/hal/drivers/local_task/task_semaphore.c:89:26
    #4 0x4f282c in iree_tooling_append_async_fence_inputs /home/muralivi/projects/iree/runtime/src/iree/tooling/vm_util.c:298:3
    #5 0x3ba8dc in iree_tooling_run_module_with_data /home/muralivi/projects/iree/runtime/src/iree/tooling/run_module.c:195:9
    #6 0x3ba0cc in iree_tooling_run_module_from_flags /home/muralivi/projects/iree/runtime/src/iree/tooling/run_module.c:321:10
    #7 0x398ae6 in main /home/muralivi/projects/iree/tools/iree-run-module-main.c:43:14
    #8 0x7f0c7d4e2d09 in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x23d09)

SUMMARY: AddressSanitizer: 136 byte(s) leaked in 2 allocation(s).
```

Note that these errors occur only wiith `iree-run-module` with `--iree-execution-model=async-external` invocation - it doesn't occur if we have a custom run module which explicitly frees the allocated fences in `main` or if we use `async-internal`.

### Steps to reproduce your issue

Exact command:
```
cmake -B ../iree-build/ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DIREE_ENABLE_ASAN=ON . && \
cmake --build ../iree-build/ && \
cmake --build ../iree-build/     --target iree-run-module     --target iree_samples_custom_module_dynamic_module && \
iree-compile --iree-execution-model=async-external         --iree-hal-target-backends=llvm-cpu samples/custom_module/dynamic/test/example.mlir | \
../iree-build/tools/iree-run-module    --module=../iree-build/samples/custom_module/dynamic/module.so@create_custom_module    --module=- --function=main
```

### What component(s) does this issue relate to?

Runtime

### Version information

_No response_

### Additional context

_No response_

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

iree-run-module invokes a synchronous custom import under the async-external execution model.

## Observed implementation decision

The tooling path creates an input semaphore and fence for the invocation but the report observes no matching release before process exit.

## Physical effect

The fence and its task semaphore retain their heap allocations after the invocation completes.

## Observed consequence

LeakSanitizer reports 136 leaked bytes rooted at iree_tooling_append_async_fence_inputs.

## Reported proposal or workaround

A custom runner that explicitly frees the allocated fences and the async-internal execution model do not leak.

## Repair trace

The issue has no maintainer diagnosis, correcting source change, or fixed regression, and it was closed without a repair reference. The allocation stack and workaround do not establish the exact missed ownership transition or its landed correction.

## Archived evidence

- Raw report: `data/raw/github/iree/13590/report.json`
- SHA-256: `f8ff0e14cfb0e4fa6c213ecd5a4dd5c9ad79fa3bf410f7f51246e1eeca81ac92`
