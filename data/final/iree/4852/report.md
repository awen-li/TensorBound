# Benchmark allocates buffers in each iteration without releasing after #4767

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/4852](https://github.com/iree-org/iree/issues/4852)
- Opened: 2021-02-16T10:05:51Z
- Closed: 2021-02-16T20:59:31Z
- Labels: `bug 🐞`, `hal/api`, `runtime/vm`

## Reported behavior

**Describe the bug**
See a sample: https://buildkite.com/iree/iree-android-benchmark/builds/1877

The error log:

```
F ../iree/tools/iree-benchmark-module-main.cc:77] Check failed: ::iree::IsOk(_status77) ../iree/hal/vulkan/status_util.c:53: RESOURCE_EXHAUSTED; VK_ERROR_OUT_OF_DEVICE_MEMORY; vmaCreateBuffer; while invoking native function hal.allocator.allocate; while calling import
```

I can reproduce this on local side. One workaround is to reduce the number of `--benchmark_repetitions`. It is working with `--benchmark_repetitions=5` but not `--benchmark_repetitions=10`.

**To Reproduce**

- Download artifacts from https://buildkite.com/iree/iree-android-benchmark/builds/1877
- Upload files to an Android phone
- Run the model, e.g., `adb shell LD_LIBRARY_PATH=/data/local/tmp taskset 80 /data/local/tmp/benchmark_tmpdir/iree-benchmark-module --flagfile=/data/local/tmp/benchmark_tmpdir/flagfile --module_file=/data/local/tmp/benchmark_tmpdir/mobile-bert-vulkan-spirv.vmfb --driver=vulkan --benchmark_repetitions=10`

## Trigger

iree-benchmark-module repeatedly invokes a model whose result list contains reference values.

## Relevant entities

callee ref register; return VM list; owned HAL buffer reference; benchmark iteration; list resize

## Implemented relation

Each returned reference has one owner in the result list and is released when that list is reset before the next invocation.

## Pre-state

An iteration returns newly allocated HAL buffer references.

## Binding transition

The ref calling convention retains/copies results while list resize/reset does not release truncated reference elements correctly.

## Post-state

References from prior iterations remain live while new buffers are allocated.

## Validity obligation

VM list truncation releases owned references, and call transfer semantics move or retain each reference exactly once.

## Invalid decision

The return calling convention and iree_vm_list_resize path did not implement matched ownership transfer and destruction for reference elements.

## Physical effect

Device and host allocations accumulate across benchmark repetitions.

## Observed consequence

Repeated execution reaches VK_ERROR_OUT_OF_DEVICE_MEMORY; reducing repetitions only delays it.

## Repair site

iree/vm/list.c, bytecode_dispatch.c, module_abi_packing.h, and benchmark result reset

## Repair

PR 4858 implements reference-aware VM list resize/truncation, switches ref return handling to move semantics where appropriate, resets benchmark outputs, and adds resize tests for refs and variants.

## Why the repair works

Old result references are released or moved out before list reuse, so each iteration's allocations can be reclaimed.

## Causal basis

The allocation trace shows per-iteration retention, and the linked patch names #4852 while changing result ownership and adding reference truncation tests.

## Analysis implication

A lifetime check needs reference ownership, retain/move operation, list element type, resize range, invocation boundary, and release count.

## Fix patches

- [`fixes/iree-4852-pr4858.patch`](fixes/iree-4852-pr4858.patch) — https://github.com/iree-org/iree/pull/4858
