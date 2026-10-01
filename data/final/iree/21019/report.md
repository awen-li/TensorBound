# [Flux] Empty output from the flux models

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/21019](https://github.com/iree-org/iree/issues/21019)
- Opened: 2025-06-05T19:37:40Z
- Closed: 2025-07-17T13:57:38Z
- Labels: `bug 🐞`

## Reported behavior

### What happened?

When the flux models are compiled and executed, the generated output seems to be empty. This issue is visible from the commit https://github.com/iree-org/iree/commit/d7bb36baa9df2a4f88abc984bd8c9cc7efc3e37c


### Steps to reproduce your issue

This can be reproduced using the steps in https://github.com/nod-ai/shark-ai/blob/e3beeed98eda98aaf88eabda282f8ba6985687a8/shortfin/python/shortfin_apps/flux/README.md?plain=1#L19

### What component(s) does this issue relate to?

Compiler, Runtime

### Version information

IREE commit: 4889cb5fb73ac2e8f546ab36998a760ba2064b01

### Additional context

The issue occurs due to `sharedMemBytes` being set to 0 in the call to **hipModuleLaunchKernel**

https://github.com/iree-org/iree/commit/d7bb36baa9df2a4f88abc984bd8c9cc7efc3e37c#diff-faa5750ca692b9798a643f340a08b4bc22c7fedc5b304f4bb7d722e2df8209b8R581

## Trigger

Shortfin loads a VMFB compiled by an older IREE compiler into a runtime that no longer consumes block_shared_memory_size.

## Relevant entities

VMFB export; block_shared_memory_size; HIP executable loader; hipModuleLaunchKernel; dynamic shared memory

## Implemented relation

A compiled kernel export records the dynamic shared-memory amount that its matching runtime must either honor or reject as incompatible.

## Pre-state

The older VMFB contains a nonzero dynamic shared-memory request for a kernel.

## Binding transition

The newer runtime accepts the still-decodable export while ignoring its deprecated block_shared_memory_size field.

## Post-state

The kernel launches with sharedMemBytes equal to zero although its compiled code expects dynamic shared storage.

## Validity obligation

Compiler and runtime agree on launch-resource fields, or the loader rejects an artifact whose nonzero requirement it cannot honor.

## Invalid decision

The runtime deprecated and ignored the field without rejecting older artifacts that still carried a nonzero value.

## Physical effect

The HIP launch provides no dynamically allocated shared memory to a kernel compiled to use it.

## Observed consequence

Flux execution completes with empty or garbage output instead of raising a compatibility error.

## Repair site

runtime/src/iree/hal/drivers/hip/native_executable.c::iree_hal_hip_native_executable_create

## Repair

PR 21118 restores access to the serialized field and rejects any nonzero value with a compiler/runtime-version diagnostic.

## Why the repair works

An incompatible artifact can no longer reach kernel launch under a silently changed shared-memory contract.

## Causal basis

The issue's size comparison identifies a stale VMFB, and the patch explicitly prevents silent execution of precisely that nonzero legacy field.

## Analysis implication

A handoff check needs artifact producer version, serialized launch-resource fields, loader compatibility policy, and final launch arguments.

## Fix patches

- [`fixes/iree-21019-pr21118.patch`](fixes/iree-21019-pr21118.patch) — https://github.com/iree-org/iree/pull/21118
