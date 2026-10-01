# Leaked buffers for unused inputs

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/2455](https://github.com/iree-org/iree/issues/2455)
- Opened: 2020-07-09T13:23:33Z
- Closed: 2020-07-10T00:13:22Z
- Labels: `bug 🐞`, `runtime`

## Reported behavior

At the moment IREE runtime is leaking buffers allocated for unused inputs. A repro case:

```mlir
func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> attributes {iree.module.export} {
  return %arg1 : tensor<4xf32>
}
```

```shell
$ iree-translate -iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=vulkan-spirv ../unused-input.mlir > unused-input.vmfb

$ iree-run-module --input_file=unused-input-vmla.vmfb --entry_function=main --inputs="4xf32,4xf32" --driver=vulkan

F ../third_party/vulkan_memory_allocator/src/vk_mem_alloc.h:11175] Check failed: m_pMetadata->IsEmpty() && "Some allocations were not freed before destruction of this memory block!"
```

This is made obvious by Vulkan's VMA allocator. But it's not specific to Vulkan HAL backend; the code in the above does not perform any computation at all. It should also be revealed via ASAN for other backends.

## Trigger

A cross-module call returns one tensor reference while another input reference remains in the result register selected for replacement.

## Relevant entities

caller-frame register; owned input reference; returned reference; ResultPack::Store; HAL buffer

## Implemented relation

Replacing a reference-register binding accounts for ownership of the displaced reference before installing the new reference.

## Pre-state

The selected register owns a live reference to an input buffer.

## Binding transition

ResultPack::Store clears the register bytes with memset before performing the reference assignment.

## Post-state

The old reference identity is erased without a release or ownership transfer.

## Validity obligation

Reference-register replacement preserves the old identity until the VM reference operation can release or transfer it.

## Invalid decision

The old result-pack variants zeroed opaque_ref, ref<T>, and optional<ref<T>> slots before managed assignment.

## Physical effect

The displaced input buffer retains an unbalanced reference count.

## Observed consequence

The unused-input program leaves a buffer allocated at shutdown and Vulkan reports the outstanding allocation.

## Repair site

iree/vm/module_abi_packing.h::ResultPack::Store

## Repair

Commit 15e51a85 removes the pre-assignment memset from reference result-pack variants and adds an unused-arguments regression.

## Why the repair works

Managed reference assignment can observe and correctly account for the displaced reference instead of receiving a zeroed slot.

## Causal basis

The fixing patch changes the exact register-clearing operations implicated by the leak and includes the reported unused-input pattern.

## Analysis implication

Register reuse analysis needs the old reference identity, ownership state, replacement operation, and release/transfer edge.

## Fix patches

- [`fixes/iree-2455-fix.patch`](fixes/iree-2455-fix.patch) — https://github.com/iree-org/iree/commit/15e51a85dcc580168e8455b04b4dd9b9b76976f2
