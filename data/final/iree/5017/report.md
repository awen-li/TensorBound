# NVVM lowering assumes contiguous dense bindings, which is not always the case

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/5017](https://github.com/iree-org/iree/issues/5017)
- Opened: 2021-03-08T19:17:21Z
- Closed: 2021-06-06T15:29:40Z
- Labels: `bug 🐞`, `codegen/nvvm`

## Reported behavior

This code right here is making an assumption that the bindings map 1:1 in dense order with the arguments of the function:
https://github.com/google/iree/blob/132f5a343a3ada65e923f0dec5c7e9907fde7678/iree/compiler/Conversion/LinalgToNVVM/ConvertToNVVM.cpp#L114-L115

This means that if there is a sparse binding set things go awry (out of bounds accesses/heap corruption/etc).

If the bindings are expanded to individual arguments then we'll need to have a mapping of binding to argument index or walk the binding list to figure out the appropriate index.

Work on buffer allocation is going to change these assumptions and this code will no longer work.

## Trigger

A CUDA dispatch interface uses sparse or out-of-order binding identifiers, such as bindings 0, 4, and 7, instead of a dense sequence beginning at zero.

## Relevant entities

HAL interface binding identifier; interface binding subspan; LLVM kernel argument; descriptor-set binding; CUDA kernel argument array; device pointer

## Implemented relation

The compiler and CUDA runtime translate each HAL binding identifier through the same sorted, compressed mapping to a dense kernel-argument position.

## Pre-state

A HAL interface gives each buffer a stable binding identifier that may be sparse and may appear in an order different from the kernel ABI.

## Binding transition

The old lowering used the numeric binding identifier directly in llvmFuncOp.getArgument, while the old CUDA command buffer used the same identifier directly as current_descriptor's argument slot.

## Post-state

A sparse identifier was treated as a physical position in a compact argument vector, so the selected position could be outside the vector or could refer to a different buffer.

## Validity obligation

Every participating layer uses one bijection from symbolic binding identifiers to dense kernel-argument positions; the mapping is independent of identifier density and descriptor enumeration order.

## Invalid decision

The lowering and runtime equated a symbolic binding number with a dense array index without establishing that the binding set was dense and ordered.

## Physical effect

The generated function or CUDA launch selected an absent or incorrect argument slot, which supplied the wrong device pointer or accessed storage outside the argument array.

## Observed consequence

The report records out-of-bounds accesses and heap corruption when bindings were not a dense ordered set.

## Repair site

iree/compiler/Conversion/LinalgToLLVMGPU/ConvertToLLVM.cpp::getKernelArgMapping and ConvertIREEBindingOp; iree/hal/cuda/graph_command_buffer.c::iree_hal_cuda_graph_command_buffer_push_descriptor_set

## Repair

Commit 0dbd1ac collects binding identifiers, sorts them, and maps them to dense argument positions during lowering. The CUDA command buffer independently sorts descriptor bindings by identifier and writes their device pointers to consecutive argument slots. The compiler regression uses out-of-order sparse identifiers 4, 0, and 7 with distinct element types to check the resulting argument order.

## Why the repair works

Both sides now derive the same order from binding identifiers and compact that order before indexing physical argument arrays. Sparse values therefore preserve binding identity without creating holes or changing which pointer reaches each kernel parameter.

## Causal basis

Issue 5017 identifies the direct binding-number indexing assumption and its sparse-binding failure. GitHub connects the issue to merged PR 6044, and the issue closes at the PR's merge time. The landed commit names the dense-set assumption, changes both the compiler and CUDA runtime to the same sorted-and-compressed convention, and adds a compiler test whose sparse, out-of-order bindings expose the old mapping.

## Analysis implication

Binding analysis needs the symbolic binding identifier, the compiler's identifier-to-argument map, the runtime descriptor order, and the final kernel-argument position together.

## Fix patches

- [`fixes/iree-5017-commit-0dbd1ac.patch`](fixes/iree-5017-commit-0dbd1ac.patch) — https://github.com/iree-org/iree/commit/0dbd1ac8730c5c73a0ef74c888badddca1d2dde5
