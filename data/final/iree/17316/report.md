# Using torch.overwrite.tensor.contents to overwrite input argument fails at runtime

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/17316](https://github.com/iree-org/iree/issues/17316)
- Opened: 2024-05-08T21:26:25Z
- Closed: 2024-05-11T15:51:55Z
- Labels: `bug 🐞`, `integrations/pytorch`

## Reported behavior

### What happened?

The lowering comes from when we use index_copy_ to try to update an input argument's values. When trying to run the vmfb after compiling [index_copy_repro.mlir](https://gist.github.com/aviator19941/96e9870807c96b80ec997cd4d5d7c653), I get this error:
```
EXEC @test_index_copy
iree/runtime/src/iree/hal/command_buffer_validation.c:363: INVALID_ARGUMENT; source and target ranges overlap within the same buffer; while invoking native function hal.command_buffer.copy_buffer; while calling import;
[ 2]   native hal.command_buffer.copy_buffer:0 -
[ 1] bytecode module.test_index_copy$async:1142 ../batch_llama_v4.mlir:2:3
[ 0] bytecode module.test_index_copy:66 ../batch_llama_v4.mlir:2:3; invoking function 'test_index_copy'
```


### Steps to reproduce your issue

1. Compile with:
`../iree-build/tools/iree-compile --iree-input-type=torch --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-hal-target-backends=rocm --mlir-print-debuginfo --mlir-print-op-on-diagnostic=false --iree-hal-target-backends=rocm --iree-rocm-target-chip=gfx940 --iree-opt-const-eval=false --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode --iree-opt-strip-assertions=true ../index_copy_repro.mlir -o index_copy_repro.vmfb`
2. Run with:
`../iree-build/tools/iree-run-module --module=llama_v4.vmfb --device=rocm
--function=test_index_copy --input=8192x16x8x128xf32 --input=4xi64 --input=4x16x8x128xf32 --output=@output.npy
EXEC @test_index_copy`
3. Expected error output:
```
EXEC @test_index_copy
iree/runtime/src/iree/hal/command_buffer_validation.c:363: INVALID_ARGUMENT; source and target ranges overlap within the same buffer; while invoking native function hal.command_buffer.copy_buffer; while calling import;
[ 2]   native hal.command_buffer.copy_buffer:0 -
[ 1] bytecode module.test_index_copy$async:1142 ../batch_llama_v4.mlir:2:3
[ 0] bytecode module.test_index_copy:66 ../batch_llama_v4.mlir:2:3; invoking function 'test_index_copy'
```


### What component(s) does this issue relate to?

Runtime

### Version information

355f56b5588ddf89565a2bb3d2a65524262d5508

### Additional context

_No response_

## Trigger

index_put produces a tensor tied to an input and overwrite requests that result in the input's external buffer.

## Relevant entities

Torch tensor; tied result; HAL tensor export; external buffer; stream update

## Implemented relation

An in-place result's logical alias to caller-provided storage is represented before allocation and copy planning.

## Pre-state

The result is logically intended to reside in the external input/output buffer.

## Binding transition

The old lowering exported the computed tensor into target storage as a late copy.

## Post-state

The copy planner sees overlapping source and target ranges in the same buffer.

## Validity obligation

Aliasing intent enters the use-def and allocation chain before a physical transfer is selected.

## Invalid decision

hal.tensor.export encoded output storage as a copy destination instead of an explicit tensor-storage alias.

## Physical effect

Lowering emits a copy_buffer operation whose source and target overlap.

## Observed consequence

Runtime command-buffer validation rejects the invocation with INVALID_ARGUMENT.

## Repair site

HAL tensor export/alias IR and HAL-to-Stream conversion

## Repair

PR 17339 introduces hal.tensor.alias, moves output-storage binding before export, and lowers the alias through import, update, slice, and transfer operations.

## Why the repair works

Allocation placement can produce into the external storage while the logical alias stays in the SSA use-def chain, eliminating the accidental overlapping copy.

## Causal basis

The issue discussion proposes an alias operation and the merged patch implements that operation at the exact export boundary.

## Analysis implication

In-place analysis needs the logical tied value, external storage identity, range overlap, wait fence, and allocation-placement decision.

## Fix patches

- [`fixes/iree-17316-pr17339.patch`](fixes/iree-17316-pr17339.patch) — https://github.com/iree-org/iree/pull/17339
