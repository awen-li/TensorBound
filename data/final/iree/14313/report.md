# INVALID_ARGUMENT; bindings[1] is indirect but the command buffer does not support binding tables

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/14313](https://github.com/iree-org/iree/issues/14313)
- Opened: 2023-07-06T06:06:39Z
- Closed: 2024-07-09T01:26:36Z
- Labels: `bug 🐞`

## Reported behavior

### What happened?

I am trying to run a StableHLO model exported from JAX. It compiles without errors, but then at runtime I get:
```
  File "/home/petkantchin/ws/iree/build/ninja/Debug/runtime/bindings/python/iree/runtime/function.py", line 137, in __call__
    self._invoke(arg_list, ret_list)
  File "/home/petkantchin/ws/iree/build/ninja/Debug/runtime/bindings/python/iree/runtime/function.py", line 162, in _invoke
    self._vm_context.invoke(self._vm_function, arg_list, ret_list)
ValueError: Error invoking function: repo/runtime/src/iree/hal/command_buffer_validation.c:514: INVALID_ARGUMENT; bindings[1] is indirect but the command buffer does not support binding tables; while invoking native function hal.command_buffer.push_descriptor_set; while calling import;
[ 1]   native hal.command_buffer.push_descriptor_set:0 -
[ 0] bytecode jit_train_step.main:92338 [
    stable_diffusion_train_step.mlir:695:12
      at stable_diffusion_train_step.mlir:2:3,
    stable_diffusion_train_step.mlir:16518:10
      at stable_diffusion_train_step.mlir:16468:10
      at stable_diffusion_train_step.mlir:693:12
      at stable_diffusion_train_step.mlir:2:3,
    stable_diffusion_train_step.mlir:16500:11
      at stable_diffusion_train_step.mlir:693:12
      at stable_diffusion_train_step.mlir:2:3,
    stable_diffusion_train_step.mlir:16509:27
      at stable_diffusion_train_step.mlir:693:12
      at stable_diffusion_train_step.mlir:2:3,
    stable_diffusion_train_step.mlir:16510:11
      at stable_diffusion_train_step.mlir:693:12
      at stable_diffusion_train_step.mlir:2:3,
    stable_diffusion_train_step.mlir:16574:11
      at stable_diffusion_train_step.mlir:694:12
      at stable_diffusion_train_step.mlir:2:3,
    stable_diffusion_train_step.mlir:16578:10
      at stable_diffusion_train_step.mlir:16528:10
      at stable_diffusion_train_step.mlir:694:12
      at stable_diffusion_train_step.mlir:2:3,
    stable_diffusion_train_step.mlir:16463:10
      at stable_diffusion_train_step.mlir:693:12
      at stable_diffusion_train_step.mlir:2:3,
    stable_diffusion_train_step.mlir:16560:11
      at stable_diffusion_train_step.mlir:694:12
      at stable_diffusion_train_step.mlir:2:3,
    stable_diffusion_train_step.mlir:16570:11
      at stable_diffusion_train_step.mlir:694:12
      at stable_diffusion_train_step.mlir:2:3,
    stable_diffusion_train_step.mlir:16527:10
      at stable_diffusion_train_step.mlir:694:12
      at stable_diffusion_train_step.mlir:2:3,
    stable_diffusion_train_step.mlir:2:3,
    stable_diffusion_train_step.mlir:16467:10
      at stable_diffusion_train_step.mlir:693:12
      at stable_diffusion_train_step.mlir:2:3
  ]
```

### Steps to reproduce your issue

Download [iree_no_binding_table_bug.zip](https://github.com/openxla/iree/files/11964868/iree_no_binding_table_bug.zip), unzip it.

```bash
cd iree_no_binding_table_bug
python iree_no_binding_table_bug.py
```



### What component(s) does this issue relate to?

Python, Runtime

### Version information

I had to fix another bug related to 0-rank tensors since some of the inputs are of this form. The PR for this is pending at https://github.com/openxla/iree/pull/14298.

### Additional context

_No response_

## Trigger

A large control-flow function remaps a 64-bit integer at the highest allocated i32 register ordinal.

## Relevant entities

64-bit SSA value; two i32 register words; branch remap; scratch register bank; adjacent ref register

## Implemented relation

The declared i32 register bank includes every word and scratch slot used by branch remapping without overlapping the reference-register bank.

## Pre-state

The reference register and 64-bit branch value occupy disjoint logical VM registers.

## Binding transition

Register allocation reserves scratch capacity per remapped value rather than per 32-bit word.

## Post-state

The second word writes one slot past the declared i32 bank and overwrites the low word of an adjacent reference.

## Validity obligation

Multiword remaps reserve one scratch register for each emitted 32-bit word, and bytecode verification bounds-checks every remap ordinal.

## Invalid decision

RegisterAllocation counted a >4-byte remapping as one scratch slot even though bytecode dispatch performs two 32-bit moves.

## Physical effect

Branch remapping clears or replaces the low 32 bits of a live HAL buffer reference.

## Observed consequence

The corrupted pointer produces a descriptor-binding error, heap-buffer-overflow, or segmentation fault.

## Repair site

compiler/src/iree/compiler/Dialect/VM/Analysis/RegisterAllocation.cpp

## Repair

PR 14332 reserves ceil(bitwidth/32) scratch registers for multiword remaps; PR 17829 makes IREE_VM_VERIFY_REG_ANY reject out-of-range i32 and ref ordinals.

## Why the repair works

Generated bytecode has sufficient disjoint storage, and malformed older bytecode is rejected before dispatch.

## Causal basis

The issue trace identifies the exact remap that crosses the i32/ref boundary, while the two patches repair generation and validation of that same ordinal relation.

## Analysis implication

A VM allocation check needs value bit width, emitted word count, remap source/target ordinals, scratch demand, bank bounds, and adjacent-bank layout.

## Fix patches

- [`fixes/iree-14313-pr14332.patch`](fixes/iree-14313-pr14332.patch) — https://github.com/iree-org/iree/pull/14332
- [`fixes/iree-14313-pr17829.patch`](fixes/iree-14313-pr17829.patch) — https://github.com/iree-org/iree/pull/17829
