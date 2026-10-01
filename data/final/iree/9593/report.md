# IREE hardfaults on Arm bare-metal target when dispatching a 64bit op

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/9593](https://github.com/iree-org/iree/issues/9593)
- Opened: 2022-06-23T14:46:25Z
- Closed: 2022-07-04T15:14:43Z
- Labels: `bug 🐞`, `runtime/vm`

## Reported behavior

### What happened?

After bumping the IREE submodule in the [iree-bare-metal-arm](https://github.com/iml130/iree-bare-metal-arm) from candidate-20220518.142 to candidate-20220519.143, a hardfault occurs on Arm bare-metal targets. This issue is related to the changes introduced with #9145 or at least this change reveals the issue. The issue was verified on several STM32 (Cortex-M4 / Armv7-M) targets and also occurs with Arm's FVP (Fixed Virtual Platform) for the Corstone-300 (Cortex-M55 / Armv8.1-M).

### Steps to reproduce your issue

See https://gist.github.com/marbre/f26c1fc96c3dd4796ff2f31ec42f46b9

### What component(s) does this issue relate to?

Runtime

### Version information

Commit google/iree@395d248 and [candidate-20220519.143](https://github.com/google/iree/releases/tag/candidate-20220519.143), respectively.

### Additional context

Debugging on an Nulceo-64 with STM32F411RE, I could trace down the error to occur when dispatching a `ConstI64` op:
https://github.com/google/iree/blob/395d248e8bb9c7c221ec2e0dd18b833074e1f336/runtime/src/iree/vm/bytecode_dispatch.c#L872

The issue can be bypassed by adding `-mno-unaligned-access` to our toolchain file. With this, the preprocessor symbol `__ARM_FEATURE_UNALIGNED` isn't defined and `IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED`is set accordingly:

https://github.com/google/iree/blob/395d248e8bb9c7c221ec2e0dd18b833074e1f336/runtime/src/iree/base/target_platform.h#L149-L156

(Please note that just passing `IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED` via CMake's `add_compile_definitions` wasn't sufficient.)
With `IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED` set, the functions for aligned memory access are defined in [alignment.h](https://github.com/google/iree/blob/candidate-20220519.143/runtime/src/iree/base/alignment.h). Else the following code is compiled:

https://github.com/google/iree/blob/395d248e8bb9c7c221ec2e0dd18b833074e1f336/runtime/src/iree/base/alignment.h#L185-L197

This implements loads and stores for all data types. However, according to the [Cortex-M4 Technical Reference Manual](https://developer.arm.com/documentation/ddi0439/b/Programmers-Model/System-address-map/Unaligned-accesses-that-cross-regions?lang=en) unaligned access isn't available for all bit widths:

> Unaligned support is only available for load/store singles (LDR, STR). Load/store double already supports word aligned accesses, but does not permit other unaligned accesses, and generates a fault if this is attempted.

I am not sure if we should enforce `-mno-unaligned-access` to be set or if we should rather refactor [alignment.h](https://github.com/google/iree/blob/candidate-20220519.143/runtime/src/iree/base/alignment.h) in a way to only use unalignment memory access where supported.
I further have to mention that there might be more that goes wrong, since just defining `IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED` didn't resolved the issue.

For the sake of completeness, `-mno-unaligned-access` is documented in [GCC's docs](https://gcc.gnu.org/onlinedocs/gcc/ARM-Options.html) as follows:

> Enables (or disables) reading and writing of 16- and 32- bit values from addresses that are not 16- or 32- bit aligned. By default unaligned access is disabled for all pre-ARMv6, all ARMv6-M and for ARMv8-M Baseline architectures, and enabled for all other architectures. If unaligned access is not enabled then words in packed data structures are accessed a byte at a time.

> The ARM attribute Tag_CPU_unaligned_access is set in the generated object file to either true or false, depending upon the setting of this option. If unaligned access is enabled then the preprocessor symbol __ARM_FEATURE_UNALIGNED is also defined.

## Trigger

ConstI64 dispatch reads a 64-bit immediate from a bytecode address without eight-byte alignment on Cortex-M.

## Relevant entities

bytecode stream; ConstI64 immediate; target alignment flags; unaligned-load helper; Arm load instruction

## Implemented relation

A bytecode operand is reconstructed with accesses permitted by the target for that operand width.

## Pre-state

The eight immediate bytes are valid in the compact bytecode stream but are not naturally aligned.

## Binding transition

Target configuration disables the general alignment workaround, so the u64 helper dereferences a uint64_t pointer directly.

## Post-state

The compiler emits a native 64-bit load at the unaligned address.

## Validity obligation

Target policy distinguishes 64-bit alignment requirements from supported narrower unaligned accesses.

## Invalid decision

One alignment flag represented every width and was also unconditionally reset in one configuration path.

## Physical effect

The processor raises an alignment fault while decoding ConstI64.

## Observed consequence

IREE hardfaults when dispatching a function containing a 64-bit VM operation on Arm bare-metal targets.

## Repair site

runtime/src/iree/base/target_platform.h and alignment.h

## Repair

PR 9697 preserves an explicitly required alignment policy; PR 9705 adds IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED_64 and uses bytewise u64/f64 loads and stores on Arm while retaining native narrower accesses.

## Why the repair works

The VM reconstructs unaligned 64-bit operands byte by byte on targets where a native doubleword access can fault.

## Causal basis

The issue pinpoints ConstI64's unaligned read, and PR 9705 states that it fixes 9593 by separating 64-bit Arm alignment.

## Analysis implication

Memory-access checks need address alignment, access width, target ISA support by width, configuration overrides, and the helper selected after preprocessing.

## Fix patches

- [`fixes/iree-9593-pr9705.patch`](fixes/iree-9593-pr9705.patch) — https://github.com/iree-org/iree/pull/9705
- [`fixes/iree-9593-pr9697.patch`](fixes/iree-9593-pr9697.patch) — https://github.com/iree-org/iree/pull/9697
