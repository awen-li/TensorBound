# mhlo.gather to mhlo.select results in incorrect slice selected

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/8790](https://github.com/iree-org/iree/issues/8790)
- Opened: 2022-04-06T20:53:22Z
- Closed: 2022-04-11T20:05:23Z
- Labels: `bug 🐞`, `codegen`

## Reported behavior

It appears that a slice into select is grabbing the wrong section of buffer before executing the select operation. Speculatively, I am guessing the dispatch changes are grabbing the full parent buffer rather than the slice section.

Reproduction:

1. Download attached [file](https://github.com/google/iree/files/8430430/test.mlir.zip) and unzip test.mlir
2. Compile to target dylib:
  `iree-compile /tmp/test.mlir --iree-input-type=mhlo --iree-hal-target-backends=dylib -o /tmp/test.vmfb`
3. Execute with iree-run-module:
  `iree-run-module --driver=dylib --module_file=/tmp/test.vmfb --entry_function=main --function_input="3x4xf32=[[0.1 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]]"`
 4. Results are as shown below:
` 2x4xf32=[0.1 0.2 0.3 0][0.2 0.3 0 0]`
but should be:
`2x4xf32=[0.2 0.3 0 0][0.3 0 0 0]`

## Trigger

A gather/select computation loads and stores tensor slices whose source view has nonzero offsets or non-unit strides.

## Relevant entities

flow.dispatch.tensor.load; flow.dispatch.tensor.store; source slice; destination slice; rewritten destructive update

## Implemented relation

The load/store offsets and strides identify the physical source and destination regions represented by each tensor slice.

## Pre-state

The tensor slice denotes a specific subregion of its parent buffer.

## Binding transition

BufferizeCopyOnlyDispatches and destructive-update rewriting replace the slice load/store with extract/insert-slice operations.

## Post-state

The rewritten operations use the parent buffer region without the original slice offsets and strides.

## Validity obligation

A destructive-update rewrite preserves the physical region denoted by every source and destination slice.

## Invalid decision

The rewrite constructed replacement slice operations without folding the load/store offsets and strides into their access descriptors.

## Physical effect

The select reads a different section of the parent buffer from the section named by the source slice.

## Observed consequence

The two output rows are shifted to values from the preceding input rows instead of the expected slices.

## Repair site

compiler/Codegen/Common/BufferizeCopyOnlyDispatchesPass.cpp and DestructiveUpdateUtils.cpp

## Repair

PR 8793 folds the dispatch load/store offsets and strides into the replacement extract_slice and insert_slice operations and adds offset, stride, and rank-reduction tests.

## Why the repair works

The replacement operations now carry the same region calculation as the original dispatch tensor accesses.

## Causal basis

The issue isolates a wrong subregion, and the linked patch explicitly identifies omitted offsets and strides, restores them in both load and store rewrites, and adds regression coverage for those access descriptors.

## Analysis implication

A rewrite check needs the parent buffer, slice offsets, slice sizes, and strides before and after destructive-update conversion.

## Fix patches

- [`fixes/iree-8790-pr8793.patch`](fixes/iree-8790-pr8793.patch) — https://github.com/iree-org/iree/pull/8793
