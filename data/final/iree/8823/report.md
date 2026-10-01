# Add a verification rule on HAL::EntryPointOp block to be non-empty.

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/8823](https://github.com/iree-org/iree/issues/8823)
- Opened: 2022-04-08T17:41:32Z
- Closed: open at collection time
- Labels: `bug 🐞`, `help wanted`, `compiler/dialects`

## Reported behavior

`hal.entry_point` may have an empty block when reaching execution, seemingly triggering UB as the number of threads seems to vary from run to tun.

This bug report is to add a verification rule in some relevant place.
The challenge is the op may legitimately have an empty block in early phases of the compiler.

## Trigger

A transform-interpreter pipeline applies workgroup-count construction through a pass nested under a special non-Module operation.

## Relevant entities

HAL executable variant; executable entry point; workgroup-count region; nested pass manager; transform operation; runtime dispatch grid

## Implemented relation

Every executable entry point carries a populated workgroup-count region that deterministically maps its workload to a runtime dispatch grid.

## Pre-state

The entry point legitimately has no workgroup-count body during early transformation and requires code generation to define one before execution.

## Binding transition

The old pipeline asks a nested pass manager to run the workgroup-count pass on a special non-Module root.

## Post-state

Pass nesting can perform no transformation without reporting failure, leaving the entry-point workgroup-count block empty at execution.

## Validity obligation

The workgroup-count transformation is applied directly to the executable variant and returns failure when it cannot populate the entry-point region.

## Invalid decision

The pipeline treated successful pass-manager invocation as proof that a pass anchored on the special root had executed.

## Physical effect

The runtime reaches a dispatch without a defined mapping from workload dimensions to workgroup counts.

## Observed consequence

The reported execution exhibits undefined, run-varying thread counts and flaky single-thread results.

## Repair site

iree/compiler/Codegen/Common/SetNumWorkgroupsPass.cpp::setNumWorkgroupsImpl and the transform-dialect extension

## Repair

Commit 1b9e84c extracts setNumWorkgroupsImpl as a directly callable function, adds iree_set_num_workgroups_to_one to the transform dialect, invokes the implementation on the top-level HAL executable variant, and removes the unreliable nested pass path.

## Why the repair works

The transform now calls the workgroup-count construction on the exact executable variant and propagates failure, so execution cannot silently inherit an empty region from a skipped pass.

## Causal basis

The landed commit names bug 8823, explains that pass nesting on non-Module roots may do nothing without a report, replaces that path with a direct implementation call, and states that the old behavior produced flaky single-thread results.

## Analysis implication

Dispatch-grid analysis needs the entry-point identity, workload dimensions, workgroup-count region state, pass or transform anchor, application result, and runtime grid together.

## Fix patches

- [`fixes/iree-8823-commit-1b9e84c.patch`](fixes/iree-8823-commit-1b9e84c.patch) — https://github.com/iree-org/iree/commit/1b9e84cf81836814e8fa4feb652a1e27b96a2b6f
