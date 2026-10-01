# There is an issue with the circular buffer in hip_device.c

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/21757](https://github.com/iree-org/iree/issues/21757)
- Opened: 2025-08-22T00:14:11Z
- Closed: open at collection time
- Labels: `bug 🐞`

## Reported behavior

### What happened?

We are seeing non-determinism in VERY large models. It looks to be related to file reads in hip_device. This is likely due to an issue with the circular parameter buffer logic in queue_read_now() and related code.

### Steps to reproduce your issue

1

### What component(s) does this issue relate to?

Runtime

### Version information

_No response_

### Additional context

_No response_

## Trigger

A parameter file larger than the staging buffer is uploaded in multiple chunks while earlier upload work remains active.

## Relevant entities

parameter file range; transfer chunk; circular staging buffer; free-byte count; HIP upload queue; model parameter storage

## Implemented relation

Each in-flight file range retains a distinct staging-buffer region until the queued HIP transfer has consumed that region.

## Pre-state

A chunk occupies part of the circular staging buffer and its asynchronous transfer has not yet completed.

## Binding transition

The old admission condition permits another chunk whenever the currently free byte count is at least that chunk's size.

## Post-state

Large uploads continue to circulate through the shared staging buffer while prior parameter transfers can still depend on its contents.

## Validity obligation

A staging-buffer region is not reused until every transfer reading that region has completed.

## Invalid decision

The circular upload path admitted continued reuse without establishing region-specific completion for the bytes selected next.

## Physical effect

Bytes used by queued parameter transfers can be replaced by later file chunks.

## Observed consequence

Very large models receive corrupted parameters and exhibit nondeterministic execution.

## Repair site

runtime/src/iree/hal/drivers/hip/hip_device.c transfer-buffer admission condition

## Repair

PR 21758 changes admission from enough free bytes for one chunk to the entire transfer buffer being free, disabling overlapping circular-buffer reuse.

## Why the repair works

A new upload begins only after the staging buffer has returned to its fully free state, so no later chunk can overwrite bytes still used by an earlier transfer.

## Causal basis

The issue and merged change both attribute large-model parameter corruption to circular upload behavior; the change serializes reuse by waiting for the full buffer to become free.

## Analysis implication

Upload analysis needs file offsets, staging ranges, free-space accounting, queue completion, and the reuse point for each physical byte range.

## Fix patches

- [`fixes/iree-21757-pr21758.patch`](fixes/iree-21757-pr21758.patch) — https://github.com/iree-org/iree/pull/21758
