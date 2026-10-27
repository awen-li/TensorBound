# Semaphore identity, ownership, and the cleanup read

This 2026-10-27 source check uses pre-fix revision
`d42e002ec10444ac05f77a58d7396ff3658fbfd4`, the archived ThreadSanitizer
trace and maintainer comments, and repair
`d83f4e46b518ae3fdc46ceab82502e88ecb289ba`. It adds no diagnostic rerun or
human-review result.

## Ownership contract

[`semaphore_base.h`, lines 52–55](https://github.com/iree-org/iree/blob/d42e002ec10444ac05f77a58d7396ff3658fbfd4/runtime/src/iree/hal/utils/semaphore_base.h#L52)
states that an active timepoint retains its semaphore. Acquisition performs
that retain in `semaphore_base.c:221–222`. Callback processing releases it at
line 88; cancellation releases it at lines 267–268 when the timepoint is
still active. The callback code explicitly states that the caller also holds
a reference, so releasing the timepoint's reference should not destroy the
semaphore.

[`task_semaphore.c`, lines 26–35](https://github.com/iree-org/iree/blob/d42e002ec10444ac05f77a58d7396ff3658fbfd4/runtime/src/iree/hal/drivers/local_task/task_semaphore.c#L26)
assigns ownership of the enclosing task timepoint to its caller. In multi-wait,
the enclosing records remain in the waiter's arena until cleanup ends at
line 397. These are real ownership contracts, but the reported race does not
show a destroyed semaphore or premature arena reclamation.

## Selected decision and violated condition

The selected operation is the read of `timepoints[i].base.semaphore` used as
the first argument to `iree_hal_semaphore_cancel_timepoint`, at
[`task_semaphore.c:392`](https://github.com/iree-org/iree/blob/d42e002ec10444ac05f77a58d7396ff3658fbfd4/runtime/src/iree/hal/drivers/local_task/task_semaphore.c#L392).
The archived trace identifies that read and the reset at
`semaphore_base.c:80` as conflicting accesses without a happens-before edge.

The base timepoint reset is performed under `timepoint_mutex`. Its source
comment requires clearing the base before the callback, allowing the handler
to reuse that storage. Cancellation also acquires `timepoint_mutex`, at line
244, but argument evaluation reads the field before entering that locked
function. The existing mutex therefore protects the reset and cancellation,
not the caller's argument read.

The source-level condition is that a cleanup read of callback-mutated state
is ordered with the conflicting reset. Its primary family is **happens-before
ordering**. The read remains the invalid decision; a valid reset is the
competing access. The family **ownership and lifetime** does not describe
the condition violated by this read: the caller's semaphore reference and
the enclosing timepoint storage remain valid.

## Repair

The [patch](fixes/iree-9392-fix.patch) stores the acquired semaphore identity
in `iree_hal_task_timepoint_t.semaphore` and reads that field during cleanup.
The callback clears only the embedded `base`, so its write no longer conflicts
with the cleanup read. The separate field also identifies successful
acquisitions for conditional cancellation and event release. The semaphore
retain/release implementation is unchanged.

The repair category is **preserve or propagate relation**: it carries the
acquired semaphore identity in a field outside the callback reset. It removes
the conflicting access rather than adding a lock or extending semaphore
lifetime. The synchronization requirement at the pre-fix read and the
identity-preserving repair are separate coding dimensions.

The ownership/lifetime relation and transition still describe the timepoint
and event resources managed by cancellation. The existing failure family
(`premature or unordered transition`, displayed as *operation before required
event*) describes the argument read before mutex acquisition. These labels
are unchanged; the primary obligation and repair category are corrected.

The maintainer's archived comments identify `timepoint.base.semaphore` as the
wrong field for the cancel argument and prescribe a separate pointer in the
enclosing task timepoint. This check preserves the historical evidence flags;
it does not re-audit the study's evidence-strength subsets.
