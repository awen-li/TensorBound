# Shutdown deadlock in `iree_task_worker_deinitialize` -> `iree_notification_await` as the thread is already gone

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/8871](https://github.com/iree-org/iree/issues/8871)
- Opened: 2022-04-13T15:14:18Z
- Closed: 2022-06-30T21:07:43Z
- Labels: `bug 🐞`, `runtime`

## Reported behavior

This reproduces every ~1000 runs of `dylib` unit tests:
```
ctest -j20 -R iree/hal/dylib/cts/dylib --repeat-until-fail 1000 --output-on-failure
```

The test just hangs. Attaching GDB to it, I get [this stack](https://gist.github.com/bjacob/04bf9cd81f4ddf7786f4c0af0de3141c). According to GDB there is no other thread anymore (other than a TSan-internal thread) --- this is a single-threaded program at this point.

So `iree_task_worker_deinitialize` is awating a notification,

```
  if (worker->thread) {
    iree_notification_await(&worker->state_notification,
                            (iree_condition_fn_t)iree_task_worker_is_zombie,
                            worker, iree_infinite_timeout());
  }
```

but it will never come as there is no other thread left to send it.

## Trigger

A worker reaches zombie state and posts its state notification after the waiting thread checks the condition but before that thread enters pthread_cond_timedwait.

## Relevant entities

worker state; notification epoch; wait token; pthread mutex; condition variable; waiting thread

## Implemented relation

A notification wait either observes a satisfied condition or detects every notification posted after its wait token was captured.

## Pre-state

The waiting thread has captured the current epoch and found that the worker is not yet zombie.

## Binding transition

The old pthread fallback checks the condition outside the mutex and then enters pthread_cond_timedwait without comparing a protected epoch to the captured wait token.

## Post-state

The worker's post occurs before the wait starts, and the waiting thread sleeps although the condition is already true and no thread remains to signal again.

## Validity obligation

Prepare, condition check, and commit use a monotonic notification token so a post in the check-to-wait interval cancels the sleep.

## Invalid decision

The pthread implementation signaled a condition variable but did not maintain the event-count epoch used by the notification contract.

## Physical effect

The main thread waits indefinitely after the worker has exited and entered zombie state.

## Observed consequence

The dylib test intermittently hangs during iree_task_worker_deinitialize, roughly once per thousand repeated runs.

## Repair site

iree/base/internal/synchronization.c pthread implementation of iree_notification_t

## Repair

PR 8877 gives the pthread path a mutex-protected epoch and waiter count; prepare snapshots the epoch, post increments it, and commit waits only while the epoch still equals the captured token.

## Why the repair works

A post between the condition check and commit changes the epoch, so commit observes the mismatch and returns instead of entering a wait that no producer can wake.

## Causal basis

The issue discussion derives the missed-post interleaving, the reporter validates the fix over repeated tests, and the merged PR states that it integrates the fix for this pthread deadlock.

## Analysis implication

Shutdown-wait analysis needs condition state, notification epoch, token capture, post order, mutex coverage, waiter registration, and thread lifetime.

## Fix patches

- [`fixes/iree-8871-pr8877.patch`](fixes/iree-8871-pr8877.patch) — https://github.com/iree-org/iree/pull/8877
