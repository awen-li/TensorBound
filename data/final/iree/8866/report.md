# (Another!) data race: `iree_task_worker_try_steal_task` vs `iree_task_executor_destroy`

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/8866](https://github.com/iree-org/iree/issues/8866)
- Opened: 2022-04-13T02:43:14Z
- Closed: 2022-04-13T15:01:54Z
- Labels: `bug 🐞`

## Reported behavior

[TSan report](https://gist.github.com/bjacob/150028432a6a791f87a203994d5f6763)

This might be the same as #8863?

During teardown, one worker is trying to steal work while the main thread is already destroying the executor.

Reproduce: `cmake -DIREE_ENABLE_TSAN=ON` and run these tests hundreds of times in parallel (running multiple tests in parallel introduces timing fluctuations that are in my experience key to triggering TSan failures).
```
ctest -j128 --timeout 30 -R iree/hal/dylib/cts/dylib --repeat-until-fail 100
```

This particular failure occurred in `dylib_buffer_mapping_test ` but it is harder to reproduce in a `ctest` command running only that test.

## Duplicate mapping

- Primary report: `github-iree-org-iree-8863`
- Shared cause: `cause-iree-8863-worker-deinitialized-before-peer-exit`

The reporter closes the case with the explicit statement that it is the same failure as issue 8863; the reported try_steal versus executor_destroy race is the primary case's exact access pair.

The sections below describe the shared cause reconstructed from the primary report. The reported behavior above remains the observation specific to this report.

## Trigger

Executor destruction overlaps a live worker's attempt to steal from another worker selected for teardown.

## Relevant entities

executor; worker threads; peer worker queues; task poller; worker data structures

## Implemented relation

Every worker and poller data structure remains valid until all threads that can access it have exited.

## Pre-state

Multiple workers are live and may inspect peer queues while processing shutdown.

## Binding transition

The old executor teardown deinitializes workers one at a time while later workers may still run.

## Post-state

A live worker can steal from a peer whose queue and synchronization state have already been deinitialized.

## Validity obligation

Shutdown requests all exits, joins every worker and the poller, and only then deinitializes shared worker structures.

## Invalid decision

Executor teardown combined join and deinitialization per worker instead of separating global thread exit from data-structure destruction.

## Physical effect

A worker reads peer state concurrently with destruction of that state.

## Observed consequence

ThreadSanitizer reports a race between iree_task_worker_try_steal_task and iree_task_executor_destroy.

## Repair site

iree/task/executor.c with worker.c and poller.c shutdown APIs

## Repair

PR 8873 splits request, await-exit, and deinitialize operations; the executor requests all exits, joins all workers and the poller, then deinitializes them.

## Why the repair works

No worker or poller thread remains capable of touching peer state when teardown begins destroying that state.

## Causal basis

The report identifies peer stealing during destruction, and the merged patch explicitly separates global thread exit from worker and poller deinitialization.

## Analysis implication

Concurrent executor-lifetime analysis needs the complete request/join/deinitialize order across all mutually accessible workers, not one object's local shutdown state.

## Fix patches

- [`fixes/iree-8866-pr8873.patch`](fixes/iree-8866-pr8873.patch) — https://github.com/iree-org/iree/pull/8873
