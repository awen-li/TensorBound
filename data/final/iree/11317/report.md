# ASan failure in iree_task_list_erase

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/11317](https://github.com/iree-org/iree/issues/11317)
- Opened: 2022-11-28T17:38:48Z
- Closed: 2022-11-29T20:25:53Z
- Labels: `bug 🐞`, `hal/cpu`

## Reported behavior

I encountered a failure test but it seems to be unrelated to my change:
https://github.com/iree-org/iree/actions/runs/3536604916/jobs/5935790728

The issue also disappeared after I made a minor change and rerun.

ASan dump:
```
==56644==ERROR: AddressSanitizer: stack-use-after-return on address 0x7fbc3272e130 at pc 0x557e07d99d87 bp 0x7fbc310febe0 sp 0x7fbc310febd8
READ of size 8 at 0x7fbc3272e130 thread T255
    #0 0x557e07d99d86 in iree_task_list_erase /work/runtime/src/iree/task/list.c:90:24
    #1 0x557e07d9ad2d in iree_task_poller_prepare_wait /work/runtime/src/iree/task/poller.c:387:9
    #2 0x557e07d9ad2d in iree_task_poller_pump_until_exit /work/runtime/src/iree/task/poller.c:499:5
    #3 0x557e07d9ad2d in iree_task_poller_main /work/runtime/src/iree/task/poller.c:533:5
    #4 0x557e07da945c in iree_thread_start_routine /work/runtime/src/iree/base/internal/threading_pthreads.c:130:29
    #5 0x7fbc3429ab42  (/lib/x86_64-linux-gnu/libc.so.6+0x94b42) (BuildId: 69389d485a9793dbe873f0ea2c93e02efaa9aa3d)
    #6 0x7fbc3432bbb3 in __clone (/lib/x86_64-linux-gnu/libc.so.6+0x125bb3) (BuildId: 69389d485a9793dbe873f0ea2c93e02efaa9aa3d)

Address 0x7fbc3272e130 is located in stack of thread T0 at offset 304 in frame
    #0 0x557e07d92edf in (anonymous namespace)::TaskWaitTest_WaitAllTimeout_Test::TestBody() /work/runtime/src/iree/task/task_test_wait.cc:167
```

## Trigger

A wait retires while the poller still holds the task in its current traversal and wait list.

## Relevant entities

wait task; poller wait list; poller traversal; task owner; pending submission; retire status

## Implemented relation

A wait task remains alive while the poller holds a list or traversal reference to it.

## Pre-state

The task is present in the poller wait list and the prepare loop still holds its address.

## Binding transition

iree_task_poller_prepare_task calls iree_task_wait_retire before its caller erases the task from the wait list.

## Post-state

Retirement can wake the owner, which can destroy the task while the poller still reads it to erase the list entry.

## Validity obligation

The poller removes every poller-owned reference before retirement publishes completion to the task owner.

## Invalid decision

The old poller retired the wait task before removing it from the poller's working set and wait list.

## Physical effect

The poller dereferences task memory after the owner has reclaimed the stack-resident task.

## Observed consequence

AddressSanitizer reports a stack-use-after-return in iree_task_list_erase.

## Repair site

runtime/src/iree/task/poller.c::iree_task_poller_prepare_task and iree_task_poller_prepare_wait

## Repair

PR 11352 returns the retire status to the caller, erases the task from the wait list, retires it afterward, and clears the now-invalid local pointer.

## Why the repair works

No poller list or traversal reference remains when retirement wakes code that may reclaim the task.

## Causal basis

The maintainer diagnosis names the retire-before-erase order, and the linked patch moves those two operations in that exact path.

## Analysis implication

Asynchronous lifetime analysis needs list membership, traversal ownership, completion publication, wakeup, and reclamation order.

## Fix patches

- [`fixes/iree-11317-pr11352.patch`](fixes/iree-11317-pr11352.patch) — https://github.com/iree-org/iree/pull/11352
