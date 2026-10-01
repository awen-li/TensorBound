# rare shutdown deadlock in `executor_test`

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/9723](https://github.com/iree-org/iree/issues/9723)
- Opened: 2022-07-06T18:22:11Z
- Closed: 2022-07-06T20:04:55Z
- Labels: `bug 🐞`

## Reported behavior

I'm running 1000 runs of `executor_test` built with TSan and about 0.5% of them end up in a deadlock like this:

```
Thread 1 (Thread 0x7fb2c31ca340 (LWP 2757465) "executor_test"):
#0  0x00007fb2c36080fa in __futex_abstimed_wait_common64 (futex_word=futex_word@entry=0x7fb2c1ebf180, expected=expected@entry=0, clockid=clockid@entry=0, abstime=abstime@entry=0x7ffd8c09b5d8, private=private@entry=0, cancel=cancel@entry=true) at ../sysdeps/nptl/futex-internal.c:74
#1  0x00007fb2c360815b in __GI___futex_abstimed_wait_cancelable64 (futex_word=futex_word@entry=0x7fb2c1ebf180, expected=expected@entry=0, clockid=clockid@entry=0, abstime=abstime@entry=0x7ffd8c09b5d8, private=private@entry=0) at ../sysdeps/nptl/futex-internal.c:123
#2  0x00007fb2c3601f44 in __pthread_cond_wait_common (abstime=0x7ffd8c09b5d8, clockid=0, mutex=0x7fb2c1ebf130, cond=0x7fb2c1ebf158) at pthread_cond_wait.c:504
#3  __pthread_cond_timedwait (cond=0x7fb2c1ebf158, mutex=0x7fb2c1ebf130, abstime=0x7ffd8c09b5d8) at pthread_cond_wait.c:637
#4  0x0000000000315fa0 in __tsan::call_pthread_cancel_with_cleanup(int (*)(void*), void (*)(void*), void*) ()
#5  0x00000000002a39b6 in pthread_cond_timedwait ()
#6  0x0000000000327f4d in iree_notification_commit_wait (notification=notification@entry=0x7fb2c1ebf130, wait_token=wait_token@entry=0, deadline_ns=deadline_ns@entry=9223372036854775807) at /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/internal/synchronization.c:618
#7  0x00000000003280ce in iree_notification_await (notification=notification@entry=0x7fb2c1ebf130, condition_fn=0x31eac0 <iree_task_poller_is_zombie>, condition_arg=condition_arg@entry=0x7fb2c1ebf120, timeout=...) at /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/internal/synchronization.c:775
#8  0x000000000031eab3 in iree_task_poller_await_exit (poller=poller@entry=0x7fb2c1ebf120) at /usr/local/google/home/benoitjacob/iree/runtime/src/iree/task/poller.c:124
#9  0x000000000031ccb4 in iree_task_executor_destroy (executor=executor@entry=0x7fb2c1ebf000) at /usr/local/google/home/benoitjacob/iree/runtime/src/iree/task/executor.c:196
#10 0x000000000031cde2 in iree_task_executor_release (executor=0x7fb2c1ebf000) at /usr/local/google/home/benoitjacob/iree/runtime/src/iree/task/executor.c:224
#11 0x0000000000318bfa in (anonymous namespace)::ExecutorTest_Lifetime_Test::TestBody (this=this@entry=0x7b04000000c0) at /usr/local/google/home/benoitjacob/iree/runtime/src/iree/task/executor_test.cc:33
#12 0x0000000000370d90 in testing::internal::HandleSehExceptionsInMethodIfSupported<testing::Test, void> (method=<optimized out>, location=0x238279 "the test body", object=<optimized out>) at /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2607
#13 testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void> (object=object@entry=0x7b04000000c0, method=<optimized out>, location=0x238279 "the test body") at /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2643
#14 0x000000000033b670 in testing::Test::Run (this=this@entry=0x7b04000000c0) at /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2682
#15 0x000000000033d263 in testing::TestInfo::Run (this=0x7b4400000000) at /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2861
#16 0x000000000033e14d in testing::TestSuite::Run (this=<optimized out>) at /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:3015
#17 0x0000000000354c19 in testing::internal::UnitTestImpl::RunAllTests (this=this@entry=0x7b5400000000) at /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:5855
#18 0x00000000003720c0 in testing::internal::HandleSehExceptionsInMethodIfSupported<testing::internal::UnitTestImpl, bool> (method=<optimized out>, location=0x2299d5 "auxiliary test code (environments or event listeners)", object=<optimized out>) at /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2607
#19 testing::internal::HandleExceptionsInMethodIfSupported<testing::internal::UnitTestImpl, bool> (object=object@entry=0x7b5400000000, method=<optimized out>, location=0x2299d5 "auxiliary test code (environments or event listeners)") at /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2643
#20 0x0000000000353d61 in testing::UnitTest::Run (this=<optimized out>) at /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:5438
#21 0x0000000000324bec in RUN_ALL_TESTS () at /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/include/gtest/gtest.h:2490
#22 main (argc=2, argv=<optimized out>) at /usr/local/google/home/benoitjacob/iree/runtime/src/iree/testing/gtest_main.cc:17
(gdb) thread apply 2 bt

Thread 2 (Thread 0x7fb2c0fff640 (LWP 2758612) "executor_test"):
#0  0x0000000000282ffe in __sanitizer::internal_usleep(unsigned long long) ()
#1  0x000000000030a46f in __tsan::BackgroundThread(void*) ()
#2  0x00007fb2c35fbd80 in start_thread (arg=0x7fb2c0fff640) at pthread_create.c:481
#3  0x00007fb2c32c876f in clone () at ../sysdeps/unix/sysv/linux/x86_64/clone.S:95
(gdb) thread apply 3 bt

Thread 3 (Thread 0x7fb2c07fe640 (LWP 2809666) "iree-poller"):
#0  0x00007fb2c32bc976 in __ppoll (fds=0x7b6c00000420, nfds=1, timeout=<optimized out>, sigmask=0x0) at ../sysdeps/unix/sysv/linux/ppoll.c:48
#1  0x00000000002be6bb in ppoll ()
#2  0x000000000032710f in iree_syscall_poll (fds=0x7b6c00000420, nfds=nfds@entry=1, deadline_ns=deadline_ns@entry=9223372036854775807, out_signaled_count=out_signaled_count@entry=0x7fb2c07be07c) at /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/internal/wait_handle_poll.c:90
#3  0x0000000000327204 in iree_wait_any (set=0x7b6c00000000, deadline_ns=deadline_ns@entry=9223372036854775807, out_wake_handle=out_wake_handle@entry=0x7fb2c07be178) at /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/internal/wait_handle_poll.c:353
#4  0x000000000031e948 in iree_task_poller_commit_wait (poller=0x7fb2c1ebf120, deadline_ns=9223372036854775807) at /usr/local/google/home/benoitjacob/iree/runtime/src/iree/task/poller.c:426
#5  iree_task_poller_pump_until_exit (poller=0x7fb2c1ebf120) at /usr/local/google/home/benoitjacob/iree/runtime/src/iree/task/poller.c:504
#6  iree_task_poller_main (poller=poller@entry=0x7fb2c1ebf120) at /usr/local/google/home/benoitjacob/iree/runtime/src/iree/task/poller.c:527
#7  0x00000000003262be in iree_thread_start_routine (param=0x7b4000000100) at /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/internal/threading_pthreads.c:130
#8  0x00000000002a29cd in __tsan_thread_start_func ()
#9  0x00007fb2c35fbd80 in start_thread (arg=0x7fb2c07fe640) at pthread_create.c:481
#10 0x00007fb2c32c876f in clone () at ../sysdeps/unix/sysv/linux/x86_64/clone.S:95
```

## Trigger

The owner requests exit after the poller checks its state but before the poller resets and enters its wait.

## Relevant entities

poller state; wake event; owner thread; poller thread; system wait

## Implemented relation

An exit request remains observable until the poller either consumes it or declines to block.

## Pre-state

The poller has observed a running state and is preparing to wait; the owner can concurrently set EXITING and signal wake_event.

## Binding transition

The poller resets wake_event after the owner's signal and enters an infinite system wait without rechecking state.

## Post-state

The EXITING state is set but its only wake event has been lost, leaving the poller asleep.

## Validity obligation

The poller rechecks the exit state at the wait-commit boundary after any operation that can clear a concurrent wake signal.

## Invalid decision

The old wait commit path could clear an exit wakeup and block without another state check.

## Physical effect

The poller never reaches its zombie state and executor teardown waits indefinitely.

## Observed consequence

Approximately 0.5 percent of repeated executor tests deadlock during shutdown.

## Repair site

runtime/src/iree/task/poller.c::iree_task_poller_commit_wait

## Repair

PR 9724 adds a sequentially consistent EXITING check immediately before entering the system wait and returns when shutdown is pending.

## Why the repair works

Even when the wake event is cleared by the race, the persistent state check prevents the poller from blocking.

## Causal basis

The PR description reconstructs the exact lost-wakeup interleaving, and the patch adds the state check at the wait boundary where that interleaving blocks.

## Analysis implication

Wait/notification analysis needs both event state and the associated persistent predicate at the final blocking boundary.

## Fix patches

- [`fixes/iree-9723-pr9724.patch`](fixes/iree-9723-pr9724.patch) — https://github.com/iree-org/iree/pull/9724
