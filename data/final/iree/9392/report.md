# data race in `iree_hal_task_semaphore_multi_wait`

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/9392](https://github.com/iree-org/iree/issues/9392)
- Opened: 2022-06-08T17:48:04Z
- Closed: 2022-06-09T03:08:22Z
- Labels: `bug 🐞`

## Reported behavior

Repro: enable TSan and run this test:

```
ctest --output-on-failure -R iree/hal/drivers/local_task/cts/local-task_embedded-elf_semaphore_submission_test
```

TSan error:

```
WARNING: ThreadSanitizer: data race (pid=3419395)
  Read of size 8 at 0x7bb000008070 by main thread:
    #0 iree_hal_task_semaphore_multi_wait /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/drivers/local_task/task_semaphore.c:392:60 (local-task_embedded-elf_semaphore_submission_test+0x34a3ad)
    #1 iree_hal_task_device_wait_semaphores /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/drivers/local_task/task_device.c:353:10 (local-task_embedded-elf_semaphore_submission_test+0x3465ae)
    #2 iree_hal_device_wait_semaphores /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/device.c:283:26 (local-task_embedded-elf_semaphore_submission_test+0x344325)
    #3 iree::hal::cts::semaphore_submission_test_SubmitWithMultipleSemaphores_Test::TestBody() /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/cts/semaphore_submission_test.h:194:3 (local-task_embedded-elf_semaphore_submission_test+0x330288)
    #4 void testing::internal::HandleSehExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2607:10 (local-task_embedded-elf_semaphore_submission_test+0x3ccfbf)
    #5 void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2643:14 (local-task_embedded-elf_semaphore_submission_test+0x3ccfbf)
    #6 testing::Test::Run() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2682:5 (local-task_embedded-elf_semaphore_submission_test+0x39791f)
    #7 testing::TestInfo::Run() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2861:11 (local-task_embedded-elf_semaphore_submission_test+0x399512)
    #8 testing::TestSuite::Run() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:3015:28 (local-task_embedded-elf_semaphore_submission_test+0x39a3fc)
    #9 testing::internal::UnitTestImpl::RunAllTests() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:5855:44 (local-task_embedded-elf_semaphore_submission_test+0x3b0ec8)
    #10 bool testing::internal::HandleSehExceptionsInMethodIfSupported<testing::internal::UnitTestImpl, bool>(testing::internal::UnitTestImpl*, bool (testing::internal::UnitTestImpl::*)(), char const*) /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2607:10 (local-task_embedded-elf_semaphore_submission_test+0x3ce2ef)
    #11 bool testing::internal::HandleExceptionsInMethodIfSupported<testing::internal::UnitTestImpl, bool>(testing::internal::UnitTestImpl*, bool (testing::internal::UnitTestImpl::*)(), char const*) /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2643:14 (local-task_embedded-elf_semaphore_submission_test+0x3ce2ef)
    #12 testing::UnitTest::Run() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:5438:10 (local-task_embedded-elf_semaphore_submission_test+0x3b0010)
    #13 RUN_ALL_TESTS() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/include/gtest/gtest.h:2490:46 (local-task_embedded-elf_semaphore_submission_test+0x3456db)
    #14 main /usr/local/google/home/benoitjacob/iree/runtime/src/iree/testing/gtest_main.cc:17:10 (local-task_embedded-elf_semaphore_submission_test+0x3456db)

  Previous write of size 8 at 0x7bb000008070 by thread T5 (mutexes: write M623):
    #0 memset <null> (local-task_embedded-elf_semaphore_submission_test+0x2c0ebb)
    #1 iree_hal_semaphore_issue_timepoint_callback /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/utils/semaphore_base.c:80:3 (local-task_embedded-elf_semaphore_submission_test+0x34c8e4)
    #2 iree_hal_semaphore_issue_timepoint_callbacks /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/utils/semaphore_base.c:105:5 (local-task_embedded-elf_semaphore_submission_test+0x34c8e4)
    #3 iree_hal_semaphore_resolve_timepoints /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/utils/semaphore_base.c:157:3 (local-task_embedded-elf_semaphore_submission_test+0x34c8e4)
    #4 iree_hal_semaphore_notify /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/utils/semaphore_base.c:286:5 (local-task_embedded-elf_semaphore_submission_test+0x34c8e4)
    #5 iree_hal_task_semaphore_signal /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/drivers/local_task/task_semaphore.c:169:3 (local-task_embedded-elf_semaphore_submission_test+0x34a5d6)
    #6 iree_hal_semaphore_signal /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/semaphore.c:52:7 (local-task_embedded-elf_semaphore_submission_test+0x3453a4)
    #7 iree_hal_task_queue_retire_cmd /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/drivers/local_task/task_queue.c:308:9 (local-task_embedded-elf_semaphore_submission_test+0x349449)
    #8 iree_task_call_execute /usr/local/google/home/benoitjacob/iree/runtime/src/iree/task/task.c:256:28 (local-task_embedded-elf_semaphore_submission_test+0x374926)
    #9 iree_task_worker_execute /usr/local/google/home/benoitjacob/iree/runtime/src/iree/task/worker.c:193:7 (local-task_embedded-elf_semaphore_submission_test+0x376bda)
    #10 iree_task_worker_pump_once /usr/local/google/home/benoitjacob/iree/runtime/src/iree/task/worker.c:256:3 (local-task_embedded-elf_semaphore_submission_test+0x376bda)
    #11 iree_task_worker_pump_until_exit /usr/local/google/home/benoitjacob/iree/runtime/src/iree/task/worker.c:304:12 (local-task_embedded-elf_semaphore_submission_test+0x376bda)
    #12 iree_task_worker_main /usr/local/google/home/benoitjacob/iree/runtime/src/iree/task/worker.c:378:5 (local-task_embedded-elf_semaphore_submission_test+0x376bda)
    #13 iree_thread_start_routine /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/internal/threading_pthreads.c:130:29 (local-task_embedded-elf_semaphore_submission_test+0x38215d)

  Location is heap block of size 32768 at 0x7bb000008000 allocated by main thread:
    #0 malloc <null> (local-task_embedded-elf_semaphore_submission_test+0x2b5f0d)
    #1 iree_allocator_system_alloc /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/allocator.c:88:17 (local-task_embedded-elf_semaphore_submission_test+0x339f67)
    #2 iree_allocator_system_ctl /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/allocator.c:126:14 (local-task_embedded-elf_semaphore_submission_test+0x339f67)
    #3 iree_allocator_issue_alloc /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/allocator.c:27:10 (local-task_embedded-elf_semaphore_submission_test+0x339cf3)
    #4 iree_allocator_malloc_uninitialized /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/allocator.c:38:10 (local-task_embedded-elf_semaphore_submission_test+0x339cf3)
    #5 iree_arena_block_pool_acquire /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/internal/arena.c:75:5 (local-task_embedded-elf_semaphore_submission_test+0x34c1f5)
    #6 iree_arena_allocate /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/internal/arena.c:172:5 (local-task_embedded-elf_semaphore_submission_test+0x34c1f5)
    #7 iree_arena_allocator_ctl /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/internal/arena.c:200:7 (local-task_embedded-elf_semaphore_submission_test+0x34c356)
    #8 iree_allocator_issue_alloc /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/allocator.c:27:10 (local-task_embedded-elf_semaphore_submission_test+0x339c76)
    #9 iree_allocator_malloc /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/allocator.c:32:10 (local-task_embedded-elf_semaphore_submission_test+0x339c76)
    #10 iree_wait_set_allocate /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/internal/wait_handle_poll.c:157:3 (local-task_embedded-elf_semaphore_submission_test+0x377761)
    #11 iree_hal_task_semaphore_multi_wait /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/drivers/local_task/task_semaphore.c:345:26 (local-task_embedded-elf_semaphore_submission_test+0x34a139)
    #12 iree_hal_task_device_wait_semaphores /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/drivers/local_task/task_device.c:353:10 (local-task_embedded-elf_semaphore_submission_test+0x3465ae)
    #13 iree_hal_device_wait_semaphores /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/device.c:283:26 (local-task_embedded-elf_semaphore_submission_test+0x344325)
    #14 iree::hal::cts::semaphore_submission_test_SubmitWithMultipleSemaphores_Test::TestBody() /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/cts/semaphore_submission_test.h:194:3 (local-task_embedded-elf_semaphore_submission_test+0x330288)
    #15 void testing::internal::HandleSehExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2607:10 (local-task_embedded-elf_semaphore_submission_test+0x3ccfbf)
    #16 void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2643:14 (local-task_embedded-elf_semaphore_submission_test+0x3ccfbf)
    #17 testing::Test::Run() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2682:5 (local-task_embedded-elf_semaphore_submission_test+0x39791f)
    #18 testing::TestInfo::Run() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2861:11 (local-task_embedded-elf_semaphore_submission_test+0x399512)
    #19 testing::TestSuite::Run() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:3015:28 (local-task_embedded-elf_semaphore_submission_test+0x39a3fc)
    #20 testing::internal::UnitTestImpl::RunAllTests() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:5855:44 (local-task_embedded-elf_semaphore_submission_test+0x3b0ec8)
    #21 bool testing::internal::HandleSehExceptionsInMethodIfSupported<testing::internal::UnitTestImpl, bool>(testing::internal::UnitTestImpl*, bool (testing::internal::UnitTestImpl::*)(), char const*) /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2607:10 (local-task_embedded-elf_semaphore_submission_test+0x3ce2ef)
    #22 bool testing::internal::HandleExceptionsInMethodIfSupported<testing::internal::UnitTestImpl, bool>(testing::internal::UnitTestImpl*, bool (testing::internal::UnitTestImpl::*)(), char const*) /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2643:14 (local-task_embedded-elf_semaphore_submission_test+0x3ce2ef)
    #23 testing::UnitTest::Run() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:5438:10 (local-task_embedded-elf_semaphore_submission_test+0x3b0010)
    #24 RUN_ALL_TESTS() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/include/gtest/gtest.h:2490:46 (local-task_embedded-elf_semaphore_submission_test+0x3456db)
    #25 main /usr/local/google/home/benoitjacob/iree/runtime/src/iree/testing/gtest_main.cc:17:10 (local-task_embedded-elf_semaphore_submission_test+0x3456db)

  Mutex M623 (0x7b2800000150) created at:
    #0 pthread_mutex_init <null> (local-task_embedded-elf_semaphore_submission_test+0x2b8d0f)
    #1 iree_mutex_initialize /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/internal/synchronization.c:240:3 (local-task_embedded-elf_semaphore_submission_test+0x382526)
    #2 iree_slim_mutex_initialize /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/internal/synchronization.c:494:3 (local-task_embedded-elf_semaphore_submission_test+0x382526)
    #3 iree_hal_semaphore_initialize /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/utils/semaphore_base.c:197:3 (local-task_embedded-elf_semaphore_submission_test+0x34c418)
    #4 iree_hal_task_semaphore_create /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/drivers/local_task/task_semaphore.c:91:5 (local-task_embedded-elf_semaphore_submission_test+0x349ce9)
    #5 iree_hal_task_device_create_semaphore /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/drivers/local_task/task_device.c:302:10 (local-task_embedded-elf_semaphore_submission_test+0x3463bd)
    #6 iree_hal_semaphore_create /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/semaphore.c:29:7 (local-task_embedded-elf_semaphore_submission_test+0x3452ed)
    #7 iree::hal::cts::semaphore_submission_test_SubmitWithMultipleSemaphores_Test::TestBody() /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/cts/semaphore_submission_test.h:155:3 (local-task_embedded-elf_semaphore_submission_test+0x32f6ab)
    #8 void testing::internal::HandleSehExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2607:10 (local-task_embedded-elf_semaphore_submission_test+0x3ccfbf)
    #9 void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2643:14 (local-task_embedded-elf_semaphore_submission_test+0x3ccfbf)
    #10 testing::Test::Run() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2682:5 (local-task_embedded-elf_semaphore_submission_test+0x39791f)
    #11 testing::TestInfo::Run() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2861:11 (local-task_embedded-elf_semaphore_submission_test+0x399512)
    #12 testing::TestSuite::Run() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:3015:28 (local-task_embedded-elf_semaphore_submission_test+0x39a3fc)
    #13 testing::internal::UnitTestImpl::RunAllTests() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:5855:44 (local-task_embedded-elf_semaphore_submission_test+0x3b0ec8)
    #14 bool testing::internal::HandleSehExceptionsInMethodIfSupported<testing::internal::UnitTestImpl, bool>(testing::internal::UnitTestImpl*, bool (testing::internal::UnitTestImpl::*)(), char const*) /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2607:10 (local-task_embedded-elf_semaphore_submission_test+0x3ce2ef)
    #15 bool testing::internal::HandleExceptionsInMethodIfSupported<testing::internal::UnitTestImpl, bool>(testing::internal::UnitTestImpl*, bool (testing::internal::UnitTestImpl::*)(), char const*) /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2643:14 (local-task_embedded-elf_semaphore_submission_test+0x3ce2ef)
    #16 testing::UnitTest::Run() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:5438:10 (local-task_embedded-elf_semaphore_submission_test+0x3b0010)
    #17 RUN_ALL_TESTS() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/include/gtest/gtest.h:2490:46 (local-task_embedded-elf_semaphore_submission_test+0x3456db)
    #18 main /usr/local/google/home/benoitjacob/iree/runtime/src/iree/testing/gtest_main.cc:17:10 (local-task_embedded-elf_semaphore_submission_test+0x3456db)

  Thread T5 'iree-worker-0' (tid=3419425, running) created by main thread at:
    #0 pthread_create <null> (local-task_embedded-elf_semaphore_submission_test+0x2b759d)
    #1 iree_thread_create /usr/local/google/home/benoitjacob/iree/runtime/src/iree/base/internal/threading_pthreads.c:180:10 (local-task_embedded-elf_semaphore_submission_test+0x381c69)
    #2 iree_task_worker_initialize /usr/local/google/home/benoitjacob/iree/runtime/src/iree/task/worker.c:71:26 (local-task_embedded-elf_semaphore_submission_test+0x37683e)
    #3 iree_task_executor_create /usr/local/google/home/benoitjacob/iree/runtime/src/iree/task/executor.c:143:16 (local-task_embedded-elf_semaphore_submission_test+0x370709)
    #4 iree_task_executor_create_from_flags /usr/local/google/home/benoitjacob/iree/runtime/src/iree/task/api.c:104:14 (local-task_embedded-elf_semaphore_submission_test+0x3701c7)
    #5 iree_hal_local_task_driver_factory_try_create /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/drivers/local_task/registration/driver_module.c:55:14 (local-task_embedded-elf_semaphore_submission_test+0x339ab8)
    #6 iree_hal_driver_registry_try_create_by_name /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/driver_registry.c:349:14 (local-task_embedded-elf_semaphore_submission_test+0x3451b8)
    #7 iree::hal::cts::CtsTestBase::TryGetDriver(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, iree_hal_driver_t**) /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/cts/cts_test_base.h:151:28 (local-task_embedded-elf_semaphore_submission_test+0x33597b)
    #8 iree::hal::cts::CtsTestBase::SetUp() /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/cts/cts_test_base.h:48:28 (local-task_embedded-elf_semaphore_submission_test+0x332908)
    #9 void testing::internal::HandleSehExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2607:10 (local-task_embedded-elf_semaphore_submission_test+0x3ccfbf)
    #10 void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*) /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2643:14 (local-task_embedded-elf_semaphore_submission_test+0x3ccfbf)
    #11 testing::Test::Run() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2677:3 (local-task_embedded-elf_semaphore_submission_test+0x39780f)
    #12 testing::TestInfo::Run() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2861:11 (local-task_embedded-elf_semaphore_submission_test+0x399512)
    #13 testing::TestSuite::Run() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:3015:28 (local-task_embedded-elf_semaphore_submission_test+0x39a3fc)
    #14 testing::internal::UnitTestImpl::RunAllTests() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:5855:44 (local-task_embedded-elf_semaphore_submission_test+0x3b0ec8)
    #15 bool testing::internal::HandleSehExceptionsInMethodIfSupported<testing::internal::UnitTestImpl, bool>(testing::internal::UnitTestImpl*, bool (testing::internal::UnitTestImpl::*)(), char const*) /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2607:10 (local-task_embedded-elf_semaphore_submission_test+0x3ce2ef)
    #16 bool testing::internal::HandleExceptionsInMethodIfSupported<testing::internal::UnitTestImpl, bool>(testing::internal::UnitTestImpl*, bool (testing::internal::UnitTestImpl::*)(), char const*) /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:2643:14 (local-task_embedded-elf_semaphore_submission_test+0x3ce2ef)
    #17 testing::UnitTest::Run() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/src/gtest.cc:5438:10 (local-task_embedded-elf_semaphore_submission_test+0x3b0010)
    #18 RUN_ALL_TESTS() /usr/local/google/home/benoitjacob/iree/third_party/googletest/googletest/include/gtest/gtest.h:2490:46 (local-task_embedded-elf_semaphore_submission_test+0x3456db)
    #19 main /usr/local/google/home/benoitjacob/iree/runtime/src/iree/testing/gtest_main.cc:17:10 (local-task_embedded-elf_semaphore_submission_test+0x3456db)

SUMMARY: ThreadSanitizer: data race /usr/local/google/home/benoitjacob/iree/runtime/src/iree/hal/drivers/local_task/task_semaphore.c:392:60 in iree_hal_task_semaphore_multi_wait
```

## Trigger

A worker resolves a task timepoint while the waiting thread begins cancellation cleanup.

## Relevant entities

task timepoint; base timepoint; semaphore pointer; worker callback; waiter cancellation

## Implemented relation

Cancellation uses a stable semaphore identity throughout asynchronous callback reset and waiter cleanup.

## Pre-state

The embedded base timepoint refers to the semaphore acquired for the wait.

## Binding transition

The worker callback clears the base timepoint while the waiter reads base.semaphore to cancel it.

## Post-state

The cancellation target is concurrently mutated and may no longer identify the acquired semaphore.

## Validity obligation

A cleanup read of callback-mutated state is ordered with the conflicting reset.
The primary family is happens-before ordering. The caller keeps the semaphore
alive, and the enclosing timepoint storage remains allocated through cleanup.
These ownership contracts do not order the read against the callback write.

## Invalid decision

The old waiter recovered the semaphore identity from the same base subobject that asynchronous resolution resets.
The selected operation is the read of `timepoints[i].base.semaphore` at
pre-fix `task_semaphore.c:392`, before cancellation acquires `timepoint_mutex`.
The callback reset is permitted by its documented contract.

## Physical effect

Worker memset and waiter pointer read race on base.semaphore.

## Observed consequence

ThreadSanitizer reports a data race in iree_hal_task_semaphore_multi_wait.

## Repair site

runtime/src/iree/hal/drivers/local_task/task_semaphore.c

## Repair

Commit d83f4e46 stores a separate semaphore pointer in iree_hal_task_timepoint_t after acquisition and uses it for conditional cancellation.

## Why the repair works

The callback may clear the embedded base while the enclosing task timepoint retains the cleanup identity until the waiter finishes.
The repair therefore removes the conflicting access by preserving the identity
in a separate field. It leaves semaphore reference counting unchanged.

## Coding correction

The [source check](source-evidence.md) changes the primary obligation from
ownership and lifetime to happens-before ordering, and the repair category
from correct ownership or lifetime to preserve or propagate relation.
The cleanup read remains the selected decision. This is a revision-time
correction, separate from the earlier human review and archived A/B labels.

## Causal basis

The archived TSan trace and fixing commit identify the same base.semaphore read/reset race and the repair introduces the proposed stable pointer.

## Analysis implication

Asynchronous cleanup analysis needs stable object identity separated from callback-owned mutable completion state.

## Fix patches

- [`fixes/iree-9392-fix.patch`](fixes/iree-9392-fix.patch) — https://github.com/iree-org/iree/commit/d83f4e46b518ae3fdc46ceab82502e88ecb289ba
