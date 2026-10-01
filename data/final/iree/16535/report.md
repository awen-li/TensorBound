# Switch from cuda1 to cuda2 introduces regression in NCCL collectives

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/16535](https://github.com/iree-org/iree/issues/16535)
- Opened: 2024-02-22T21:06:04Z
- Closed: 2024-03-02T22:24:00Z
- Labels: `bug 🐞`

## Reported behavior

### What happened?

I noticed that the swap from cuda1 to cuda2 driver may have caused a regression in the collectives HAL implementation.
```
boian-nvidia-gpus:7003:7022 [0] group.cc:274 NCCL WARN Cuda failure 'operation not permitted'

boian-nvidia-gpus:7003:7022 [0] init.cc:1786 NCCL WARN Cuda failure 'operation not permitted'
[boian-nvidia-gpus:07003] *** Process received signal ***
[boian-nvidia-gpus:07003] Signal: Segmentation fault (11)
[boian-nvidia-gpus:07003] Signal code:  (128)
[boian-nvidia-gpus:07003] Failing at address: (nil)
```

On commit e31b5a2e0476f0b7d0cf1b32869509554044c2c7, just before switching to cuda2. I don't get an error.

On another system there is no error at all.

### Steps to reproduce your issue

Running the test
```
NCCL_DEBUG=WARN python tests/e2e/collectives/collectives_test.py --target_backend=cuda --driver=cuda --verbose -k SingleRank.test_stablehlo_all_reduce
```


### What component(s) does this issue relate to?

Runtime

### Version information

4a80ee36d52b35be25bf30ea6a0f9b423477f6e9

### Additional context

Configure command:
```bash
export BUILD_TYPE=Debug

cmake \
  -GNinja \
  -B "$SCRIPT_DIR/build/ninja/$BUILD_TYPE" \
  -S "$SCRIPT_DIR/repo" \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DIREE_ENABLE_ASSERTIONS=ON \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_INCLUDE_TESTS=ON \
  -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF \
  -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DCMAKE_C_COMPILER=clang-15 \
  -DCMAKE_CXX_COMPILER=clang++-15 \
  -DPython3_EXECUTABLE="$(which python)" \
  -DIREE_TARGET_BACKEND_VULKAN_SPIRV=ON \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
  -DIREE_TARGET_BACKEND_CUDA=ON \
  -DIREE_HAL_DRIVER_CUDA=ON \
  -DIREE_ENABLE_LLD=ON \
  -DIREE_ENABLE_CCACHE=OFF \
  -DCMAKE_CXX_FLAGS=-gdwarf-4 \
  -DCMAKE_C_FLAGS=-gdwarf-4 \
  -DIREE_BUILD_EXPERIMENTAL_SPLIT_MLIR=OFF \
  -UIREE_COMPILER_PLUGIN_PATHS \
  -UIREE_EXTERNAL_HAL_DRIVERS \
  -DIREE_BUILD_EXPERIMENTAL_DISTRIBUTED=ON \
  -DIREE_ENABLE_COLLECTIVE_RUNTIME_TESTS=ON
```
Platform:
Tesla T4 GPU
CUDA version 12.3
NCCL version 2.19.3

## Trigger

Queue shutdown overlaps a launched CUDA host callback that will re-enqueue and clean up an action.

## Relevant entities

worker thread; pending action; CUDA host callback; ready list; state notification

## Implemented relation

An action remains live and the worker remains available until its asynchronous host callback completes and cleanup finishes.

## Pre-state

The worker launches a host callback for an action that is not yet complete.

## Binding transition

The old exit condition observes an empty ready list and idle worker without counting the callback-owned action.

## Post-state

The worker may exit or sleep before the callback re-enqueues work, and cleanup may race action access.

## Validity obligation

Shutdown waits for pending_action_count to reach zero, and callback enqueue wakes the worker.

## Invalid decision

Worker liveness tracked only immediately visible ready work, not actions executing through host callbacks.

## Physical effect

The callback accesses queue/action state after the worker's lifecycle decision or leaves new work unprocessed.

## Observed consequence

NCCL collectives report CUDA operation-not-permitted errors and may segfault.

## Repair site

runtime/src/iree/hal/drivers/cuda/pending_queue_actions.c

## Repair

PR 16621 counts pending actions until cleanup, posts the state notification from callbacks, delays worker exit until the count is zero, and mirrors the protocol in HIP.

## Why the repair works

The lifetime counter covers the callback gap and the notification makes callback-produced work visible to a sleeping worker.

## Causal basis

Maintainers identify the callback/worker race, and the patch implements both proposed counter and wakeup edges.

## Analysis implication

Async queue analysis needs action state, callback ownership, pending count, worker state, notification edges, and cleanup order.

## Fix patches

- [`fixes/iree-16535-pr16621.patch`](fixes/iree-16535-pr16621.patch) — https://github.com/iree-org/iree/pull/16621
