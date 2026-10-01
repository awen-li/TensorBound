# keras_vision_model_ResNet50_test fails with weights='imagenet'

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/1660](https://github.com/iree-org/iree/issues/1660)
- Opened: 2020-04-24T17:02:02Z
- Closed: 2020-05-06T17:33:31Z
- Labels: `bug 🐞`, `integrations/tensorflow`, `integrations`

## Reported behavior

I tried using `ResNet50` with `weights=imagenet` in a Colab notebook using the `vmla` driver/backend and hit issues when trying to call the `predict` function on the compiled module. I see the same errors when I run `keras_vision_model_ResNet50_test` with some local changes.

Test file: https://github.com/google/iree/blob/master/integrations/tensorflow/e2e/keras_vision_model_test.py

Local changes I have when trying:

```
-INPUT_SHAPE = [1, 32, 32, 3]
+INPUT_SHAPE = [1, 224, 224, 3]

-        weights=None, include_top=False, input_shape=INPUT_SHAPE[1:])
+        weights="imagenet", include_top=True, input_shape=tuple(INPUT_SHAPE[1:]))
```

(the test passes with `weights=None` and the other changes)

Error message appears to be `Parameter 0 contains a reference to the wrong type; have hal.buffer but expected hal.allocator`, in context:

```
$ bazel test integrations/tensorflow/e2e:keras_vision_model_ResNet50_test --test_output=all
...
I0424 16:49:53.328980 139798114543424 tf_test_utils.py:520] Using backends from command line: tf,iree_vmla
...
I0424 16:50:19.922265 139798114543424 tf_test_utils.py:91] Saving raw TF input MLIR to: /usr/local/google/home/scotttodd/.cache/bazel/_bazel_scotttodd/9e82052937ca42451e1b3d0e47f9949a/sandbox/linux-sandbox/10517/execroot/iree_core/_tmp/ccf1a5e1919931ba41e4207858fa92c2/AppTest/raw_vmla.mlir
I0424 16:50:21.032403 139798114543424 tf_test_utils.py:101] Saving IREE input MLIR to: /usr/local/google/home/scotttodd/.cache/bazel/_bazel_scotttodd/9e82052937ca42451e1b3d0e47f9949a/sandbox/linux-sandbox/10517/execroot/iree_core/_tmp/ccf1a5e1919931ba41e4207858fa92c2/AppTest/input_vmla.mlir
I0424 16:50:23.630982 139798114543424 tf_test_utils.py:109] Saving compiled IREE module to: /usr/local/google/home/scotttodd/.cache/bazel/_bazel_scotttodd/9e82052937ca42451e1b3d0e47f9949a/sandbox/linux-sandbox/10517/execroot/iree_core/_tmp/ccf1a5e1919931ba41e4207858fa92c2/AppTest/compiled_vmla.vmfb
[ RUN      ] AppTest.test_application
E ./iree/base/api_util.h:30] INVALID_ARGUMENT: Parameter 0 contains a reference to the wrong type; have hal.buffer but expected hal.allocator (N4iree2vm3refI18iree_hal_allocatorEE); while executing hal.allocator.allocate.const
Created IREE driver vmla: <pyiree.rt.binding.HalDriver object at 0x7f225413edb0>
SystemContext driver=<pyiree.rt.binding.HalDriver object at 0x7f225413edb0>
[  FAILED  ] AppTest.test_application
[ RUN      ] AppTest.test_session
[  SKIPPED ] AppTest.test_session
======================================================================
ERROR: test_application (__main__.AppTest)
test_application (__main__.AppTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/usr/local/google/home/scotttodd/.cache/bazel/_bazel_scotttodd/9e82052937ca42451e1b3d0e47f9949a/sandbox/linux-sandbox/10517/execroot/iree_core/bazel-out/k8-opt/bin/integrations/tensorflow/e2e/keras_vision_model_ResNet50_test.runfiles/iree_core/integrations/tensorflow/e2e/keras_vision_model_test.py", line 103, in test_application
    self.modules.applications.all.predict(input_data).print().assert_all_close(
  File "/usr/local/google/home/scotttodd/.cache/bazel/_bazel_scotttodd/9e82052937ca42451e1b3d0e47f9949a/sandbox/linux-sandbox/10517/execroot/iree_core/bazel-out/k8-opt/bin/integrations/tensorflow/e2e/keras_vision_model_ResNet50_test.runfiles/iree_core/integrations/tensorflow/bindings/python/pyiree/tf/support/tf_test_utils.py", line 305, in __call__
    for backend, f in self._backend_function_dict.items()
  File "/usr/local/google/home/scotttodd/.cache/bazel/_bazel_scotttodd/9e82052937ca42451e1b3d0e47f9949a/sandbox/linux-sandbox/10517/execroot/iree_core/bazel-out/k8-opt/bin/integrations/tensorflow/e2e/keras_vision_model_ResNet50_test.runfiles/iree_core/integrations/tensorflow/bindings/python/pyiree/tf/support/tf_test_utils.py", line 305, in <dictcomp>
    for backend, f in self._backend_function_dict.items()
  File "/usr/local/google/home/scotttodd/.cache/bazel/_bazel_scotttodd/9e82052937ca42451e1b3d0e47f9949a/sandbox/linux-sandbox/10517/execroot/iree_core/bazel-out/k8-opt/bin/integrations/tensorflow/e2e/keras_vision_model_ResNet50_test.runfiles/iree_core/integrations/tensorflow/bindings/python/pyiree/tf/support/tf_test_utils.py", line 264, in __call__
    return self._f(*args)
  File "/usr/local/google/home/scotttodd/.cache/bazel/_bazel_scotttodd/9e82052937ca42451e1b3d0e47f9949a/sandbox/linux-sandbox/10517/execroot/iree_core/bazel-out/k8-opt/bin/integrations/tensorflow/e2e/keras_vision_model_ResNet50_test.runfiles/iree_core/bindings/python/pyiree/rt/system_api.py", line 130, in __call__
    self._context._vm_context.invoke(self._vm_function, inputs, results)
ValueError: Error invoking function: 3
```


## Trigger

Compiling and executing ImageNet ResNet50 with real weights creates a VM function whose live reference-register ordinal exceeds 0x3F.

## Relevant entities

VM SSA reference; encoded register number; reference-register bitset; live range; physical register slot; HAL reference type

## Implemented relation

A live VM reference retains an exclusive physical register slot until the allocator releases that exact register ordinal.

## Pre-state

More than 63 reference values are live, so at least one allocated reference register uses ordinal bits above the low six bits.

## Binding transition

releaseRegister clears the allocation bit selected by masking the encoded register number.

## Post-state

The hard-coded 0x3F mask clears a different low ordinal while leaving the intended high ordinal marked live.

## Validity obligation

Register release decodes the full ordinal range defined by the current integer or reference register bank.

## Invalid decision

The release path retained masks from the earlier 8-bit register encoding after register numbers expanded to 16 bits.

## Physical effect

The allocator reuses a physical reference-register slot whose prior value is still live, replacing one HAL reference with another reference type.

## Observed consequence

ResNet50 execution reaches hal.allocator.allocate.const with a hal.buffer in the register where a hal.allocator is required and fails with INVALID_ARGUMENT.

## Repair site

iree/compiler/Dialect/VM/Analysis/RegisterAllocation.cpp::RegisterUsage::releaseRegister

## Repair

Commit b0adcaf replaces the 0x3F and 0x7F release masks with kRefRegisterCount and kIntRegisterCount, matching the expanded register banks.

## Why the repair works

Releasing a high register now clears its own allocation bit, so no unrelated live slot becomes available for reuse.

## Causal basis

The landed commit explicitly states that it fixes 1660, identifies the stale masks after the uint8-to-uint16 expansion, changes the two release masks, and describes reuse while the displaced value remains live. The report's final diagnosis identifies VM register allocation and its runtime error exhibits the resulting reference-type replacement.

## Analysis implication

Register-binding analysis needs the encoded register class, full ordinal width, bank capacity, live interval, release point, and runtime reference type.

## Fix patches

- [`fixes/iree-1660-commit-b0adcaf.patch`](fixes/iree-1660-commit-b0adcaf.patch) — https://github.com/iree-org/iree/commit/b0adcaf3529363509dcdf3f680b31f874803e27b
