# VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00293(ERROR / SPEC) with Vulkan AMD 6900XT

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/10414](https://github.com/iree-org/iree/issues/10414)
- Opened: 2022-09-14T22:55:07Z
- Closed: 2022-09-20T16:40:44Z
- Labels: `bug 🐞`, `hal/vulkan`

## Reported behavior

### What happened?

Running Resnet50 on AMD 6900XT with Vulkan reports

```
VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00293(ERROR / SPEC): msgNum: 2143647270 - Validation Error: [ VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00293 ] Object 0: handle = 0x1ce1d63d250, type = VK_OBJECT_TYPE_DEVICE; | MessageID = 0x7fc57626 | vkCreatePipelineLayout() Multiple push descriptor sets found. The Vulkan spec states: pSetLayouts must not contain more than one descriptor set layout that was created with VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR set (https://vulkan.lunarg.com/doc/view/1.3.224.1/windows/1.3-extensions/vkspec.html#VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00293)
    Objects: 1
        [0] 0x1ce1d63d250, type: 3, name: NULL
```

### Steps to reproduce your issue

Steps to reproduce:

```
 $  wget https://storage.googleapis.com/shark_tank/latest/resnet50_tf/resnet50_tf.mlir
  $  /home/anush/github/shark/shark.venv/lib/python3.10/site-packages/iree/compiler/tools/../_mlir_libs/iree-compile --iree-input-type=mhlo --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-hal-target-backends=vulkan --iree-llvm-embedded-linker-path=/home/anush/github/shark/shark.venv/lib/python3.10/site-packages/iree/compiler/tools/../_mlir_libs/iree-lld --mlir-print-debuginfo --mlir-print-op-on-diagnostic=false --mlir-pass-pipeline-crash-reproducer=ist/core-reproducer.mlir --iree-llvm-target-cpu-features=host -iree-vulkan-target-triple=rdna2-unknown-linux --iree-stream-resource-index-bits=64 --iree-vm-target-index-bits=64 resnet50_tf.mlir -o resnet50_tf.vmfb
  $  iree-benchmark-module --module_file=resnet50_tf.vmfb --device=vulkan --entry_function=forward --function_input="1x224x224x3xf32"
```


### What component(s) does this issue relate to?

Compiler

### Version information

amdgpu from upstream 6.0.0-rc4 kernel with latest `amdvulkan` drivers built from source.

IREE is top of master.

### Additional context

_No response_

## Trigger

A built-in Vulkan executable creates placeholder descriptor-set layouts for set numbers preceding its active storage-buffer set.

## Relevant entities

pipeline layout; active descriptor set; dummy descriptor sets; descriptor bindings; push-descriptor flag

## Implemented relation

Each pipeline set position refers either to the active bound resources or to an empty placeholder needed only to preserve set numbering.

## Pre-state

One active set contains a storage-buffer binding, while preceding set positions contain no resources.

## Binding transition

The Vulkan runtime creates native descriptor-set layouts and assembles them into a pipeline layout.

## Post-state

Every placeholder is created with a synthetic binding and the push-descriptor flag, so the pipeline layout contains multiple push-descriptor sets.

## Validity obligation

Placeholder set layouts preserve numbering without claiming active bindings or a second push-descriptor role.

## Invalid decision

BuiltinExecutables created every placeholder with binding count one, and native layout creation enabled the push-descriptor flag independently of whether a set had real bindings.

## Physical effect

VkPipelineLayoutCreateInfo contains more than one set layout marked VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR.

## Observed consequence

Vulkan validation rejects the pipeline layout; on the reported AMD path no dispatch executes and dependent output is invalid.

## Repair site

runtime/src/iree/hal/drivers/vulkan/builtin_executables.cc and native_pipeline_layout.cc

## Repair

PR 10448 creates dummy layouts with zero bindings and enables the push-descriptor flag only when a layout contains at least one binding.

## Why the repair works

Only the active resource-bearing set is marked as a push-descriptor set, while placeholders retain their set positions without introducing binding identities.

## Causal basis

The report's validation error names multiple push-descriptor sets, and the linked patch removes that state by making placeholder sets empty and conditioning the native flag on a nonzero binding count.

## Analysis implication

A pipeline-binding check needs set position, binding count, placeholder status, and native descriptor-set flags.

## Fix patches

- [`fixes/iree-10414-pr10448.patch`](fixes/iree-10414-pr10448.patch) — https://github.com/iree-org/iree/pull/10448
