# [Vulkan] unaligned buffer transfer hangs

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/24771](https://github.com/iree-org/iree/issues/24771)
- Opened: 2026-07-30T09:37:00Z
- Closed: 2026-08-24T11:45:47Z
- Labels: `bug 🐞`, `hal/vulkan`, `regression`

## Reported behavior

### What happened?

When copying a small and unaligned buffer from host to device on vulkan, the iree_hal_device_transfer_h2d function prints some vulkan validation errros (if validation is enabled) and hangs forever.

Minimal reproducer:
```c
#include <stdio.h>

#include "iree/runtime/api.h"

void test_buffer_copy(iree_runtime_session_t* session, iree_hal_dim_t size) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  iree_hal_dim_t shape[] = {size};
  char *data = malloc(size);
  IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(
      iree_runtime_session_device(session),
      iree_runtime_session_device_allocator(session),
      IREE_ARRAYSIZE(shape), shape, IREE_HAL_ELEMENT_TYPE_INT_8,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(data, size), &buffer_view));
  iree_hal_buffer_view_release(buffer_view);
  free(data);
}

int main(int argc, char** argv) {
  // init an empty vulkan session
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance));
  iree_hal_device_t* device = NULL;
  IREE_CHECK_OK(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view("vulkan"), &device));
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  IREE_CHECK_OK(iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session));
  iree_hal_device_release(device);

  // PASS: larger than IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE, unaligned
  test_buffer_copy(session, IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE + 1);
  printf("test case 1 passed\n");

  // PASS: larger than IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE, aligned
  // alignment doesn't matter when using a staging buffer, but test it anyway
  test_buffer_copy(session, IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE + 4);
  printf("test case 2 passed\n");

  // PASS: smaller than IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE, aligned
  // this translates to a single vkCmdUpdateBuffer
  test_buffer_copy(session, 4);
  printf("test case 3 passed\n");

  // FAIL: smaller than IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE, unaligned
  // this will use an additional compute pipeline to copy the unaligned edges
  test_buffer_copy(session, 5);
  printf("test case 4 passed\n"); // never reached

  iree_runtime_session_release(session);
  iree_runtime_instance_release(instance);
  return 0;
}
```

Program output:
```
test case 1 passed
test case 2 passed
test case 3 passed
Validation Error: [ VUID-vkCmdBindPipeline-commandBuffer-cmdpool ] | MessageID = 0xdadc65f
vkCmdBindPipeline(): called in VkCommandBuffer 0x584045045e10 which was allocated from the VkCommandPool 0x20150000002015 which was created with queueFamilyIndex 1 which contains the capability flags VK_QUEUE_TRANSFER_BIT|VK_QUEUE_SPARSE_BINDING_BIT (but requires VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT or VK_QUEUE_DATA_GRAPH_BIT_ARM).
The Vulkan spec states: The VkCommandPool that commandBuffer was allocated from must support VK_QUEUE_COMPUTE_BIT, VK_QUEUE_DATA_GRAPH_BIT_ARM, or VK_QUEUE_GRAPHICS_BIT operations (https://docs.vulkan.org/spec/latest/chapters/pipelines.html#VUID-vkCmdBindPipeline-commandBuffer-cmdpool)
Objects: 2
    [0] VkCommandBuffer 0x584045045e10
    [1] VkCommandPool 0x20150000002015

Validation Error: [ VUID-vkCmdBindPipeline-pipelineBindPoint-00777 ] | MessageID = 0x16eade23
vkCmdBindPipeline(): VkCommandBuffer 0x584045045e10 was allocated from VkCommandPool 0x20150000002015 that does not support VK_PIPELINE_BIND_POINT_COMPUTE.
The Vulkan spec states: If pipelineBindPoint is VK_PIPELINE_BIND_POINT_COMPUTE, the VkCommandPool that commandBuffer was allocated from must support compute operations (https://docs.vulkan.org/spec/latest/chapters/pipelines.html#VUID-vkCmdBindPipeline-pipelineBindPoint-00777)
Objects: 2
    [0] VkCommandBuffer 0x584045045e10
    [1] VkCommandPool 0x20150000002015

Validation Error: [ VUID-vkCmdBindDescriptorSets-commandBuffer-cmdpool ] | MessageID = 0x4355e0c6
vkCmdBindDescriptorSets(): called in VkCommandBuffer 0x584045045e10 which was allocated from the VkCommandPool 0x20150000002015 which was created with queueFamilyIndex 1 which contains the capability flags VK_QUEUE_TRANSFER_BIT|VK_QUEUE_SPARSE_BINDING_BIT (but requires VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT or VK_QUEUE_DATA_GRAPH_BIT_ARM).
The Vulkan spec states: The VkCommandPool that commandBuffer was allocated from must support VK_QUEUE_COMPUTE_BIT, VK_QUEUE_DATA_GRAPH_BIT_ARM, or VK_QUEUE_GRAPHICS_BIT operations (https://docs.vulkan.org/spec/latest/chapters/descriptorsets.html#VUID-vkCmdBindDescriptorSets-commandBuffer-cmdpool)
Objects: 2
    [0] VkCommandBuffer 0x584045045e10
    [1] VkCommandPool 0x20150000002015

Validation Error: [ VUID-vkCmdBindDescriptorSets-pipelineBindPoint-00361 ] | MessageID = 0x48a0e8b4
vkCmdBindDescriptorSets(): VkCommandBuffer 0x584045045e10 was allocated from VkCommandPool 0x20150000002015 that does not support VK_PIPELINE_BIND_POINT_COMPUTE.
The Vulkan spec states: pipelineBindPoint must be supported by the commandBuffer's parent VkCommandPool's queue family (https://docs.vulkan.org/spec/latest/chapters/descriptorsets.html#VUID-vkCmdBindDescriptorSets-pipelineBindPoint-00361)
Objects: 2
    [0] VkCommandBuffer 0x584045045e10
    [1] VkCommandPool 0x20150000002015

Validation Error: [ VUID-vkCmdPushConstants-commandBuffer-cmdpool ] | MessageID = 0x27afdb33
vkCmdPushConstants(): called in VkCommandBuffer 0x584045045e10 which was allocated from the VkCommandPool 0x20150000002015 which was created with queueFamilyIndex 1 which contains the capability flags VK_QUEUE_TRANSFER_BIT|VK_QUEUE_SPARSE_BINDING_BIT (but requires VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT).
The Vulkan spec states: The VkCommandPool that commandBuffer was allocated from must support VK_QUEUE_COMPUTE_BIT, or VK_QUEUE_GRAPHICS_BIT operations (https://docs.vulkan.org/spec/latest/chapters/descriptorsets.html#VUID-vkCmdPushConstants-commandBuffer-cmdpool)
Objects: 2
    [0] VkCommandBuffer 0x584045045e10
    [1] VkCommandPool 0x20150000002015

Validation Error: [ VUID-vkCmdDispatch-commandBuffer-cmdpool ] | MessageID = 0xce915240
vkCmdDispatch(): called in VkCommandBuffer 0x584045045e10 which was allocated from the VkCommandPool 0x20150000002015 which was created with queueFamilyIndex 1 which contains the capability flags VK_QUEUE_TRANSFER_BIT|VK_QUEUE_SPARSE_BINDING_BIT (but requires VK_QUEUE_COMPUTE_BIT).
The Vulkan spec states: The VkCommandPool that commandBuffer was allocated from must support VK_QUEUE_COMPUTE_BIT operations (https://docs.vulkan.org/spec/latest/chapters/dispatch.html#VUID-vkCmdDispatch-commandBuffer-cmdpool)
Objects: 2
    [0] VkCommandBuffer 0x584045045e10
    [1] VkCommandPool 0x20150000002015

```

vulkaninfo output:

[vulkaninfo.txt](https://github.com/user-attachments/files/30540315/vulkaninfo.txt)

### What component(s) does this issue relate to?

Runtime

### Version information

Locally built on git commit bc908fd620d7378d1e2da0bff6ee70d77a965110

## Trigger

A small host buffer produces an unaligned Vulkan update that cannot use the native transfer instruction.

## Relevant entities

transfer command category; unaligned update; compute built-in; queue family; semaphore

## Implemented relation

The selected Vulkan queue supports every command used by the concrete aligned or emulated transfer path.

## Pre-state

The API operation is categorized as a transfer and carries a caller queue affinity.

## Binding transition

Queue selection requires only VK_QUEUE_TRANSFER_BIT before replay chooses a compute built-in for the unaligned range.

## Post-state

A compute dispatch is recorded on a transfer-only queue family.

## Validity obligation

Queue capability selection includes compute for transfer operations that may take the unaligned emulation path.

## Invalid decision

The old logical-device code selected the queue from the abstract transfer category without the concrete replay path's compute requirement.

## Physical effect

Vulkan rejects or drops the invalid command while a signal semaphore remains unsatisfied.

## Observed consequence

Validation reports errors and the host-to-device transfer hangs indefinitely.

## Repair site

runtime/src/iree/hal/drivers/vulkan/logical_device.c

## Repair

PR 24838 requires both transfer and compute flags for fill/update selection and retries with any affinity when the requested lanes cannot satisfy them.

## Why the repair works

Every chosen lane can execute the native transfer and the compute-emulated unaligned path, so the submission reaches its semaphore signal.

## Causal basis

The patch names unaligned fill/update replay and explains the exact transfer-only queue hang reported by issue 24771.

## Analysis implication

Queue checks need the eventual lowering path, required queue flags, requested affinity, fallback affinity, and semaphore completion edge.

## Fix patches

- [`fixes/iree-24771-pr24838.patch`](fixes/iree-24771-pr24838.patch) — https://github.com/iree-org/iree/pull/24838
