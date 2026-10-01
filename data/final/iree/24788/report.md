# [Vulkan] mismatch between memory type selection heuristics

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/24788](https://github.com/iree-org/iree/issues/24788)
- Opened: 2026-08-06T07:43:09Z
- Closed: Open at collection
- Labels: `bug 🐞`, `hal/vulkan`

## Reported behavior

### What happened?

Hello, running the Vulkan SPIR-V backend on a PowerVR GPU, I've found a mismatch in the way memory types are being selected.

### Steps to reproduce your issue

Running a simple TOSA as an example:
```
module {
  func.func @main(%arg0: tensor<1x10x10x3xf32>, %arg1: tensor<10x2x2x3xf32>) -> tensor<1x9x9x10xf32> {
    %0 = "tosa.const"() <{values = dense<-3.185550e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.conv2d %arg0, %arg1, %0, %1, %2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x10x10x3xf32>, tensor<10x2x2x3xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x9x9x10xf32>
    %4 = tosa.cast %3 : (tensor<1x9x9x10xf32>) -> tensor<1x9x9x10xf32>
    return %4 : tensor<1x9x9x10xf32>
  }
}
```
Compile command:
```
iree-compile --iree-hal-target-backends=vulkan-spirv test.mlir -o test.vmfb
```
Execute command:
```
iree-run-module   --device=vulkan   --vulkan_sparse_residency=false --vulkan_sparse_binding=false  -vulkan_buffer_device_addresses=false  --vulkan_validation_layers=false --module=test.vmfb   --function=main   --input='1x10x10x3xf32=0' --input='10x2x2x3xf32=0'
```
Produces the following error:
```
EXEC @main
iree/runtime/src/iree/hal/drivers/vulkan/allocator.c:1455: NOT_FOUND; no Vulkan queue allocation pool can satisfy allocation of 3240 bytes; while invoking native function hal.device.queue.alloca; while calling import;
[ 0] bytecode module@1:396 -; invoking function 'main'; `sync func @main(%input0: tensor<1x10x10x3xf32>, %input1: tensor<10x2x2x3xf32>) -> (%output0: tensor<1x9x9x10xf32>)`
```

### What component(s) does this issue relate to?

Vulkan - SPIR-V backend.

### Version information

IREE: `18c3da6eaab13e4a972d6fcb93bb3f183e490046`

### Additional context

On my device vulkaninfo shows the following memory types:
```
        memoryTypes[0]:
                heapIndex     = 0
                propertyFlags = 0x0001: count = 1
                        MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                usable for:
                        IMAGE_TILING_OPTIMAL:
                                color images
                                FORMAT_D16_UNORM
                                FORMAT_X8_D24_UNORM_PACK32
                                FORMAT_D32_SFLOAT
                                FORMAT_S8_UINT
                                FORMAT_D24_UNORM_S8_UINT
                                FORMAT_D32_SFLOAT_S8_UINT
                        IMAGE_TILING_LINEAR:
                                color images
        memoryTypes[3]:
                heapIndex     = 0
                propertyFlags = 0x0007: count = 3
                        MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                        MEMORY_PROPERTY_HOST_VISIBLE_BIT
                        MEMORY_PROPERTY_HOST_COHERENT_BIT
                usable for:
                        IMAGE_TILING_OPTIMAL:
                                color images
                                FORMAT_D16_UNORM
                                FORMAT_X8_D24_UNORM_PACK32
                                FORMAT_D32_SFLOAT
                                FORMAT_S8_UINT
                                FORMAT_D24_UNORM_S8_UINT
                                FORMAT_D32_SFLOAT_S8_UINT
                        IMAGE_TILING_LINEAR:
                                color images
        memoryTypes[4]:
                heapIndex     = 0
                propertyFlags = 0x000b: count = 3
                        MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                        MEMORY_PROPERTY_HOST_VISIBLE_BIT
                        MEMORY_PROPERTY_HOST_CACHED_BIT
                usable for:
                        IMAGE_TILING_OPTIMAL:
                                color images
                                FORMAT_D16_UNORM
                                FORMAT_X8_D24_UNORM_PACK32
                                FORMAT_D32_SFLOAT
                                FORMAT_S8_UINT
                                FORMAT_D24_UNORM_S8_UINT
                                FORMAT_D32_SFLOAT_S8_UINT
                        IMAGE_TILING_LINEAR:
                                color images
        memoryTypes[5]:
                heapIndex     = 0
                propertyFlags = 0x0011: count = 2
                        MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                        MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT
                usable for:
                        IMAGE_TILING_OPTIMAL:
                                color images
                                FORMAT_D16_UNORM
                                FORMAT_X8_D24_UNORM_PACK32
                                FORMAT_D32_SFLOAT
                                FORMAT_S8_UINT
                                FORMAT_D24_UNORM_S8_UINT
                                FORMAT_D32_SFLOAT_S8_UINT
                                (transient only)
                        IMAGE_TILING_LINEAR:
                                color images
                                (transient only)
```

Narrowed down the cause of the issue to the following.
First we try to find the `best_device` pool [here](https://github.com/iree-org/iree/blob/18c3da6eaab13e4a972d6fcb93bb3f183e490046/runtime/src/iree/hal/drivers/vulkan/allocator.c#L693). According to `iree_hal_vulkan_allocator_memory_type_priority` the 4 memory types get the scores:
- no.0 100
- no.3 100 + 10 + 2
- no.4 100 + 10 + 5
- no.5 100
So no.4 gets selected as best device pool.

But later when we get `hal.device.queue.alloca` [here](https://github.com/iree-org/iree/blob/18c3da6eaab13e4a972d6fcb93bb3f183e490046/runtime/src/iree/hal/drivers/vulkan/allocator.c#L1368) `iree_hal_vulkan_allocator_resolve_memory_placement` -> `iree_hal_vulkan_allocator_score_memory_type`. Here both `HOST_CACHED`  and `HOST_COHERENT` add a score of 0 when not requested, this means no.3 gets the same score as no.4, so it gets selected as it appears first.

This then causes `iree_hal_vulkan_allocator_select_default_pool` -> `iree_hal_vulkan_allocator_pool_matches` to fail at `(capabilities.memory_type & required_type) != required_type)`, as requested type now expects `HOST_COHERENT` which the pool does not have.

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises iREE Vulkan memory-type selection consistency between direct allocation and pool-selection heuristics.

## Observed implementation decision

Two selection paths break equal-score memory-type ties differently.

## Physical effect

A request can be assigned to a pool whose memory properties differ from the direct allocator's selected type.

## Observed consequence

The report observes mismatched allocation behavior for the same compatibility set.

## Reported proposal or workaround

PRs 24798 and 24799 propose aligning the tie-break order.

## Repair trace

Both proposals were closed without merge, and no alternative landed change or regression is linked.

## Archived evidence

- Raw report: `data/raw/github/iree/24788/report.json`
- SHA-256: `2c70eb5d963cda4ba211cd7d09feefaa38b009379f15394f77481e25db770283`
