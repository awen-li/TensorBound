# HAL allocator infers/declares wrong usage type when loading large flatbuffers on RDNA3 devices.

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/13516](https://github.com/iree-org/iree/issues/13516)
- Opened: 2023-05-10T11:20:10Z
- Closed: Open at collection
- Labels: `bug 🐞`, `runtime`, `hal/vulkan`

## Reported behavior

### What happened?

With a device whose VRAM and memory heaps should be plenty sufficient for this allocation:
```
RuntimeError: Error registering modules: C:\E\SHARK-Runtime\runtime\src\iree\hal\drivers\vulkan\vma_allocator.cc:693: RESOURCE_EXHAUSTED; VK_ERROR_OUT_OF_DEVICE_MEMORY; vmaCreateBuffer; failed to allocate buffer of length 26953666560; while invoking native function hal.allocator.allocate.initialized; while calling import;
[ 1]   native hal.allocator.allocate.initialized:0 -
[ 0] bytecode module@1:2938 -
```
The resource allocation issue seems to be happening because the HAL allocator gets the wrong memory type (DEVICE_VISIBLE over DEVICE_LOCAL) when trying to set up the buffer allocation.

Here I show that the 26GB buffer can allocate successfully if the DEVICE_LOCAL memory type is explicitly specified:
```
(Pdb) config.device.allocator.allocate_buffer(memory_type=ireert.MemoryType.DEVICE_VISIBLE,allowed_usage=ireert.BufferUsage.DEFAULT,allocation_size=26953666560)
*** RuntimeError: could not allocate buffer: D:\a\SHARK-Runtime\SHARK-Runtime\c\runtime\src\iree\hal\drivers\vulkan\vma_allocator.cc:693: RESOURCE_EXHAUSTED; VK_ERROR_OUT_OF_DEVICE_MEMORY; vmaCreateBuffer

(Pdb) config.device.allocator.allocate_buffer(memory_type=ireert.MemoryType.DEVICE_LOCAL,allowed_usage=ireert.BufferUsage.DEFAULT,allocation_size=26953666560)
<HalBuffer 26953666560 bytes (at offset 0 into 26953666560), memory_type=DEVICE_LOCAL, allowed_access=ALL, allowed_usage=TRANSFER|DISPATCH_STORAGE>
```
I put a few return_on_error prints in iree/hal/drivers/vulkan/vma_allocator.cc to see which memory type is specified for the failing allocation and it is always DEVICE_VISIBLE for w7900 in the stack below ireert.VmModule.from_flatbuffer() where we load the .vmfb into device memory in SHARK.

Can this be explicitly changed on the python level or will it require refinement of the vma_allocator memory type populating?
Can we specify the memory type or have it inferred from a compile-time option? I tried getting the right case to happen [in StreamToHAL](https://github.com/openxla/iree/blob/4d6c2b8699383598817352f19b5888b38be06ccb/compiler/src/iree/compiler/Dialect/HAL/Conversion/StreamToHAL/Patterns.cpp#L187) by compiling the .vmfb with `--iree-execution-model=async-external` but didn't see an effect on the inferred memory type.


### Steps to reproduce your issue

_No response_

### What component(s) does this issue relate to?

Runtime

### Version information

[5b89a14](https://github.com/openxla/iree/commit/5b89a145d02fcad42cd375b7ae447bfd672471af)

### Additional context

_No response_

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises iREE HAL buffer usage and memory-type declaration for large flatbuffer-backed dispatch storage on RDNA3.

## Observed implementation decision

The report initially attributes allocation failure to memory selection, while maintainer discussion points to an incorrect declared buffer usage and asks for a Stream-level trace.

## Physical effect

A large dispatch resource is requested with a host-visible/device-local combination inappropriate for ordinary constants or transients.

## Observed consequence

Loading the large model allocation fails on the target device.

## Reported proposal or workaround

Maintainers suggest inspecting the Stream dialect to locate the incorrect declaration.

## Repair trace

No offending source site, correcting patch, or fixed regression is identified; the issue remains open.

## Archived evidence

- Raw report: `data/raw/github/iree/13516/report.json`
- SHA-256: `3939f8e373c83b5c5152ed676e6f008f766e8002e4220b7174a647989572292e`
