# Allocator ownership during buffer initialization

This source check was completed on 2026-10-27. It uses the pre-fix source,
archived patch, and regression test; it does not rerun the failure or add a
human-review result.

## Historical operation

Pre-fix revision: `c0be21d0f16c793ba44bd367e81c8910b9a38521`.
Repair revision: `52e2fd8f554db42b2e4052a17b3a7351dbe940fd`.

The pre-fix [`iree_hal_buffer_initialize`, lines 179–202](https://github.com/iree-org/iree/blob/c0be21d0f16c793ba44bd367e81c8910b9a38521/runtime/src/iree/hal/buffer.c#L179)
contains this complete body:

```c
  iree_hal_resource_initialize(vtable, &buffer->resource);
  buffer->host_allocator = host_allocator;
  buffer->device_allocator = device_allocator;
  buffer->allocated_buffer = allocated_buffer;
  buffer->allocation_size = allocation_size;
  buffer->byte_offset = byte_offset;
  buffer->byte_length = byte_length;
  buffer->memory_type = memory_type;
  buffer->allowed_access = allowed_access;
  buffer->allowed_usage = allowed_usage;

  // Retain the base allocated buffer if it's unique from the buffer we are
  // initializing.
  if (allocated_buffer != buffer) {
    iree_hal_buffer_retain(buffer->allocated_buffer);
  }
}
```

The selected decision is the initializer's completion (its implicit return
at line 202). The buffer then holds `device_allocator` without an allocator
reference that protects its later deallocation. The assignment at line 188
records the correct pointer. It can be followed by a retain before the
initializer completes, so the assignment alone does not establish the
omission. The later deallocation call exposes the resulting lifetime failure.

The validity condition is that the backing allocator remains alive through
buffer destruction. At lines 207–208, `iree_hal_buffer_recycle` uses the saved
pointer:

```c
    if (buffer->device_allocator) {
      iree_hal_allocator_deallocate_buffer(buffer->device_allocator, buffer);
```

## Repair and supporting test

The [archived patch](fixes/iree-9033-fix.patch) adds
`iree_hal_allocator_retain(device_allocator)` before initialization completes.
In `iree_hal_buffer_destroy`, it saves the allocator pointer, destroys the
buffer, and then calls `iree_hal_allocator_release`.

The added `testGcShutdownFiasco` drops allocator, device, and driver references
before dropping the array. Its stated requirement is that the array retain
the resources needed to stay live. This test and the retain/release changes
support the lifetime condition; the pre-fix header has no explicit
allocator-retain contract attached to the initializer declaration.

## Coding decision

The failure family is **omitted operation**: a required retain is absent.
The allocator identity is stored and reaches deallocation, so this is not a
binding fact dropped during propagation. The previous **lost or omitted
relation** assignment is corrected under the existing codebook rule.

The relation remains **storage--ownership/lifetime**, the transition remains
**ownership or lifetime change**, and the primary obligation remains
**ownership and lifetime**. The repair strategy remains **correct ownership
or lifetime**, because it adds a matched retain/release interval through the
allocator's last use. The initializer completion is the invalid decision;
the retain and release are repair operations.

Upstream source excerpts retain IREE's Apache-2.0 WITH LLVM-exception license.
