# [Python] Failure to send a tensor of < 4 bytes to host

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/15508](https://github.com/iree-org/iree/issues/15508)
- Opened: 2023-11-09T14:53:46Z
- Closed: 2023-11-09T21:36:18Z
- Labels: `bug 🐞`, `runtime`, `bindings/python`

## Reported behavior

# Problem description
The following python code will fail with the following error
```
Traceback (most recent call last):
  File "/home/quinn/model_testing/only_conv.py", line 116, in <module>
    print(iree_output.to_host())
          ^^^^^^^^^^^^^^^^^^^^^
  File "/home/quinn/iree-build/runtime/bindings/python/iree/runtime/array_interop.py", line 107, in to_host
    self._transfer_to_host(False)
  File "/home/quinn/iree-build/runtime/bindings/python/iree/runtime/array_interop.py", line 135, in _transfer_to_host
    self._host_array = self._copy_to_host()
                       ^^^^^^^^^^^^^^^^^^^^
  File "/home/quinn/iree-build/runtime/bindings/python/iree/runtime/array_interop.py", line 162, in _copy_to_host
    self._device.queue_copy(
ValueError: Source and target buffer length must match and it does not. Please check allocations

```

```python
import torch

module = """
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @forward(%arg0: tensor<1x?xf16>) -> tensor<1xf16> {
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c1 : tensor<1x?xf16>
    %0 = tensor.empty(%dim) : tensor<1x?xf16>
    %c1_0 = arith.constant 1 : index
    %dim_1 = tensor.dim %arg0, %c1_0 : tensor<1x?xf16>
    %1 = tensor.empty() : tensor<1xf16>
    %cst = arith.constant -6.550400e+04 : f16
    %res = flow.dispatch.region -> (tensor<1xf16>) {
      %2 = linalg.fill ins(%cst : f16) outs(%1 : tensor<1xf16>) -> tensor<1xf16>
      %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1x?xf16>) outs(%2 : tensor<1xf16>) {
      ^bb0(%in: f16, %out: f16):
        %7 = arith.maximumf %in, %out : f16
        linalg.yield %7 : f16
      } -> tensor<1xf16>
      %cst_2 = arith.constant 0.000000e+00 : f16
      %4 = linalg.fill ins(%cst_2 : f16) outs(%1 : tensor<1xf16>) -> tensor<1xf16>
      %5 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0, %3 : tensor<1x?xf16>, tensor<1xf16>) outs(%4 : tensor<1xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %7 = arith.subf %in, %in_3 : f16
        %8 = math.exp %7 : f16
        %9 = arith.addf %8, %out : f16
        linalg.yield %9 : f16
      } -> tensor<1xf16>
      flow.return %5 : tensor<1xf16>
    }
    return %res : tensor<1xf16>
  }
}
"""

example_input = torch.FloatTensor([[2.0, 0.0]]).half()

# Run through IREE vulkan
import iree.runtime as ireert
from iree.runtime import get_driver, get_device
import iree.compiler as ireec

device = "vulkan"
runtime_device = "vulkan"

flatbuffer = ireec.compile_str(
    module,
    target_backends=[device],
    extra_args=[
        "--iree-vulkan-target-triple=rdna3-unknown-linux",
        "--iree-stream-resource-index-bits=64",
        "--iree-vm-target-index-bits=64",
        ],
)

config = ireert.Config(driver_name=runtime_device)
vm_module = ireert.VmModule.from_flatbuffer(config.vm_instance, flatbuffer)
ctx = ireert.SystemContext(config=config)
ctx.add_vm_module(vm_module)
forward = ctx.modules.module["forward"]
iree_output = forward(example_input.numpy())

# Fails due to a size mismatch between the source and target buffer
print(iree_output.to_host())
```

# Investigation

The source (device) buffer will be `2` bytes for the single f16, however when trying to allocate the host buffer to copy into (https://github.com/openxla/iree/blob/d991dc9330d486b804a204e6e18f8287c256def1/runtime/bindings/python/iree/runtime/array_interop.py#L153), it ends up padding the allocation to `4` bytes, causing the sizes to mismatch.

## Trigger

A two-byte f16 tensor is copied into a host-visible buffer allocated at a four-byte minimum size.

## Relevant entities

logical tensor byte length; source buffer; aligned target buffer; queue_copy

## Implemented relation

A logical source range may be copied into a target buffer when the target contains that entire range.

## Pre-state

The two-byte source range fits within the four-byte target allocation.

## Binding transition

The Python binding compares complete source and target buffer lengths for equality.

## Post-state

A safe bounded copy is rejected solely because the target has unused trailing capacity.

## Validity obligation

The copy guard checks source_length less than or equal to target_capacity; it rejects only overflow.

## Invalid decision

queue_copy treated equal backing-buffer lengths as necessary for a full-source copy.

## Physical effect

No transfer is submitted for a valid sub-four-byte tensor.

## Observed consequence

to_host raises ValueError for a tensor containing one f16 element.

## Repair site

runtime/bindings/python/hal.cc::HalDevice::QueueCopy

## Repair

PR 15511 changes the equality check to source_length > target_length and adds tests for smaller-source success and larger-source rejection.

## Why the repair works

Every logical source byte is still bounded by target capacity while allocator padding no longer changes copy validity.

## Causal basis

The report identifies two-versus-four-byte lengths and the patch changes only that guard with both boundary tests.

## Analysis implication

Copy validation needs logical range length and target capacity separately instead of equating two backing allocations.

## Fix patches

- [`fixes/iree-15508-pr15511.patch`](fixes/iree-15508-pr15511.patch) — https://github.com/iree-org/iree/pull/15511
