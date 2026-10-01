# Exported I/O buffer views improperly alias with tied ops.

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/16168](https://github.com/iree-org/iree/issues/16168)
- Opened: 2024-01-19T17:52:14Z
- Closed: 2025-12-05T16:49:56Z
- Labels: `bug 🐞`, `compiler/dialects`

## Reported behavior

This is likely a missing case in the MaterializeCopyOnWrite pass. An input program that has a single tied operation (in my case from a linalg.matmul) will not have a clone on the input inserted and will mutate the original input in-place when that's not what the user indicated. We should only do this when iree.abi.output is assigned (the hal.tensor.export has a tied buffer attached).

The current workaround is to pass in copies of the arguments but that's pretty terrible.

```mlir
#config = #iree_codegen.lowering_config<tile_sizes = [[32, 128], [4, 4], [0, 0, 32]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader, GroupNonUniform], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, api=Vulkan, #spirv.resource_limits<max_compute_workgroup_size = [128, 128, 64], subgroup_size = 64, cooperative_matrix_properties_khr = []>>}>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize pipeline_depth = 3>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation, workgroup_size = [32, 8, 1]>
#device_target_vulkan = #hal.device.target<"vulkan", {executable_targets = [#executable_target_vulkan_spirv_fb], legacy_sync}>
module attributes {hal.device.targets = [#device_target_vulkan]} {
  func.func @matmul_accumulate_256x128xi8_times_128x512xi8_into_256x512xi32_for_SPIRVVectorizeNVIDIA_32_128_4_4_0_0_32_32_8_1(%arg0: tensor<256x128xi8>, %arg1: tensor<128x512xi8>, %arg2: tensor<256x512xi32>) -> tensor<256x512xi32> {
    %0 = linalg.matmul {compilation_info = #compilation} ins(%arg0, %arg1 : tensor<256x128xi8>, tensor<128x512xi8>) outs(%arg2 : tensor<256x512xi32>) -> tensor<256x512xi32>
    return %0 : tensor<256x512xi32>
  }
}
```

```mlir
// -----// IR Dump After FormDispatchWorkgroups (iree-flow-form-dispatch-workgroups) //----- //
func.func @matmul_accumulate_256x128xi8_times_128x512xi8_into_256x512xi32_for_SPIRVVectorizeNVIDIA_32_128_4_4_0_0_32_32_8_1(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @matmul_accumulate_256x128xi8_times_128x512xi8_into_256x512xi32_for_SPIRVVectorizeNVIDIA_32_128_4_4_0_0_32_32_8_1(%input0: tensor<256x128xi8>, %input1: tensor<128x512xi8>, %input2: tensor<256x512xi32>) -> (%output0: tensor<256x512xi32>)"}} {
  %0 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<256x128xi8>
  %1 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<128x512xi8>
  %2 = hal.tensor.import %arg2 "input2" : !hal.buffer_view -> tensor<256x512xi32>
  %3 = flow.dispatch.workgroups(%0, %1, %2) : (tensor<256x128xi8>, tensor<128x512xi8>, tensor<256x512xi32>) -> %2 =
      (%arg3: !flow.dispatch.tensor<readonly:tensor<256x128xi8>>, %arg4: !flow.dispatch.tensor<readonly:tensor<128x512xi8>>, %arg5: !flow.dispatch.tensor<readwrite:tensor<256x512xi32>>) {
    %5 = flow.dispatch.tensor.load %arg3, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x128xi8>> -> tensor<256x128xi8>
    %6 = flow.dispatch.tensor.load %arg4, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x512xi8>> -> tensor<128x512xi8>
    %7 = flow.dispatch.tensor.load %arg5, offsets = [0, 0], sizes = [256, 512], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<256x512xi32>> -> tensor<256x512xi32>
    %8 = linalg.matmul {compilation_info = #iree_codegen.compilation_info<lowering_config = <tile_sizes = [[32, 128], [4, 4], [0, 0, 32]]>, translation_info = <SPIRVBaseVectorize pipeline_depth = 3>, workgroup_size = [32, 8, 1]>} ins(%5, %6 : tensor<256x128xi8>, tensor<128x512xi8>) outs(%7 : tensor<256x512xi32>) -> tensor<256x512xi32>
    flow.dispatch.tensor.store %8, %arg5, offsets = [0, 0], sizes = [256, 512], strides = [1, 1] : tensor<256x512xi32> -> !flow.dispatch.tensor<readwrite:tensor<256x512xi32>>
    flow.return
  } count() -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_slice
    flow.return %x, %y, %z : index, index, index
  }
  %4 = hal.tensor.export %3 "output0" : tensor<256x512xi32> -> !hal.buffer_view
  return %4 : !hal.buffer_view
}
```

```mlir
// -----// IR Dump After EncodeHostTensorsPass (iree-stream-encode-host-tensors) //----- //
func.func @matmul_accumulate_256x128xi8_times_128x512xi8_into_256x512xi32_for_SPIRVVectorizeNVIDIA_32_128_4_4_0_0_32_32_8_1(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @matmul_accumulate_256x128xi8_times_128x512xi8_into_256x512xi32_for_SPIRVVectorizeNVIDIA_32_128_4_4_0_0_32_32_8_1(%input0: tensor<256x128xi8>, %input1: tensor<128x512xi8>, %input2: tensor<256x512xi32>) -> (%output0: tensor<256x512xi32>)"}} {
  %c524288 = arith.constant 524288 : index
  %c65536 = arith.constant 65536 : index
  %c32768 = arith.constant 32768 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %element_type_i8 = hal.element_type<i8> : i32
  %dense_row_major = hal.encoding_type<dense_row_major> : i32
  hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c256, %c128]) type(%element_type_i8) encoding(%dense_row_major)
  %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<256x128xi8> in !stream.resource<external>{%c32768}
  %1 = stream.async.transfer %0 : !stream.resource<external>{%c32768} -> !stream.resource<*>{%c32768}
  hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%c128, %c512]) type(%element_type_i8) encoding(%dense_row_major)
  %2 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<128x512xi8> in !stream.resource<external>{%c65536}
  %3 = stream.async.transfer %2 : !stream.resource<external>{%c65536} -> !stream.resource<*>{%c65536}
  %element_type_i32 = hal.element_type<i32> : i32
  hal.buffer_view.assert<%arg2 : !hal.buffer_view> message("input2") shape([%c256, %c512]) type(%element_type_i32) encoding(%dense_row_major)
  %4 = stream.tensor.import %arg2 : !hal.buffer_view -> tensor<256x512xi32> in !stream.resource<external>{%c524288}
  %5 = stream.async.transfer %4 : !stream.resource<external>{%c524288} -> !stream.resource<*>{%c524288}
  %6 = stream.async.dispatch @matmul_accumulate_256x128xi8_times_128x512xi8_into_256x512xi32_for_SPIRVVectorizeNVIDIA_32_128_4_4_0_0_32_32_8_1_dispatch_0::@matmul_accumulate_256x128xi8_times_128x512xi8_into_256x512xi32_for_SPIRVVectorizeNVIDIA_32_128_4_4_0_0_32_32_8_1_dispatch_0_matmul_256x512x128_i8xi8xi32(%1[%c0 to %c32768 for %c32768], %3[%c0 to %c65536 for %c65536], %5[%c0 to %c524288 for %c524288]) : (!stream.resource<*>{%c32768}, !stream.resource<*>{%c65536}, !stream.resource<*>{%c524288}) -> %5{%c524288}
  %7 = stream.async.transfer %6 : !stream.resource<*>{%c524288} -> !stream.resource<external>{%c524288}
  %8 = stream.tensor.export %7 : tensor<256x512xi32> in !stream.resource<external>{%c524288} -> !hal.buffer_view
  return %8 : !hal.buffer_view
}
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

Exported I/O buffer views improperly alias with tied ops.

## Observed implementation decision

The copy-on-write safety test permits a single-user external import without consulting consume semantics, so no clone is created before the read-write dispatch.

## Physical effect

MaterializeCopyOnWrite treats a borrowed imported input as safe to update in place when a tied operation returns that storage, even though the ABI does not authorize input consumption.

## Observed consequence

Basic matmul, sort, and top-k paths mutate the caller's input buffer and can return incorrect paired outputs on CPU and GPU backends.

## Reported proposal or workaround

The report documents copy-based workarounds and later analysis shows that a separate clone-preservation fix does not create the missing clone; no complete fix was archived by the cutoff.

## Repair trace

The report establishes unsafe update of a borrowed import, but the available clone-preservation change does not create the required clone and no completed consume-semantics repair is linked.

## Archived evidence

- Raw report: `data/raw/github/iree/16168/report.json`
- SHA-256: `b57f1d67c45f4797324612b55c2371ad5bb234bfd16f3ccfddc9ac8131e95b4a`
