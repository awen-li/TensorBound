# [GPU] very poor numerics for certain conv layouts

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/20473](https://github.com/iree-org/iree/issues/20473)
- Opened: 2025-04-07T03:24:53Z
- Closed: 2025-04-15T18:50:06Z
- Labels: `bug 🐞`

## Reported behavior

### What happened?

I'm trying to decompose backwards convolutions into forward convolutions, and this sometimes results in unusual layouts. Here is a some simple `conv_1d_nhc_chf` IR which reproduces a significant numerical error:

```mlir
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 + d4, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
module @module {
  util.func public @conv_1d_float32_forward_1x3x2_nhc_2x2x2_chf_nhf_1s_0p_1d_1g$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.fence, %arg3: !hal.fence) -> !hal.buffer_view attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub, preprocessing_pipeline = #util.preprocessing_pipeline<"iree-preprocessing-make-single-dispatch">} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import wait(%arg2) => %arg0 : !hal.buffer_view -> tensor<1x3x2xf32>
    %1 = hal.tensor.import wait(%arg2) => %arg1 : !hal.buffer_view -> tensor<2x2x2xf32>
    %2 = tensor.empty() : tensor<1x2x2xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<1x3x2xf32>, tensor<2x2x2xf32>) outs(%3 : tensor<1x2x2xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.mulf %in, %in_0 : f32
      %8 = arith.addf %out, %7 : f32
      linalg.yield %8 : f32
    } -> tensor<1x2x2xf32>
    %5 = hal.tensor.barrier join(%4 : tensor<1x2x2xf32>) => %arg3 : !hal.fence
    %6 = hal.tensor.export %5 : tensor<1x2x2xf32> -> !hal.buffer_view
    util.return %6 : !hal.buffer_view
  }
  util.func public @conv_1d_float32_forward_1x3x2_nhc_2x2x2_chf_nhf_1s_0p_1d_1g(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = util.null : !hal.fence
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    %1 = util.call @conv_1d_float32_forward_1x3x2_nhc_2x2x2_chf_nhf_1s_0p_1d_1g$async(%arg0, %arg1, %0, %fence) : (!hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> !hal.buffer_view
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) flags("None") : i32
    util.return %1 : !hal.buffer_view
  }
}
```



### Steps to reproduce your issue

Compile the above IR with

```
iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 -o conv_repro.vmfb
```

Then run on some simple arguments:

```
iree-run-module --device=hip \
   --module="conv_repro.vmfb" \
   --input=1x3x2xf32=[[[0,1],[0,0],[0,0]]] \
   --input=2x2x2xf32=[[[1,2],[3,4]],[[5,6],[7,8]]]
```

The result of this run gives:

```
result[0]: hal.buffer_view
1x2x2xf32=[[3 4][0 0]]
```

But it should be `[[5 6][0 0]]` (you can check this by hand, or just compile and run this same IR for `llvm-cpu`).

### What component(s) does this issue relate to?

Compiler

### Version information

iree-base-compiler  3.4.0rc20250406
iree-base-runtime   3.4.0rc20250406

### Additional context

This issue is present for similar 2d convolution layouts. I suspect that something is going wrong during im2col, since:

```
// -----// IR Dump After ConvolutionToIGEMMPass (iree-codegen-convolution-to-igemm) //----- //
func.func @conv_1d_float32_forward_1x3x2_nhc_2x2x2_chf_nhf_1s_0p_1d_1g$async_dispatch_0_conv_2x2x2x2_f32() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>}>} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !flow.dispatch.tensor<readonly:tensor<3x2xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<4x2xf32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !flow.dispatch.tensor<writeonly:tensor<2x2xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [3, 2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<3x2xf32>> -> tensor<3x2xf32>
  %4 = tensor.empty() : tensor<2x2xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %6 = tensor.empty() : tensor<2x4xf32>
  %7 = iree_linalg_ext.im2col strides = [1] dilations = [1] kernel_size = [2] m_offset = [0] * [1] k_offset = [0] * [1] batch_pos = [] m_pos = [0] k_pos = [1] ins(%3 : tensor<3x2xf32>) outs(%6 : tensor<2x4xf32>) -> tensor<2x4xf32>
  %8 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4, 2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4x2xf32>> -> tensor<4x2xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7, %8 : tensor<2x4xf32>, tensor<4x2xf32>) outs(%5 : tensor<2x2xf32>) attrs =  {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>, padding = [16, 16, 4], promote_operands = [0, 1, 2], reduction = [0, 0, 1], subgroup = [1, 1, 0], workgroup = [16, 16, 0]}>} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %10 = arith.mulf %in, %in_0 : f32
    %11 = arith.addf %10, %out : f32
    linalg.yield %11 : f32
  } -> tensor<2x2xf32>
  flow.dispatch.tensor.store %9, %2, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : tensor<2x2xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x2xf32>>
  return
}
```

Indicates that the `ck` weight dims get flattened together (in that order), but I have no reason to suspect that the `im2col` for the input is also performing the necessary logic to transpose `Hkc` to `Hck` (`H`=output `h`) before flattening the last two dims. This would make sense with the numerical results shown above (since the index of the `1` in `x` is `n=0, h=0, c=1`, and this is getting multiplied by the `3` in `w` at `c=0,k=1,f=0` for the output value at `n=0, ho = 0, f=0`. However, `h != ho + k` for these indices, and this equation would hold if `k` and `c` indices were confused in `w`. This is corroborated by other shapes and example args.

## Trigger

A convolution uses input and filter layouts whose shared reduction dimensions appear in different orders, such as HWC input and CHW filter order.

## Relevant entities

input indexing map; filter indexing map; shared reduction dimensions; im2col K dimension; flattened filter K dimension

## Implemented relation

Corresponding input and filter reduction dimensions must occupy the same positions in the flattened matrix-multiplication K dimension.

## Pre-state

The convolution indexing maps explicitly identify matching reduction dimensions despite their different physical order.

## Binding transition

ConvertConvToIm2ColOp flattens the input patches and filter reduction dimensions into matrix operands.

## Post-state

The input patch is flattened in input order while the filter is flattened in filter order, with no permutation relating the two.

## Validity obligation

Convolution lowering preserves the one-to-one correspondence and order of input and filter reduction coordinates in the flattened K dimension.

## Invalid decision

The im2col representation recorded positions and strides but omitted the permutation from input shared-dimension order to filter shared-dimension order.

## Physical effect

Each flattened input element is multiplied by a filter element from a different logical reduction coordinate.

## Observed consequence

The reported one-dimensional convolution returns 3 and 4 where the corresponding correct values are 5 and 6; CPU lowering remains correct.

## Repair site

IREE LinalgExt Im2colOp metadata, convolution conversion, and im2col decomposition

## Repair

PR 20531 adds and verifies input_filter_perm, derives it from the two indexing maps, and applies it when constructing the im2col K basis; tests include the reported NHC/CHF permutation.

## Why the repair works

The input patch coordinates are reordered into the filter's reduction-dimension order before flattening, so equal K positions again denote equal logical coordinates.

## Causal basis

The issue isolates a swapped reduction-coordinate order, and the patch derives, stores, validates, and consumes exactly that missing permutation while adding the reported layout as a regression case.

## Analysis implication

A convolution-lowering check needs both indexing maps, the derived shared-dimension correspondence, the im2col permutation, and the final K flattening order.

## Fix patches

- [`fixes/iree-20473-pr20531.patch`](fixes/iree-20473-pr20531.patch) — https://github.com/iree-org/iree/pull/20531
