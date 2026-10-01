# [Codegen][LLVMGPU] Correctness issue with softmax due to TileDispatchUsingForall failing to fuse

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/19189](https://github.com/iree-org/iree/issues/19189)
- Opened: 2024-11-18T18:00:26Z
- Closed: 2024-12-07T02:14:02Z
- Labels: `bug 🐞`, `codegen/rocm`

## Reported behavior

The following dispatch produces incorrect results due to a silent failure to distribute all ops to workgroups.
```mlir
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx1100", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<WMMA_F32_16x16x16_F16>, <WMMA_F16_16x16x16_F16>, <WMMA_I32_16x16x16_I8>, <WMMA_I32_16x16x16_I8>, <WMMA_I32_16x16x16_I8>], subgroup_size_choices = [32, 64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 8192>>, ukernels = "none"}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>
module {
  hal.executable public @torch_jit$async_dispatch_9 {
    hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @torch_jit$async_dispatch_9_softmax_12x197x197xf32_dispatch_tensor_store ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @torch_jit$async_dispatch_9_softmax_12x197x197xf32_dispatch_tensor_store() attributes {translation_info = #translation} {
          %cst = arith.constant 0.000000e+00 : f32
          %cst_0 = arith.constant -3.40282347E+38 : f32
          %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
          %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
          %2 = arith.index_castui %0 : i32 to index
          %3 = arith.index_castui %1 : i32 to index
          %4:2 = util.assume.int
              %2[<umin = 4841472, umax = 4841472, udiv = 4841472>, <umin = 5446656, umax = 5446656, udiv = 5446656>, <umin = 5446656, umax = 5446656, udiv = 5446656>, <umin = 5446656, umax = 5446656, udiv = 5446656>, <umin = 5446656, umax = 5446656, udiv = 5446656>, <umin = 5446656, umax = 5446656, udiv = 5446656>, <umin = 5446656, umax = 5446656, udiv = 5446656>, <umin = 5446656, umax = 5446656, udiv = 5446656>, <umin = 5446656, umax = 5446656, udiv = 5446656>, <umin = 5446656, umax = 5446656, udiv = 5446656>, <umin = 5446656, umax = 5446656, udiv = 5446656>, <umin = 5446656, umax = 5446656, udiv = 5446656>],
              %3[<umin = 0, umax = 0>, <umin = 1210368, umax = 1210368, udiv = 1210368>, <umin = 605184, umax = 605184, udiv = 605184>, <umin = 605184, umax = 605184, udiv = 605184>, <umin = 605184, umax = 605184, udiv = 605184>, <umin = 605184, umax = 605184, udiv = 605184>, <umin = 605184, umax = 605184, udiv = 605184>, <umin = 605184, umax = 605184, udiv = 605184>, <umin = 605184, umax = 605184, udiv = 605184>, <umin = 605184, umax = 605184, udiv = 605184>, <umin = 605184, umax = 605184, udiv = 605184>, <umin = 605184, umax = 605184, udiv = 605184>]
            : index, index
          %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%4#0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<12x197x197xf32>>
          %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%4#1) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<12x197x197xf32>>
          %7 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0], sizes = [12, 197, 197], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<12x197x197xf32>> -> tensor<12x197x197xf32>
          %8 = tensor.empty() : tensor<12x197x197xf32>
          %9 = tensor.empty() : tensor<12x197xf32>
          %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<12x197xf32>) -> tensor<12x197xf32>
          %11 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7 : tensor<12x197x197xf32>) outs(%10 : tensor<12x197xf32>) {
          ^bb0(%in: f32, %out: f32):
            %15 = arith.maxnumf %in, %out : f32
            linalg.yield %15 : f32
          } -> tensor<12x197xf32>
          %12 = linalg.fill ins(%cst : f32) outs(%9 : tensor<12x197xf32>) -> tensor<12x197xf32>
          %13 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7, %11 : tensor<12x197x197xf32>, tensor<12x197xf32>) outs(%12 : tensor<12x197xf32>) attrs =  {lowering_config = #iree_gpu.lowering_config<{reduction = [0, 0, 1], thread = [1, 1, 0], workgroup = [4, 8, 0]}>} {
          ^bb0(%in: f32, %in_1: f32, %out: f32):
            %15 = arith.subf %in, %in_1 : f32
            %16 = math.exp %15 : f32
            %17 = arith.addf %16, %out : f32
            linalg.yield %17 : f32
          } -> tensor<12x197xf32>
          %14:2 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7, %11, %13 : tensor<12x197x197xf32>, tensor<12x197xf32>, tensor<12x197xf32>) outs(%8, %8 : tensor<12x197x197xf32>, tensor<12x197x197xf32>) {
          ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32):
            %15 = arith.subf %in, %in_1 : f32
            %16 = math.exp %15 : f32
            %17 = arith.divf %16, %in_2 : f32
            linalg.yield %16, %17 : f32, f32
          } -> (tensor<12x197x197xf32>, tensor<12x197x197xf32>)
          flow.dispatch.tensor.store %14#1, %6, offsets = [0, 0, 0], sizes = [12, 197, 197], strides = [1, 1, 1] : tensor<12x197x197xf32> -> !flow.dispatch.tensor<writeonly:tensor<12x197x197xf32>>
          return
        }
      }
    }
  }
}
```

## Trigger

One producer feeds multiple tilable consumers after consumer fusion creates additional producer opportunities.

## Relevant entities

tiling root; producer; multiple consumers; loop-yielded replacements; workgroup-distributed operations

## Implemented relation

A fused producer value is yielded from a tiled loop only when one compatible consumer can receive that replacement.

## Pre-state

The softmax graph contains a producer whose value reaches more than one tilable consumer.

## Binding transition

The old pass selects yielded replacements from dominance alone and tries to use one yielded value across incompatible consumer fusion paths.

## Post-state

Fusion silently stops before every operation receives the intended workgroup distribution.

## Validity obligation

Yield selection accounts for the number and compatibility of tilable consumers and revisits producer opportunities after consumer fusion.

## Invalid decision

TileDispatchUsingForall yielded a replacement for a value with multiple tilable consumers.

## Physical effect

Part of the dispatch remains undistributed and executes with an incorrect workgroup-to-tensor mapping.

## Observed consequence

The LLVMGPU softmax dispatch returns incorrect numerical results.

## Repair site

compiler/src/iree/compiler/Codegen/Common/TileDispatchUsingForall.cpp

## Repair

PR 19212 declines yielded replacement when a fused value has multiple tilable consumers, queues fusion work, and performs another producer-fusion pass with a softmax-shaped regression.

## Why the repair works

The pass avoids an incompatible shared replacement and still revisits slices exposed by consumer fusion, so all eligible operations receive coherent distribution.

## Causal basis

The report's multi-use softmax IR is named as resolved by PR 19212, whose regression uses the same dimensions and whose guard targets multi-consumer yielded values.

## Analysis implication

Fusion validation needs producer-consumer multiplicity, dominance, yielded replacements, generated slices, and final distribution coverage.

## Fix patches

- [`fixes/iree-19189-pr19212.patch`](fixes/iree-19189-pr19212.patch) — https://github.com/iree-org/iree/pull/19212
