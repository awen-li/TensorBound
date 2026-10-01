# Crash on CPU: abstract_cpu_buffer.cc:92 literal.size_bytes() == size

- Repository: `openxla/xla`
- Report: [https://github.com/openxla/xla/issues/47379](https://github.com/openxla/xla/issues/47379)
- Opened: 2026-08-16T05:53:42Z
- Closed: Open at collection
- Labels: `bug`, `stat:awaiting openxla-eng`, `CPU`

## Reported behavior

Running the attached StableHLO module crashes `run_hlo_module` on the CPU platform (SIGABRT). Crash signature: `abstract_cpu_buffer.cc:92 literal.size_bytes() == size`.

The crash also reproduces on:

- `run_hlo_module` on CPU (HLO proto input) — failed (rc=-6, SIGABRT)

The same module runs successfully on:

- `run_hlo_module` on Interpreter (StableHLO input)
- `run_hlo_module` on GPU (StableHLO input)
- `run_hlo_module` on Interpreter (HLO proto input)
- `run_hlo_module` on GPU (HLO proto input)

The module also compiles cleanly on:

- `hlo_runner_main` on CPU (HLO proto input, compile-only via `--run=false`)
- `hlo_runner_main` on GPU (HLO proto input, compile-only via `--run=false`)

The StableHLO→HLO-proto conversion:

- `hlo-translate` — succeeds

**Command:**

```
hlo-translate \
  --mlir-to-hlo \
  --emit-proto \
  0c9939881072605b.mlir \
  -o \
  0c9939881072605b.hlo
sed -i '1{/^goo\.gle\//d}' 0c9939881072605b.hlo
```

## Environment

- **CPU**: `Intel(R) Core(TM) i9-14900HX`
- **GPU**: `NVIDIA GeForce RTX 4060 Laptop GPU`
- **CUDA Driver**: `580.126.09`
- **XLA commit**: `6f1f2d916dcef0428bfa7ea66765be539126c4f2`
- **StableHLO commit**: `e6f81ebd06b3509f2c7fa6175430aadbd4d724ca`

## IR

```mlir
module @DynamicDimensions attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<?xf32, #stablehlo.bounds<10>>) -> tensor<?xf32, #stablehlo.bounds<10>> {
    %0 = stablehlo.logistic %arg0 : tensor<?xf32, #stablehlo.bounds<10>>
    %1 = stablehlo.convert %0 : (tensor<?xf32, #stablehlo.bounds<10>>) -> tensor<?xi32>
    %2 = stablehlo.slice %1 [0:1] : (tensor<?xi32>) -> tensor<1xi32>
    %3 = stablehlo.reshape %2 : (tensor<1xi32>) -> tensor<i32>
    %4:2 = call @wrapped_11473(%3) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    %5 = stablehlo.convert %4#0 : (tensor<i32>) -> tensor<f32>
    %6 = stablehlo.reshape %5 : (tensor<f32>) -> tensor<f32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<?xf32, #stablehlo.bounds<10>>
    return %7 : tensor<?xf32, #stablehlo.bounds<10>>
  }
  func.func private @wrapped_11473(%arg0: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0) : tensor<i32>, tensor<i32>
    cond {
      %c_1 = stablehlo.constant dense<100> : tensor<i32>
      %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %c_1 = stablehlo.constant dense<1> : tensor<i32>
      %1 = stablehlo.add %iterArg, %c_1 : tensor<i32>
      %2 = stablehlo.add %iterArg_0, %iterArg : tensor<i32>
      stablehlo.return %1, %2 : tensor<i32>, tensor<i32>
    }
    return %0#0, %0#1 : tensor<i32>, tensor<i32>
  }
  func.func private @while_body_11473(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.add %arg0, %c : tensor<i32>
    %1 = stablehlo.add %arg1, %arg0 : tensor<i32>
    return %0, %1 : tensor<i32>, tensor<i32>
  }
  func.func private @while_cond_11473(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i1> {
    %c = stablehlo.constant dense<100> : tensor<i32>
    %0 = stablehlo.compare  LT, %arg0, %c : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}
```

## Reproduction (CPU (StableHLO input) — failed (rc=-6, SIGABRT))

**Command:**

```bash
export XLA_FLAGS='--xla_backend_extra_options=xla_cpu_use_multi_output_fusion --xla_enable_enzyme_comms_opt=true --xla_gpu_async_dot=true --xla_gpu_enable_cudnn_layer_norm=true --xla_gpu_enable_dus_accumulator_zero_init_elimination=true --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_scatter_determinism_expander=true --xla_gpu_experimental_stream_annotation=true --xla_gpu_experimental_use_ragged_dot_fusion --xla_gpu_gemm_rewrite_size_threshold=0 --xla_gpu_multi_streamed_windowed_einsum=true --xla_recognize_reduction_optimization_level=1 --xla_reduce_window_rewrite_base_length=1'
run_hlo_module \
  --platform=CPU \
  --reference_platform= \
  --input_format=stablehlo \
  0c9939881072605b.mlir
```

**Output:**

```

 ** Running 0c9939881072605b.mlir**
Running HLO module with runner HloRunner...
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
F0000 00:00:1786848841.904323 3849807 abstract_cpu_buffer.cc:92] Check failed: literal.size_bytes() == size (40 vs. 44)
*** Check failure stack trace: ***
    @     0x62dc42a34414  absl::lts_20260107::log_internal::LogMessage::SendToLog()
    @     0x62dc42a34396  absl::lts_20260107::log_internal::LogMessage::Flush()
    @     0x62dc385b4c22  xla::PackOrCopy()
    @     0x62dc385b860c  absl::lts_20260107::internal_any_invocable::RemoteInvoker<>()
    @     0x62dc38660eaf  std::_Function_handler<>::_M_invoke()
    @     0x62dc428bd654  Eigen::ThreadPoolTempl<>::WorkerLoop()
    @     0x62dc428bd4e1  std::__invoke_impl<>()
    @     0x62dc428ace4f  tsl::(anonymous namespace)::PThread::ThreadFn()
    @     0x74b43969caa4  (unknown)
    @     0x74b439729c6c  (unknown)

```

## Reference run (GPU (StableHLO input) — succeeds)

**Command:**

```bash
export XLA_FLAGS='--xla_backend_extra_options=xla_cpu_use_multi_output_fusion --xla_enable_enzyme_comms_opt=true --xla_gpu_async_dot=true --xla_gpu_enable_cudnn_layer_norm=true --xla_gpu_enable_dus_accumulator_zero_init_elimination=true --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_scatter_determinism_expander=true --xla_gpu_experimental_stream_annotation=true --xla_gpu_experimental_use_ragged_dot_fusion --xla_gpu_gemm_rewrite_size_threshold=0 --xla_gpu_multi_streamed_windowed_einsum=true --xla_recognize_reduction_optimization_level=1 --xla_reduce_window_rewrite_base_length=1'
run_hlo_module \
  --platform=CUDA \
  --reference_platform= \
  --input_format=stablehlo \
  0c9939881072605b.mlir
```

**Output:**

```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1786848847.043978 3850207 service.cc:178] XLA service 0x5b2db547b210 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
I0000 00:00:1786848847.044003 3850207 service.cc:194]   StreamExecutor [0]: NVIDIA GeForce RTX 4060 Laptop GPU, Compute Capability 8.9 (Driver: 13.0.0[580.126.9]; Runtime: 12.8.0; Toolkit: 12.9.0; DNN: 9.10.2)
I0000 00:00:1786848847.044379 3850207 se_gpu_pjrt_client.cc:1325] Using BFC allocator.
I0000 00:00:1786848847.044397 3850207 gpu_helpers.cc:161] XLA backend will use up to 5.70GiB (6119489536 bytes) on device 0 for BFCAllocator.
I0000 00:00:1786848847.044429 3850207 gpu_helpers.cc:201] XLA backend will use up to 2040528896 bytes on device 0 for CollectiveBFCAllocator.
I0000 00:00:1786848847.050374 3850207 cuda_dnn.cc:436] Loaded cuDNN version 91002

 ** Running 0c9939881072605b.mlir**
Running HLO module with runner HloRunner...
... compiled and ran in 0.0646289s.
Skipping reference runner

```

## Contact

- **Email**: ``ch395@njit.edu, zhihao.yao@njit.edu, benquike@gmail.com``

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises pJRT CPU BufferFromHostLiteral handling of dynamic-shaped literals.

## Observed implementation decision

Literal byte-size/linearization metadata does not include valid dynamic dimension sizes.

## Physical effect

The host literal size disagrees with the destination buffer allocation.

## Observed consequence

Buffer creation crashes a size equality check.

## Reported proposal or workaround

PRs 47804 and 47805 propose dynamic-literal support and metadata initialization.

## Repair trace

The direct PRs were unmerged at the study boundary; no landed repair exists.

## Archived evidence

- Raw report: `data/raw/github/xla/47379/report.json`
- SHA-256: `c72add2c00178d50f6c6df32a5454eecd9c7282f8e1eb1b1b3f11a80d6de8114`
