# 🚨 [GPU][AOT/JIT] Runtime crash when executing bounded dynamic shape HLO: slice extent must be smaller than buffer size

- Repository: `openxla/xla`
- Report: [https://github.com/openxla/xla/issues/33194](https://github.com/openxla/xla/issues/33194)
- Opened: 2025-10-28T10:31:48Z
- Closed: 2026-05-27T06:12:21Z
- Labels: `bug`, `GPU`, `stat:awaiting response from contributor`, `err: Runtime`

## Reported behavior

# Description

When running AOT-compiled GPU executables that contain bounded dynamic shapes, the execution crashes with:

```
Check failed: extent <= base.size() (108 vs. 100)
slice extent 108 must be smaller than buffer #1 size 100
```

The issue reproduces **both in AOT and JIT** modes.

# Reproduction Steps

1. Modify and rebuild the following test file:
`xla/service/gpu/gpu_aot_compilation_test.cc`

2. Add an HLO module that includes dynamic dimensions (bounded shapes), e.g.:

```c++
const absl::string_view hlo_string = R"(
HloModule Test

ENTRY main {
  a = f32[<=5, 5]{1,0} parameter(0)
  ROOT b = f32[<=5, 5]{0,1} copy(a)
}
)";
```

3. Compile Ahead-Of-Time:

```c++
AotCompilationOptions aot_options(compiler->PlatformId());
aot_options.set_executor(stream_exec);
TF_ASSERT_OK_AND_ASSIGN(
    std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
    compiler->CompileAheadOfTime(std::move(module), aot_options));
```

4. Load and execute the AOT executable:

```c++
TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
    std::move(*aot_result).LoadExecutable(&compiler, stream_exec));

TF_ASSERT_OK_AND_ASSIGN(xla::Literal result,
    test_runner_as_hlo_runner().ExecuteWithExecutable(
        wrapped_executable.get(), {&argument}));
```

5. Execution crashes with:

```
Check failed: extent <= base.size() (108 vs. 100)
slice extent 108 must be smaller than buffer #1 size 100
```

# Expected behavior
- The program should execute successfully on GPU when using **bounded dynamic shapes**,
or at least report a clear **“unsupported”** error instead of a segmentation fault.

# Actual behavior
- AOT compilation **succeeds**, but runtime execution **crashes**.
- JIT execution with similar HLO also crashes with the same buffer extent mismatch.

# UT Code
```c++
TEST_F(GpuAotCompilationTest, ExportAndLoadExecutable) {
  const absl::string_view hlo_string = R"(
HloModule Test

ENTRY main {
  a = f32[<=5, 5]{1,0} parameter(0)
  ROOT b = f32[<=5, 5]{0,1} copy(a)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto compiler = backend().compiler();
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          se::PlatformManager::PlatformWithName(name));
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                          platform->ExecutorForDevice(0));

  // Compile AOT.
  AotCompilationOptions aot_options(compiler->PlatformId());
  aot_options.set_executor(stream_exec);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
      compiler->CompileAheadOfTime(std::move(module), aot_options));

  // Serialize-deserialize AOT compilation result.
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                          aot_results[0]->SerializeAsString());
  // aot_results[0] converts to GpuCompiler
  LOG(INFO) << "Debug string: " << aot_results[0]->DebugString();
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AotCompilationResult> aot_result,
      compiler->LoadAotCompilationResult(serialized_aot_result));

  // Load Executable from AOT compilation result.
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      std::move(*aot_result).LoadExecutable(compiler, stream_exec));
  std::unique_ptr<OpaqueExecutable> wrapped_executable =
      test_runner_as_hlo_runner().WrapExecutable(std::move(executable));
  const xla::Literal argument = xla::LiteralUtil::CreateR2<float>(
      {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1},
       {1, 1, 1, 1, 1}});
  TF_ASSERT_OK_AND_ASSIGN(
        xla::Literal result,
        test_runner_as_hlo_runner().ExecuteWithExecutable(
            wrapped_executable.get(), {&argument}));
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
      result, xla::LiteralUtil::CreateR2<float>(
                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1},
                   {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}})));
}
```

# UT test.log
```
exec ${PAGER:-/usr/bin/less} "$0" || exit 1
Executing tests from //xla/service/gpu:gpu_aot_compilation_test
-----------------------------------------------------------------------------
Note: Randomizing tests' orders with a seed of 41023 .
[==========] Running 3 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 3 tests from GpuAotCompilationTest
[ RUN      ] GpuAotCompilationTest.ExportAndLoadExecutable
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1761646825.427641  457517 cuda_dnn.cc:463] Loaded cuDNN version 90300
I0000 00:00:1761646825.485168  457517 gpu_aot_compilation_test.cc:79] Debug string: goo.gle/debugonly
hlo_module_with_config {
  hlo_module {
    name: "Test"
    entry_computation_name: "main"
    computations {
      name: "main"
      instructions {
        name: "a"
        opcode: "parameter"
        shape {
          element_type: F32
          dimensions: 5
          dimensions: 5
          layout {
            minor_to_major: 1
            minor_to_major: 0
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: true
          is_dynamic_dimension: false
        }
        metadata {
        }
        frontend_attributes {
        }
        statistics_viz {
        }
      }
      instructions {
        name: "a.padded"
        opcode: "custom-call"
        shape {
          element_type: TUPLE
          tuple_shapes {
            element_type: F32
            dimensions: 5
            dimensions: 5
            layout {
              minor_to_major: 1
              minor_to_major: 0
              tail_padding_alignment_in_elements: 1
            }
            is_dynamic_dimension: false
            is_dynamic_dimension: false
          }
          tuple_shapes {
            element_type: S32
            layout {
              tail_padding_alignment_in_elements: 1
            }
          }
          tuple_shapes {
            element_type: S32
            layout {
              tail_padding_alignment_in_elements: 1
            }
          }
        }
        metadata {
        }
        custom_call_target: "PadToStatic"
        id: 1
        operand_ids: 0
        feature_group_count: 1
        precision_config {
        }
        batch_group_count: 1
        frontend_attributes {
        }
        custom_call_api_version: API_VERSION_ORIGINAL
        statistics_viz {
        }
      }
      instructions {
        name: "a.data.1"
        opcode: "get-tuple-element"
        shape {
          element_type: F32
          dimensions: 5
          dimensions: 5
          layout {
            minor_to_major: 1
            minor_to_major: 0
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: false
          is_dynamic_dimension: false
        }
        metadata {
        }
        id: 2
        operand_ids: 1
        frontend_attributes {
        }
        statistics_viz {
        }
      }
      instructions {
        name: "a.size.1"
        opcode: "get-tuple-element"
        shape {
          element_type: S32
          layout {
            tail_padding_alignment_in_elements: 1
          }
        }
        metadata {
        }
        tuple_index: 1
        id: 3
        operand_ids: 1
        frontend_attributes {
        }
        statistics_viz {
        }
      }
      instructions {
        name: "constant_1"
        opcode: "constant"
        shape {
          element_type: S32
          layout {
            tail_padding_alignment_in_elements: 1
          }
        }
        metadata {
        }
        literal {
          shape {
            element_type: S32
            layout {
              tail_padding_alignment_in_elements: 1
            }
          }
          s32s: 5
        }
        id: 4
        frontend_attributes {
        }
        statistics_viz {
        }
      }
      instructions {
        name: "custom-call"
        opcode: "custom-call"
        shape {
          element_type: F32
          dimensions: 5
          dimensions: 5
          layout {
            minor_to_major: 0
            minor_to_major: 1
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: true
          is_dynamic_dimension: false
        }
        metadata {
        }
        custom_call_target: "SliceToDynamic"
        id: 5
        operand_ids: 2
        operand_ids: 3
        operand_ids: 4
        feature_group_count: 1
        precision_config {
        }
        batch_group_count: 1
        frontend_attributes {
        }
        custom_call_api_version: API_VERSION_ORIGINAL
        statistics_viz {
        }
      }
      program_shape {
        parameters {
          element_type: F32
          dimensions: 5
          dimensions: 5
          layout {
            minor_to_major: 1
            minor_to_major: 0
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: true
          is_dynamic_dimension: false
        }
        result {
          element_type: F32
          dimensions: 5
          dimensions: 5
          layout {
            minor_to_major: 0
            minor_to_major: 1
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: true
          is_dynamic_dimension: false
        }
        parameter_names: "a"
      }
      root_id: 5
    }
    host_program_shape {
      parameters {
        element_type: F32
        dimensions: 5
        dimensions: 5
        layout {
          minor_to_major: 1
          minor_to_major: 0
          tail_padding_alignment_in_elements: 1
        }
        is_dynamic_dimension: true
        is_dynamic_dimension: false
      }
      result {
        element_type: F32
        dimensions: 5
        dimensions: 5
        layout {
          minor_to_major: 0
          minor_to_major: 1
          tail_padding_alignment_in_elements: 1
        }
        is_dynamic_dimension: true
        is_dynamic_dimension: false
      }
      parameter_names: "p0"
    }
    schedule {
      sequences {
        key: 0
        value {
          instruction_ids: 4
          instruction_ids: 0
          instruction_ids: 1
          instruction_ids: 3
          instruction_ids: 2
          instruction_ids: 5
        }
      }
    }
    input_output_alias {
    }
    is_dynamic: true
    device_assignment {
      replica_count: 1
      computation_count: 1
      computation_devices {
        replica_device_ids: 0
      }
    }
    buffer_donor {
    }
    frontend_attributes {
      map {
        key: "fingerprint_before_lhs"
        value: "a905fb396bb09b6ad42563c2d4e4675d"
      }
      map {
        key: "suggested_combiner_threshold"
        value: "22603077737"
      }
    }
  }
  config {
    entry_computation_layout {
      parameters {
        element_type: F32
        dimensions: 5
        dimensions: 5
        layout {
          minor_to_major: 1
          minor_to_major: 0
          tail_padding_alignment_in_elements: 1
        }
        is_dynamic_dimension: true
        is_dynamic_dimension: false
      }
      result {
        element_type: F32
        dimensions: 5
        dimensions: 5
        layout {
          minor_to_major: 0
          minor_to_major: 1
          tail_padding_alignment_in_elements: 1
        }
        is_dynamic_dimension: true
        is_dynamic_dimension: false
      }
      parameter_names: "p0"
    }
    replica_count: 1
    num_partitions: 1
    intra_op_parallelism_threads: -1
    debug_options {
      xla_disable_hlo_passes: "constant_folding"
      xla_backend_optimization_level: 3
      xla_eliminate_hlo_implicit_broadcast: true
      xla_cpu_multi_thread_eigen: true
      xla_gpu_cuda_data_dir: "./cuda_sdk_lib"
      xla_llvm_enable_alias_scope_metadata: true
      xla_llvm_enable_noalias_metadata: true
      xla_llvm_enable_invariant_load_metadata: true
      xla_llvm_disable_expensive_passes: false
      xla_cpu_use_onednn: false
      xla_cpu_enable_fast_math: false
      xla_gpu_enable_fast_min_max: false
      xla_force_host_platform_device_count: 1
      xla_hlo_evaluator_use_fast_path: true
      xla_dump_hlo_as_html: false
      xla_cpu_fast_math_honor_nans: true
      xla_cpu_fast_math_honor_infs: true
      xla_allow_excess_precision: true
      xla_gpu_autotune_level: 4
      xla_cpu_fast_math_honor_division: true
      xla_cpu_fast_math_honor_functions: true
      xla_dump_include_timestamp: false
      xla_dump_max_hlo_modules: -1
      xla_cpu_enable_xprof_traceme: false
      xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found: false
      xla_cpu_enable_fast_min_max: true
      xla_multiheap_size_constraint_per_heap: -1
      xla_dump_module_metadata: false
      xla_dump_fusion_visualization: false
      xla_gpu_strict_conv_algorithm_picker: true
      xla_gpu_all_reduce_combine_threshold_bytes: 31457287
      xla_gpu_nccl_termination_timeout_seconds: -1
      xla_dump_hlo_as_long_text: true
      xla_gpu_enable_shared_constants: true
      xla_gpu_enable_cublaslt: false
      xla_gpu_shape_checks: RUNTIME
      xla_gpu_use_runtime_fusion: false
      xla_dump_latency_hiding_schedule: false
      xla_dump_enable_mlir_pretty_form: true
      xla_gpu_enable_latency_hiding_scheduler: false
      xla_partitioning_algorithm: PARTITIONING_ALGORITHM_NOOP
      xla_gpu_enable_triton_gemm: true
      xla_gpu_enable_cudnn_int8x32_convolution_reordering: true
      xla_gpu_triton_gemm_any: true
      xla_gpu_enable_while_loop_reduce_scatter_code_motion: false
      xla_gpu_collective_inflation_factor: 1
      xla_gpu_graph_min_graph_size: 5
      xla_gpu_enable_reassociation_for_converted_ar: true
      xla_gpu_pgle_profile_file_or_directory_path: ""
      xla_gpu_all_gather_combine_threshold_bytes: 31457287
      xla_gpu_reduce_scatter_combine_threshold_bytes: 31457287
      xla_gpu_enable_highest_priority_async_stream: true
      xla_gpu_enable_pipelined_all_reduce: false
      xla_gpu_exhaustive_tiling_search: false
      xla_gpu_auto_spmd_partitioning_memory_budget_gb: 0
      xla_gpu_auto_spmd_partitioning_memory_budget_ratio: 1.1
      xla_gpu_enable_pipelined_all_gather: false
      xla_gpu_redzone_padding_bytes: 8388608
      xla_gpu_enable_pipelined_reduce_scatter: true
      xla_gpu_fused_attention_use_cudnn_rng: false
      xla_gpu_copy_insertion_use_region_analysis: false
      xla_gpu_collective_permute_decomposer_threshold: 9223372036854775807
      xla_gpu_collect_cost_model_stats: false
      xla_gpu_enable_split_k_autotuning: true
      xla_gpu_enable_reduction_epilogue_fusion: true
      xla_gpu_enable_pipelined_p2p: false
      xla_gpu_cublas_fallback: true
      xla_gpu_enable_while_loop_double_buffering: false
      xla_gpu_filter_kernels_spilling_registers_on_autotuning: true
      xla_debug_buffer_assignment_show_max: 15
      xla_detailed_logging: true
      xla_enable_dumping: true
      xla_gpu_enable_all_gather_combine_by_dim: false
      xla_gpu_enable_analytical_latency_estimator: false
      xla_gpu_llvm_verification_level: 0
      xla_gpu_enable_reduce_scatter_combine_by_dim: false
      xla_gpu_enable_command_buffer: FUSION
      xla_gpu_enable_command_buffer: CUBLAS
      xla_gpu_enable_command_buffer: CUBLASLT
      xla_gpu_enable_command_buffer: CUSTOM_CALL
      xla_gpu_enable_command_buffer: CUDNN
      xla_gpu_enable_cub_radix_sort: true
      xla_gpu_memory_limit_slop_factor: 95
      xla_gpu_target_config_filename: ""
      xla_gpu_enable_cudnn_layer_norm: false
      xla_gpu_threshold_for_windowed_einsum_mib: 100000
      xla_gpu_enable_nccl_user_buffers: false
      xla_gpu_enable_llvm_module_compilation_parallelism: false
      xla_gpu_enable_libnvptxcompiler: false
      xla_gpu_enable_nccl_comm_splitting: true
      xla_gpu_nccl_collective_max_nchannels: 0
      xla_gpu_nccl_p2p_max_nchannels: 0
      xla_gpu_nccl_init_max_rank_per_root_ratio: 0
      xla_gpu_multi_streamed_windowed_einsum: true
      xla_gpu_gemm_rewrite_size_threshold: 100
      xla_gpu_require_complete_aot_autotune_results: false
      xla_gpu_cudnn_gemm_fusion_level: 0
      xla_gpu_use_memcpy_local_p2p: false
      xla_gpu_autotune_max_solutions: 0
      xla_dump_large_constants: false
      xla_gpu_verify_triton_fusion_numerics: false
      xla_reduce_window_rewrite_base_length: 16
      xla_gpu_enable_while_loop_unrolling: WHILE_LOOP_UNROLLING_AUTO_UNROLL
      xla_gpu_enable_host_memory_offloading: false
      xla_llvm_force_inline_before_split: true
      xla_gpu_nccl_terminate_on_error: false
      xla_gpu_shard_autotuning: true
      xla_gpu_enable_approx_costly_collectives: false
      xla_cpu_enable_concurrency_optimized_scheduler: true
      xla_cpu_prefer_vector_width: 256
      xla_gpu_per_fusion_autotune_cache_dir: ""
      xla_cmd_buffer_trace_cache_size: 16
      xla_gpu_temp_buffer_use_separate_color: false
      xla_syntax_sugar_async_ops: false
      xla_gpu_autotune_gemm_rtol: 0.1
      xla_enable_command_buffers_during_profiling: false
      xla_gpu_cudnn_gemm_max_plans: 5
      xla_cpu_parallel_codegen_split_count: 32
      xla_gpu_experimental_autotune_cache_mode: AUTOTUNE_CACHE_MODE_UPDATE
      xla_gpu_executable_warn_stuck_timeout_seconds: 10
      xla_gpu_executable_terminate_timeout_seconds: 30
      xla_gpu_experimental_disable_binary_libraries: false
      xla_ignore_channel_id: false
      xla_gpu_dot_merger_threshold_mb: 32
      xla_cpu_max_isa: ""
      xla_gpu_experimental_enable_fusion_block_level_rewriter: false
      xla_enable_fast_math: false
      xla_gpu_experimental_parallel_collective_overlap_limit: 1
      xla_cpu_copy_insertion_use_region_analysis: false
      xla_gpu_operand_bytes_threshold_for_windowed_einsum: -1
      xla_gpu_experimental_enable_triton_heroless_priority_fusion: false
      xla_gpu_pgle_accuracy_checker: PGLE_STRICTNESS_LEVEL_WARN
      xla_gpu_experimental_stream_annotation: true
      xla_gpu_libnvjitlink_mode: LIB_NV_JIT_LINK_MODE_AUTO
      xla_pjrt_allow_auto_layout_in_hlo: false
      xla_gpu_enable_scatter_determinism_expander: false
      xla_gpu_require_exclusive_lock: false
      xla_gpu_generate_debug_info: false
      xla_gpu_generate_line_info: false
      xla_gpu_unsupported_enable_ragged_all_to_all_decomposer: false
      xla_gpu_experimental_pipeline_parallelism_opt_level: PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE
      xla_gpu_fail_ptx_compilation_on_register_spilling: false
      xla_gpu_collectives_use_persistent_cliques: false
      xla_gpu_experimental_enable_triton_tma: false
      xla_gpu_enable_analytical_sol_latency_estimator: true
      xla_gpu_analytical_latency_estimator_options {
        key: "chunk_prep_us"
        value: "-1"
      }
      xla_gpu_analytical_latency_estimator_options {
        key: "chunk_size_bytes"
        value: "-1"
      }
      xla_gpu_analytical_latency_estimator_options {
        key: "gpus_per_node"
        value: "-1"
      }
      xla_gpu_analytical_latency_estimator_options {
        key: "nccl_op_launch_us"
        value: "-1"
      }
      xla_gpu_analytical_latency_estimator_options {
        key: "nic_speed_gbps"
        value: "-1"
      }
      xla_gpu_analytical_latency_estimator_options {
        key: "rtt_us"
        value: "-1"
      }
      xla_gpu_unsupported_annotate_with_emitter_loc: false
      xla_cpu_use_xnnpack: true
      xla_gpu_experimental_pack_dot_operands_along_k_dimension: true
      xla_unsupported_crash_on_hlo_pass_fix_max_iterations: false
      xla_cpu_experimental_xnn_grap
F0000 00:00:1761646825.524539  457517 buffer_allocations.cc:79] Check failed: extent <= base.size() (108 vs. 100) slice extent 108 must be smaller than buffer #1 size 100
*** Check failure stack trace: ***
    @     0x55873e10dff4  absl::lts_20250814::log_internal::LogMessage::SendToLog()
    @     0x55873e10df76  absl::lts_20250814::log_internal::LogMessage::Flush()
    @     0x55873c339fc4  xla::gpu::BufferAllocations::GetDeviceAddress()
    @     0x5587352f0f16  xla::gpu::KernelThunk::ExecuteOnStream()
    @     0x55873531587e  xla::gpu::SequentialThunk::ExecuteOnStream()
    @     0x558735272a8c  xla::gpu::(anonymous namespace)::ExecuteThunksImpl()
    @     0x558735270196  xla::gpu::GpuExecutable::ExecuteThunks()
    @     0x55873526e9ba  xla::gpu::GpuExecutable::ExecuteAsyncOnStreamImpl()
    @     0x55873526e47d  xla::gpu::GpuExecutable::ExecuteAsyncOnStream()
    @     0x55873cfe84e8  xla::Executable::ExecuteAsyncOnStreamWrapper()
    @     0x55873cfe8216  xla::Executable::ExecuteOnStreamWrapper()
    @     0x55873ccec144  xla::HloRunner::ExecuteWithExecutionInputs()
    @     0x55873ccebc9e  xla::HloRunner::ExecuteWithDeviceBuffers()
    @     0x55873cceb064  xla::HloRunner::ExecuteWithExecutable()
    @     0x55873cd04c00  xla::HloRunnerInterface::ExecuteWithExecutable()
    @     0x5587344e2cf3  xla::gpu::GpuAotCompilationTest_ExportAndLoadExecutable_Test::TestBody()
    @     0x55873db8f52e  testing::internal::HandleExceptionsInMethodIfSupported<>()
    @     0x55873db8f420  testing::Test::Run()
    @     0x55873db903a2  testing::TestInfo::Run()
    @     0x55873db914d5  testing::TestSuite::Run()
    @     0x55873dba24a1  testing::internal::UnitTestImpl::RunAllTests()
    @     0x55873dba1abe  testing::internal::HandleExceptionsInMethodIfSupported<>()
    @     0x55873dba1939  testing::UnitTest::Run()
    @     0x55873db0fb35  main
    @     0x7f76e764b9d0  (unknown)
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

🚨 [GPU][AOT/JIT] Runtime crash when executing bounded dynamic shape HLO: slice extent must be smaller than buffer size

## Observed implementation decision

The host-to-device transfer binds only the literal payload to the parameter allocation while the compiled PadToStatic custom call binds that allocation to a larger data-plus-metadata representation.

## Physical effect

A bounded-dynamic parameter reaches device memory without the dimension-size metadata that PadToStatic expects adjacent to its padded data buffer.

## Observed consequence

Both AOT and JIT execution request a 108-byte slice from a 100-byte device buffer and abort in BufferAllocations::GetDeviceAddress.

## Reported proposal or workaround

A maintainer traces the size mismatch to omitted dynamic-shape metadata; supplying the dimension size explicitly through set-dimension-size avoids PadToStatic and executes the bounded-shape program.

## Repair trace

The missing dynamic-shape metadata is diagnosed and a set-dimension-size workaround executes, but no repair of the original transfer path or post-fix run is available.

## Archived evidence

- Raw report: `data/raw/github/xla/33194/report.json`
- SHA-256: `5ec8b4c1630c058d08b1ff0f3bf3c80d393e71fb46b3d715489448c8b346ac26`
