# Failure after converting HLO to StableHLO: RET_CHECK failure (xla/backends/cpu/runtime/collective_thunk.cc:167) params.device_assignment Device assignment is null

- Repository: `openxla/xla`
- Report: [https://github.com/openxla/xla/issues/39825](https://github.com/openxla/xla/issues/39825)
- Opened: 2026-03-25T05:45:52Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

# Bug Report:

This bug is triggered by the HLO-to-StableHLO conversion: the HLO module runs successfully via `run_hlo_module --input_format=hlo`, and the conversion via `hlo-translate --hlo-to-mlir` also succeeds, but running the translated StableHLO via `run_hlo_module --input_format=stablehlo` fails with:

> `RET_CHECK failure (xla/backends/cpu/runtime/collective_thunk.cc:167) params.device_assignment Device assignment is null`

## Environment

- **XLA commit**: `5ce7908a2d32a9f91fd99380435cda1b645c8cc7`
- **CPU**: `Intel(R) Core(TM) i9-14900HX`
- **GPU**: `NVIDIA GeForce RTX 4060 Laptop GPU`
- **CUDA Driver**: `580.126.09`

## run_hlo_module (HLO) — Success

**HLO:**

```
HloModule jit_matmul, entry_computation_layout={(f64[8,3]{1,0}, f64[])->f64[8,3]{1,0}}, allow_spmd_sharding_propagation_to_parameters={false,true}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=2

ENTRY main.4 {
  Arg_1.2 = f64[] parameter(1)
  Arg_0.1 = f64[8,3]{1,0} parameter(0), sharding={devices=[2,1]0,1}
  ROOT dot.3 = f64[8,3]{1,0} dot(Arg_1.2, Arg_0.1), lhs_contracting_dims={}, rhs_contracting_dims={}
}
```

**Execution Command:**

```bash
run_hlo_module \
  --platform=CPU \
  --reference_platform= \
  --input_format=hlo \
  jit_matmul_204e03cb_361.hlo
```

**Output:**

```

 ** Running jit_matmul_204e03cb_361.hlo**
Running HLO module with runner Host...
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1774416778.019256 2088405 sharding_propagation.cc:3127] GSPMD sharding propagation is going to be deprecated and not supported in the future. Please consider migrating to Shardy (https://openxla.org/shardy). For reference, Shardy is already the default partitioner in JAX.
... compiled and ran in 0.0247959s.
Skipping reference runner

```

## run_hlo_module (StableHLO) — FAIL

**Translation Command:**

```bash
hlo-translate \
  --hlo-to-mlir \
  jit_matmul_204e03cb_361.hlo \
  -o \
  jit_matmul_204e03cb_361.mlir
```

**IR After Translation:**

```mlir
module @jit_matmul attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.num_partitions = 2 : i32, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<8x3xf64> {mhlo.sharding = "{devices=[2,1]0,1}"}, %arg1: tensor<f64>) -> tensor<8x3xf64> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<f64>, tensor<8x3xf64>) -> tensor<8x3xf64>
    return %0 : tensor<8x3xf64>
  }
}
```

**Execution Command:**

```bash
run_hlo_module \
  --platform=CPU \
  --reference_platform= \
  --input_format=stablehlo \
  jit_matmul_204e03cb_361.mlir
```

**Output:**

```

 ** Running jit_matmul_204e03cb_361.mlir**
Running HLO module with runner Host...
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1774416778.184489 2088505 sharding_propagation.cc:3127] GSPMD sharding propagation is going to be deprecated and not supported in the future. Please consider migrating to Shardy (https://openxla.org/shardy). For reference, Shardy is already the default partitioner in JAX.
E0000 00:00:1774416778.204544 2088505 status_macros.cc:58] INTERNAL: RET_CHECK failure (xla/backends/cpu/runtime/collective_thunk.cc:167) params.device_assignment Device assignment is null
*** Begin stack trace ***

























	__libc_start_main

*** End stack trace ***

... compiled and ran in 0.0206898s.
INTERNAL: RET_CHECK failure (xla/backends/cpu/runtime/collective_thunk.cc:167) params.device_assignment Device assignment is null
	Failed to execute on Host

```

## Contact

- **Email**: `ch395@njit.edu, zhihao.yao@njit.edu, benquike@gmail.com`

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises xLA CPU collective execution after HLO-to-StableHLO conversion.

## Observed implementation decision

The converted executable reaches CollectiveThunk without a device_assignment binding.

## Physical effect

Runtime collective setup dereferences absent assignment metadata.

## Observed consequence

Execution fails a RET_CHECK.

## Reported proposal or workaround

The report provides the failing conversion path.

## Repair trace

No source-localized metadata-preservation repair or fixed regression is linked.

## Archived evidence

- Raw report: `data/raw/github/xla/39825/report.json`
- SHA-256: `beef412e059733557bbfbbdd5c70bae2021712daea364a2d75d123c56c441a2d`
