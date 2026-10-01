# [CUDA] Runtime export lookup uses unsanitized name while compilation sanitizes export names

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/24653](https://github.com/iree-org/iree/issues/24653)
- Opened: 2026-06-30T16:31:07Z
- Closed: open at collection time
- Labels: `bug 🐞`

## Reported behavior

### What happened?

In `native_executable.c` the export function lookup uses a direct string equality check: https://github.com/iree-org/iree/blob/a4d0287253fbe7211f545cda304b487218c2a711/runtime/src/iree/hal/drivers/cuda/native_executable.c#L559

However, during compilation, the export names are sanitized due to strict naming requirements in PTX:
https://github.com/iree-org/iree/blob/a4d0287253fbe7211f545cda304b487218c2a711/compiler/plugins/target/CUDA/CUDATarget.cpp#L545
https://github.com/iree-org/iree/blob/a4d0287253fbe7211f545cda304b487218c2a711/compiler/plugins/target/CUDA/CUDATarget.cpp#L659



### Steps to reproduce your issue

MRE:
```mlir
module @repro {
  util.func public @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = util.call @callee$async(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    util.return %0 : tensor<4xf32>
  }
  util.func private @callee$async(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>
      attributes {inlining_policy = #util.inline.never} {
    %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
    util.return %0 : tensor<4xf32>
  }
}
```

```bash
$ iree-compile --iree-hal-target-device=cuda --iree-cuda-target=ampere \
             --iree-execution-model=async-external \
             iree_cuda_dollar_repro.mlir -o repro_cuda.vmfb

# NOT_FOUND; function 'callee$async_dispatch_0_elementwise_4_f32' not found in executable
$ iree-run-module --module=repro_cuda.vmfb --device=cuda \
                --function=main --input=4xf32=1 --input=4xf32=2
```

Compiles and runs fine on CPU:
```bash
$ iree-compile --iree-hal-target-device=local \
             --iree-hal-local-target-device-backends=llvm-cpu \
             --iree-execution-model=async-external \
             iree_cuda_dollar_repro.mlir -o repro_cpu.vmfb
$ iree-run-module --module=repro_cpu.vmfb --device=local-task \
                --function=main --input=4xf32=1 --input=4xf32=2
```

### What component(s) does this issue relate to?

Runtime

### Version information

Commit hash 59f952c5a83777e78f251e238338d7345a1bfb36

### Additional context

Discovered when compiling a CNN model exported from pytorch using `iree-turbine`. I believe this same workflow ran fine around a year ago but cannot pinpoint exactly when the regression occurred, since bisecting that many commits would take forever.

## Trigger

Compilation creates a CUDA dispatch named `callee$async_dispatch_0_elementwise_4_f32` and sanitizes the PTX kernel symbol while the runtime later looks up the original export name.

## Relevant entities

IREE executable export name; sanitized PTX kernel symbol; CUDA executable FlatBuffer; runtime export table; cuModuleGetFunction; hal.executable.lookup.function

## Implemented relation

Each logical IREE export identity maps to one backend-valid PTX kernel symbol, with both names retained for their respective consumers.

## Pre-state

The compiler owns the original export name and derives a sanitized PTX identifier by replacing characters that PTX does not allow.

## Binding transition

The old serializer stored only the sanitized value as kernel_name, and executable creation copied that value into the runtime export table used for logical-name lookup.

## Post-state

The runtime table contained `callee_async_dispatch_0_elementwise_4_f32` while the VM requested `callee$async_dispatch_0_elementwise_4_f32`.

## Validity obligation

Backend symbol sanitization changes only the physical kernel identifier; it preserves an explicit mapping from the unchanged logical export name to that identifier.

## Invalid decision

CUDA serialization reused the sanitized backend symbol as the logical export identity, collapsing two names with different naming rules into one field.

## Physical effect

The runtime's direct export-name comparison could not resolve the requested function even though the compiled PTX kernel was present.

## Observed consequence

iree-run-module failed with `NOT_FOUND; function 'callee$async_dispatch_0_elementwise_4_f32' not found in executable`, while the CPU backend ran the same module.

## Repair site

compiler/plugins/target/CUDA/CUDATarget.cpp; runtime/src/iree/schemas/cuda_executable_def.fbs; runtime/src/iree/hal/drivers/cuda/native_executable.c

## Repair

PR 24889 appends an optional export_name field to the CUDA executable schema, serializes the original IREE name there, keeps the sanitized kernel_name for cuModuleGetFunction, and stores export_name in the runtime lookup table. New runtimes fall back to kernel_name for older VMFBs.

## Why the repair works

PTX loading and IREE export lookup now consume the names defined by their own interfaces, while the two serialized fields preserve the correspondence between the logical function and its physical kernel.

## Causal basis

The issue identifies the compiler sanitization and runtime equality check and supplies a `$`-containing reproducer. The first proposed repair, PR 24718, was closed by an IREE maintainer as addressed by PR 24889. The merged PR describes the same lookup failure, separates the original export name from the PTX symbol, and reports that its CUDA reproduction succeeds after the change.

## Analysis implication

Name-binding analysis needs the logical symbol, every backend normalization, the serialized mapping, and the lookup key at each consumer rather than treating a sanitized symbol as an identity-preserving rename.

## Fix patches

- [`fixes/iree-24653-pr24889.patch`](fixes/iree-24653-pr24889.patch) — https://github.com/iree-org/iree/pull/24889
