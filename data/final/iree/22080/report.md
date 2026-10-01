# [iree-run-module] Issues in using --expected_output flag for si64 type

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/22080](https://github.com/iree-org/iree/issues/22080)
- Opened: 2025-09-24T08:00:20Z
- Closed: Open at collection
- Labels: `bug 🐞`

## Reported behavior

### What happened?

When the result tensor type is si64, the --expected_output flag with iree-run-module seems to identify it as a different type than i64 resulting in the following error.

```
[FAILED] result[0]: metadata is i64; expected that the view matches si64; expected that the view is equal to contents of a view of si64
  expected:
si64=8
  actual:
i64=8
```

### Steps to reproduce your issue

_**test.mlir**_

```
func.func @simple_add_onnx(%arg0: !torch.vtensor<[],si64>, %arg1: !torch.vtensor<[],si64>) -> !torch.vtensor<[],si64>
    attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.0"} {
  %0 = torch.operator "onnx.Add"(%arg0, %arg1) : (!torch.vtensor<[],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}
```

Compile
```
iree-compile test.mlir --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu  --iree-llvmcpu-target-cpu=x86-64 -o outs.vmfb
```

Save output to npy
```
iree-run-module --device=local-sync --module=outs.vmfb --function=simple_add_onnx --input=si64=4 --input=si64=4 --output=@out.npy
```

Verify output again using the generated npy from previous invocation
```
iree-run-module --device=local-sync --module=outs.vmfb --function=simple_add_onnx --input=si64=4 --input=si64=4 --expected_output=@out.npy
```


Results in the following error

```
EXEC @simple_add_onnx
[FAILED] result[0]: metadata is i64; expected that the view matches si64; expected that the view is equal to contents of a view of si64
  expected:
si64=8
  actual:
i64=8
```

### What component(s) does this issue relate to?

Runtime

### Version information

a03fdad

### Additional context

Fixing this would be helpful when running larger models like llama

### Options for solving the issue

The following are options for a solution to the issue (see also discussion #24804):
1. Generally treat signless integer and signed integer of same width as matching in output comparisons.
   - advantages: easy to use, small code change
   - disadvantages: loses element type strictness for values from .bin files and for immediate values on command line
2. Carry source of reference (.npy file or not) to the actual comparison and treat signless/signed as matching only for .npy.
   - advantages: easy to use, relaxing i/si comparison only for .npy files
   - disadvantages: larger code changes touching several .c files and several internal data structures
3. Support explicit metadata (`=123x456x789xi64=@file.npy`) for .npy files.
   - advantages: keeping element type matching strict
   - disadvantages: cumbersome to use for .npy with tensors (need to manually specify full shape/dtype although most of the information is contained in the .npy file)
4. Support a new minimal syntax to change signed integer data from .npy file to signless (e.g. `=@#file.npy`)
   - advantages: keeping element type matching strict, quite easy to use, small code change
   - disadvantages: introducing a new syntax for this case

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises iree-run-module comparison of signless integer results against signed NPY expected-output files.

## Observed implementation decision

The tooling cannot represent signless integers in NPY and its comparison path does not have an explicit user-supplied type binding.

## Physical effect

Expected and actual buffers with compatible bits are rejected or can be conflated under ambiguous signedness.

## Observed consequence

The --expected_output check fails for si64/signless i64 combinations.

## Reported proposal or workaround

PR 24804 proposes an explicit policy; a downstream CI comparison change does not repair IREE tooling.

## Repair trace

PR 24804 was not merged, and discussion has not selected and landed a comparison contract with regression coverage.

## Archived evidence

- Raw report: `data/raw/github/iree/22080/report.json`
- SHA-256: `83b1a5e8f9bc40dc4842dc3b500d4b8e87479ba30b3cfea2b50a5573f985a294`
