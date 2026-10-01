# [Stream] SplitParameterEncoding in consolidate mode doesn't work

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/24920](https://github.com/iree-org/iree/issues/24920)
- Opened: 2026-09-11T11:24:30Z
- Closed: Open at collection
- Labels: `bug 🐞`

## Reported behavior

### What happened?

SplitParameterEncoding consolidate mode states:

https://github.com/iree-org/iree/blob/0df50f9a8e40ab1878479124b952f45d9f8c0650/compiler/src/iree/compiler/Dialect/Stream/Transforms/Passes.h#L97

```
"Merge all encoded parameters and original parameters, into a single consolidated scope."
```

and

https://github.com/iree-org/iree/blob/0df50f9a8e40ab1878479124b952f45d9f8c0650/compiler/src/iree/compiler/Dialect/Stream/Transforms/SplitParameterEncoder.cpp#L867

    // - Consolidate mode (includeUnmodified=true): includes all parameters

However there are multiple cases where this does not happen. This is a major issue since this basically completely prohibits the use of input parameter files as the parameter encoder themself will not remove the encoded parameters from the original scope. So in the end if you want to run anything while using SplitParameterEncoding you have to ship both the inputs and outputs of the encoder to your final .vmfb which means you probably have to carry your entire model parameters twice, where most of the input file is simply dead bytes.

### Steps to reproduce your issue

```mlir
util.global private @weight_a = #flow.parameter.named<"model"::"weight_a"> : tensor<256xi32>
util.global private @weight_b = #flow.parameter.named<"model"::"weight_b"> : tensor<256xi32>
util.global private @shifted : tensor<256xi32>
util.global private @padded : tensor<512xi32>

util.initializer {
  %a = util.global.load @weight_a : tensor<256xi32>
  %c42 = arith.constant 42 : i32
  %init = tensor.empty() : tensor<256xi32>
  %c42s = linalg.fill ins(%c42 : i32) outs(%init : tensor<256xi32>) -> tensor<256xi32>
  %shifted = linalg.add ins(%a, %c42s : tensor<256xi32>, tensor<256xi32>) outs(%init : tensor<256xi32>) -> tensor<256xi32>
  util.global.store %shifted, @shifted : tensor<256xi32>

  %b = util.global.load @weight_b : tensor<256xi32>
  %c0 = arith.constant 0 : i32
  %padded = tensor.pad %b low[0] high[256] {
  ^bb0(%i: index):
    tensor.yield %c0 : i32
  } : tensor<256xi32> to tensor<512xi32>
  util.global.store %padded, @padded : tensor<512xi32>
  util.return
}

func.func @main() -> (tensor<256xi32>, tensor<512xi32>) {
  %shifted = util.global.load @shifted : tensor<256xi32>
  %padded = util.global.load @padded : tensor<512xi32>
  return %shifted, %padded : tensor<256xi32>, tensor<512xi32>
}
```

```

iree-create-parameters --splat=weight_a=256xi32=1 --splat=weight_b=256xi32=7 --output=input.irpa --quiet

iree-compile consolidate_leak.mlir --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx  --iree-parameter-encoder-output-file=encoder.mlir -o main.vmfb

iree-compile encoder.mlir --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx -o encoder.vmfb

iree-encode-parameters --module=encoder.vmfb --parameters=model=input.irpa --output=encoded=output.irpa --quiet

iree-run-module --device=local-sync --module=main.vmfb --function=main --parameters=encoded=output.irpa
```

Error:

```
runtime/src/iree/modules/io/parameters/module.c:130: NOT_FOUND; no provider registered that handles scopes like 'model'; while invoking native function io_parameters.load
```

### What component(s) does this issue relate to?

Compiler

### Version information

_No response_

### Additional context

_No response_

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises iREE Stream SplitParameterEncoding in consolidate mode.

## Observed implementation decision

The pass's consolidation logic does not preserve all parameter ranges/encodings when splitting resources.

## Physical effect

Generated consolidated parameter references bind logical slices to incorrect physical segments.

## Observed consequence

The reported compilation path fails or produces invalid parameter access.

## Reported proposal or workaround

Maintainer discussion states that the pass requires a broader rework.

## Repair trace

No implemented rework, landed patch, or fixed regression is available.

## Archived evidence

- Raw report: `data/raw/github/iree/24920/report.json`
- SHA-256: `c28a5c450ea978eae33754b3af33fea3ceb8695413a6c483aedd539147ec2c4c`
