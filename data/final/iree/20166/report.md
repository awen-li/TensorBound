# `scf.for` loop with zero-sized tensor hangs on `local-task` device but works on `local-sync`

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/20166](https://github.com/iree-org/iree/issues/20166)
- Opened: 2025-03-05T18:25:04Z
- Closed: Open at collection
- Labels: `bug 🐞`

## Reported behavior

### What happened?

This following input IR hangs during execution with the `local-task` device:
```mlir
// input.mlir
func.func @main(%iters: i64) -> tensor<?xi64> {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %init = flow.tensor.dynamic_constant dense<> : tensor<0xi64> -> tensor<?xi64>
  %result = scf.for %i = %c0 to %iters step %c1 iter_args(%iter_arg = %init) -> (tensor<?xi64>) : i64 {
    %empty = tensor.empty() : tensor<1xi64>
    %fill = linalg.fill ins(%i : i64) outs(%empty : tensor<1xi64>) -> tensor<1xi64>
    %concat = tensor.concat dim(0) %iter_arg, %fill : (tensor<?xi64>, tensor<1xi64>) -> tensor<?xi64>
    scf.yield %concat : tensor<?xi64>
  }
  return %result : tensor<?xi64>
}
```
The expected output is a tensor of size `iters` with values `result[i] = i`. Running on `local-sync` gives the expected result.

### Steps to reproduce your issue

```console
$ iree-compile  input.mlir --iree-hal-target-device=vmvx -o tmp.vmfb

# local-sync executes succesfully:
$ iree-run-module --module=tmp.vmfb  --input='10' --device=local-sync
EXEC @main
result[0]: hal.buffer_view
10xi64=0 1 2 3 4 5 6 7 8 9

# local-task hangs:
$ iree-run-module --module=tmp.vmfb  --input='10' --device=local-task
EXEC @main
```

### What component(s) does this issue relate to?

Runtime

### Version information

7782778ae3c3e37dc206831a330946e61b95a118

### Additional context

some discussion on discord here: https://discord.com/channels/689900678990135345/689906000043573354/1346912899079208991

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises iREE local-task command-buffer completion for a loop over a zero-sized tensor.

## Observed implementation decision

The task path can end with a trailing barrier that never receives a progress-producing operation.

## Physical effect

The command buffer waits indefinitely although local-sync completes the same zero-sized computation.

## Observed consequence

Execution hangs.

## Reported proposal or workaround

PR 24225 proposes handling a command buffer ending in a trailing barrier.

## Repair trace

PR 24225 was not merged, so the open report has no landed completion repair or fixed regression.

## Archived evidence

- Raw report: `data/raw/github/iree/20166/report.json`
- SHA-256: `1679f1c275462a8ab7a33817fc8537efd249208306bfdf5c219a76de86faf867`
