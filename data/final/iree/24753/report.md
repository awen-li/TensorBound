# `hal.command_buffer.dispatch` OUT_OF_RANGE on 3.12.0rc but works on 3.11.0

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/24753](https://github.com/iree-org/iree/issues/24753)
- Opened: 2026-07-23T21:13:12Z
- Closed: Open at collection
- Labels: `bug 🐞`

## Reported behavior

### What happened?

The following short JAX snippet generates StableHLO output that compiles successfully with `--iree-hal-indirect-command-buffers=false`, but when ran it fails on 3.12rc but succeeds on 3.11.

The error:
```
EXEC @main
c/runtime/src/iree/hal/command_buffer_validation.c:185: OUT_OF_RANGE; indirect buffer reference slot 0 is out range of the declared binding capacity of the command buffer 0; binding[0]; while invoking native function hal.command_buffer.dispatch; while calling import;
[ 0] bytecode jit_entry.__init:3170 -; invoking function 'main'; `sync func @main(%input0: tensor<7xi32>, %input1: tensor<7xi32>, %input2: tensor<7xi32>, %input3: tensor<i32>, %input4: tensor<i32>, %input5: tensor<i1>, %input6: tensor<i32>) -> (%output0: tensor<7xi32>, %output1: tensor<7xi32>, %output2: tensor<7xi32>, %output3: tensor<i32>, %output4: tensor<i32>, %output5: tensor<i1>)`
```

If you invert the failure mode selector (see standalone.py) it fails with:
```
EXEC @main
c/runtime/src/iree/vm/ref.h:223: INVALID_ARGUMENT; ref is null; while invoking native function hal.buffer_view.create; while calling import;
[ 0] bytecode jit_entry.__init:4362 -; invoking function 'main'; `sync func @main(%input0: tensor<7xi32>, %input1: tensor<7xi32>, %input2: tensor<7xi32>, %input3: tensor<i32>, %input4: tensor<i32>, %input5: tensor<i1>, %input6: tensor<i32>) -> (%output0: tensor<7xi32>, %output1: tensor<7xi32>, %output2: tensor<7xi32>, %output3: tensor<i32>, %output4: tensor<i32>, %output5: tensor<i1>)`
```

### Steps to reproduce the issue

Use a CPU jax env (i.e. make sure default device is CPU).

First install these:
```bash
pip install jax==0.5.3 absl-py
```

Then either IREE 3.11:
```bash
pip install iree-base-compiler==3.11.0 iree-base-runtime==3.11.0
```

Or IREE nightly (currently 3.12rc):
```bash
python -m pip install \
  --find-links https://iree.dev/pip-release-links.html \
  --upgrade --pre iree-base-compiler &&
python -m pip install \
  --find-links https://iree.dev/pip-release-links.html \
  --upgrade --pre iree-base-runtime
```

Then generate the module in the current directory with:
```bash
python standalone.py
```

Compile and run with:
```bash
iree-compile module.mlir \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=x86-64 \
    --iree-opt-const-expr-hoisting=false \
    --iree-hal-indirect-command-buffers=false \
    -o module.vmfb &&
iree-run-module \
    --device=local-sync \
    --module=module.vmfb \
    --function=main \
    --input=7xi32=0 \
    --input=7xi32=0 \
    --input=7xi32=0 \
    --input=i32=0 \
    --input=i32=0 \
    --input=i1=01 \
    --input=i32=2
```
Failure mode selector is the second to last input.

[standalone.py](https://github.com/user-attachments/files/30344968/standalone.py)

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises iREE VM branch-remap ownership for references passed to hal.command_buffer.dispatch under Python 3.12.

## Observed implementation decision

A branch remap applies an incorrect retain/release treatment to a reference consumed by the dispatch path.

## Physical effect

The live reference reaching the HAL call is missing or invalid.

## Observed consequence

Execution fails with OUT_OF_RANGE on Python 3.12 but succeeds on 3.11.

## Reported proposal or workaround

PR 24781 proposes fixing reference ownership in branch remaps.

## Repair trace

PR 24781 was not merged, leaving no landed ownership repair or fixed regression.

## Archived evidence

- Raw report: `data/raw/github/iree/24753/report.json`
- SHA-256: `eb47579ba20700102ac8a8b410fac0860ba72a57f14fc9eeae26d95b3d05b47f`
