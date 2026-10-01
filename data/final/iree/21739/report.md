# Getting non-deterministic inf, -inf, and nan's with optimized Wave kernel

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/21739](https://github.com/iree-org/iree/issues/21739)
- Opened: 2025-08-20T14:23:06Z
- Closed: Open at collection
- Labels: `bug 🐞`

## Reported behavior

### What happened?

When I add Wave optimizations to the Wave kernel and use `util.global.load` to load the arguments and run it with iree-run-module, I get inf, -inf, and nan's from the Wave gemm. However, passing in the arguments into the function using numpy arrays works fine. Reverting the Wave kernel back to the slower one with `util.global.load` also works fine.

### Steps to reproduce your issue

1. Download the inputs:
```
azcopy copy 'https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/llm-dev/llama3_405b/fp4/inputs/wave_nan_inputs' '.' --recursive
```
2. Download global_load_fast_kernel.mlir:
```
wget https://gist.githubusercontent.com/aviator19941/c30232ae586f76c3c2cabeda8e4068b9/raw/284cbc0aa3cd3eb9ee4daddba5630e83955cf4f2/global_load_fast_kernel.mlir
```
3. Download global_load_slow_kernel.mlir:
```
wget https://gist.githubusercontent.com/aviator19941/b8950e531a616cebc535998a07829023/raw/cba25bc31db7f462b86539c5e9e505be21d03220/global_load_slow_kernel.mlir
```
4. Download args_fast_kernel.mlir:
```
wget https://gist.githubusercontent.com/aviator19941/c3689ac470f33a2d3f1ec31924060e90/raw/06aafb9017c6f5c78822af4dd55d165b13b047ac/args_fast_kernel.mlir
```
5. Download compile.sh: wget https://gist.githubusercontent.com/aviator19941/d4b73813e638ffc7d214a88260344e64/raw/9c6f16c757797cb56d9a60c0e585e2f3fd02116e/compile.sh
6. Download run_prefill.sh: wget https://gist.githubusercontent.com/aviator19941/f376781fc123625b5be3c502858bda45/raw/01658b206500fe5ab74628dbad42b965fce451e2/run_prefill.sh
7. Run compile.sh and run_prefill.sh

### What component(s) does this issue relate to?

_No response_

### Version information

3.7.0rc20250818

### Additional context

_No response_

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises iREE Wave-generated buffer loads for tensors imported from parameter archives with nonzero physical offsets.

## Observed implementation decision

The optimized kernel uses stream.binding.subspan from offset zero rather than extracting the actual strided metadata offset.

## Physical effect

Loads address unrelated bytes from the imported allocation.

## Observed consequence

Optimized execution produces nondeterministic infinities and NaNs.

## Reported proposal or workaround

The reporter suggests memref.extract_strided_metadata to carry the actual offset.

## Repair trace

No linked patch, maintainer adjudication, or fixed regression establishes the proposed repair.

## Archived evidence

- Raw report: `data/raw/github/iree/21739/report.json`
- SHA-256: `268f66ad6f417df0b8bc80c16c089057431d98b197496f782627a587e5d45f2f`
