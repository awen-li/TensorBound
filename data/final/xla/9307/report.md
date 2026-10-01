# [BUG] Returning a matrix crashes the HLO runner

- Repository: `openxla/xla`
- Report: [https://github.com/openxla/xla/issues/9307](https://github.com/openxla/xla/issues/9307)
- Opened: 2024-02-08T09:16:17Z
- Closed: 2024-12-11T07:11:07Z
- Labels: `bug`, `CPU`

## Reported behavior

Following up on https://github.com/openxla/xla/issues/9164#issuecomment-1932144981
Running the following program
```
ENTRY main {
  ROOT tmp_2 = c128[3,3]{0,1} constant({ { (-3.4683985041370184, 0.2818515524340488), (-2.0531999581789173, 0.76532647668268594), (5.269823805569831, 3.5613196166646359) }, { (-2.559036532612764, -0.3698252257987209), (-0.25420681272147494, -1.8022993744788929), (1.667097371428415, 2.7456472301998929) }, { (1.3575242846983384, -4.1063133419096447), (1.4836047923312523, -1.9767224256595632), (-6.86256834124183, -0.81498779940376331) } })
}
```
e.g., with
```
bazel run --config=dbg -- //xla/tools:run_hlo_module --input_format=hlo --run_test_hlo_passes=false --run_reference_hlo_passes=false --platform=cpu --random_init_input_literals=true /home/evgeny/general_code/xla/fn_hlo_tmp.txt
```
crashes with
```
FAILED_PRECONDITION: Source allocation on device not large enough for data transfer: 0 < 144
	Failed to execute on Host
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The HLO entry computation returns a c128 matrix constant directly as its root.

## Observed implementation decision

The execution path exposes a zero-byte source allocation for a 144-byte logical result.

## Physical effect

The host transfer attempts to read the matrix from an allocation smaller than the requested literal.

## Observed consequence

The HLO runner fails with a source-allocation-size precondition error.

## Reported proposal or workaround

Adding an explicit HLO copy materializes the constant into transferable storage.

## Repair trace

The issue provides only the copy workaround and was closed without a linked fix. No source-localized allocation/publication repair or fixed regression explains how direct constant roots became transferable.

## Archived evidence

- Raw report: `data/raw/github/xla/9307/report.json`
- SHA-256: `7b5d3a0e49166eb14c81a10473eab744440c73f7741e20827bd2993ad8361e80`
