# host_callback_test.py is failing on multi-GPU platforms

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/6447](https://github.com/jax-ml/jax/issues/6447)
- Opened: 2021-04-15T00:20:12Z
- Closed: 2022-08-24T19:23:10Z
- Labels: `bug`, `P1 (soon)`, `NVIDIA GPU`

## Reported behavior

`~/jax$ python3 tests/host_callback_test.py` fails with:

```
[ RUN      ] CallJaxTest.test_jax_grad
2021-04-15 00:16:53.333864: E external/org_tensorflow/tensorflow/compiler/xla/status_macros.cc:56] Internal: RET_CHECK failure (external/org_tensorflow/tensorflow/compiler/xla/service/gpu/outfeed_thunk.cc:80) ShapeUtil::Equal(source_slices_[index].shape, output_shape) Mismatch between outfeed output buffer shape u32[2]{0} and outfeed source buffer shape f32[]
```
`Mismatch between outfeed output buffer shape u32[2]{0} and outfeed source buffer shape f32[]`

All test cases seem to fail this way. It's a little hard to tell because after a few test cases run and fail with this, the test will just hang until you kill it (probably because computations are blocked waiting for infeed).

I haven't dug into this at all, but for now I'm planning to disable all of host_callback_test on multi-GPU platforms in order to make our test suite runnable on multi-GPU (since right now it hangs).

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The host callback test suite runs while several GPU devices are visible.

## Observed implementation decision

The discussion attributes failures to one shared feed queue instead of device-specific queues and marks the report as a duplicate of 5577.

## Physical effect

Requests for one callback are matched to buffers from another callback or device, producing incompatible shapes and blocked feed operations.

## Observed consequence

Tests fail with an outfeed shape mismatch and later hang waiting for feed progress.

## Reported proposal or workaround

Merged PR 6448 disables the test suite on multi-GPU platforms.

## Repair trace

The merged change suppresses execution of the failing tests and does not repair queue identity or callback routing. The primary report 5577 also lacks a landed host_callback repair, so this duplicate cannot yet map to a supported repaired cause.

## Archived evidence

- Raw report: `data/raw/github/jax/6447/report.json`
- SHA-256: `83423811108a6afab0739bdc089cdc76509ec1bbea36438679aedc561000f6b0`
