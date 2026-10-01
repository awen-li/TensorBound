# Multi-device 2D convolution numerical imprecision

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/18283](https://github.com/iree-org/iree/issues/18283)
- Opened: 2024-08-19T15:19:51Z
- Closed: open at collection time
- Labels: `bug 🐞`, `compiler/dialects`

## Reported behavior

I encountered some numerical imprecisions when executing on 2 local-task (most likely it is actually compiler lowering bug) devices.
I have narrowed it down to a 2D convolution without padding or bias.
Interestingly, with padding the result is correct.

[conv2d-multi-device-numerical-imprecision.zip (updated)](https://github.com/user-attachments/files/16723207/conv2d-multi-device-numerical-imprecision.zip) contains a reproducer.

My assumption is that the lowering of the different preparation of the main conv dispatch is wrong. When using padding we first 0-fill a tensor and then insert the input according to the padding size. Without padding we use the tensor directly.

## Trigger

Turbine emits a statically known convolution input as a dynamic-shaped flow.tensor.transfer result that is cast back to a static shape inside the dispatch.

## Relevant entities

logical convolution input; flow.tensor.transfer; dynamic tensor type; static tensor.cast; memref.subview rank reduction; vector loads

## Implemented relation

Tensor shape and strided memref metadata identify which source dimension a rank-reduced convolution subview drops.

## Pre-state

The transferred tensor's concrete dimensions are statically known at export time.

## Binding transition

The compiler lowers a dynamic-shaped transfer plus static cast through memref.subview folding.

## Post-state

With ambiguous dynamic strides, rank reduction drops dimension zero instead of the unit spatial dimension and vector loads advance the wrong index.

## Validity obligation

Rank reduction preserves the source-to-result dimension mapping used by every subsequent load.

## Invalid decision

The FoldMemRefAlias path selected a dropped dimension from ambiguous dynamic stride information.

## Physical effect

Convolution vector loads advance a spatial coordinate where the channel coordinate should advance.

## Observed consequence

The convolution returns incorrect numerical results on the isolated CPU dispatch; the static-shaped variant is correct.

## Repair site

iree/turbine/ops/iree.py::transfer_to_logical_device.select (scoped upstream workaround)

## Repair

Turbine PR 205 specializes all input and output dimensions when generating flow.tensor.transfer, preventing this path from introducing the ambiguous dynamic-shaped subview.

## Why the repair works

Static dimensions make the dropped-dimension mapping unambiguous for the reported pipeline, so lowering retains the channel and spatial coordinate correspondence.

## Causal basis

The issue isolates wrong dimension advancement after ambiguous rank reduction, and the linked Turbine patch is explicitly described as solving the report by removing the dynamic representation that exposes it.

## Analysis implication

A shape-lowering check needs provenance of static dimensions, transfer type refinement, cast movement, subview rank-reduction mapping, and final load indices.

## Fix patches

- [`fixes/turbine-18283-pr205.patch`](fixes/turbine-18283-pr205.patch) — https://github.com/iree-org/iree-turbine/pull/205
