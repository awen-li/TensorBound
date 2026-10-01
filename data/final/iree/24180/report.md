# [Runtime] float64 operations produce corrupted output (buffer size mismatch)

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/24180](https://github.com/iree-org/iree/issues/24180)
- Opened: 2026-04-21T11:16:54Z
- Closed: Open at collection
- Labels: `bug 🐞`

## Reported behavior

### What happened?


Any operation on `float64` tensors produces **corrupted output**. For example, `Abs([1.0, 2.0, 3.0, 4.0])` returns `[1.0, 2.0, 0.0, 0.0]`. This happens both with and without `--iree-vm-target-extension-f64`.

`int64` (same 8-byte element width) works correctly, so this is specific to float64.

## Evidence that this is f64-specific

| Type | Operation | Input | Output | Correct? |
|------|-----------|-------|--------|----------|
| float64 | `Abs([1,2,3,4])` | f64 | `[1,2,0,0]` | **NO** |
| float64 | `Add([1,2,3], [10,20,30])` | f64 | `[960, 0, 0]` | **NO** |
| float32 | `Abs([-1,-2,-3,-4])` | f32 | `[1,2,3,4]` | YES |
| int64   | `Add([1,2,3,4], [10,20,30,40])` | i64 | `[11,22,33,44]` | YES |

The corruption pattern (first ~half of elements partially readable, rest zeroed) suggests the runtime allocates `N * sizeof(float32)` bytes instead of `N * sizeof(float64)` for the output buffer, so only the first `N/2` float64 values fit in the undersized buffer.

## Notes

- Tested with and without `--iree-vm-target-extension-f64` — both produce the same corrupted output
- `--iree-input-demote-f64-to-f32` doesn't help: the function signature still expects f64 inputs at runtime
- The MLIR correctly shows `f64` types throughout (`!torch.vtensor<[4],f64>`)
- The compiler produces no warnings or errors


### Steps to reproduce your issue


## Reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto
import subprocess
from iree.compiler import compile_str
from iree import runtime as ireert

x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
X = helper.make_tensor_value_info("X", TensorProto.DOUBLE, [4])
Y = helper.make_tensor_value_info("Y", TensorProto.DOUBLE, None)
node = helper.make_node("Abs", ["X"], ["Y"])
graph = helper.make_graph([node], "test", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model = onnx.shape_inference.infer_shapes(model)

onnx.save(model, "/tmp/f64.onnx")
r = subprocess.run(["iree-import-onnx", "/tmp/f64.onnx"], capture_output=True, text=True)
compiled = compile_str(r.stdout, input_type="onnx", target_backends=["llvm-cpu"],
                       extra_args=["--iree-vm-target-extension-f64"])
config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled)
ctx.add_vm_module(vm_module)
out = ctx.modules.module.test(x).to_host()

print(f"Input:    {x}")        # [1. 2. 3. 4.]
print(f"Expected: {x}")        # [1. 2. 3. 4.]
print(f"Got:      {out}")      # [1. 2. 0. 0.]  ← WRONG
```

### What component(s) does this issue relate to?

_No response_

### Version information

- iree-base-compiler: 3.11.0 (IREE compiler version 3.11.0rc20260316)
- iree-base-runtime: 3.11.0
- Backend: llvm-cpu
- Python: 3.11
- OS: Linux

### Additional context

_No response_

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

[Runtime] float64 operations produce corrupted output (buffer size mismatch)

## Observed implementation decision

DemoteF64ToF32Pass emits hal.tensor import/export with f64 encoding and f32 storage, while the compatibility verifier permits the size-changing bitcast.

## Physical effect

F64 demotion keeps the public ABI typed as f64 while representing the function body as f32 and bridges them with a bitcast whose source and destination have different total byte sizes.

## Observed consequence

HAL allocates 16 bytes for four f64 values that require 32 bytes, so the tail is dropped and Python observes truncated or corrupted results.

## Reported proposal or workaround

The report compares unaffected types; the follow-up traces the mismatch through post-pass and final HAL IR and specifies explicit truncation/extension plus verifier checks as the repair.

## Repair trace

The report proposes explicit conversion and verifier checks, but no implemented patch or fixed-version execution demonstrates the repair.

## Archived evidence

- Raw report: `data/raw/github/iree/24180/report.json`
- SHA-256: `004aa449bcc56ddd3a54c39a10a6b83f896d412612c83bb0756168d163a50720`
