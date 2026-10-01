# Segfault when calling module using hal.tensor.export

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/17087](https://github.com/iree-org/iree/issues/17087)
- Opened: 2024-04-18T12:10:33Z
- Closed: open at collection time
- Labels: `bug 🐞`, `runtime`

## Reported behavior

### What happened?

Calling functions that use the `{iree.abi.output = 1 : index}` or calls `hal.tensor.export` manually sometimes segfaults. (Usually involving invalid values in the session or session.context I think, but not sure about that). This happens if I call it from rust through eerie and if I call it in python.

### Steps to reproduce your issue

The following python code segfaults when calling the compiled function (at least for me reliably)

```python
import iree.runtime as ir_runtime
import iree.compiler as ir_compiler
import numpy as np

segfault = True

if segfault:
    module_str = """
    module @jit_jax_funcified_fgraph {
      func.func public @logp(%arg0: tensor<2xf32>, %arg1: !hal.buffer_view) -> (tensor<2xf32>, !hal.buffer_view) {
        %0 = stablehlo.add %arg0, %arg0 : tensor<2xf32>
        %1 = hal.tensor.export %0 into(%arg1 : !hal.buffer_view) : tensor<2xf32> -> !hal.buffer_view
        return %0, %1 : tensor<2xf32>, !hal.buffer_view
      }
    }
    """

else:
    module_str = """
    module @jit_jax_funcified_fgraph {
      func.func public @logp(%arg0: tensor<2xf32>, %arg1: !hal.buffer_view) -> (tensor<2xf32>, !hal.buffer_view) {
        %0 = stablehlo.add %arg0, %arg0 : tensor<2xf32>
        return %0, %arg1 : tensor<2xf32>, !hal.buffer_view
      }
    }
    """

compiled = ir_compiler.compile_str(
    module_str,
    target_backends=["llvm-cpu"],
)

config = ir_runtime.Config("local-task")
ctx = ir_runtime.SystemContext(config=config)
vm_module = ir_runtime.VmModule.copy_buffer(ctx.instance, compiled)
ctx.add_vm_module(vm_module)

device = ir_runtime.get_device("local-task")

arg0 = np.ones(2, dtype=np.float32)
arg1 = np.zeros(2, dtype=np.float32)

buffer0 = ir_runtime.asdevicearray(device, arg0)
buffer1 = ir_runtime.asdevicearray(device, arg1)

f = ctx.modules["jit_jax_funcified_fgraph"]["logp"]
results = f(buffer0, buffer1)
```

### What component(s) does this issue relate to?

_No response_

### Version information

iree-compiler==20240410.859
iree-runtime==20240410.859


### Additional context

CC @benvanik (This is pretty much a follow up from the discussion on discord...)

## Trigger

A computed tensor is returned normally and also exported into a caller-provided output buffer.

## Relevant entities

computed result; stream.async.update; external target resource; tied output buffer

## Implemented relation

The update connects the computed value to an externally supplied target that may be tied to another value.

## Pre-state

The external target remains the required destination for one returned result.

## Binding transition

The folder replaces a whole-target stream.async.update with its update operand.

## Post-state

The import/update of the external target disappears, and the computed resource is exported in its place.

## Validity obligation

An update may be folded only when replacing the target cannot discard an externally observable alias or tied binding.

## Invalid decision

The folder used type/extent equality as sufficient evidence for folding without proving value semantics.

## Physical effect

The caller-provided output buffer is not updated through the required resource path.

## Observed consequence

The minimal invocation segfaults before the fix; the reporter confirms it works with the fix.

## Repair site

compiler/src/iree/compiler/Dialect/Stream/IR/StreamOpFolders.cpp

## Repair

The fold now requires hasValueSemantics(target), excluding external, tied, and subview resources.

## Why the repair works

The update is preserved whenever the target can carry an externally observable binding.

## Causal basis

The source guard and regression cases directly encode the distinction between value-semantic and externally tied targets.

## Analysis implication

A folding check needs target provenance and alias/tie semantics in addition to matching type and range.

## Fix patches

- [`fixes/iree-17087-fix.patch`](fixes/iree-17087-fix.patch) — https://github.com/iree-org/iree/pull/17135
