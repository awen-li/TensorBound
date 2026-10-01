# Unsigned integer outputs become signed

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/9282](https://github.com/iree-org/iree/issues/9282)
- Opened: 2022-06-02T14:01:28Z
- Closed: 2023-05-11T18:31:35Z
- Labels: `bug 🐞`, `help wanted`, `integrations/stablehlo`, `collab/nvidia`

## Reported behavior

Repro:
```
import platform
import iree.compiler
import iree.runtime
import numpy as np

CODE = """
#loc0 = loc(unknown)
module @jit_prim_fun.0 {
  func.func public @main(%arg0: tensor<5x7xui32> loc(unknown)) -> tensor<5x7xui32> {
    %0 = "mhlo.sort"(%arg0) ({
    ^bb0(%arg1: tensor<ui32> loc(unknown), %arg2: tensor<ui32> loc(unknown)):
      %1 = "mhlo.compare"(%arg1, %arg2) {compare_type = #mhlo<"comparison_type UNSIGNED">, comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<ui32>, tensor<ui32>) -> tensor<i1> loc(#loc2)
      "mhlo.return"(%1) : (tensor<i1>) -> () loc(#loc1)
    }) {dimension = 1 : i64, is_stable = false} : (tensor<5x7xui32>) -> tensor<5x7xui32> loc(#loc1)
    return %0 : tensor<5x7xui32> loc(#loc0)
  } loc(#loc0)
} loc(#loc0)
#loc1 = loc("jit(sort)/jit(main)/sort[dimension=1 is_stable=False num_keys=1]"("/Users/phawkins/p/jax/tests/lax_test.py":2039:1))
#loc2 = loc("jit(sort)/jit(main)/lt"("/Users/phawkins/p/jax/tests/lax_test.py":2039:1))
"""

# Work around https://github.com/google/iree/issues/9028
extra_args = []
if platform.system() == "Darwin" and platform.machine() == "arm64":
  extra_args += ["--iree-llvm-target-triple=arm64-apple-darwin21.5.0"]

iree_config = iree.runtime.system_api.Config("dylib")
iree_binary = iree.compiler.compile_str(
    CODE, target_backends=["dylib"], input_type="mhlo", extra_args=extra_args)
vm_module = iree.runtime.VmModule.from_flatbuffer(iree_binary)
module_object = iree.runtime.load_vm_module(vm_module, iree_config)

out = module_object["main"](np.ones((5, 7), dtype=np.uint32))
assert out.dtype == np.uint32, out.dtype
```

Get:
```
Traceback (most recent call last):
  File "/Users/phawkins/p/jax/t.py", line 34, in <module>
    assert out.dtype == np.uint32, out.dtype
AssertionError: int32
```

The MHLO has a ui32 output, so I expect a uint32 result from IREE.

## Trigger

An MHLO function accepts and returns a tensor<ui32>, and Python consumes the compiled result.

## Relevant entities

MHLO unsigned tensor; public function signature; signless Linalg tensor; runtime element type; NumPy dtype

## Implemented relation

The public function type carries logical unsignedness across signless internal arithmetic so the runtime result is decoded with the original dtype.

## Pre-state

The MHLO argument, comparison semantics, and result are unsigned 32-bit values.

## Binding transition

MHLOToLinalgOnTensors converts integer types to signless types, including the public function result.

## Post-state

The computation retains unsigned comparison behavior, but the exported result type is signless and Python maps it to a signed int32 array.

## Validity obligation

Internal sign erasure does not erase signedness that remains observable at an external function boundary.

## Invalid decision

The conversion target required signless integer types for func.func inputs, results, and returns instead of preserving the public unsigned boundary.

## Physical effect

The exported element type identifies the result as generic INT_32, which the Python binding decodes as signed int32.

## Observed consequence

The reported sort computes correct bits but returns a signed array unless the result is manually bitcast to uint32.

## Repair site

compiler/src/iree/compiler/InputConversion/MHLO/MHLOToLinalgOnTensors.cpp

## Repair

PR 13090 keeps unsigned types legal at public inputs and outputs and inserts explicit tensor.bitcast operations around the signless internal representation; it adds unsigned-boundary conversion tests.

## Why the repair works

The interior remains signless for operation-defined arithmetic while the function boundary again exposes ui32 to later runtime layers.

## Causal basis

The issue shows correct bits under bitcast, and the linked patch changes the exact boundary-type conversion while preserving signless internal tensors.

## Analysis implication

A representation check needs the source dtype, internal sign convention, public ABI type, operation-level signedness, and host dtype decoder.

## Fix patches

- [`fixes/iree-9282-pr13090.patch`](fixes/iree-9282-pr13090.patch) — https://github.com/iree-org/iree/pull/13090
