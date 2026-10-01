# Sequence return types failing in the python bindings.

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/5320](https://github.com/iree-org/iree/issues/5320)
- Opened: 2021-04-05T21:37:08Z
- Closed: 2021-04-09T00:37:05Z
- Labels: `bug 🐞`, `bindings/python`

## Reported behavior

MLIR modules bound with the Python runtime bindings currently provide incorrect results when returning multiple values (e.g. `return a, b`) and when returning sequences (e.g. `return (a,)` or `return (a, b)`). This is documented in [this colab](https://colab.research.google.com/drive/1n2d2poVTcak4dgKRiNE62qzb8pR5u1az) and below:

```python
class PythonTypesModule(tf.Module):

  @tf.function(input_signature=[])
  def output_tuple_len_0(self):
    return tuple()

  @tf.function(input_signature=[])
  def output_tuple_len_1(self):
    return (1.,)  # Need to explicitly wrap.

  @tf.function(input_signature=[])
  def output_tuple_len_2(self):
    return 1., 2.

  @tf.function(input_signature=[])
  def output_tuple_len_3(self):
    return 1., 2., 3.
```

```mlir
module  {
  func @output_tuple_len_0() attributes {
        iree.module.export,
        iree.reflection = {abi = "sip", abiv = 1 : i32, sip = "I1!R1!"}
    } {
    return
  }
  func @output_tuple_len_1() -> tensor<f32> attributes {
        iree.module.export,
        iree.reflection = {abi = "sip", abiv = 1 : i32, sip = "I1!R8!S5!k0_0"}
    } {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    return %0 : tensor<f32>
  }
  func @output_tuple_len_2() -> (tensor<f32>, tensor<f32>) attributes {
        iree.module.export,
        iree.reflection = {abi = "sip", abiv = 1 : i32, sip = "I1!R12!S9!k0_0k1_1"}
    } {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<2.000000e+00> : tensor<f32>
    return %0, %1 : tensor<f32>, tensor<f32>
  }
  func @output_tuple_len_3() -> (tensor<f32>, tensor<f32>, tensor<f32>) attributes {
        iree.module.export,
        iree.reflection = {abi = "sip", abiv = 1 : i32, sip = "I1!R17!S13!k0_0k1_1k2_2"}
    } {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<2.000000e+00> : tensor<f32>
    %2 = mhlo.constant dense<3.000000e+00> : tensor<f32>
    return %0, %1, %2 : tensor<f32>, tensor<f32>, tensor<f32>
  }
}
```

Expected vs actual results:

```python
module.output_tuple_len_1()
expected: [array([1.])]
actual:   [None, array([1.])]

module.output_tuple_len_2()
expected: [array([1.]), array([2.])]
actual:   [None, array([2.])]

module.output_tuple_len_3()
expected: [array([1.]), array([2.]), array([3.])]
actual:   [None, array([2.]), None, array([3.])]
```

IREE itself produces the correct values via `iree-run-mlir`:

```
iree/tools/iree-run-mlir \
  -export-all \
  -iree-hal-target-backends=dylib-llvm-aot \
  sequences.mlir

EXEC @output_tuple_len_1
I iree/iree/tools/utils/vm_util.cc:201] result[0]: Buffer<float32[2]>
1xf32=1

EXEC @output_tuple_len_2
I iree/iree/tools/utils/vm_util.cc:201] result[0]: Buffer<float32[2]>
1xf32=1
I iree/iree/tools/utils/vm_util.cc:201] result[1]: Buffer<float32[2]>
1xf32=2

EXEC @output_tuple_len_3
I iree/iree/tools/utils/vm_util.cc:201] result[0]: Buffer<float32[2]>
1xf32=1
I iree/iree/tools/utils/vm_util.cc:201] result[1]: Buffer<float32[2]>
1xf32=2
I iree/iree/tools/utils/vm_util.cc:201] result[2]: Buffer<float32[2]>
1xf32=3
```

Given this I would guess that there is some error in how the python bindings are interpreting the SIP or with the SIP itself.

## Trigger

An exported function returns a tuple containing one or more scalar tensors.

## Relevant entities

SIP result descriptors; VM output list; result ordinal; Python sequence; tensor values

## Implemented relation

SIP result ordinal k selects VM output k and occupies Python sequence position k.

## Pre-state

The VM returns each tensor in consecutive output positions and the SIP metadata describes those positions.

## Binding transition

The old function ABI reads the VM list with an ordinal advanced by one.

## Post-state

Some Python result positions receive None or a later tensor while the first tensor is skipped.

## Validity obligation

ABI decoding uses the same zero-based result identity in SIP metadata, VM storage, and the Python sequence.

## Invalid decision

The Python function ABI applied an off-by-one offset when retrieving result values.

## Physical effect

Returned tensor handles bind to the wrong Python positions or no position.

## Observed consequence

Tuples such as (1,2,3) are returned as sequences containing None and missing values.

## Repair site

bindings/python/iree/runtime/function_abi.cc with system_api.py sequence conversion

## Repair

PR 5347 corrects result retrieval to outputs->Get(ordinal), normalizes ABI sequences to tuples, and adds tuple lengths zero through three plus nested pytree tests.

## Why the repair works

The decoder now uses one ordinal across the VM list and Python structure, and preserves the public tuple representation.

## Causal basis

The PR identifies an ABI off-by-one error, changes the output-list access by one index, and adds the report's tuple-arity cases.

## Analysis implication

Boundary-ABI checking needs descriptor ordinals, VM list indices, arity, and host-language container structure.

## Fix patches

- [`fixes/iree-5320-pr5347.patch`](fixes/iree-5320-pr5347.patch) — https://github.com/iree-org/iree/pull/5347
