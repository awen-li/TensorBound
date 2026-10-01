# Python bindings silently accept f64 scalars and return garbage when the module accepts f32

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/12967](https://github.com/iree-org/iree/issues/12967)
- Opened: 2023-04-07T02:26:54Z
- Closed: Open at collection
- Labels: `bug 🐞`

## Reported behavior

### What happened?

I have a module function that accepts a f32 scalar.
When I call it with
```python
my_module.f(1.0)
```
I get garbage.

Same thing happens if I call it with `int`.
```python
my_module.f(1)
```
It should either raise an error or do the correct conversion.





### Steps to reproduce your issue

1. Compile this MLIR
```mlir
func.func @f(%arg0: f32) -> f32 {
  %0 = math.exp2 %arg0 : f32
  return %0 : f32
}
```
with
```
iree-compile --iree-hal-target-backends=llvm-cpu -o f.vmfb f.mlir
```

2. Load into the module
```python
import iree.runtime
vm_module = iree.runtime.load_vm_flatbuffer_file("f.vmfb", backend="local-task")
```

3. Run the function
```python
vm_module.f(2.0)
```

This returns `1.0` when it should return `4.0`.

### What component(s) does this issue relate to?

Python

### Version information

On top of d1ae167ad685e068f55a736e2e0861e324175313.

### Additional context

_No response_

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises iREE Python scalar-to-VM argument conversion when a function expects f32 but receives a Python f64 value.

## Observed implementation decision

The binding accepts a scalar whose type does not match the VM signature and transfers bits under an incompatible width/type contract.

## Physical effect

The callee reads a value different from the supplied scalar.

## Observed consequence

The invocation returns garbage instead of rejecting the argument.

## Reported proposal or workaround

Discussion proposes validating or converting scalar types at the binding boundary.

## Repair trace

The issue remains open with no linked implementation change or fixed regression.

## Archived evidence

- Raw report: `data/raw/github/iree/12967/report.json`
- SHA-256: `39ccdeecdacefd44c68a7c21450a671be126e755d0a8d356cefbc5bd4248abb0`
