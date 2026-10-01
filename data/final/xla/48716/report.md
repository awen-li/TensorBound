# [XLA:CPU] scatter with an empty index vector applies only the last update

- Repository: `openxla/xla`
- Report: [https://github.com/openxla/xla/issues/48716](https://github.com/openxla/xla/issues/48716)
- Opened: 2026-09-11T12:46:45Z
- Closed: 2026-09-15T12:35:31Z
- Labels: `bug`, `CPU`

## Reported behavior

A `scatter` whose index vector is empty (`scatter_dims_to_operand_dims = []`, so every update
targets the same, loop-invariant position) and whose update window holds a single element applies
**only the last update** instead of combining all of them. Duplicate-target `scatter_add` / `scatter_mul`
are fully specified for a commutative combiner, so this is a wrong result, not an ordering choice.

### Reproducer

```python
import numpy as np, jax
from jax import lax
from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc
# operand [10]; 3 updates [1],[2],[4] addressed by EMPTY index vectors (indices shape (3,0)); window = 1 element
# -> every update targets operand[0]; spec (operation_semantics#scatter): 10+1+2+4 = 17
operand = np.array([10], np.int32); indices = np.zeros((3, 0), np.int32); updates = np.array([[1], [2], [4]], np.int32)
dn = lax.ScatterDimensionNumbers(update_window_dims=(1,), inserted_window_dims=(), scatter_dims_to_operand_dims=())
for mode in ('promise_in_bounds', 'clip', 'fill'):
    got = jax.jit(lambda o, i, u: lax.scatter_add(o, i, u, dn, mode=mode))(operand, indices, updates)
    print(f'lax.scatter_add mode={mode}: expected [17] actual {np.asarray(got).tolist()}')
print('lax.scatter_mul: expected [210] actual', np.asarray(lax.scatter_mul(np.array([2], np.int32), indices, np.array([[3], [5], [7]], np.int32), dn)).tolist())
# same scatter as raw StableHLO compiled through the PJRT CPU client (no JAX lowering involved)
text = '''module @m {
  func.func public @main(%arg0: tensor<1xi32>, %arg1: tensor<3x0xi32>, %arg2: tensor<3x1xi32>) -> tensor<1xi32> {
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [], scatter_dims_to_operand_dims = [], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%a: tensor<i32>, %b: tensor<i32>):
      %1 = stablehlo.add %a, %b : tensor<i32>
      stablehlo.return %1 : tensor<i32>
    }) : (tensor<1xi32>, tensor<3x0xi32>, tensor<3x1xi32>) -> tensor<1xi32>
    return %0 : tensor<1xi32>
  }
}'''
be = xb.get_backend()
exe = be.compile_and_load(text, be.devices(), xc.CompileOptions())
got = exe.execute_sharded([jax.device_put(a) for a in (operand, indices, updates)]).disassemble_into_single_device_arrays()[0][0]
print('raw StableHLO via PJRT CPU: expected [17] actual', np.asarray(got).tolist())
# controls (both correct): a 2-element window, or an explicit index vector [0]
print('control 2-elem window: expected [17, 90] actual', np.asarray(lax.scatter_add(np.array([10, 20], np.int32), indices, np.array([[1, 10], [2, 20], [4, 40]], np.int32), dn)).tolist())
dn1 = lax.ScatterDimensionNumbers(update_window_dims=(1,), inserted_window_dims=(), scatter_dims_to_operand_dims=(0,))
print('control explicit index [0]: expected [17] actual', np.asarray(lax.scatter_add(operand, np.zeros((3, 1), np.int32), updates, dn1)).tolist())
```

Output on my machine:

```
lax.scatter_add mode=promise_in_bounds: expected [17] actual [14]
lax.scatter_add mode=clip:                expected [17] actual [14]
lax.scatter_add mode=fill:                expected [17] actual [14]
lax.scatter_mul:                          expected [210] actual [14]
raw StableHLO via PJRT CPU:               expected [17] actual [14]
control 2-elem window:                    expected [17, 90] actual [17, 90]
control explicit index [0]:               expected [17] actual [17]
```

### Analysis

`14` is `10 + 4`, i.e. the initial operand plus the final update alone; `scatter_mul` likewise gives
`2 * 7` rather than the product of all updates.

The two controls isolate it: the same scatter with a two-element update window, and the same scatter
written with an explicit index operand, both combine every update correctly. Only the
empty-index-vector plus single-element-window shape is wrong.

This looks like the CPU scatter fusion emitter
(`xla/backends/cpu/codegen/emitters/cpu_scatter_emitter.cc`): `EmitScatterComputation` reads the
current operand value through the immutable fusion-argument tensor rather than the loop-carried
output tensor. When the target index is loop-invariant the `tensor.extract` is loop-invariant too and
gets hoisted out of the update loop, so every iteration combines against the *original* operand value
and the last write wins.

### Environment

- `jax` 0.11.1, `jaxlib` 0.11.1 (pip), CPU backend, `JAX_PLATFORMS=cpu`
- Linux x86-64 (AVX-512), Python 3.12


### Notes

Found by a differential fuzzer comparing XLA:CPU against a NumPy oracle written from the operation semantics, then minimised. Each case is also reproduced from raw StableHLO through the PJRT CPU client, so it is not a JAX lowering artefact. I am filing this as an issue rather than a pull request, per the repository's contribution guidelines: this report was prepared with the help of an AI coding assistant (Claude), and I reproduced every result above myself on the environment listed.

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

An empty-index-vector scatter repeatedly updates one loop-invariant destination.

## Observed implementation decision

EmitScatterComputation reads the entry operand instead of the loop-carried output tensor.

## Physical effect

Loop-invariant code motion can hoist the initial operand read, so each update combines with the initial value.

## Observed consequence

Only the last update affects the output.

## Reported proposal or workaround

PR 48781 changed the read to the loop-carried output and added tests, but PR 48853 reverted that commit later the same day.

## Repair trace

The archived correcting patch was reverted, and the evidence bundle contains no durable replacement or fixed-version regression; a completed repair relationship is therefore not established.

## Archived evidence

- Raw report: `data/raw/github/xla/48716/report.json`
- SHA-256: `491046798dfd48d2b28bd2f03808701ca2b2bc8889b802cd2b157d411247a0d8`
