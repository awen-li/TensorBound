# [XLA:CPU] In-place dynamic-update-slice fusion reads its own output when the update is a shifted slice of the operand

- Repository: `openxla/xla`
- Report: [https://github.com/openxla/xla/issues/49176](https://github.com/openxla/xla/issues/49176)
- Opened: 2026-09-21T14:57:24Z
- Closed: Open at collection
- Labels: `bug`, `stat:awaiting openxla-eng`, `CPU`, `err: Runtime`

## Reported behavior

## Summary

On CPU, `dynamic-update-slice(base, slice(base), ...)` followed by a `reshape` returns wrong values. `AlgebraicSimplifier` moves the reshape across the DUS, CPU fusion puts the shifted `slice` of the operand into the same loop fusion as the DUS, and the fusion is still emitted as an in-place DUS kernel whose input and output share one allocation. The forward copy then reads elements it has already overwritten. GPU, eager op-by-op execution, and `--xla_disable_hlo_passes=algsimp` are all correct.

## Reproducer

```python
import numpy as np, jax
from jax import lax

def f(x):
    base = x + np.int32(1)
    return lax.dynamic_update_slice(base, base[:3], (1, 0, 0)).reshape((1, 4, 1))

x = np.arange(4, dtype=np.int32).reshape(4, 1, 1)
cpu = jax.local_devices(backend="cpu")[0]
print("cpu jit:", np.asarray(jax.jit(f)(jax.device_put(x, cpu))).ravel())   # [1 1 1 1]  wrong
with jax.disable_jit():
    print("eager:  ", np.asarray(f(jax.device_put(x, cpu))).ravel())       # [1 1 2 3]  correct
```

Expected `[1, 1, 2, 3]`: `base = [1,2,3,4]` shifted down by one row. The idiomatic 2-D form `base.at[1:].set(base[:-1]).reshape((1,) + base.shape)` with `base: s32[8,16]` is also wrong on CPU (96 of 128 elements), and at length 1024 the pattern gives 1,022 wrong elements.

Controls: removing the final `reshape` is correct

## Agent Analysis

Before `algsimp`:

```text
add   = add(x, broadcast(1))                        s32[4,1,1]
slice = slice(add), slice={[0:3],[0:1],[0:1]}
dus   = dynamic-update-slice(add, slice, 1, 0, 0)
ROOT reshape = reshape(dus)                         s32[1,4,1]
```

After optimization the CPU module is:

```text
%fused_computation (param_0.2: s32[4,1,1]) -> s32[1,4,1] {
  %bitcast.3 = s32[1,4,1] bitcast(%param_0.2)
  %slice.0   = s32[3,1,1] slice(%param_0.2), slice={[0:3], [0:1], [0:1]}
  %bitcast.2 = s32[1,3,1] bitcast(%slice.0)
  ROOT %dynamic_update_slice.2 = dynamic-update-slice(%bitcast.3, %bitcast.2, 0, 1, 0)
}
ENTRY {
  %broadcast_add_fusion = fusion(%x.1)
  %copy.1 = copy(%broadcast_add_fusion)
  ROOT %bitcast_dynamic-update-slice_fusion = fusion(%copy.1), calls=%fused_computation
}
```

Buffer assignment puts `%copy.1` and the fusion output in the same allocation (`allocation 0: size 16 … value: copy.1 … value: bitcast_dynamic-update-slice_fusion`), and the CPU fusion emitter takes the in-place DUS path (`fusion_emitter.cc` → `CanEmitFusedDynamicUpdateSlice` in `xla/codegen/ir_emission_utils.cc`). That check is meant to reject fusions that read the updated parameter at other index positions, but this fusion passes it even though `%slice.0` reads `%param_0.2` at rows 0–2 while the kernel writes rows 1–3 of the same buffer. Element 1 is written from element 0, then element 2 is read from the already-overwritten element 1, and so on, giving `[1,1,1,1]`.

On GPU the `slice` is placed in a separate fusion (`wrapped_slice`) whose result lives in its own allocation before the DUS fusion runs, so the result is correct. With `--xla_disable_hlo_passes=algsimp` the CPU pipeline also materializes the update separately and is correct.

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises xLA CPU fusion of an in-place dynamic-update-slice whose update reads a shifted slice of the operand.

## Observed implementation decision

Fusion aliases output with operand without preserving the update's pre-write source range.

## Physical effect

The update reads values already overwritten by earlier positions.

## Observed consequence

The fused CPU result is silently wrong.

## Reported proposal or workaround

The report supplies an overlap reproducer.

## Repair trace

No linked source repair, landed patch, or fixed regression is available.

## Archived evidence

- Raw report: `data/raw/github/xla/49176/report.json`
- SHA-256: `807bbd81f1939fee3ceac17436240ca2636fd3216327e48b3b5935a4fb2a71b4`
