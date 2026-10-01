# torch.compile peak-memory reordering changes native_dropout RNG order

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/192635](https://github.com/pytorch/pytorch/issues/192635)
- Opened: 2026-08-09T12:06:37Z
- Closed: Open at collection
- Labels: `triaged`, `module: random`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

## Summary

When both of the following Inductor options are enabled:

```python
torch._inductor.config.fallback_random = True
torch._inductor.config.reorder_for_peak_memory = True
```

`torch.compile` may reorder independent `aten.native_dropout` calls. The fallback operator consumes the global RNG state, but the scheduler does not see this hidden state dependency. Consequently, the same dropout calls receive different random masks in eager and compiled execution even when `torch.manual_seed()` is reset to the same value immediately before each run.

In the reproducer below, eager executes the large encoder branch first and the small decoder branch last. Inductor's peak-memory scheduling moves the decoder branch first. This changes both the output and gradients.

## Environment

```text
PyTorch: 2.12.0+cpu
Git revision: 7661cd9c6b841b62b7f411aa52ec51f05457263b
Python: 3.10
Device: CPU
```

## Minimal reproducer

```python
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile


torch._inductor.config.fallback_random = True
torch._inductor.config.reorder_for_peak_memory = True


def model(encoder, decoder):
    # The branches are independent until the final add. A sufficiently large
    # graph causes peak-memory scheduling to move the small branch first.
    for _ in range(17):
        encoder = F.dropout(F.relu(encoder), p=0.5, training=True)
    decoder = F.dropout(decoder, p=0.5, training=True)
    return encoder + decoder


def run(fn, encoder, decoder):
    torch.manual_seed(2026)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        output = fn(encoder, decoder)

    dropout_shapes = [
        tuple(event.input_shapes[0])
        for event in prof.events()
        if event.name in {"aten::dropout", "aten::native_dropout"}
        and event.input_shapes
    ]
    order = [
        "decoder" if shape == (1, 1, 64) else "encoder"
        for shape in dropout_shapes
    ]
    return output, order


# requires_grad=True selects the training graph. Calling backward is not needed
# to expose the forward RNG-order mismatch.
encoder = torch.randn(1, 128, 64, requires_grad=True)
decoder = torch.randn(1, 1, 64, requires_grad=True)
compiled_model = torch.compile(model)
compiled_model(encoder, decoder)  # compile outside the profiler

eager_output, eager_order = run(model, encoder, decoder)
compiled_output, compiled_order = run(compiled_model, encoder, decoder)

print("eager order:  ", eager_order)
print("compiled order:", compiled_order)
print("outputs equal:", torch.equal(eager_output, compiled_output))

torch.testing.assert_close(compiled_output, eager_output, rtol=0, atol=0)
```

Run with:

```bash
python debug_dropout_cpu.py
```

## Actual behavior

Abbreviated output:

```text
eager order:    [encoder x17, decoder]
compiled order: [decoder, encoder x17]
outputs equal: False

AssertionError: Tensor-likes are not equal!
```

## Expected behavior

With the same input tensors and manual seed, `torch.compile` should preserve the observable RNG ordering of fallback random operators. The compiled order should be `[encoder x17, decoder]`, and the output should exactly equal eager output.

## Root cause

The relevant path in the inspected source is:

```text
torch/_inductor/lowering.py
  native_dropout()
    fallback_random=True
      -> ir.FallbackKernel.create(aten.native_dropout.default, ...)

torch/_inductor/scheduler.py
  Scheduler initialization
    reorder_for_peak_memory=True
      -> memory.reorder_for_peak_memory(...)
```

The encoder and decoder are independent dataflow branches until the final add. Therefore, moving the decoder branch before the encoder is a valid topological reordering if `native_dropout` is treated as a pure operator.

However, fallback `native_dropout` consumes the global CPU RNG state. The actual semantic dependency is:

```text
RNG S0 -> encoder dropout 0 -> S1 -> ... -> encoder dropout 16 -> S17
        -> decoder dropout -> S18
```

That dependency is absent from the scheduler graph. Peak-memory scheduling can therefore move the small decoder branch first and assign every dropout a different part of the global random-number sequence.

Registering `aten.native_dropout.default` as an `ORDERED` effect makes AOTAutograd thread an effect token through the calls. Inductor lowers `with_effects` by adding ordering dependencies between the generated operations, which prevents the peak-memory scheduler from reversing them.

## Proposed fix: opt-in config switch

Add an option for users/backends that require eager RNG order for fallback dropout. The default remains `False`, so this has no behavior change unless the option is explicitly enabled.

### `torch/_inductor/config.py`

```diff
 # fallback to eager for random/dropout, this is slow but useful for debugging
 fallback_random = False

+# Preserve eager RNG ordering for fallback native_dropout calls by threading an
+# ordered effect token through the compiled graph.
+preserve_fallback_random_order = False
```

### `torch/_inductor/compile_fx.py`

Add this context manager immediately before `_compile_fx_main`:

```diff
+@contextlib.contextmanager
+def _preserve_fallback_random_order():
+    if not (
+        config.fallback_random
+        and config.preserve_fallback_random_order
+    ):
+        yield
+        return
+
+    from torch._higher_order_ops.effects import (
+        _EffectType,
+        _register_effectful_op,
+    )
+
+    handle = _register_effectful_op(
+        torch.ops.aten.native_dropout.default,
+        _EffectType.ORDERED,
+    )
+    try:
+        yield
+    finally:
+        handle.destroy()
```

Enter the context before AOTAutograd traces the graph:

```diff
 def _compile_fx_main(...):
     ...
     with (
+        _preserve_fallback_random_order(),
         _use_lazy_graph_module(dynamo_config.use_lazy_graph_module),
         enable_python_dispatcher(),
         ...
     ):
```

Enable it with:

```python
torch._inductor.config.fallback_random = True
torch._inductor.config.preserve_fallback_random_order = True
```

### Verification result

```text
switch=False:
  eager:    [encoder x17, decoder]
  compiled: [decoder, encoder x17]
  outputs equal: False

switch=True:
  eager:    [encoder x17, decoder]
  compiled: [encoder x17, decoder]
  outputs equal: True
```

### Trade-offs

- No default behavior or performance change.
- Users must know that the switch is required for correct fallback RNG ordering.
- The effect registry is process-global. Temporarily changing it during a scoped config patch can race with concurrent compilation in another thread. A production implementation may need a non-global conditional effect mechanism or compiler-level dependency representation.
- Preserving order may reduce peak-memory scheduling freedom for graphs containing many fallback dropout calls. Because the new switch defaults to `False`, users opt into this correctness/performance trade-off explicitly.

The current local PyTorch 2.12 source contains this switch-based implementation.

## Verification matrix

| Case | Result |
|---|---|
| Original 2.12, `fallback_random=True` | Reproduced order and output mismatch |
| New switch off | Mismatch remains; existing behavior is unchanged |
| New switch on | Dropout order and output match eager |
| New switch on with backward | Encoder and decoder gradients match eager |


cc @pbelevich @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises torchInductor peak-memory scheduling of fallback native_dropout operations that share global RNG state.

## Observed implementation decision

The scheduler treats dropout nodes as independent because their hidden RNG-state dependency is absent from the scheduling graph.

## Physical effect

Peak-memory reordering changes which operation consumes each segment of the RNG stream.

## Observed consequence

Outputs and gradients differ from eager execution under the same seed.

## Reported proposal or workaround

Disabling peak-memory reordering or fallback random avoids the mismatch.

## Repair trace

The issue has no linked landed scheduler/RNG repair or report-specific fixed regression.

## Archived evidence

- Raw report: `data/raw/github/pytorch/192635/report.json`
- SHA-256: `e4febdc88a01974a5b15db544910582ddafcaac80a163a3ed28873241ec8b606`
