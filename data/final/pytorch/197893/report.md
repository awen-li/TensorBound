# [inductor] `x + 0` / `x * 1` / `x / 1.0` as a graph output returns the input tensor itself (`out is x`), so an in-place update of the result silently corrupts the input

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/197893](https://github.com/pytorch/pytorch/issues/197893)
- Opened: 2026-09-21T14:32:36Z
- Closed: Open at collection
- Labels: `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug

**Summary.** For a function whose output is `x / temperature`, `buf * momentum` or `x + bias` called with the neutral value (`1.0`, `1.0`, `0`), eager and `aot_eager` return a fresh tensor, while Inductor returns the input object itself (`out is x`; with `requires_grad`, another object on the same storage), on torch 2.14.0 and nightly, CPU and CUDA. A caller that then updates the result in place (`new_buf.add_(g)`, `logits.clamp_(...)`) overwrites its own input with no error, only for the value that makes the operation a no-op: the same code is correct for `temperature=0.9` and wrong for `temperature=1.0`. Because `joint_graph.remove_no_ops` replaces the node by its operand without the check that `post_grad.remove_noop_ops` has ("Don't introduce new aliasing between inputs and outputs"), every graph that returns such a no-op of an input is affected.

```python
import torch

def scale(logits, temperature):
    return logits / temperature

def momentum_step(buf, momentum):
    return buf * momentum

def add_bias(x, bias):
    return x + bias

for name, f, arg in (("logits / 1.0", scale, 1.0), ("logits / 0.9", scale, 0.9),
                     ("buf * 1.0", momentum_step, 1.0), ("x + 0", add_bias, 0)):
    for backend in ("eager (no compile)", "aot_eager", "inductor"):
        torch._dynamo.reset()
        g = f if backend.startswith("eager") else torch.compile(f, backend=backend)
        x = torch.ones(3)
        out = g(x, arg)
        same = out is x
        out.add_(5)                     # the caller updates the *result* in place
        print(f"{name:13s} {backend:19s} out is x: {str(same):5s}  x afterwards: {x.tolist()}")
```

```
logits / 1.0  eager (no compile)  out is x: False  x afterwards: [1.0, 1.0, 1.0]
logits / 1.0  aot_eager           out is x: False  x afterwards: [1.0, 1.0, 1.0]
logits / 1.0  inductor            out is x: True   x afterwards: [6.0, 6.0, 6.0]
logits / 0.9  eager (no compile)  out is x: False  x afterwards: [1.0, 1.0, 1.0]
logits / 0.9  aot_eager           out is x: False  x afterwards: [1.0, 1.0, 1.0]
logits / 0.9  inductor            out is x: False  x afterwards: [1.0, 1.0, 1.0]
buf * 1.0     eager (no compile)  out is x: False  x afterwards: [1.0, 1.0, 1.0]
buf * 1.0     aot_eager           out is x: False  x afterwards: [1.0, 1.0, 1.0]
buf * 1.0     inductor            out is x: True   x afterwards: [6.0, 6.0, 6.0]
x + 0         eager (no compile)  out is x: False  x afterwards: [1.0, 1.0, 1.0]
x + 0         aot_eager           out is x: False  x afterwards: [1.0, 1.0, 1.0]
x + 0         inductor            out is x: True   x afterwards: [6.0, 6.0, 6.0]
```

Also affected: `0 + x`, `x - 0`, `x + 0.0`, `torch.add(x, 0)`, `(x + 0) + 0`, the same operations with a constant tensor built inside the function (`x * torch.ones(3)`, `x + torch.zeros(3)`, `x - torch.zeros_like(x)`, `x / torch.ones_like(x)`, `x * torch.full_like(x, 1.0)`), integer dtypes, an `nn.Module` whose `forward` is `logits / self.t` with `self.t = 1.0`, and inputs with `requires_grad=True` (there `out is x` is `False` but `out.data_ptr() == x.data_ptr()`, so `out.detach().mul_(2)` doubles `x`). Not affected: `x ** 1`, `x // 1`, `x.clone()`, `x + torch.zeros(())`.

A second, smaller consequence of the same elimination: `x + 0` no longer normalises negative zero (`torch.tensor(-0.0) + 0` is `0.` in eager and `-0.` compiled).

**Root cause and possible fix.** `torch/_inductor/fx_passes/joint_graph.py::remove_no_ops` → `replace_no_op` does `node.replace_all_uses_with(replacement)` whenever shape/dtype/device match and the replacement is not mutated (the guard added for #174187). When `replacement` is a placeholder and `node` feeds the graph output, the compiled graph returns its input, but AOTAutograd has already recorded that output as a non-alias, so the runtime wrapper hands the input tensor back to the caller as if it were fresh. `post_grad.py::remove_noop_ops` avoids the same situation with

```python
# Don't introduce new aliasing between inputs and outputs.
if (not node_is_view and node_storage in output_storages
        and (src_storage in input_storages or src_storage in output_storages)):
    continue
```

Applying the same condition in `replace_no_op` (skip, or replace the node by `aten.clone(replacement)` so the arithmetic still disappears) restores eager semantics; the check also has to cover two outputs that would become the same tensor (`return x * 1, x + 0`).

### Error logs

No error is raised.

### Versions

```
PyTorch version: 2.14.0+cpu
OS: Microsoft Windows 11 Pro for Workstations (10.0.26200 64-bit)
MSVC: Microsoft (R) C/C++ Optimizing Compiler Version 19.44.35228 for x64
Python version: 3.14.7 (64-bit runtime)
Is CUDA available: False
CPU: 11th Gen Intel(R) Core(TM) i9-11900T @ 1.50GHz
[pip3] numpy==2.5.2
[pip3] torch==2.14.0+cpu
```

Same output on nightly (2.15.0.dev20260911 on Windows, 2.15.0.dev20260920 on Linux), on Linux CPU with 2.14.0, and on CUDA (Tesla T4, 2.14.0+cu130), so the behaviour comes from the shared FX pass and not from a backend.


cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

[inductor] `x + 0` / `x * 1` / `x / 1.0` as a graph output returns the input tensor itself (`out is x`), so an in-place update of the result silently corrupts the input

## Observed implementation decision

replace_no_op calls replace_all_uses_with on output-producing identity operations without the storage-alias guard already used by the post-grad no-op pass.

## Physical effect

Joint-graph no-op elimination replaces a fresh arithmetic result at the graph boundary with its input tensor, introducing input/output aliasing absent from eager semantics.

## Observed consequence

For neutral values such as x+0 or x/1, mutating the returned tensor silently mutates the caller's input; non-neutral values remain independent.

## Reported proposal or workaround

Backend, value, device, dtype, and requires-grad controls localize the change to joint_graph.remove_no_ops; applying the existing output-alias guard restores eager behavior.

## Repair trace

A local guard restores eager alias behavior, but no landed patch or fixed-version regression establishes the repository repair.

## Archived evidence

- Raw report: `data/raw/github/pytorch/197893/report.json`
- SHA-256: `b80656362427c2f63d686acf7b2b0da02302ba7b42eff4d5b8f71af6dddf5e9a`
