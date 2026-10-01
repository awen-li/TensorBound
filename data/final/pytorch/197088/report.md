# [inductor][cudagraphs] `x.new_zeros(shape, device="cpu")` inside a compiled function returns a **CUDA** tensor when `triton.cudagraphs` is on (eager and default config return a CPU tensor)

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/197088](https://github.com/pytorch/pytorch/issues/197088)
- Opened: 2026-09-15T11:50:13Z
- Closed: Open at collection
- Labels: `triaged`, `module: correctness (silent)`, `module: cuda graphs`, `oncall: pt2`, `module: inductor`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug

**Summary.** With `triton.cudagraphs` enabled, `x.new_zeros(shape, device="cpu")` for a CUDA `x` does not return the device the program asked for: eager and the default compiled configuration return a CPU tensor, while the cudagraphs path returns a CUDA tensor (torch 2.10.0+cu128 and 2.14.0+cu130, Tesla T4). A user who enables `mode="reduce-overhead"` gets tensors on the wrong device downstream, which surfaces as device-mismatch errors far from the line that created the tensor. Because the device is decided by the cudagraphs lowering rather than by the explicit `device=` argument, it is the compilation mode and not the program that determines where the tensor lives.

A compiled function that creates a tensor with an explicit `device="cpu"` (here `x.new_zeros((2, 3), device="cpu")` with `x` on CUDA) returns a CPU tensor in eager mode and with the default Inductor configuration, but a **CUDA** tensor when CUDA graphs are enabled (`torch._inductor.config.triton.cudagraphs = True`, which is what `mode="reduce-overhead"` sets). The explicit device argument is silently overridden, so downstream code that expects a CPU tensor (e.g. feeding it to a CPU-only op, `.numpy()`, or comparing `.device`) breaks only in the cudagraphs configuration.

```python
import torch
from torch._inductor import config as IC

def f(x):
    return x.new_zeros((2, 3), device="cpu")

x = torch.randn(5, device="cuda")
print("eager            :", f(x).device)                          # cpu

torch._dynamo.reset()
print("inductor default :", torch.compile(f)(x).device)           # cpu

torch._dynamo.reset()
with IC.patch({"triton.cudagraphs": True}):
    print("inductor cudagraph:", torch.compile(f)(x).device)      # cuda:0   <-- wrong
# equivalently: torch.compile(f, mode="reduce-overhead")(x).device
```

Observed on a Tesla T4 with torch 2.14.0+cu130 (and 2.10.0+cu128):

| configuration | output device |
|---|---|
| eager | cpu |
| `torch.compile(f)` (default) | cpu |
| `torch.compile(f)` with `triton.cudagraphs=True` | **cuda:0** |

Values are all zeros in every case; only the device differs. Found by a configuration differential that compiles the same OpInfo sample under several Inductor configurations and compares the results with each other (the OpInfo sample for `new_zeros` passes `device='cpu'` explicitly).

Expected: a CPU tensor, as eager and the default configuration produce — or, if a CPU-resident output cannot live in a CUDA-graph-captured region, the graph should be partitioned around it (as #190555 proposes for cross-device *fallback* kernels) rather than moving the result to the wrong device.

### Error logs

No error; the output device differs as shown.

### Versions

```
PyTorch version: 2.14.0+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
OS: Ubuntu 22.04 (Kaggle notebook image), Linux 6.12, glibc 2.35
Python version: 3.12.13
GPU: Tesla T4 (compute capability 7.5), driver as provided by the Kaggle image
Is CUDA available: True
Versions of relevant libraries:
[pip3] torch==2.14.0+cu130 (installed from PyPI into the Kaggle image)
```

(The same reproducer on torch 2.10.0+cu128 / Tesla T4 behaves identically.)

Still present on nightly torch 2.15.0.dev20260911+cu130 (nightly, git fb8ff0f1, Kaggle: Tesla T4 + Linux CPU): with `triton.cudagraphs=True` the compiled function returns a CUDA tensor, eager and the default config return a CPU tensor.

cc @mcarilli @ezyang @eellison @penguinwu @BoyuanFeng @chauhang @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises torchInductor cudagraph lowering of tensor factory calls with an explicit CPU device and a CUDA receiver.

## Observed implementation decision

The cudagraph path derives placement from the receiver/captured graph instead of preserving the explicit device argument.

## Physical effect

A logically CPU tensor is allocated on CUDA.

## Observed consequence

The same program returns different devices under default and cudagraph configurations.

## Reported proposal or workaround

Disabling cudagraphs preserves CPU placement.

## Repair trace

The issue remains open without a linked source correction or fixed regression.

## Archived evidence

- Raw report: `data/raw/github/pytorch/197088/report.json`
- SHA-256: `c5f1a5f50ef7c92fab0e0780b3344fb3e25e12212705d9a0ff2d233135fed983`
