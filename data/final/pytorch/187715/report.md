# `torch.compile(mode='reduce-overhead')` silently degrades training on ROCm — `cudagraph_trees` has no HIP guard

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/187715](https://github.com/pytorch/pytorch/issues/187715)
- Opened: 2026-06-19T07:11:38Z
- Closed: 2026-07-18T00:24:42Z
- Labels: `module: rocm`, `triaged`, `module: correctness (silent)`, `module: cuda graphs`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

# `torch.compile(mode='reduce-overhead')` silently degrades training on ROCm — `cudagraph_trees` has no HIP guard

### 🐛 Bug Description

On AMD ROCm (MI300X, ROCm 7.0, PyTorch 2.10), `torch.compile(mode='reduce-overhead')` causes training loss to converge significantly slower than `mode='default'` or no-compile.

Isolation tests confirm the underlying HIP Graph API (`torch.cuda.CUDAGraph`) works correctly. The bug is in inductor's `cudagraph_trees.py`, which has zero ROCm awareness and unconditionally enables the new cudagraph implementation on ROCm.

### 📊 Minimal Reproduction

```python
import torch
import torch._inductor.config as icfg

# A simple training loop
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(512, 256)
        self.fc2 = torch.nn.Linear(256, 512)
    def forward(self, x):
        return self.fc2(torch.nn.functional.relu(self.fc1(x)))

def train(model, n_steps=200):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(n_steps):
        x = torch.randn(64, 512, device='cuda')
        y = torch.randn(64, 512, device='cuda')
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step(); opt.zero_grad()
        if step <= 5 or step % 20 == 0:
            losses.append((step, loss.item()))
    return losses

# ✅ Fine
model1 = Net().cuda()
loss1 = train(model1)
print(f"no compile:                step200 loss={loss1[-1][1]:.4f}")

# ✅ Fine
model2 = Net().cuda()
model2 = torch.compile(model2, mode='default')
loss2 = train(model2)
print(f"mode='default':            step200 loss={loss2[-1][1]:.4f}")

# ❌ Degraded on ROCm
model3 = Net().cuda()
model3 = torch.compile(model3, mode='reduce-overhead')
loss3 = train(model3)
print(f"mode='reduce-overhead':    step200 loss={loss3[-1][1]:.4f}")

# ❌ Also degraded — this is the underlying cause
icfg.triton.cudagraphs = True
icfg.triton.cudagraph_trees = True  # ← culprit
model4 = Net().cuda()
model4 = torch.compile(model4)
loss4 = train(model4)
print(f"cudagraphs+trees manually: step200 loss={loss4[-1][1]:.4f}")

# ✅ Fine — old cudagraph impl works on ROCm
icfg.triton.cudagraphs = True
icfg.triton.cudagraph_trees = False  # ← old impl, works
model5 = Net().cuda()
model5 = torch.compile(model5)
loss5 = train(model5)
print(f"cudagraphs ONLY (old):     step200 loss={loss5[-1][1]:.4f}")
```

### 📈 Typical Output on ROCm (MI300X, ROCm 7.0)

```
no compile:                step200 loss=0.7234
mode='default':            step200 loss=0.7310
mode='reduce-overhead':    step200 loss=0.9012   ← ~23% worse
cudagraphs+trees manually: step200 loss=0.8956   ← same degradation
cudagraphs ONLY (old):     step200 loss=0.7288   ← fine
```

### 🔍 Step-by-step Loss Trace

When tracing loss per step, the degradation shows a clear pattern of stalling during graph replay:

```
Step     default     reduce-overhead
  1      1.042        1.040           ← graph capture, same
 10      0.891        0.985           ← divergence grows during replay
 20      0.793        0.972           ← loss stalled
 30      0.751        0.968           ← still stalled
 40      0.728        0.901           ← graph re-recorded, temporary recovery
 50      0.714        0.885           ← degrading again
```

### 🎯 Root Cause

`torch.compile(mode='reduce-overhead')` unconditionally enables `triton.cudagraphs=True`, which — combined with the default `triton.cudagraph_trees=True` — routes into `cudagraph_trees.py`'s new implementation. This file has **zero ROCm/HIP handling** (0 HIP references, compared to 16+ in `utils.py` and 8+ in `config.py`).

The call chain:
```
torch.compile(model, mode='reduce-overhead')
  → torch/__init__.py: _TorchCompileInductorWrapper.apply_mode()
    → config_patches = {"triton.cudagraphs": True}
      → torch/_inductor/compile_fx.py: cudagraphify()
        → config.triton.cudagraph_trees=True (default)
        → new_cudagraphify_impl()   ← NO ROCm GUARD
```

The fix location, `torch/_inductor/__init__.py`:
```python
# Current — unconditionally enables cudagraphs on ROCm:
"reduce-overhead": {
    "triton.cudagraphs": True,
},
```

### 🧪 Isolation: HIP Graph Primitive is Fine

```python
# torch.cuda.CUDAGraph works correctly for training on ROCm:
g = torch.cuda.CUDAGraph()
model = Net().cuda()
opt = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(64, 512, device='cuda')
y = torch.randn(64, 512, device='cuda')

opt.zero_grad()
with torch.cuda.graph(g):
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()

# Replay 10 times with new data — loss changes normally:
for i in range(10):
    x.copy_(torch.randn(64, 512, device='cuda'))
    y.copy_(torch.randn(64, 512, device='cuda'))
    opt.zero_grad()
    g.replay()
    opt.step()
    # ✅ Loss decreases normally — HIP Graph API works
```

| Layer | Works on ROCm? |
|-------|:-------------:|
| `torch.cuda.CUDAGraph` (HIP Graph primitive) | ✅ |
| inductor `cudagraphs` old impl (`cudagraph_trees=False`) | ✅ |
| inductor `cudagraph_trees` new impl (`cudagraph_trees=True`) | ❌ |
| `mode='reduce-overhead'` | ❌ |

### 💡 Suggested Fix

**One-line fix** in `torch/_inductor/__init__.py`:

```python
# Before:
"triton.cudagraphs": True,

# After:
"triton.cudagraphs": False if torch.version.hip else True,
```

This safely degrades `reduce-overhead` to `default` mode on ROCm without affecting CUDA.

Alternative: add `config.rocm.disable_cudagraph_trees: bool = True` and check it in `compile_fx.py:cudagraphify()`.

### 🌍 Environment

| Item | Detail |
|------|--------|
| GPU | AMD Radeon Graphics (MI300X-class), 205 GB VRAM |
| ROCm | 7.0.51831 |
| PyTorch | 2.10.0+rocm7.0 |
| Python | 3.10 |

### 📎 Related Source Files

- `torch/_inductor/__init__.py:357` — `reduce-overhead` mode definition (fix here)
- `torch/_inductor/cudagraph_trees.py` — **zero HIP handling**
- `torch/_inductor/compile_fx.py:1804-1843` — `cudagraphify()` entry point
- `torch/_inductor/config.py:1996` — `class rocm` (no cudagraph disable option)

---

_CC: @pytorch/rocm-maintainers_


### Versions


PyTorch version: 2.10.0+rocm7.0
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 7.0.51831

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0
Clang version: 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.0)
CMake version: version 3.31.6
Libc version: glibc-2.35

Python version: 3.10.20 | packaged by conda-forge | (main, Jun 11 2026, 03:31:56) [GCC 14.3.0] (64-bit runtime)
Python platform: Linux-6.1.0-39-amd64-x86_64-with-glibc2.35
Is CUDA available: True
GPU models and configuration: AMD Radeon Graphics (gfx942:sramecc+:xnack-)
HIP runtime version: 7.0.51831
MIOpen runtime version: 3.5.0

CPU: INTEL(R) XEON(R) PLATINUM 8568Y+ (160 cores, 2 sockets)

Versions of relevant libraries:
[pip3] torch==2.10.0+rocm7.0
[pip3] torchaudio==2.10.0+rocm7.0
[pip3] torchvision==0.25.0+rocm7.0
[pip3] triton==3.7.0
[pip3] triton-rocm==3.6.0

cc @jeffdaily @sunway513 @jithunnair-amd @pruthvistony @ROCmSupport @jataylo @hongxiayang @naromero77amd @pragupta @jerrymannil @xinyazhang @mcarilli @ezyang @eellison @penguinwu @BoyuanFeng @chauhang @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

A training loop runs under torch.compile mode reduce-overhead with cudagraph trees active on an MI300X-era stack.

## Observed implementation decision

The report attributes slower convergence to enabling cudagraph trees without a HIP guard, but it does not localize a captured buffer, replay dependency, or mutation that changes model values.

## Physical effect

The original environment records a different optimization trajectory under graph replay.

## Observed consequence

Training loss is reported to converge more slowly than eager and default compilation.

## Reported proposal or workaround

Disabling cudagraph trees removes the reported difference; an AMD maintainer cannot reproduce on PyTorch main and ROCm 7.14rc3 while confirming graph recording remained active.

## Repair trace

The later run uses different hardware and software and supplies no fixing commit. The record therefore lacks both the first invalid binding decision and a landed repair that explains the original MI300X observation.

## Archived evidence

- Raw report: `data/raw/github/pytorch/187715/report.json`
- SHA-256: `59499ec3cb2d43001c5c9bbfc96b838311e48e64c2b956cec5132fe9fbc823f3`
