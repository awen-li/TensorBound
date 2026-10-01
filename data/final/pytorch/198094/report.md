# [inductor] Scatter result reinplaced into a graph input that a later, unrelated mutation overwrites: y = slice_scatter(x, src); x.zero_(); return y returns zeros

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/198094](https://github.com/pytorch/pytorch/issues/198094)
- Opened: 2026-09-22T02:35:00Z
- Closed: open at collection time
- Labels: `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `module: pt2-dispatcher`, `module: reinplacing`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug

Inductor reinplaces a functional `slice_scatter` / `select_scatter` / `diagonal_scatter` whose `self` is a graph input into that input's buffer whenever *some* `copy_` epilogue into the input exists. When that `copy_` writes an unrelated value (a later `zero_()` / `fill_()`) and the scatter result is still observed afterwards (e.g. returned), the result is silently overwritten.

```python
import torch

def fn(x, src):
    y = torch.slice_scatter(x, src[1:3], dim=0, start=1, end=3)   # fresh tensor in eager
    x.zero_()
    return y

x   = torch.arange(24, dtype=torch.int64, device="cuda").reshape(4, 6)
src = torch.arange(-12, 12, dtype=torch.int64, device="cuda").reshape(4, 6)
print(fn(x.clone(), src.clone()).flatten().tolist()[:8])
print(torch.compile(fn, backend="inductor")(x.clone(), src.clone()).flatten().tolist()[:8])
```

```
[0, 1, 2, 3, 4, 5, -6, -5]      # eager
[0, 0, 0, 0, 0, 0, 0, 0]        # inductor
```

Two more shapes of the same bug:

```python
def g(x):                                          # value-wise no-op scatter, fresh tensor in eager
    y = torch.diagonal_scatter(x, x.diagonal())    # (select_scatter(x, x.select(0, 1), 0, 1) and
    x.fill_(2)                                     #  slice_scatter(x, x[1:3], 0, 1, 3) behave the same)
    return y                                       # inductor: all 2s

def h(x):
    x.index_fill_(1, torch.tensor([1], device=x.device), 4)
    y = x.clone()                                  # snapshot of x
    x.clamp_(-4, 4)
    return y                                       # inductor: the clamped values (y aliases x)
```

`backend="aot_eager"` is correct for all three, so the graph is fine. Reproduces with `dynamic=None` and `dynamic=True` on torch 2.14.0+cu130 and 2.15.0.dev20260921+cu130 (CUDA). Replacing the scatter with `y = x.clone(); y[1:3] = src[1:3]` is correct.

### Root cause

`reinplace_inplaceable_ops_core.can_inplace` (`torch/_inductor/fx_passes/reinplace.py`), placeholder branch: it requires that a `copy_` back into the input exists and that no view of the input is used between the op and that `copy_`, but it does not check that the `copy_` writes *this node's result* back, nor whether the node's result is observed after the `copy_`. In the example above the scatter is reinplaced into `x`'s buffer, then `copy_(x, zeros)` overwrites it, and the returned `y` shares that buffer.

Generated code for the first example allocates no buffer for `y` and returns `arg0_1` after the zero-fill kernel. With the fix (PR follows) a full-size buffer is allocated only in this situation; `x[1:3] = src; return x` and `y = torch.slice_scatter(x, src, ...); x.copy_(y); return x` still reinplace without extra allocations.

Related: #197893 (arithmetic no-ops returning the input; `remove_noop_ops`, a different pass), #195451 (aliasing of the copy-back path; values were correct there).

### Versions

- torch 2.14.0+cu130 and torch 2.15.0.dev20260921+cu130, Windows 11, Python 3.12.10, CUDA 13.0, NVIDIA GeForce RTX 4070 Ti, triton-windows 3.8.0

<details><summary>collect_env (stable 2.14.0)</summary>

```
PyTorch version: 2.14.0+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCM used to build PyTorch: N/A

OS: Microsoft Windows 11 Home (10.0.26200 64��Ʈ)
GCC version: Could not collect
Clang version: Could not collect
CMake version: Could not collect
Libc version: N/A

Python version: 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)] (64-bit runtime)
Python platform: Windows-11-10.0.26200-SP0
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to:
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4070 Ti
Nvidia driver version: Could not collect
cuDNN version: Could not collect
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False
Caching allocator config: N/A

CPU:
Name: Intel(R) Core(TM) i7-14700K
Manufacturer: GenuineIntel
Family: 198
Architecture: 9
ProcessorType: 3
DeviceID: CPU0
CurrentClockSpeed: 3400
MaxClockSpeed: 3400
L2CacheSize: 28672
L2CacheSpeed: None
Revision: None

Versions of relevant libraries:
[pip3] numpy==2.5.3
[pip3] torch==2.14.0+cu130
[pip3] triton-windows==3.8.0.post28
[conda] Could not collect
```
</details>

<details><summary>collect_env (nightly 2.15.0.dev20260921)</summary>

```
PyTorch version: 2.15.0.dev20260921+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCm SDK used to build PyTorch: N/A
HIP used to build PyTorch: N/A

OS: Microsoft Windows 11 Home (10.0.26200 64��Ʈ)
GCC version: Could not collect
Clang version: Could not collect
CMake version: Could not collect
Libc version: N/A

Python version: 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)] (64-bit runtime)
Python platform: Windows-11-10.0.26200-SP0
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to:
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4070 Ti
Nvidia driver version: Could not collect
cuDNN version: Could not collect
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False
Caching allocator config: N/A

CPU:
Name: Intel(R) Core(TM) i7-14700K
Manufacturer: GenuineIntel
Family: 198
Architecture: 9
ProcessorType: 3
DeviceID: CPU0
CurrentClockSpeed: 3400
MaxClockSpeed: 3400
L2CacheSize: 28672
L2CacheSpeed: None
Revision: None

Versions of relevant libraries:
[pip3] numpy==2.5.3
[pip3] torch==2.15.0.dev20260921+cu130
[pip3] triton-windows==3.8.0.post28
[conda] Could not collect
```
</details>


cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo @bdhirsh @bobrenjc93 @aorenste

## Trigger

A functional result may be reinplaced into a graph input, an unrelated copy or mutation later overwrites that input, and the result is observed afterward.

## Relevant entities

functional result and its aliases; graph input buffer; copy_ epilogue; later graph output or use

## Implemented relation

Reinplacement makes the functional result and graph input share storage.

## Pre-state

The functional result is semantically fresh and remains live after the later input mutation.

## Binding transition

can_inplace approves reinplacement because some copy_ back to the input exists.

## Post-state

The live result occupies the input buffer and is overwritten by an unrelated later copy or mutation.

## Validity obligation

If a later write replaces the input with a value unrelated to the reinplaced result, neither the result nor any value that would share its buffer remains observable afterward.

## Invalid decision

The placeholder branch checked for a copy-back and intervening input-view uses but not the copy source or post-copy liveness of the result alias closure.

## Physical effect

The later input write replaces bytes that still represent the returned functional result.

## Observed consequence

The returned scatter result contains zeros or fill values instead of the eager result.

## Repair site

torch/_inductor/fx_passes/reinplace.py

## Repair

The pass computes values that would alias the result, checks whether copy_ writes one of them, and rejects reinplacement when an unrelated copy precedes any later use of that alias set.

## Why the repair works

A fresh buffer is retained whenever the input will be overwritten while the functional result or a derived alias remains live.

## Causal basis

The patch reproducer matches the report and the new liveness/source guard directly separates unsafe unrelated overwrites from valid same-result write-backs.

## Analysis implication

A reinplacement check needs copy-source identity and liveness over the prospective alias closure, not only the presence of a copy-back.

## Fix patches

- [`fixes/pytorch-198094-195451-fix.patch`](fixes/pytorch-198094-195451-fix.patch) — https://github.com/pytorch/pytorch/pull/198096
