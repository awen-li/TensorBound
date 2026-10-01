# Since 2.13, inductor's 4x random path corrupts CUDA dropout masks

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/198333](https://github.com/pytorch/pytorch/issues/198333)
- Opened: 2026-09-23T07:38:34Z
- Closed: open at collection time
- Labels: `triage review`, `module: activation checkpointing`, `module: random`, `module: regression`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

When attempting to fine-tune a model under pytorch 2.13, that was originally trained under pytorch 2.11 (with inductor compilation turned on), we noticed that the model collapsed. The same fine-tuning under 2.11 and 2.12 worked just fine. Using Claude, we systematically tried possible pytorch flags that had a new default value or were introduced in 2.13. Eventually, Claude was able to isolate the issue and reproduce it in a small script (see below). The workaround was also tested in our much larger model.

### Use of AI

**The following is completely written by Claude.** The "what caused this"-analysis has not been properly reviewed by a human. I'm posting it here to the benefit of whoever wants to take this on, to get a head-start. I understand, you get a lot of AI trash here... But frankly, without Claude, this bug report would not have been possible at all. It is my hope that it helps resolving the issue quickly. We made a sincere effort to figure out when and by which PR the problem has been introduced.

---

Since 2.13, inductor's 4x random path corrupts CUDA dropout masks. Two symptoms, one cause.

**A. The mask is wrong in a compiled inference forward.** Groups of 4 adjacent channels receive a
byte-identical mask, so only a quarter of it is independent and the keep rate is several sigma off.
No checkpointing and no backward are involved. On the reproducer below, at `p=0.5`, shape
`(8, 64, 4096)`:

```
   eager       channel-group duplication  50.047%   keep rate 0.499996 (-0.0 sigma)
   compiled    channel-group duplication 100.000%   keep rate 0.497803 (-6.4 sigma)
```

This appears when autograd is not recording — `torch.no_grad()`, `eval()`-time dropout, MC-dropout.
With autograd recording, the training forward compiles to a different kernel and is clean at this
shape.

**B. Under activation checkpointing, the gradients are silently wrong.** The recompute draws a
different mask than the forward applied, so the backward differentiates a mask that was never used.
Agreement is ~50%, which is what two independent masks give at `p=0.5`:

```
   eager       forward/backward mask agreement 100.000%
   compiled    forward/backward mask agreement  49.999%
```

No error, no NaN, no warning. Eager is unaffected in both cases, and 2.12 is unaffected in both.

Bisected to #184377, "Use rand4x for Inductor Triton random".

### Reproducer

One CUDA GPU, no dependency beyond torch, a few seconds. It checks both symptoms and includes an
isolating control that disables only the 4x path.

```python
"""torch.compile: inductor's 4x random path corrupts dropout masks on CUDA (regression in 2.13).

Two symptoms, one cause. No dependencies beyond torch, one CUDA GPU, a few seconds.

A. THE FORWARD MASK IS WRONG, with no checkpointing involved. Groups of 4 adjacent channels receive
   a byte-identical mask, so only a quarter of the mask is independent and the keep rate is off by
   several sigma.

B. UNDER ACTIVATION CHECKPOINTING the backward differentiates a different mask than the forward
   applied, so the gradients are silently wrong.

Both need the dropout's input to come from a materialised reshape (here `.transpose(1,2)
.contiguous()`). Plain contiguous dropout, a bare `.transpose()` and `.reshape()` are all clean, so
a three-line `rand()` test will NOT reproduce this.

Masks are recovered exactly: the input is ones, so a kept element reads 1/(1-p) and a dropped one 0.
The mask the backward used is recovered the same way from x.grad with an upstream gradient of ones.
"""

from __future__ import annotations

import torch
import torch._inductor.config
from torch.utils.checkpoint import checkpoint

P = 0.5
SHAPE = (8, 64, 4096)
# Both switches below are in the FX graph cache key, so this is not strictly required; it is here so
# that anyone adding the _can_use_4x_random control does not silently get a cached kernel.
torch._inductor.config.force_disable_caches = True


def block(x: torch.Tensor) -> torch.Tensor:
    """Dropout fed by a materialised transpose, as a transformer or STFT front end produces."""
    h = x.transpose(1, 2).contiguous()
    h = torch.nn.functional.dropout(h, P, True)
    return h.transpose(1, 2).contiguous()


def run(compiled: bool, use_checkpoint: bool, four_x: bool = True):
    """Return (mask the forward applied, mask the backward differentiated)."""
    torch._dynamo.reset()
    patched = None
    if not four_x:  # isolate the 4x path without touching anything else
        from torch._inductor.codegen.triton import TritonOverrides as T
        patched = T._can_use_4x_random
        T._can_use_4x_random = staticmethod(lambda *a, **k: False)
    try:
        def fn(x):
            if use_checkpoint:
                return checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)
            return block(x)

        f = torch.compile(fn, backend="inductor") if compiled else fn
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        x = torch.ones(SHAPE, device="cuda", requires_grad=True)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        y = f(x)
        forward = (y.detach() != 0).float()
        y.backward(torch.ones_like(y))
        return forward, (x.grad.detach() != 0).float()
    finally:
        if patched is not None:
            from torch._inductor.codegen.triton import TritonOverrides as T
            T._can_use_4x_random = patched


def duplication(mask: torch.Tensor) -> tuple[float, float]:
    """Percent of positions where channels 4k+1..3 copy channel 4k, and the keep rate's sigma."""
    dup = sum((mask[:, 0::4, :] == mask[:, j::4, :]).float().mean().item() for j in (1, 2, 3)) / 3
    keep = mask.mean().item()
    sigma = (keep - (1 - P)) / ((P * (1 - P) / mask.numel()) ** 0.5)
    return dup * 100, sigma


def main() -> None:
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}")
    print(f"dropout p={P}, shape {SHAPE} (batch, channels, time)
")

    print("A. forward mask, no checkpointing -- channels 4k+1..3 should NOT copy channel 4k")
    bad_fwd = False
    for label, kw in [("eager", dict(compiled=False)), ("compiled", dict(compiled=True)),
                      ("compiled, 4x path off", dict(compiled=True, four_x=False))]:
        fwd, _ = run(use_checkpoint=False, **kw)
        dup, sigma = duplication(fwd)
        if label == "compiled" and dup > 99.0:
            bad_fwd = True
        print(f"   {label:<24} duplication {dup:7.3f}%    keep rate {fwd.mean():.6f} "
              f"({sigma:+.1f} sigma)")

    print("
B. forward vs backward mask, with checkpoint(preserve_rng_state=True)")
    bad_bwd = False
    for label, kw in [("eager", dict(compiled=False)), ("compiled", dict(compiled=True)),
                      ("compiled, 4x path off", dict(compiled=True, four_x=False))]:
        fwd, bwd = run(use_checkpoint=True, **kw)
        agree = (fwd == bwd).float().mean().item() * 100
        if label == "compiled" and agree < 99.99:
            bad_bwd = True
        print(f"   {label:<24} agreement   {agree:7.3f}%"
              + ("" if agree > 99.99 else "    <-- backward used a different mask"))

    print()
    if bad_fwd or bad_bwd:
        print("=> AFFECTED"
              + (" (forward mask corrupted)" if bad_fwd else "")
              + (" (wrong gradients under checkpointing)" if bad_bwd else ""))
    else:
        print("=> not affected")


if __name__ == "__main__":
    main()
```

### Results

RTX PRO 6000 (sm_120), CUDA 13.0.

| torch | A: inference duplication | B: fwd/bwd agreement |
|---|---|---|
| 2.12.0+cu130 | 49.992% (clean) | 100.000% |
| 2.13.0+cu130 | **100.000%** | **49.999%** |
| 2.14.0.dev20260810+cu130 | **100.000%** | **49.992%** |
| 2.15.0.dev20260922+cu130 | **100.000%** | **49.992%** |

Still present on the most recent nightly at the time of writing.

**The 4x path alone is responsible.** Monkeypatching `TritonOverrides._can_use_4x_random()` to
return `False`, with `align_random_eager` left off and nothing else changed, restores both:
duplication 49.992%, agreement 100.000%.

That the ~50% in symptom B is two independent masks rather than a coincidence of `p` is confirmed by
a sweep — measured agreement tracks `p^2 + (1-p)^2` at every point:

| p | measured | independent-mask prediction |
|---|---|---|
| 0.10 | 81.899% | 82.000% |
| 0.25 | 62.505% | 62.500% |
| 0.50 | 49.976% | 50.000% |
| 0.75 | 62.514% | 62.500% |
| 0.90 | 82.115% | 82.000% |

### Cause

`_can_use_4x_random()` requires `triton_tensor_ndim() == 1`, which is a property of the *generated
kernel*, not of the graph. Dumping the generated code for the reproducer on 2.13:

```
forward:    triton_poi_fused_clone_native_dropout_transpose_0              1-D        -> triton_helpers.rand4x
recompute:  triton_poi_fused_native_dropout_native_dropout_backward_transpose_0   2-D tiled  -> tl.rand
```

Two separate problems follow from that one guard.

**Symptom A.** In the 1-D forward kernel the offset expression is `x1 + 64*x0 + 262144*x2`, where
`x0` is time, `x1` is channel and `x2` is batch — so within a block the offsets **stride by 64**,
not by 1. `rand4x` computes `base = tl.min(offsets)` and
`reduced_offsets = base // 4 + tl.arange(0, BLOCK // 4)`, which is only correct for contiguous,
4-aligned offsets. Blocks whose channel index differs only in the low two bits share the same
`base // 4` and therefore receive byte-identical Philox output. Hence exactly 4-fold duplication at
this shape.

The 4-fold factor is what this stride produces; it should not be read as a general figure. The
general statement is that the reconstruction is unsound whenever the offsets are not contiguous.

**Symptom B.** The forward and the recompute are separate kernels and are tiled differently, so they
take opposite branches of the guard. `tl.rand(seed, i)` returns word 0 of `philox(seed, i)`;
`rand4x` returns word `i mod 4` of `philox(seed, i div 4)`. Both are functions of the seed and the
index, but they are different functions, so the two kernels disagree.

Note this is not a `preserve_rng_state` failure. Under `torch.compile` that flag is a no-op —
compiled + 4x-disabled gives 100% agreement with `preserve_rng_state=False` too. The consistency
normally comes from inductor's functionalized RNG handing the recompute the same seed and offset,
which is why 2.12, eager and the 4x-disabled build all agree. Since 2.13 the two kernels map that
same seed and offset to different values.

### Scope

Measured on 2.13, all at shape `(8, 64, 4096)`:

- `use_reentrant=True` is equally affected (49.999%).
- fp16 and bf16 autocast are equally affected (49.999% each).
- **What does not reproduce it:** plain contiguous dropout, a bare `.transpose()` without
  `.contiguous()`, and `.reshape()` are all clean at ~50% duplication. The dropout's input has to
  come from a *materialised* reshape for the offset expression to become non-contiguous. **A
  three-line `rand()` test will not reproduce this.**

### Bisect

Dated nightly wheels: last healthy `2.13.0.dev20260609`, first affected `2.13.0.dev20260610`.
Healthy points checked: `dev20260514`, `dev20260601`, `dev20260606`, `dev20260608`, `dev20260609`.

#184377 landed as `ac6173be20bb4214950e667e2bb7a75ea47c293c` on 2026-06-09, which is that nightly
boundary. It had landed once before on 2026-05-26 (`bf5961e5`) and was reverted the same day
(`fe1b0a2a`).

The PR states that it "changes the exact CUDA Inductor random sequence for a given seed", which is
expected and fine. Neither symptom above is that. Symptom A is a wrong distribution; symptom B is two
kernels that are required to agree not agreeing.

### Workaround

```python
import torch._inductor.config as ic
ic.align_random_eager = True
```

Verified to fix both symptoms. It switches inductor to the eager-aligned RNG path, which as a side
effect disables the 4x path (`not config.align_random_eager` is one of the conditions in
`_can_use_4x_random()`).

Two caveats:

- It is a plain module attribute with no `Config(env_name_force=...)`, so there is no
  `TORCHINDUCTOR_*` environment variable for it and setting one silently does nothing. It must be set
  in Python before the first compile.
- On the 2.13.0 release this flag has a separate bug, #193938: compiled `torch.randn` / `randn_like`
  return uniform(0,1) instead of standard normal. Fixed by #194118 (2026-08-20) and not present on
  current nightlies, but it makes the workaround unsafe on 2.13.0 for any model using `randn` in a
  compiled region.

Capping fusion also works, but only at `max_fusion_size = 1`. Measured at this shape:

| | A: duplication | B: agreement |
|---|---|---|
| `max_fusion_size = 1` | 50.057% (clean) | 100.000% |
| `max_fusion_size = 2` | **100.000%** | **49.976%** |

So the cap is not a reliable workaround: it happens to help only when it forces both kernels onto the
same branch, and which value does that is shape-dependent.


### Versions

- Affected: 2.13.0+cu130 (triton 3.7.1), 2.14.0.dev20260810+cu130, 2.15.0.dev20260922+cu130 (triton 3.8.0)
- Not affected: 2.12.0+cu130 (triton 3.7.0)
- GPU: RTX PRO 6000 Blackwell (sm_120), CUDA 13.0, driver 580.173.02

2.12 ships triton 3.7.0 and is unaffected, so the triton 3.6 to 3.7 jump is not implicated.
`tl.rand4x` is a Triton builtin present in 3.7.0; #184377 adds the `triton_helpers.rand4x` wrapper
and the codegen call site. `TritonOverrides._can_use_4x_random` does not exist in 2.12.

Possibly related: #193671 (silent wrong gradients from `nn.RReLU` under non-reentrant checkpointing)
is a different op and cause, but the same failure shape.


cc @soulitzer @pbelevich @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Trigger

Dropout consumes a materialized transpose and is compiled into a one-dimensional forward kernel, while checkpoint recomputation can use a differently tiled kernel.

## Relevant entities

Philox seed; logical RNG offset; rand4x reduced offset; kernel dimensionality; dropout forward mask; checkpoint recompute mask

## Implemented relation

A given logical RNG element maps to the same independent random word regardless of physical offset order or kernel tiling.

## Pre-state

Functionalized RNG supplies stable seeds and logical offsets, but the fused forward accesses those offsets with a stride rather than as contiguous four-element groups.

## Binding transition

The old _can_use_4x_random guard selects rand4x from one-dimensional kernel shape alone; rand4x then reconstructs groups from the minimum offset and a contiguous arange assumption.

## Post-state

Distinct strided logical offsets collapse onto shared four-word groups, and a differently tiled recompute can select tl.rand instead of rand4x for the same seed-offset pairs.

## Validity obligation

Vectorized random generation is selected only when offset grouping preserves one-to-one logical RNG identity and every recomputation uses the same seed-offset-to-word mapping.

## Invalid decision

The selection rule used generated-kernel dimensionality as a proxy for offset contiguity and did not preserve the RNG mapping across alternative legal tilings.

## Physical effect

Groups of channels receive duplicate random words, while checkpoint forward and recompute obtain independent masks from identical functional RNG state.

## Observed consequence

Compiled inference shows 100% duplication within four-channel groups, and checkpoint forward/backward mask agreement falls to about 50% at p=0.5, producing silent wrong gradients.

## Repair site

torch/_inductor/codegen/triton.py::TritonOverrides._can_use_4x_random and torch/_inductor/config.py

## Repair

Commit 983d11f adds an explicit use_rand4x configuration gate that is disabled by default and updates random-codegen tests to exercise the safe tl.rand default while retaining opt-in rand4x coverage.

## Why the repair works

Default compilation no longer enters the offset-grouping path whose preconditions are not established; forward and recomputation use the scalar random mapping that is defined per logical offset.

## Causal basis

The report traces both failures to the rand4x selection path and shows that disabling only that path restores independent masks and checkpoint agreement. Landed commit 983d11f directly references 198333 and disables rand4x by default at the identified selection guard.

## Analysis implication

Random-codegen analysis needs logical offsets, physical access order, four-word grouping, seed/counter mapping, kernel tiling, forward/recompute equivalence, and selection configuration together.

## Fix patches

- [`fixes/pytorch-198333-commit-983d11f.patch`](fixes/pytorch-198333-commit-983d11f.patch) — https://github.com/pytorch/pytorch/commit/983d11f91bd577039613d36d58da6c1c729a12e2
