# `torch.compile` silently computes wrong gradients for chained modulated LayerNorms with a shared scale/shift parameter (regression 2.9.1 → 2.10.0 due to mix_order_reduction)

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/193061](https://github.com/pytorch/pytorch/issues/193061)
- Opened: 2026-08-11T23:20:27Z
- Closed: 2026-08-21T21:26:20Z
- Labels: `high priority`, `triaged`, `module: regression`, `has workaround`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

When a model applies two chained "modulated norm" blocks of the form `x = x + (layer_norm(x) * (1 + scale) + shift)`, where all `scale`/`shift` vectors are **chunks of one shared parameter** (the DiT / AdaLN `scale_shift_table` pattern used by Wan 2.1), the Inductor-compiled backward produces **silently wrong gradients for the second norm's `shift`/`scale` slices** once the total reduction size `batch*seq` is large (≳16k tokens). The first norm's slices are correct to fp32 precision; the second norm's slices come back with relative errors of up to ~1e+6 — depending on graph arrangement the corresponding region of the grad buffer is simply never written, so it can also surface as bitwise-stale values from a previous backward.

**Update: This is a regression introduced in 2.10.0**: the same script is fully clean on 2.9.1 and fails identically on 2.10.0, 2.11.0, and 2.13.0 (latest stable).

Minimal repro in plain fp32, plain `nn.LayerNorm`, no autocast, no checkpointing, no attention:

```python
import torch
import torch.nn as nn

DIM, SEQ, SMALL_SEQ = 1024, 32768, 4096
dev = 'cuda'


def run(seq=SEQ, n_norms=2, slicing='chunk'):
    torch.manual_seed(0)
    norms = [nn.LayerNorm(DIM, 1e-6, elementwise_affine=False).to(dev)
             for _ in range(n_norms)]
    if slicing == 'separate':
        params = [nn.Parameter(torch.randn(DIM, device=dev) / DIM ** 0.5)
                  for _ in range(2 * n_norms)]
    else:
        table = nn.Parameter(
            torch.randn(1, 2 * n_norms, DIM, device=dev) / DIM ** 0.5)
        params = [table]

    def fn(x):
        if slicing == 'separate':
            mods = params
        elif slicing == 'unbind':
            mods = [s.unsqueeze(1) for s in params[0].unbind(1)]
        else:  # 'chunk' — the failing case
            mods = params[0].chunk(2 * n_norms, dim=1)
        for i, norm in enumerate(norms):
            shift, scale = mods[2 * i], mods[2 * i + 1]
            x = x + (norm(x) * (1 + scale) + shift)
        return x.square().mean()

    fn_c = torch.compile(fn, mode='default', dynamic=False)
    x = torch.randn(1, seq, DIM, device=dev)

    def grads():
        g = [p.grad for p in params]
        for p in params:
            p.grad = None
        return torch.cat([t.reshape(-1) for t in g])

    fn(x).backward()
    g_eager = grads()
    fn_c(x).backward()
    g_compiled = grads()

    rel = ((g_eager - g_compiled).abs().max()
           / g_eager.abs().max().clamp(min=1e-12)).item()
    return rel


CASES = [
    ('chunked table, 2 norms, seq 32768  (THE BUG)', dict()),
    ('same but seq 4096', dict(seq=SMALL_SEQ)),
    ('same but single norm', dict(n_norms=1)),
    ('same but unbind(1)+unsqueeze(1) instead of chunk', dict(slicing='unbind')),
    ('same but 4 independent parameters', dict(slicing='separate')),
]

for name, kwargs in CASES:
    rel = run(**kwargs)
    print(f'{"FAIL" if rel > 0.1 else "ok  "}  max rel grad err {rel:9.3e}  {name}',
          flush=True)

print('\nexpected: first row FAIL (silently wrong grads), all others ok')
```

### Output

On H100 — **identical on torch 2.10.0, 2.11.0 and 2.13.0** (cu126/cu128/cu129 builds all affected; CUDA toolkit version is irrelevant). **torch 2.9.1 is clean** (first row `ok  2.141e-07`):

```
FAIL  max rel grad err 1.481e+06  chunked table, 2 norms, seq 32768  (THE BUG)
ok    max rel grad err 2.068e-07  same but seq 4096
ok    max rel grad err 2.251e-07  same but single norm
ok    max rel grad err 2.141e-07  same but unbind(1)+unsqueeze(1) instead of chunk
ok    max rel grad err 2.116e-07  same but 4 independent parameters
```

### What we bisected (each ingredient is necessary)

1. **Two chained modulated norms** — a single one is clean.
2. **Gradient flow through a multiplicative rsqrt-variance statistic** — an RMSNorm-style `x * rsqrt(mean(x^2) + eps)` fails exactly like LayerNorm; removing the norm, subtracting only the mean (`x - x.mean(-1, keepdim=True)`), or detaching the norm statistics (same forward, pointwise backward) are all clean.
3. **The modulation vectors must be `chunk`/narrow slices of one shared tensor** — `unbind(1)` (+`unsqueeze(1)` for the broadcast shape) or independent parameters are clean. The shared tensor need not be a parameter: chunking a conditioning MLP's output fails the same way (max rel err 7.2e+1 / 4.7e+5 on 2.11 / 2.13, landing in the MLP weight grads). In every failing case the slice grads assemble into one buffer — matching the never-written/stale-buffer behavior.
4. **The slices must broadcast over batch AND sequence, with total `batch*seq` ≳ 16k** — at DIM=5120: seq 8192 ok, seq 16384 FAIL; also FAIL at DIM=256 with long seq, and at batch 8 × seq 4096 (shared table) even though batch 1 × seq 4096 is ok. Per-sample modulation (a real batch dim, e.g. `lin(c).view(B, 4, D)` with B ≥ 2) is clean at every tested length (B=2 × seq 32768, B=4 × seq 65536 ok) — but at B=1 it degenerates to the broadcast case and fails (B=1 × seq 32768 FAIL). Note the two roles of batch size: within the broadcast case it only contributes to the total (B=8 × 4096 fails like B=1 × 32768), while B ≥ 2 per-sample modulation is clean even with a 65536-long reduction per grad element — so the broadcast structure selects the buggy codegen path, and reduction size is the trigger only within it.

Not required: autocast/bf16, gradient checkpointing, attention/FFN, `elementwise_affine`. No escape via compile config (`default`/`reduce-overhead`/`max-autotune`, `fullgraph` on/off, `dynamic` on/off), `TORCHINDUCTOR_SPLIT_REDUCTIONS=0`, or expression rewrites (cloning the norm output or slices, `contiguous()`, `torch.addcmul`, distributing the multiply) — only the `unbind`/separate-parameter rewrites, which change how the slice grads are routed, avoid it.

### Real-world impact

Found while training diffusers Wan 2.1 14B (video diffusion): the per-block `scale_shift_table` gradients were garbage, and since the grad-buffer region is never written, iteration N reused iteration N-1's values bitwise whenever shapes matched — a linearly growing grad-norm staircase.

Who is exposed: any chained-modulated-norm model whose modulation slices broadcast over the batch — learnable scale-shift tables (Wan-style, at any batch size), or ordinary per-sample adaLN whenever the local batch is 1 (common in long-video training and gradient accumulation). Per-sample adaLN at local batch ≥ 2 tested clean at every length (up to seq 65536), and we verified three real DiT-family ImageNet training configs are unaffected.

### Versions

Fails on 2.10.0+cu128, 2.11.0+cu128/cu129, 2.13.0+cu126/cu129; clean on 2.9.1+cu128 (regression window: 2.9.1 → 2.10.0). `collect_env` for the primary environment:

```
PyTorch version: 2.13.0+cu129
Is debug build: False
CUDA used to build PyTorch: 12.9
OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
Libc version: glibc-2.35
Python version: 3.11.10 (64-bit runtime)
Python platform: Linux-5.15.0-131-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.9.86
GPU models and configuration: 8x NVIDIA H100 80GB HBM3
Nvidia driver version: 550.127.08
cuDNN version: 9.10.2
triton: 3.7.1
```

### A related observation

On 2.11.0 only, the same setup with independent parameters of shape `[1, 1, D]` (instead of `[D]` used above) made the compiled backward crash outright (`Function CompiledFunctionBackward returned an invalid gradient at index 3 - got [1024] but expected shape compatible with [1, 1, 1024]`). 2.13.0 neither crashes nor miscomputes there — so the grad-shape handling in this area changed between 2.11 and 2.13 while the silent bug above survived; possibly a useful hint for locating it.

### AI assistance disclosure

Per the [PyTorch AI policy](https://github.com/pytorch/pytorch/blob/main/AI_POLICY.md):
The bug was hit in a real training run (Wan 2.1 14B). The debugging, ablation, and this write-up were developed with AI assistance (Claude Fable 5). Every number and code path in this report was executed and verified on our cluster (H100, both torch versions), and I have reviewed and can speak to all of it.

cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo @ezyang @gchanan @zou3519 @msaroufim @eellison @shunting314 @jansel @bdhirsh

## Trigger

A mix-order-fused reduction produces a gradient slice represented as a NonOwningLayout view into a shared concatenated gradient buffer.

## Relevant entities

reduction workspace; NonOwningLayout output view; shared gradient buffer; generated Python buffer name

## Implemented relation

The generated output name denotes a view whose writes must update a region of the shared gradient buffer.

## Pre-state

The output variable is bound to reinterpret_tensor over the shared buffer.

## Binding transition

Final-reduction code assigns a fresh reduction tensor to the output variable.

## Post-state

The variable no longer denotes the alias, and the shared-buffer region remains unwritten.

## Validity obligation

Code generation for a non-owning output writes through the existing alias instead of rebinding its name to independent storage.

## Invalid decision

_codegen_mix_order_reduction emitted variable assignment for every output layout.

## Physical effect

The reduction value stays in standalone storage while the returned shared gradient buffer contains stale or uninitialized bytes in that slice.

## Observed consequence

Compiled gradients show relative errors around 1e6 for affected slices.

## Repair site

torch/_inductor/codegen/simd.py

## Repair

For NonOwningLayout outputs, code generation emits output_alias.copy_(reduced) and retains assignment only for owning outputs.

## Why the repair works

copy_ writes the computed reduction through the alias into the shared buffer region returned to the caller.

## Causal basis

The repair branches explicitly on NonOwningLayout and regression checks require copy_ into the alias, matching the reported missing write.

## Analysis implication

Code generation needs ownership/layout information to distinguish binding an owning result from updating a non-owning destination view.

## Fix patches

- [`fixes/pytorch-193061-fix.patch`](fixes/pytorch-193061-fix.patch) — https://github.com/pytorch/pytorch/pull/193103
