# `torch.compile` silently returns stale zeros when a size-1 `scatter_add_` is gathered from in the same graph (read-before-write in a fused Triton kernel)

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/191571](https://github.com/pytorch/pytorch/issues/191571)
- Opened: 2026-07-30T02:05:16Z
- Closed: 2026-07-30T15:08:05Z
- Labels: `triage review`, `module: correctness (silent)`, `module: scatter & gather ops`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

When the target of a `scatter_add_` has a leading dimension of size 1 and the result is gathered from within the same graph, Inductor fuses the scatter and the gather into one Triton kernel and emits the gather's **load before** the scatter's `tl.atomic_add`. The gather therefore reads the pre-scatter buffer, and the compiled function silently returns zeros. No error, no warning, no NaN.

This is a wrong-results bug rather than a precision one. I hit it in a neural-network potential where the compiled model returned an energy of `2.88e21 eV` and atomic charges of `4.3e5` — plausible-looking machinery, garbage values, nothing raised.

### Minimal repro

```python
import torch

def f(x, idx):
    res = torch.zeros(1, x.shape[1], device=x.device, dtype=x.dtype)
    res.scatter_add_(0, idx.unsqueeze(-1).expand(-1, x.shape[1]), x)
    return torch.index_select(res, 0, idx)

x = torch.randn(8, 4, device="cuda")
idx = torch.zeros(8, dtype=torch.long, device="cuda")

eager = f(x, idx)
compiled = torch.compile(f)(x, idx)

print("max |eager - compiled| =", (eager - compiled).abs().max().item())
print("eager row0    :", eager[0].tolist())
print("compiled row0 :", compiled[0].tolist())
print("column sums   :", x.sum(0).tolist())
```

Output on torch 2.9.1+cu128, NVIDIA L40S:

```
max |eager - compiled| = 2.245109796524048
eager row0    : [1.2478060722351074, -1.7113100290298462, -1.244249701499939, 2.245109796524048]
compiled row0 : [0.0, 0.0, 0.0, 0.0]
column sums   : [1.2478058338165283, -1.7113100290298462, -1.2442495822906494, 2.245109796524048]
```

Eager returns the column sums, as it should. The compiled version returns **all zeros** — the scatter is not reflected in what the gather reads.

### Generated code (the mechanism)

`TORCH_LOGS=output_code` shows two kernels. The first fills the buffer with zeros. The second fuses the scatter with the gather, and the ordering inside it is the bug:

```python
def triton_poi_fused_expand_index_select_scatter_add_unsqueeze_zeros_1(
    in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr
):
    ...
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask)
    tmp8 = tl.load(out_ptr0 + (x0), xmask, eviction_policy='evict_last')   # <-- gather reads here
    ...
    tl.atomic_add(out_ptr0 + (x0), tmp2, xmask, sem='relaxed')             # <-- scatter writes here
    tl.store(out_ptr1 + (x2), tmp8, xmask)                                 # <-- stores the stale value
```

`tmp8` is loaded from `out_ptr0` before the `tl.atomic_add` that populates `out_ptr0`, so `out_ptr1` receives the pre-scatter contents.

Note that reordering alone would not be sufficient: all 32 program instances in this kernel accumulate into the same 4 output elements, and there is no barrier between the `atomic_add` and the load, so any single instance can only ever observe a partial sum. The scatter and its consumer do not appear to be safely fusible into one kernel at all here.

### Conditions (measured)

Each factor varied independently, `max |eager - compiled|` over 8×4 inputs:

| condition | result |
|---|---|
| `n_out = 1` | **3.43e+00 — wrong** |
| `n_out = 2, 3, 4, 8, 64` | 0.0 — exact |
| CUDA, float32 | **wrong** |
| CUDA, float64 | **wrong** (3.28e+00) |
| CPU, float32 / float64 | 0.0 — exact |
| result **not** gathered from in-graph | 0.0 — exact |
| `index_add_` instead of `scatter_add_` | **wrong**, identically |
| `res[idx]` instead of `index_select` | **wrong**, identically |
| with / without an intervening `+ 1e-6` | **wrong** either way |

So it requires all three of: a size-1 leading dimension on the scatter target, a gather from that result in the same graph, and CUDA. It is independent of dtype, of the scatter/gather op spelling, and of intervening pointwise ops.

`n_out > 1` being exact suggests the fusion decision is only taken when the index is provably a single element, which is also what makes the read-before-write possible.

### Versions

```
torch 2.9.1+cu128
CUDA 12.8
NVIDIA L40S
Linux 6.17
```

### Workaround, for anyone who lands here

Avoid the in-graph gather of a size-1 scatter result — either special-case the single-slot path (a plain `sum` over the scattered axis is equivalent when there is one output slot), or break the graph between the scatter and the gather. Note that the backward pass of `index_select` is itself a scatter, so a model can hit this through autograd even if its forward looks safe: in my case fixing only the forward left gradients wrong by 8 eV/Å while the energy looked correct.

I am happy to test a patch or provide the full `output_code` dump for the failing case.

cc @pragupta @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Duplicate mapping

- Primary report: `github-pytorch-pytorch-171493`
- Shared cause: `cause-pytorch-171493-atomic-scatter-fused-with-dependent-read`

Report 191571 exhibits the same atomic scatter followed by a dependent read in one fused Triton kernel. Its generated code shows the read before atomic_add, and a maintainer closes it as fixed by PR 172301, the landed repair for report 171493.

The sections below describe the shared cause reconstructed from the primary report. The reported behavior above remains the observation specific to this report.

## Trigger

All indices target a size-one leading dimension, and compiled code consumes the accumulated buffer later in the same graph.

## Relevant entities

scatter input; index tensor; zero-filled target buffer; atomic_add store; dependent gather load; fused Triton kernel

## Implemented relation

A consumer of an atomic scatter target reads the buffer only after every contributing program instance has completed its updates.

## Pre-state

The target buffer contains zeros and many CUDA program instances contribute values to the same output locations.

## Binding transition

The scheduler treats the scatter write and dependent gather read as fusible because their normalized index expressions match.

## Post-state

One Triton kernel contains both operations without a grid-wide synchronization boundary, and the generated load executes before or concurrently with the atomic updates.

## Validity obligation

An atomic scatter that requires contributions across program instances completes in a separate kernel before any dependent read of its target begins.

## Invalid decision

The fusion check compared read and write indices but did not reject writes whose store mode requires cross-thread synchronization.

## Physical effect

The gather observes the zero-filled or partially accumulated target rather than the completed reduction.

## Observed consequence

The report's compiled index_add_ model returns a mean near 1,009,876.5 instead of the eager value near 0.000977; the later duplicate returns all-zero gathered rows.

## Repair site

torch/_inductor/scheduler.py::Scheduler.fusable_read_and_write

## Repair

Commit 1687bbf classifies non-default store modes as requiring synchronization and rejects fusion between such writes and dependent reads. It adds CUDA regressions for the reported index_add_ pattern and a second atomic-add/read pattern.

## Why the repair works

Separating the atomic scatter and its consumer into distinct kernels supplies the global completion boundary that an in-kernel barrier cannot provide for arbitrary grid sizes.

## Causal basis

The landed commit states that it fixes 171493, describes the missing all-thread completion requirement, changes fusable_read_and_write, and includes the report's index_add_ reproducer. Report 191571 later names the same repair and shows the identical generated read-before-atomic sequence.

## Analysis implication

Fusion safety analysis needs memory dependence, store mode, reduction fan-in, launch geometry, synchronization scope, and the consumer's read timing together.

## Fix patches

- [`fixes/pytorch-191571-commit-1687bbf.patch`](fixes/pytorch-191571-commit-1687bbf.patch) — https://github.com/pytorch/pytorch/commit/1687bbf6293f6886af4b0916dfaaa98d74272840
