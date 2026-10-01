# [inductor] Generated reduction kernel reloads global scratch written by its own earlier loop with no synchronization, giving non-deterministic gradients

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/198562](https://github.com/pytorch/pytorch/issues/198562)
- Opened: 2026-09-24T22:22:53Z
- Closed: 2026-09-24T22:36:51Z
- Labels: `triage review`, `needs reproduction`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

Inductor generates fused reduction kernels in which one loop stores partial
results to a **global** scratch buffer and a second loop in the **same kernel**
reloads that buffer, with no synchronization between the two loops. Nothing
guarantees that the stores from loop 1 are visible to the waves executing loop 2,
so the reload can read stale or partially-written data. The result is a
compiled backward that returns different input gradients on every call from
fixed inputs and a fixed upstream gradient.

The forward output is bitwise stable. Eager is bitwise stable. Only the compiled
backward moves.

Because it is a race, the observed behaviour depends on wave scheduling. Anything
that perturbs code generation or timing — reordering operations, forcing a graph
break, changing fusion — makes it appear to go away without removing the hazard.
We think that is worth stating explicitly, because it makes the bug easy to
misdiagnose as a scheduling or version artifact.

#### The generated pattern

Schematic, not a verbatim dump — the structure is what matters:

```python
@triton.jit
def triton_red_fused_..._0(in_ptr0, out_ptr0, xnumel, r0_numel, ...):
    # loop 1 — partial results go to a global scratch buffer
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        ...
        tl.store(out_ptr0 + (...), tmp_partial, mask)

    # <-- nothing here

    # loop 2 — reload what loop 1 wrote, same global buffer
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        tmp_reloaded = tl.load(out_ptr0 + (...), mask)
        ...
```

Triton does not analyse global read/write dependencies across loops on any
backend, and discharging that is left to whoever writes the kernel — see
triton-lang/triton#9887. When the kernel author is Inductor, that obligation
falls on Inductor's code generation, and here it is not being met.

### Evidence

Two kernels matching this pattern were identified by hash in the generated
output. Inserting a single `tl.debug_barrier()` before each reload loop, by
patching the Inductor-generated source post-hoc, is the only change between the
two arms below.

Denominator: the harness runs N iterations and the first sets the reference, so
it compares against itself and can never differ. 128 iterations gives 127
comparisons. Each arm is 127 comparisons × 8 GPUs = 1016.

| arm | compiled input gradients changed |
|---|---|
| unmodified source | 127 of 127 on 8 of 8 GPUs (1016 of 1016) |
| recompile of byte-identical source | 127 of 127 on 8 of 8 GPUs (1016 of 1016) |
| one `tl.debug_barrier()` before each reload loop | 0 on 8 of 8 (0 of 1016) |
| eager | 0 on 8 of 8 (0 of 1016) |
| graph break around the fused region | 0 on 8 of 8 (0 of 1016) |

The recompile arm is the control that matters: recompiling unchanged source stays
broken, so the clean arm is not a cache effect or an artifact of recompilation.
The barrier is the only difference between dirty and clean. Inductor and Triton
caches were redirected per arm and per GPU so that no arm could read another
arm's compiled binary.

Relative L2 drift on the affected gradient reaches ~5%. It fires at the same rate
on all eight GPUs on every replay — there is no device lottery here.

### What we have not established

- **We have not checked whether current `main` still emits this pattern.** Our
  measurements are all on a 2.11 build. A 2.12 build in a separate matrix did not
  reproduce, but we never identified a mechanism for that, so we cannot say
  whether it was fixed, or is merely latent there. If this was already addressed,
  pointing us at the change would be the most useful possible answer.
- **We have not measured the cost of the barrier.** We are not proposing an
  unconditional `tl.debug_barrier()` before every cross-loop scratch reload; that
  would be a throughput regression on every backend and we have no numbers for
  it.
- **We have not reproduced this on NVIDIA hardware.** We only have AMD parts. The
  hazard as written is backend-independent, but whether it fires elsewhere is an
  open question we cannot answer.
- **We have not confirmed the pattern appears in production models.** A static
  scan of every Triton kernel Inductor generated for a real transformer-style
  backward flagged 0 of 205 across two shapes, with the scanner qualified in the
  same job against the case that does reproduce. So this is a demonstrated defect
  in a demonstrated reproducer; we are not claiming a known blast radius.
- Single GPU per process, fp32, one shape. Nothing here says anything about
  multi-GPU or bf16.

### What we are asking

Mainly whether the barrier is the right fix at all. The alternative reading is
that the scheduler should not be fusing these two loops into one kernel while
routing the intermediate through global memory, in which case a barrier is
patching over a fusion decision. We would rather hear which one you consider
correct before anyone writes a patch. If it is the barrier, we are happy to do
the work on the codegen change and the benchmarking.

A self-contained reproducer is in progress. The case we measured is third-party
code that we are not able to publish. We can reproduce on request against
anything you would like us to try.


### Error logs

_No response_

### Versions

PyTorch: 2.11.0+rocm7.2
Triton: 3.6.0
HIP: 7.2.26015
GPU: AMD Instinct MI355X (gfx950), 8 per node
OS: Linux

Filing here rather than against the ROCm fork because the code generation path is common and the missing synchronization is in Inductor, not in a backend-specific component. Happy to be redirected.

cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Duplicate mapping

- Primary report: `github-pytorch-pytorch-190517`
- Shared cause: `cause-pytorch-190517-global-scratch-reload-lacks-cross-warp-barrier`

Issue 198562 describes the same two-loop global store/reload pattern and experimentally shows that inserting tl.debug_barrier changes 1,016 of 1,016 drifting comparisons to 0. A maintainer identifies PR 190519 as the fix, and the reporter confirms correctness after that change.

The sections below describe the shared cause reconstructed from the primary report. The reported behavior above remains the observation specific to this report.

## Trigger

A fused normalization kernel materializes partial values to a global output/scratch buffer and a later reduction loop reads those values back.

## Relevant entities

producer reduction loop; global scratch buffer; store warp; consumer reduction loop; load warp; invalidated-store set; barrier

## Implemented relation

A load in a later loop observes all earlier writes to the same global scratch range, including writes issued by other warps in the kernel.

## Pre-state

Different warps execute stores for different portions of the materialized intermediate.

## Binding transition

Code generation emits the later reload after the reduction-loop boundary without a cross-warp barrier, even though the buffer is recorded in cse.invalidated_stores.

## Post-state

A consumer warp can reach its load before another warp's corresponding store is visible.

## Validity obligation

A same-kernel global read-after-write edge that crosses warps is ordered by a barrier before the first reload.

## Invalid decision

The load path did not discharge the read-after-write synchronization obligation tracked by invalidated_stores.

## Physical effect

The second loop reads stale or partially written global scratch values and incorporates them into the reduction result.

## Observed consequence

The reported pit_b_224 XPU inference fails accuracy with RMSE around 0.4; the second report observes nondeterministic compiled gradients on AMD GPUs.

## Repair site

torch/_inductor/codegen/triton.py load code generation

## Repair

Commit 62a4f51 emits tl.debug_barrier before a non-CPU load when its buffer name is in cse.invalidated_stores and adds a compiled normalization regression that checks for the barrier and correct output.

## Why the repair works

All warps complete and publish the earlier global stores before any warp performs the tracked reload, restoring the required read-after-write order.

## Causal basis

The landed commit explicitly fixes report 190517, names the cross-warp read-after-write race, adds the barrier at the tracked reload, and reports that the failing model now passes.

## Analysis implication

Fused-kernel analysis needs buffer identity, store and load loop regions, warp mapping, invalidation state, memory space, barrier placement, and reload consumers together.

## Fix patches

- [`fixes/pytorch-198562-commit-62a4f51.patch`](fixes/pytorch-198562-commit-62a4f51.patch) — https://github.com/pytorch/pytorch/commit/62a4f51cf34e32503ae138c8f977b434bb90a325
