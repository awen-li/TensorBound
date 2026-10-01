# Pipelined dot-input load appears before gdc_wait and produces incorrect results with PDL

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/11905](https://github.com/triton-lang/triton/issues/11905)
- Opened: 2026-09-22T09:13:38Z
- Closed: 2026-09-23T17:42:14Z
- Labels: `bug`

## Reported behavior

### Describe the bug

With Triton 3.7.1 on an H20 (SM90), a consumer kernel using `num_stages=2` produces incorrect results when `tl.extra.cuda.gdc_wait()` is inside its K loop, immediately before the dependent input load.

The generated TTGIR contains an asynchronous copy of the dependent input in the loop prologue, before the GDC wait inside the loop. Disabling PDL, using `num_stages=1`, or adding a GDC wait before the loop makes this standalone reproducer pass.

Is a GDC wait inside a software-pipelined loop expected to prevent this load from being scheduled before the wait? If this placement is unsupported, could you clarify the required pattern? Based on the documented semantics, I expected the dependent load to remain after the wait.

### Environment details

- GPU: NVIDIA H20-3e, compute capability 9.0
- NVIDIA driver: 580.105.08
- Triton: 3.7.1
- PyTorch: 2.13.0+cu132
- `torch.version.cuda`: 13.2
- Python: 3.12.14
- OS: Linux 5.10.134-16.3.al8.x86_64, x86_64
- Tested with the installed versions above; not yet tested against Triton main/nightly.

## Reproducer

Both kernels launch one CTA. The producer signals its dependent, briefly sleeps, and writes ones to X. The store depends on the inline-assembly result to keep it after the delay. The delay widens the overlap window; reproducibility on other devices is not guaranteed by the sleep duration.

Before each pair, X is reset to zero. The producer is launched with `launch_pdl=False` so this reset keeps its normal stream dependency. Only the consumer enables PDL. W is always one, so all output values must be exactly 64.

K is a runtime parameter so the K loop remains available for software pipelining. The source order inside that loop is:

```python
w = tl.load(wp)               # independent weights
tl.extra.cuda.gdc_wait()
x = tl.load(xp).to(tl.bfloat16)  # dependent producer output
acc += tl.dot(x, w)
```

Save the following as `repro_gdc_pipeline.py` and run:

```bash
python repro_gdc_pipeline.py
```

On the affected setup it prints all four controls, then exits with an AssertionError. It also writes TTGIR and PTX for every variant into `repro_gdc_asm/`.

```python
"""Standalone reproducer for GDC wait + software-pipelined dot input loads.

Requires an NVIDIA SM90+ GPU, PyTorch and Triton. No vLLM dependency.
Run: python repro_gdc_pipeline.py
"""
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def producer(X, SIZE: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    # Keep the producer alive after signaling. The store depends on the asm
    # result, so it cannot move before the delay.
    value = tl.inline_asm_elementwise(
        "nanosleep.u32 1000000; mov.u32 $0, 1;",
        constraints="=r", args=[], dtype=tl.int32, is_pure=False, pack=1,
    )
    i = tl.arange(0, SIZE)
    tl.store(X + i, value.to(tl.float32))


@triton.jit
def consumer(X, W, Y, K, N: tl.constexpr, WAIT_OUTSIDE: tl.constexpr):
    m = tl.arange(0, 64)
    n = tl.program_id(0) * 128 + tl.arange(0, 128)
    k = tl.arange(0, 32)
    xp = X + m[:, None] * K + k[None, :]
    wp = W + k[:, None] * N + n[None, :]
    acc = tl.zeros((64, 128), tl.float32)
    if WAIT_OUTSIDE:
        tl.extra.cuda.gdc_wait()
    for _ in range(tl.cdiv(K, 32)):
        w = tl.load(wp)
        tl.extra.cuda.gdc_wait()
        x = tl.load(xp).to(tl.bfloat16)
        acc += tl.dot(x, w)
        xp += 32
        wp += 32 * N
    tl.store(Y + m[:, None] * N + n[None, :], acc)


def run_case(pdl, stages, outside=False):
    M, N, K = 64, 128, 64
    x = torch.zeros((M, K), device="cuda", dtype=torch.float32)
    w = torch.ones((K, N), device="cuda", dtype=torch.bfloat16)
    y = torch.empty((M, N), device="cuda", dtype=torch.float32)

    def launch():
        x.zero_()
        # Normal producer launch preserves the dependency on x.zero_().
        producer[(1,)](x, M * K, num_warps=4, launch_pdl=False)
        return consumer[(N // 128,)](
            x, w, y, K, N, outside,
            num_warps=4, num_stages=stages, launch_pdl=pdl,
        )

    kernel = launch()
    torch.cuda.synchronize()
    stem = f"pdl{int(pdl)}_stages{stages}_outside{int(outside)}"
    output = Path(__file__).with_name("repro_gdc_asm")
    output.mkdir(exist_ok=True)
    for ext in ("ttgir", "ptx"):
        (output / f"{stem}.{ext}").write_text(kernel.asm[ext])

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    failures = 0
    worst = 0.0
    for _ in range(20):
        graph.replay()
        torch.cuda.synchronize()
        # All inputs are one after the producer: every output must equal K.
        error = (y - K).abs().max().item()
        failures += int(not torch.equal(y, torch.full_like(y, K)))
        worst = max(worst, error)
    print(f"{stem}: failed_replays={failures}/20 max_abs_error={worst}", flush=True)
    return failures


def main():
    print(f"torch={torch.__version__}, CUDA={torch.version.cuda}, "
          f"triton={triton.__version__}", flush=True)
    print(f"GPU={torch.cuda.get_device_name()}, "
          f"capability={torch.cuda.get_device_capability()}", flush=True)
    assert torch.cuda.get_device_capability()[0] >= 9, "Requires SM90+"
    assert run_case(False, 2) == 0, "Non-PDL control failed"
    assert run_case(True, 1) == 0, "Single-stage control failed"
    failures = run_case(True, 2)
    assert run_case(True, 2, outside=True) == 0, "Outside-wait control failed"
    assert failures == 0, (
        "Dependent loads appear to cross gdc_wait with num_stages=2: "
        f"{failures}/20 graph replays produced incorrect results"
    )


if __name__ == "__main__":
    main()
```

## Actual result

```text
torch=2.13.0+cu132, CUDA=13.2, triton=3.7.1
GPU=NVIDIA H20-3e, capability=(9, 0)
pdl0_stages2_outside0: failed_replays=0/20 max_abs_error=0.0
pdl1_stages1_outside0: failed_replays=0/20 max_abs_error=0.0
pdl1_stages2_outside0: failed_replays=20/20 max_abs_error=32.0
pdl1_stages2_outside1: failed_replays=0/20 max_abs_error=0.0
Traceback (most recent call last):
  File "/home/linitra24/vllm_my/tmp/lora_pdl/repro_gdc_pipeline.py", line 102, in <module>
    main()
  File "/home/linitra24/vllm_my/tmp/lora_pdl/repro_gdc_pipeline.py", line 95, in main
    assert failures == 0, (
           ^^^^^^^^^^^^^
AssertionError: Dependent loads appear to cross gdc_wait with num_stages=2: 20/20 graph replays produced incorrect results
```

The outside-wait control adds one wait before the loop and leaves the original in-loop wait in place.

Expected: all four variants produce exactly 64 in every output element.

## Generated-code evidence

In `pdl1_stages2_outside0.ttgir`, `%xp_22` is derived from the producer output pointer `%X`. Its asynchronous copy appears at line 72, before the `scf.for` at line 74. The GDC wait is inside the loop at line 85.

Abbreviated for readability:

```text
%xp_16 = tt.splat %X ...
...
%x_38 = ttg.async_copy_global_to_local %xp_22, %x_36 ...
...
%acc_40:8 = scf.for ... {
    ...
    %9 = tt.elementwise_inline_asm "griddepcontrol.wait; // dummy $0"
         {constraints = "=r", packed_element = 1 : i32, pure = false} -> i32
    ...
}
```

The PTX likewise contains dependent-input `cp.async.cg.shared.global` instructions at lines 123–135, before the `griddepcontrol.wait` at line 323. Full TTGIR, PTX and the execution log are attached.

This suggests the asynchronous input copy is moved ahead of the synchronization during software pipelining. I have not identified the responsible compiler pass or verified a compiler fix.

For this reproducer, both `num_stages=1` and an additional wait before the loop avoid the incorrect results.

Reference: [Triton GDC wait documentation](https://triton-lang.org/main/python-api/generated/triton.language.extra.cuda.gdc_wait.html).

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

Pipelined dot-input load appears before gdc_wait and produces incorrect results with PDL

## Observed implementation decision

The pipeliner hoists the dependent loop load into the prologue across the side-effecting gdc_wait inside the loop.

## Physical effect

Software pipelining starts an asynchronous copy from producer-owned X before the programmatic-dependency wait has established that the producer finished writing X.

## Observed consequence

The consumer observes stale zero values and produces outputs with maximum absolute error 32 in all affected graph replays.

## Reported proposal or workaround

Generated TTGIR and PTX place cp.async before griddepcontrol.wait; non-PDL, single-stage, and pre-loop-wait controls all produce the expected value while the affected configuration fails 20 of 20 replays.

## Repair trace

The report localizes a load hoisted across gdc_wait, but the responsible source transformation, corrective patch, and fixed-version execution are unavailable.

## Archived evidence

- Raw report: `data/raw/github/triton/11905/report.json`
- SHA-256: `76e6abefe057bfeb29c6c720f059b115ba6b2083f07b22fd13e5a63b4db2771e`
