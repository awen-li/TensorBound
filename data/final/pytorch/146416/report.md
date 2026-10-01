# Silent correctness bug in Inductor when fusing transpose into other ops

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/146416](https://github.com/pytorch/pytorch/issues/146416)
- Opened: 2025-02-04T16:17:38Z
- Closed: 2025-02-10T19:15:54Z
- Labels: `high priority`, `triage review`, `oncall: distributed`, `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`

## Reported behavior

### 🐛 Describe the bug

```py
import os

import torch
import torch.distributed._functional_collectives as funcol

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "2743"
torch.distributed.init_process_group(backend="nccl")


def scale(t):
    scale = torch.finfo(torch.float8_e4m3fn).max / t.abs().amax(dim=-1, keepdim=True).float()
    t = t.mul(scale).to(torch.float8_e4m3fn)
    return t, scale


def fp8_rowwise_backward(in_, w, out_grad):
    out_grad_fp8, scale_out_grad = scale(out_grad)
    w_fp8, scale_w = scale(w.t().contiguous())
    out_grad_fp8 = funcol.all_gather_tensor(
        out_grad_fp8, gather_dim=0, group=torch.distributed.group.WORLD
    )
    scale_out_grad = funcol.all_gather_tensor(
        scale_out_grad, gather_dim=0, group=torch.distributed.group.WORLD
    )
    in_grad = torch._scaled_mm(
        out_grad_fp8, w_fp8.t(), scale_a=scale_out_grad, scale_b=scale_w.t(), out_dtype=torch.bfloat16
    )

    out_grad = funcol.all_gather_tensor(
        out_grad.t().contiguous(), gather_dim=0, group=torch.distributed.group.WORLD
    )
    w_grad = out_grad @ in_

    return in_grad, w_grad


in_ = torch.randn((3072, 4096), device="cuda", dtype=torch.bfloat16)
w = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
out_grad = torch.randn((3072, 4096), device="cuda", dtype=torch.bfloat16)

eager_in_grad, eager_w_grad = fp8_rowwise_backward(in_, w, out_grad)
compile_in_grad, compile_w_grad = torch.compile(fp8_rowwise_backward)(in_, w, out_grad)

assert torch.testing.assert_close(compile_w_grad, eager_w_grad)
```
The issue seems to come from Inductor trying to fuse the `scale(out_grad)` operation (which is a row-wise reduction + pointwises) with the `out_grad.t().contiguous()` step, probably in order to load `out_grad` only once. However, these ops aren't easy to fuse (one works on rows, the other one on blocks), and indeed Inductor ends up messing up the transposition. The result it produces for the transposition is in fact the same storage "reinterpreted" as a transposed shape.

### Error logs

```
AssertionError: Tensor-likes are not close!

Mismatched elements: 16690833 / 16777216 (99.5%)
Greatest absolute difference: 422.0 at index (2854, 3714) (up to 1e-05 allowed)
Greatest relative difference: 15335424.0 at index (803, 2794) (up to 0.016 allowed)
```

### Versions

PyTorch nightly `2.7.0.dev20250120+cu126`

cc @ezyang @gchanan @zou3519 @kadeng @msaroufim @H-Huang @awgu @kwen2501 @wanchaol @fegin @fduwjj @wz337 @wconstab @d4l3k @c-p-i-o @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @yf225 @chenyang78 @muchulee8 @amjames @desertfire @aakhundov

## Trigger

The reported backward path sends out_grad.t().contiguous() through all_gather_tensor before multiplying the gathered result by the input.

## Relevant entities

FX tensor metadata; transpose; contiguous result; Inductor layout; functional all-gather input; native collective; gathered tensor

## Implemented relation

The tensor entering a functional collective has the physical strides required by its FX value, and the native collective consumes that same contiguous element order.

## Pre-state

The FX graph represents a transposed tensor followed by contiguous materialization, defining a specific row-major logical order for the collective input.

## Binding transition

The old functional-collective lowering registers no FX-stride layout constraint, allowing fusion and layout selection to pass a differently strided physical buffer to the collective.

## Post-state

The collective interprets the buffer as contiguous even though its physical order does not implement the FX value's required strides.

## Validity obligation

Compiler lowering enforces the FX stride contract at collective boundaries, and the native operation rejects inputs that do not satisfy its contiguous-memory requirement.

## Invalid decision

Functional collectives were treated as layout-polymorphic even though their native implementations consumed contiguous storage order.

## Physical effect

The all-gather output contains the same storage values in the wrong transposed order, so the following matrix multiplication pairs incorrect elements.

## Observed consequence

Compiled w_grad differs at 16,690,833 of 16,777,216 elements, with a maximum absolute error of 422 in the report.

## Repair site

torch/_inductor/comm_lowering.py collective registration and torch/csrc/distributed/c10d/Functional.cpp

## Repair

Commit 97f6480 introduces register_comm_lowering, attaches constrain_to_fx_strides to functional collective lowerings, adds native contiguity checks, and includes the report's FP8 rowwise-backward all-gather regression.

## Why the repair works

Inductor now materializes the input in the element order encoded by FX before the collective, while the native checks prevent a future incompatible layout from being silently consumed.

## Causal basis

The landed commit explicitly fixes report 146416, identifies missing FX-stride forcing for functional collectives, adds the reported computation as a regression, and enforces the same contract in both compiler and native code.

## Analysis implication

Collective-boundary analysis needs logical shape, FX strides, compiler-selected strides, contiguity, collective contract, storage order, and downstream interpretation together.

## Fix patches

- [`fixes/pytorch-146416-commit-97f6480.patch`](fixes/pytorch-146416-commit-97f6480.patch) — https://github.com/pytorch/pytorch/commit/97f6480cf55e72965269437df708fc4a1ca1e387
