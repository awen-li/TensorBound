# Inductor: `unsqueeze` lowering loses source-tensor strides when source has non-standard layout — produces fused Triton kernel that  reads OOB

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/186850](https://github.com/pytorch/pytorch/issues/186850)
- Opened: 2026-06-09T23:08:45Z
- Closed: 2026-06-09T23:51:34Z
- Labels: `triage review`, `module: crash`, `module: cuda`, `module: custom-operators`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `module: pt2-dispatcher`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

  When a `torch.library.custom_op` returns a tensor with non-standard strides, and a downstream chain of `unsqueeze → expand → contiguous → view → permute → view` ops fuses with another input plus a bias
  into a single Triton kernel, Inductor's lowering of that chain reads the source custom_op output with **stride multipliers that match a standard-contiguous layout, not the actual allocated strides**.

This will cause
 - **Silent wrong numerics:** `max_abs ≈ 9.0` between eager and compile at production shape (vs `fp16` ULP of `~5e-4`). 100% reproducible.
 - **`cudaErrorIllegalAddress`:** rare; ~1 in 20 trials when `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is *not* set (fragmented allocator increases the chance the racy read lands in an unmapped page)

  ### Direct evidence (from a real reproducing run)

  From the Inductor `call()` wrapper for the failing graph:

  ```python
  # Allocation honours register_fake's unusual strides:
  buf139 = empty_strided_cuda(
      (1, 2048, 120, 45, 80),
      (884736000, 120, 1, 19660800, 245760),   # T innermost, then C, W, H, B
      torch.float16,
  )
  ```

  FX annotations recorded by Inductor in the same wrapper file:

  ```
  %add_27      : "f16[1, 2048, 120, 45, 80][884736000, 120,    1,    19660800, 245760]"
  %unsqueeze_2 : "f16[1, 2048, 1, 120, 45, 80][884736000, 432000, 432000, 3600, 80, 1]"
                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                standard-contiguous; doesn't match add_27
  ```

  `unsqueeze` should be metadata-only — it should preserve all existing strides. But the post-`unsqueeze` strides are computed as `(2048*1*120*45*80, 1*120*45*80, 1*120*45*80, 45*80, 80, 1)`, ignoring `add_27`'s actual stride pattern.

  `cuda-gdb` on the captured coredump pinpoints the faulting load:

  ```
  CUDA Exception: Warp Illegal Address
  PC: 0xffe325e3b800
  Kernel: triton_poi_fused__to_copy__unsafe_view_add_clone_convolution_expand_permute_unsqueeze_view_15
  Source: <kernel>.py:28
    tmp0 = tl.load(in_ptr0 + (80*(x1 // 2) + 3648*x2 + 437760*(((x0 % 2)) // 2)
                            + 437760*(x1 % 2) + 875520*x3 + (x0 // 2)), ...)
  ```

### A minimal repro script
```
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch


# Custom op that returns a tensor with a non-standard stride layout
# (T-axis innermost, B outermost). Inductor sees this stride from
# register_fake and is supposed to honour it when lowering downstream ops.
@torch.library.custom_op(
    "reprolib::opaque_strided",
    mutates_args=(),
)
def opaque_strided(x: torch.Tensor) -> torch.Tensor:
    B, C, T, H, W = x.shape
    # Strides: (B=C*T*H*W, C=T, T=1, H=C*T*W, W=C*T) — T innermost.
    strides = (C * T * H * W, T, 1, C * T * W, C * T)
    out = torch.empty_strided((B, C, T, H, W), strides, dtype=x.dtype, device=x.device)
    out.copy_(x)  # logical values match x; physical layout differs.
    return out

@opaque_strided.register_fake
def _opaque_strided_fake(x: torch.Tensor) -> torch.Tensor:
    B, C, T, H, W = x.shape
    strides = (C * T * H * W, T, 1, C * T * W, C * T)
    return torch.empty_strided((B, C, T, H, W), strides, dtype=x.dtype, device=x.device)


def pixel_shuffle_add(x, conv_out, bias):
    """Mirror the buggy chain: pixel-shuffle x from (B, 2C, T, H, W) to
    (B, C, T, 2H, 2W) via the unsqueeze→expand→contiguous→reshape chain;
    pixel-shuffle conv_out from (B, 4C, T, H, W) to (B, C, T, 2H, 2W) via
    the simpler reshape→permute chain; sum both plus a per-channel bias.

    Inductor fuses ALL THREE reads into one Triton kernel, and that fused
    kernel reads x with stride multipliers that don't match x's actual
    (non-standard) strides.
    """
    B, C2, T, H, W = x.shape
    C = C2 // 2

    # Pixel-shuffle x: (B, 2C, T, H, W) → (B, C, T, 2H, 2W) via the
    # unsqueeze → expand → clone → view → permute → view idiom.
    x_unsq = x.unsqueeze(2)  # (B, 2C, 1, T, H, W)
    x_exp = x_unsq.expand(B, 2 * C, 2, T, H, W)  # broadcast new dim
    x_cl = x_exp.contiguous()  # materialise (1.77 GB at prod)
    x_v1 = x_cl.reshape(B, 4 * C, T, H, W)
    x_v2 = x_v1.reshape(B, C, 2, 2, T, H, W)
    x_perm = x_v2.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
    x_out = x_perm.reshape(B, C, T, 2 * H, 2 * W)

    # Pixel-shuffle conv_out: (B, 4C, T, H, W) → (B, C, T, 2H, 2W).
    co_v = conv_out.reshape(B, C, 2, 2, T, H, W)
    co_perm = co_v.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
    co_out = co_perm.reshape(B, C, T, 2 * H, 2 * W)

    return x_out + co_out + bias.view(1, C, 1, 1, 1)

def chain(latent: torch.Tensor) -> torch.Tensor:
    # 1. Opaque op produces a non-standard-strided tensor.
    add_27 = torch.ops.reprolib.opaque_strided(latent)

    # 2. A second, normally-strided tensor (in the real graph this is the
    #    output of a conv3d that expanded channels 4×).
    B, C2, T, H, W = add_27.shape
    C = C2 // 2
    conv_out = torch.empty(
        (B, 4 * C, T, H, W), dtype=add_27.dtype, device=add_27.device
    )
    conv_out.normal_()

    # 3. Bias broadcast over (1, C, 1, 1, 1).
    bias = torch.zeros(C, dtype=add_27.dtype, device=add_27.device)

    return pixel_shuffle_add(add_27, conv_out, bias)

def main():
    # Production shape that triggers consistently. Smaller shapes may not.
    SHAPE = (1, 2048, 120, 45, 80)
    print(f"shape={SHAPE}", flush=True)
    x = torch.randn(*SHAPE, dtype=torch.float16, device="cuda")

    print("--- eager ---", flush=True)
    eager_out = chain(x).detach()
    print(
        f"eager NaN={eager_out.isnan().sum().item()} "
        f"range=[{eager_out.float().min():.4f}, {eager_out.float().max():.4f}]"
    )

    print("--- compiled ---", flush=True)
    torch._dynamo.reset()
    compiled_chain = torch.compile(chain, mode="default")
    compiled_out = compiled_chain(x).detach()
    torch.cuda.synchronize()
    print(
        f"compiled NaN={compiled_out.isnan().sum().item()} "
        f"range=[{compiled_out.float().min():.4f}, {compiled_out.float().max():.4f}]"
    )

    diff = (eager_out.float() - compiled_out.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"diff: max_abs={max_abs:.4f} mean_abs={mean_abs:.6f}")
    # fp16 ULP at ~1.0 is ~5e-4. Anything above ~0.01 is the bug.
    assert max_abs < 0.01, (
        f"INDUCTOR STRIDE BUG: eager vs compiled diverge by {max_abs:.2f} "
        f"(expected < 0.01 for fp16 ULP)."
    )


if __name__ == "__main__":
    main()
```

### Versions

```
  GNU nano 5.6.1                                                                                                                                                                                                                  /home/jianweif/tmp.log
<frozen runpy>:128: RuntimeWarning: 'torch.utils.collect_env' found in sys.modules after import of package 'torch.utils', but prior to execution of 'torch.utils.collect_env'; this may result in unpredictable behaviour
Collecting environment information...
PyTorch version: 2.11.0+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCM used to build PyTorch: N/A

OS: CentOS Stream 9 (aarch64)
GCC version: (GCC) 11.5.0 20240719 (Red Hat 11.5.0-14)
Clang version: 22.1.3 (CentOS 22.1.3-1.el9)
CMake version: version 3.31.8
Libc version: glibc-2.40

Python version: 3.12.13+meta (3.12:3bb231a, Mar 03 2026, 12:38:32) [Clang 19.1.2 (ssh://git.vip.facebook.com/data/gitrepos/osmeta/external/llvm-pro (64-bit runtime)
Python platform: Linux-6.16.1-0_fbk1_rc7_0_gbe6a4d97f720-aarch64-with-glibc2.40
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to:
GPU models and configuration:
GPU 0: NVIDIA GB200
GPU 1: NVIDIA GB200

Nvidia driver version: 580.105.08
cuDNN version: Could not collect
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True
Caching allocator config: N/A

CPU:
Architecture:                            aarch64
CPU op-mode(s):                          64-bit
Byte Order:                              Little Endian
CPU(s):                                  144
On-line CPU(s) list:                     0-143
Vendor ID:                               ARM
Model name:                              Neoverse-V2
Model:                                   0
Thread(s) per core:                      1
Core(s) per socket:                      72
Socket(s):                               2
Stepping:                                r0p0
Frequency boost:                         disabled
CPU(s) scaling MHz:                      100%
CPU max MHz:                             3402.0000
CPU min MHz:                             81.0000
BogoMIPS:                                2000.00
Flags:                                   fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm sb dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh
L1d cache:                               9 MiB (144 instances)
L1i cache:                               9 MiB (144 instances)
L2 cache:                                144 MiB (144 instances)
L3 cache:                                228 MiB (2 instances)
NUMA node(s):                            18
NUMA node0 CPU(s):                       0-71
NUMA node1 CPU(s):                       72-143
NUMA node2 CPU(s):
NUMA node3 CPU(s):
NUMA node4 CPU(s):
NUMA node5 CPU(s):
NUMA node6 CPU(s):
NUMA node7 CPU(s):
NUMA node8 CPU(s):
NUMA node9 CPU(s):
NUMA node10 CPU(s):
NUMA node11 CPU(s):
NUMA node12 CPU(s):
NUMA node13 CPU(s):
NUMA node14 CPU(s):
NUMA node15 CPU(s):
NUMA node16 CPU(s):
NUMA node17 CPU(s):
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Old microcode:             Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; __user pointer sanitization
Vulnerability Spectre v2:                Vulnerable: Unprivileged eBPF enabled
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Not affected

Versions of relevant libraries:
[pip3] Could not collect
[conda] Could not collect

```

cc @ptrblck @msaroufim @eqy @tinglvv @nWEIdia @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo @bdhirsh @bobrenjc93 @aorenste

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

A custom op returns an empty_strided tensor and a fused unsqueeze, expand, materialization, reshape, and permutation chain consumes it.

## Observed implementation decision

The report shows the unsqueeze result carrying standard-contiguous strides instead of the custom op output's physical strides.

## Physical effect

The fused Triton kernel addresses the source with contiguous-layout multipliers and can read outside the intended physical range.

## Observed consequence

The reporter observes large numerical divergence and intermittent illegal-address failures.

## Reported proposal or workaround

The related report 189401 was repaired by preserving a raw ReinterpretView passed directly to view lowering.

## Repair trace

Commit f2a2c90 changes only the raw-IR-node branch of view lowering; ordinary FX unsqueeze receives a TensorBox, for which the old and new implementations both unwrap .data. The shared symptom therefore does not establish that report 189401's repair corrects this custom-op path, and report 186850 has no separate landed repair or fixed regression.

## Archived evidence

- Raw report: `data/raw/github/pytorch/186850/report.json`
- SHA-256: `0d51ff1b7cbbef70555d03735d30d5514cd7cca859852ca7c942e20ba452520c`
