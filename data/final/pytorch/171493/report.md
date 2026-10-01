# incorrect fusion and re-ordering of an index_add

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/171493](https://github.com/pytorch/pytorch/issues/171493)
- Opened: 2025-12-30T01:46:11Z
- Closed: 2026-02-11T18:26:58Z
- Labels: `high priority`, `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `pt2: ubn`

## Reported behavior

### 🐛 Describe the bug

torch.compile gives incorrect output for the following example code:

```python
import torch

print('torch version:', torch.__version__)

class Example(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f, batch):

        # f are node features
        # batch is pyg style batch tensor

        # 1. some scaling
        f_u = f**2 + 0.00987654321

        # 2. reduce f_u to per batch
        n_batch = batch.max() + 1
        F_u_mol = torch.zeros((n_batch, f.shape[1]), device=f.device, dtype=f.dtype)
        F_u_mol.index_add_(0, batch, f_u)

        # uncomment this for correct output
        # torch._dynamo.graph_break()

        # expand back to per node so we can scale the node features
        # also add epsilon so we never divide by zero
        F_u_at_atom = F_u_mol[batch] + 1e-6

        # scale the node features by the graph features
        _f = f_u /F_u_at_atom

        return _f

device = 'cuda'

# Create a single graph with many nodes
f = torch.ones(1024, 1, device=device)
batch = torch.zeros(1024, dtype=torch.long, device=device)

# if we have more than one batch it will work fine
# uncomment this for correct output
# batch[-1] = 1

model = Example().to(device)
compiled_model = torch.compile(model)

# 1. Eager Mode (Correct Answer)
# F_u should be (1024 * (1^2 + 0.0098...)) approx 1034.11
# _f should be approx (1.0098 / 1034.11) = 0.00097
eager_out = model(f, batch)

# 2. Compiled Mode (Incorrect)
compiled_out = compiled_model(f, batch)


print(f"Eager Mean:    {eager_out.mean().item():.6f}")
print(f"Compiled Mean: {compiled_out.mean().item():.6f}")

# Check for discrepancy
diff = (eager_out - compiled_out).abs().max().item()
if diff > 1e-4:
    print(f"BUG REPRODUCED! Max Difference: {diff:.6f}")
else:
    print("Bug not triggered.")


```
Output:

```
torch version: 2.9.1+cu128
Eager Mean:    0.000977
Compiled Mean: 1009876.500000
BUG REPRODUCED! Max Difference: 1009876.500000
```


- Uncommenting the graph break gives correct code.
- It is also correct if there is more that 1 batch.
- It is only incorrect for the case of one batch.

Looking at the generated triton It looks like it has incorrectly re-ordered the code so that the index_add happens too late and the value that depends on it is being read before it is written:

```python
def triton_poi_fused_add_div_index_index_add_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp7 = tl.load(out_ptr0 + (0)) # this is f_u_mol being read before it is written
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK]) # undefined values broadcast
    tmp1 = tl.full([XBLOCK], 1, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 1)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 1")
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = (tmp6 / tmp10)
    tl.atomic_add(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp6, None, sem='relaxed')  # atomic add ends up being done here
    tl.store(out_ptr1 + (x0), tmp11, xmask)

```






### Error logs

_No response_

### Versions

PyTorch version: 2.9.1+cu128
Is debug build: False
CUDA used to build PyTorch: 12.8
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 12.3.0-1ubuntu1~22.04.2) 12.3.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.13.11 | packaged by conda-forge | (main, Dec  6 2025, 11:24:03) [GCC 14.3.0] (64-bit runtime)
Python platform: Linux-6.8.0-87-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.8.93
CUDA_MODULE_LOADING set to:
GPU models and configuration:
GPU 0: NVIDIA TITAN V


Nvidia driver version: 570.195.03
cuDNN version: Could not collect
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True
Caching allocator config: N/A


Versions of relevant libraries:
[pip3] numpy==2.4.0
[pip3] nvidia-cublas-cu12==12.8.4.1
[pip3] nvidia-cuda-cupti-cu12==12.8.90
[pip3] nvidia-cuda-nvrtc-cu12==12.8.93
[pip3] nvidia-cuda-runtime-cu12==12.8.90
[pip3] nvidia-cudnn-cu12==9.10.2.21
[pip3] nvidia-cufft-cu12==11.3.3.83
[pip3] nvidia-curand-cu12==10.3.9.90
[pip3] nvidia-cusolver-cu12==11.7.3.90
[pip3] nvidia-cusparse-cu12==12.5.8.93
[pip3] nvidia-cusparselt-cu12==0.7.1
[pip3] nvidia-nccl-cu12==2.27.5
[pip3] nvidia-nvjitlink-cu12==12.8.93
[pip3] nvidia-nvtx-cu12==12.8.90
[pip3] torch==2.9.1
[pip3] triton==3.5.1
[conda] numpy                       2.4.0            pypi_0                pypi
[conda] nvidia-cublas-cu12          12.8.4.1         pypi_0                pypi
[conda] nvidia-cuda-cupti-cu12      12.8.90          pypi_0                pypi
[conda] nvidia-cuda-nvrtc-cu12      12.8.93          pypi_0                pypi
[conda] nvidia-cuda-runtime-cu12    12.8.90          pypi_0                pypi
[conda] nvidia-cudnn-cu12           9.10.2.21        pypi_0                pypi
[conda] nvidia-cufft-cu12           11.3.3.83        pypi_0                pypi
[conda] nvidia-curand-cu12          10.3.9.90        pypi_0                pypi
[conda] nvidia-cusolver-cu12        11.7.3.90        pypi_0                pypi
[conda] nvidia-cusparse-cu12        12.5.8.93        pypi_0                pypi
[conda] nvidia-cusparselt-cu12      0.7.1            pypi_0                pypi
[conda] nvidia-nccl-cu12            2.27.5           pypi_0                pypi
[conda] nvidia-nvjitlink-cu12       12.8.93          pypi_0                pypi
[conda] nvidia-nvtx-cu12            12.8.90          pypi_0                pypi
[conda] torch                       2.9.1            pypi_0                pypi
[conda] triton                      3.5.1            pypi_0

cc @ezyang @gchanan @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @muchulee8 @amjames @aakhundov @coconutruben @jataylo

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

- [`fixes/pytorch-171493-commit-1687bbf.patch`](fixes/pytorch-171493-commit-1687bbf.patch) — https://github.com/pytorch/pytorch/commit/1687bbf6293f6886af4b0916dfaaa98d74272840
