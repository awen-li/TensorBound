# Torch Inductor SearchSorted SliceView Bug

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/182343](https://github.com/pytorch/pytorch/issues/182343)
- Opened: 2026-05-04T20:33:39Z
- Closed: 2026-06-04T17:51:20Z
- Labels: `module: crash`, `triaged`, `module: correctness (silent)`, `module: sorting and selection`, `oncall: pt2`, `module: inductor`, `bot-triaged`, `repro:agent:h100:repro_success:reproduced`

## Reported behavior

### 🐛 Describe the bug

**Title:** `torch.searchsorted` produces wrong results or crashes when `sorted_sequence` is a slice

**Description:**

`torch.searchsorted(sorted_sequence[1:], values)` fails under `torch.compile` and AOTI due to two bugs in Inductor's `_boundaries_helper` (`torch/_inductor/lowering.py`).

**Bug 1 — Crash (dynamic slice):**
When the sliced tensor is dynamic (e.g., `cumsum(...)[1:]`), it survives as a `SliceView` IR node. `_boundaries_helper` calls `tb.get_stride()`, but `SliceView` inherits from `GenericView` which does not implement `get_stride()`, raising `NotImplementedError`.

**Bug 2 — Silent wrong result (constant slice):**
When the sliced tensor is constant (e.g., `torch.tensor([0, 3, 7, 12])[1:]`), Inductor realizes it as a `ReinterpretView` on the original buffer with a non-zero layout offset. `_boundaries_helper` extracts the buffer name via `get_name()` but never accounts for the layout offset. The generated Triton `bucketize_binary_search` kernel binary-searches from position 0 of the original buffer, producing off-by-one results.

```python
import torch

@torch.compile
def f(offsets, values):
    return torch.searchsorted(offsets[1:], values, right=True)

offsets = torch.tensor([0, 3, 7, 12], device="cuda")
values = torch.arange(12, device="cuda")

print(f(offsets, values))
# Compiled: [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]  (WRONG, off by 1)
# Expected: [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]
```

**Reproduce:**

```python
from __future__ import annotations

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase


class TestSearchSortedSlice(TestCase):
    """Regression tests for searchsorted on sliced sorted_sequence."""

    device = "cuda"

    def _check_compile(
        self,
        model: nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> None:
        model.eval()
        with torch.no_grad():
            eager_result = model(*args)
            compiled = torch.compile(model, backend="inductor")
            compiled_result = compiled(*args)
        self.assertEqual(eager_result, compiled_result)

    def _check_aoti(
        self,
        model: nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> None:
        import tempfile

        model.eval()
        with torch.no_grad():
            eager_result = model(*args)
        ep = torch.export.export(model, args, strict=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            pt2_path = torch._inductor.aoti_compile_and_package(
                ep, package_path=f"{tmpdir}/model.pt2"
            )
            engine = torch._inductor.aoti_load_package(pt2_path)
            aoti_result = engine(*args)
        self.assertEqual(eager_result, aoti_result)

    # -- Bug 2: constant slice -> silent wrong result --

    def test_searchsorted_constant_slice_compile(self) -> None:
        class Model(nn.Module):
            def forward(
                self, offsets: torch.Tensor, positions: torch.Tensor
            ) -> torch.Tensor:
                return torch.searchsorted(offsets[1:], positions, right=True)

        offsets = torch.tensor([0, 3, 7, 12], device=self.device)
        positions = torch.arange(12, device=self.device)
        self._check_compile(Model().to(self.device), (offsets, positions))

    def test_searchsorted_constant_slice_aoti(self) -> None:
        class Model(nn.Module):
            def forward(
                self, offsets: torch.Tensor, positions: torch.Tensor
            ) -> torch.Tensor:
                return torch.searchsorted(offsets[1:], positions, right=True)

        offsets = torch.tensor([0, 3, 7, 12], device=self.device)
        positions = torch.arange(12, device=self.device)
        self._check_aoti(Model().to(self.device), (offsets, positions))

    def test_searchsorted_constant_slice_from_end_compile(self) -> None:
        class Model(nn.Module):
            def forward(
                self, sorted_ids: torch.Tensor, query: torch.Tensor
            ) -> torch.Tensor:
                return torch.searchsorted(sorted_ids[:-1], query)

        sorted_ids = torch.tensor([10, 20, 30, 40, 50], device=self.device)
        query = torch.tensor([15, 25, 35], device=self.device)
        self._check_compile(Model().to(self.device), (sorted_ids, query))

    # -- Bug 1: dynamic slice -> SliceView crash --

    def test_searchsorted_dynamic_slice_compile(self) -> None:
        class Model(nn.Module):
            def forward(
                self, lengths: torch.Tensor, positions: torch.Tensor
            ) -> torch.Tensor:
                offsets = torch.zeros(
                    lengths.shape[0] + 1,
                    dtype=lengths.dtype,
                    device=lengths.device,
                )
                offsets[1:] = torch.cumsum(lengths, dim=0)
                return torch.searchsorted(offsets[1:], positions, right=True)

        lengths = torch.tensor([3, 4, 5], device=self.device)
        positions = torch.arange(12, device=self.device)
        self._check_compile(Model().to(self.device), (lengths, positions))

    def test_searchsorted_dynamic_slice_aoti(self) -> None:
        class Model(nn.Module):
            def forward(
                self, lengths: torch.Tensor, positions: torch.Tensor
            ) -> torch.Tensor:
                offsets = torch.zeros(
                    lengths.shape[0] + 1,
                    dtype=lengths.dtype,
                    device=lengths.device,
                )
                offsets[1:] = torch.cumsum(lengths, dim=0)
                return torch.searchsorted(offsets[1:], positions, right=True)

        lengths = torch.tensor([3, 4, 5], device=self.device)
        positions = torch.arange(12, device=self.device)
        self._check_aoti(Model().to(self.device), (lengths, positions))

    # -- Verify non-sliced inputs still work (no regression) --

    def test_searchsorted_no_slice_compile(self) -> None:
        class Model(nn.Module):
            def forward(
                self, boundaries: torch.Tensor, values: torch.Tensor
            ) -> torch.Tensor:
                return torch.searchsorted(boundaries, values, right=True)

        boundaries = torch.tensor([3, 7, 12], device=self.device)
        values = torch.arange(12, device=self.device)
        self._check_compile(Model().to(self.device), (boundaries, values))

    # -- Multi-dimensional sorted_sequence with slice --

    def test_searchsorted_2d_slice_compile(self) -> None:
        class Model(nn.Module):
            def forward(
                self, boundaries: torch.Tensor, values: torch.Tensor
            ) -> torch.Tensor:
                return torch.searchsorted(boundaries[:, 1:], values)

        boundaries = torch.tensor(
            [[0, 2, 5, 9], [0, 3, 6, 10]], device=self.device
        )
        values = torch.tensor([[1, 4, 7], [2, 5, 8]], device=self.device)
        self._check_compile(Model().to(self.device), (boundaries, values))

    # -- Step slices --

    def test_searchsorted_step_slice_compile(self) -> None:
        class Model(nn.Module):
            def forward(
                self, boundaries: torch.Tensor, values: torch.Tensor
            ) -> torch.Tensor:
                return torch.searchsorted(boundaries[::2], values)

        boundaries = torch.tensor(
            [1, 3, 5, 7, 9, 11], device=self.device
        )
        values = torch.tensor([2, 6, 10], device=self.device)
        self._check_compile(Model().to(self.device), (boundaries, values))

    # -- Jagged offset pattern (the real-world use case) --

    def test_searchsorted_jagged_batch_index_compile(self) -> None:
        class JaggedBatchIndex(nn.Module):
            def forward(
                self, offsets: torch.Tensor, values: torch.Tensor
            ) -> torch.Tensor:
                batch_idx = torch.searchsorted(
                    offsets[1:], values, right=True
                )
                local_pos = values - offsets[batch_idx]
                return batch_idx * 1000 + local_pos

        offsets = torch.tensor([0, 3, 7, 12], device=self.device)
        values = torch.arange(12, device=self.device)
        self._check_compile(JaggedBatchIndex().to(self.device), (offsets, values))

    def test_searchsorted_jagged_batch_index_aoti(self) -> None:
        class JaggedBatchIndex(nn.Module):
            def forward(
                self, offsets: torch.Tensor, values: torch.Tensor
            ) -> torch.Tensor:
                batch_idx = torch.searchsorted(
                    offsets[1:], values, right=True
                )
                local_pos = values - offsets[batch_idx]
                return batch_idx * 1000 + local_pos

        offsets = torch.tensor([0, 3, 7, 12], device=self.device)
        values = torch.arange(12, device=self.device)
        self._check_aoti(JaggedBatchIndex().to(self.device), (offsets, values))


if __name__ == "__main__":
    run_tests()
```

**Root cause:**
`_boundaries_helper` assumes `tb` is a realized contiguous buffer. When `tb` is a view (from slicing), `get_name()` returns the original pre-slice buffer and the layout offset is ignored.

**Fix:**
Clone view tensors into contiguous buffers before extracting boundary info, so `get_name()` and `get_stride()` refer to the actual sliced data:

```python
def _clone_if_view(tb):
    if isinstance(tb.data, BaseView):
        tb = clone(tb)
        tb.realize()
    return tb
```

Applied in both `_boundaries_helper` and the `searchsorted` lowering after `sorted_sequence.realize()`.

**Affects:** `torch.compile`, AOTI, and `torch.bucketize` (same lowering path).


### Versions

PyTorch version: N/A
Is debug build: N/A
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.12 (main, Mar  3 2026, 11:56:32) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-1044-azure-x86_64-with-glibc2.35
Is CUDA available: N/A
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: Could not collect
Nvidia driver version: Could not collect
cuDNN version: Could not collect
Is XPU available: N/A
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: N/A
Caching allocator config: N/A

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        48 bits physical, 48 bits virtual
Byte Order:                           Little Endian
CPU(s):                               4
On-line CPU(s) list:                  0-3
Vendor ID:                            AuthenticAMD
Model name:                           AMD EPYC 7763 64-Core Processor
CPU family:                           25
Model:                                1
Thread(s) per core:                   2
Core(s) per socket:                   2
Socket(s):                            1
Stepping:                             1
BogoMIPS:                             4890.86
Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves user_shstk clzero xsaveerptr rdpru arat npt nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload umip vaes vpclmulqdq rdpid fsrm
Virtualization:                       AMD-V
Hypervisor vendor:                    Microsoft
Virtualization type:                  full
L1d cache:                            64 KiB (2 instances)
L1i cache:                            64 KiB (2 instances)
L2 cache:                             1 MiB (2 instances)
L3 cache:                             32 MiB (1 instance)
NUMA node(s):                         1
NUMA node0 CPU(s):                    0-3
Vulnerability Gather data sampling:   Not affected
Vulnerability Itlb multihit:          Not affected
Vulnerability L1tf:                   Not affected
Vulnerability Mds:                    Not affected
Vulnerability Meltdown:               Not affected
Vulnerability Mmio stale data:        Not affected
Vulnerability Reg file data sampling: Not affected
Vulnerability Retbleed:               Not affected
Vulnerability Spec rstack overflow:   Vulnerable: Safe RET, no microcode
Vulnerability Spec store bypass:      Vulnerable
Vulnerability Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:             Mitigation; Retpolines; STIBP disabled; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                  Not affected
Vulnerability Tsx async abort:        Not affected
Vulnerability Vmscape:                Not affected

Versions of relevant libraries:
[pip3] No relevant packages
[conda] Could not collect

cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Trigger

searchsorted receives a sliced boundary tensor, including a dynamic SliceView or a realized ReinterpretView with a nonzero storage offset.

## Relevant entities

logical boundary tensor; SliceView; ReinterpretView; named storage buffer; layout offset; strides; bucketize binary-search indices

## Implemented relation

Each logical boundary index is bound to the storage address formed from the realized buffer identity, layout offset, and per-dimension strides of the lookup view.

## Pre-state

The slice carries its logical extent and either an unrealized view transform or a realized strided layout with a nonzero offset.

## Binding transition

The old lowering calls realize on the existing TensorBox, then obtains a buffer name and stride while passing zero as the lookup base index.

## Post-state

A dynamic SliceView can remain without the required strided interface, while a ReinterpretView names its backing buffer but loses the slice offset at the binary-search call.

## Validity obligation

Before code generation, the lookup is represented as a named strided tensor and every binary-search base index includes that tensor's layout offset and outer-dimensional stride contribution.

## Invalid decision

The lowering treated a logical lookup view as though its storage began at index zero in a directly queryable named buffer.

## Physical effect

One path raises NotImplementedError while requesting a SliceView stride; the other searches the unsliced prefix of the backing buffer and shifts every returned insertion position.

## Observed consequence

Compiled searchsorted either crashes for dynamic slices or returns values one position above the eager result for offsets[1:].

## Repair site

torch/_inductor/lowering.py::_realize_bucketize_lookup_tensor, _bucketize_indices, searchsorted, and bucketize

## Repair

Commit f1987e1 realizes lookup tensors through ExternKernel.realize_input, computes binary-search indices from layout.offset and strides, uses the layout storage span for masking, and adds sliced, offset, multidimensional, sorter, zero-stride, and dynamic-shape regressions.

## Why the repair works

The generated helper now receives a concrete strided representation and starts each search at the physical location that represents logical element zero of the slice.

## Causal basis

A report comment identifies PR 184043 as the fix; its landed commit preserves lookup-view offsets and adds searchsorted regressions for sliced computed boundaries, including nonzero offsets and dynamic shapes.

## Analysis implication

View-sensitive lowering needs logical extent, realized buffer identity, layout offset, strides, storage span, and the generated helper's base index together.

## Fix patches

- [`fixes/pytorch-182343-commit-f1987e1.patch`](fixes/pytorch-182343-commit-f1987e1.patch) — https://github.com/pytorch/pytorch/commit/f1987e19015dd85d698eda2c188ba1c8aefabfc5
