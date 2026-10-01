# Inductor doesn't support tensor.view(dtype).copy_(...)

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/151156](https://github.com/pytorch/pytorch/issues/151156)
- Opened: 2025-04-12T04:33:07Z
- Closed: 2025-04-21T08:02:38Z
- Labels: `high priority`, `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`

## Reported behavior

### 🐛 Describe the bug

```python
import os
import torch

from torch import Tensor

os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
os.environ["TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"] = "1"
os.environ["TORCHINDUCTOR_BENCHMARK_KERNEL"] = "1"


@torch.compile
def view_copy(target: Tensor, source: Tensor):
    assert target.dtype == torch.bfloat16
    assert source.dtype == torch.uint16
    target.view(torch.uint16).copy_(source)


target: Tensor = torch.ones(65536 * 1024, dtype=torch.bfloat16, device="cuda")
source = torch.full_like(target, 4, dtype=torch.uint16)

target.view(torch.uint16).copy_(source)
print(target[0])  # 3.6734e-40
view_copy(target, source)
print(target[0])  # 4.
```

### Error logs

The generated triton code is wrong
```python
@triton_heuristics.pointwise(
    size_hints={'x': 67108864},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*u16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'kernel_num_gb': 0.268435456},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32, bitcast=False)
    tl.store(out_ptr0 + (x0), tmp1, None)
```

Full code:
```python
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/5m/c5mmszhvnmnuimm6a3j7emw2wh7vx6mwt6uar6eqk5mygz5jqgw4.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg0_1, %view_2), kwargs = {})
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid

@triton_heuristics.pointwise(
    size_hints={'x': 67108864},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*u16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'kernel_num_gb': 0.268435456},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32, bitcast=False)
    tl.store(out_ptr0 + (x0), tmp1, None)


def get_args():
    arg_0 = rand_strided((67108864,), (1,), device='cuda:0', dtype=torch.uint16)
    arg_1 = rand_strided((67108864,), (1,), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(*args, 67108864, grid=grid(67108864), stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_0.benchmark_all_configs(*args, 67108864, grid=grid(67108864))


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.268435456
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (67108864, ), (1, ))
    assert_size_stride(arg1_1, (67108864, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(arg1_1, arg0_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg0_1
        del arg1_1
    return ()


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((67108864, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((67108864, ), (1, ), device='cuda:0', dtype=torch.uint16)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

```

### Versions

Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.12.9 (main, Mar 11 2025, 17:26:57) [Clang 20.1.0 ] (64-bit runtime)
Python platform: Linux-5.4.250-2-velinux1u1-amd64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA H100 80GB HBM3
GPU 1: NVIDIA H100 80GB HBM3
GPU 2: NVIDIA H100 80GB HBM3
GPU 3: NVIDIA H100 80GB HBM3

Nvidia driver version: 535.129.03
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Address sizes:                   52 bits physical, 57 bits virtual
Byte Order:                      Little Endian
CPU(s):                          168
On-line CPU(s) list:             0-74
Off-line CPU(s) list:            75-167
Vendor ID:                       GenuineIntel
Model name:                      Intel(R) Xeon(R) Platinum 8457C
CPU family:                      6
Model:                           143
Thread(s) per core:              2
Core(s) per socket:              42
Socket(s):                       2
Stepping:                        8
BogoMIPS:                        5199.99
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch cpuid_fault invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512_bf16 wbnoinvd arat avx512vbmi umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid cldemote movdiri movdir64b md_clear arch_capabilities
Hypervisor vendor:               KVM
Virtualization type:             full
L1d cache:                       3.9 MiB (84 instances)
L1i cache:                       2.6 MiB (84 instances)
L2 cache:                        168 MiB (84 instances)
L3 cache:                        195 MiB (2 instances)
NUMA node(s):                    2
NUMA node0 CPU(s):               0-83
NUMA node1 CPU(s):               84-167
Vulnerability Itlb multihit:     Not affected
Vulnerability L1tf:              Not affected
Vulnerability Mds:               Not affected
Vulnerability Meltdown:          Not affected
Vulnerability Mmio stale data:   Unknown: No mitigations
Vulnerability Retbleed:          Not affected
Vulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:        Mitigation; Enhanced IBRS, IBPB conditional, RSB filling, PBRSB-eIBRS SW sequence
Vulnerability Srbds:             Not affected
Vulnerability Tsx async abort:   Mitigation; TSX disabled

Versions of relevant libraries:
[pip3] numpy==2.2.4
[pip3] nvidia-cublas-cu12==12.4.5.8
[pip3] nvidia-cuda-cupti-cu12==12.4.127
[pip3] nvidia-cuda-nvrtc-cu12==12.4.127
[pip3] nvidia-cuda-runtime-cu12==12.4.127
[pip3] nvidia-cudnn-cu12==9.1.0.70
[pip3] nvidia-cufft-cu12==11.2.1.3
[pip3] nvidia-curand-cu12==10.3.5.147
[pip3] nvidia-cusolver-cu12==11.6.1.9
[pip3] nvidia-cusparse-cu12==12.3.1.170
[pip3] nvidia-cusparselt-cu12==0.6.2
[pip3] nvidia-nccl-cu12==2.21.5
[pip3] nvidia-nvjitlink-cu12==12.4.127
[pip3] nvidia-nvtx-cu12==12.4.127
[pip3] torch==2.6.0
[pip3] triton==3.2.0
[conda] Could not collect

cc @ezyang @gchanan @zou3519 @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @chenyang78 @muchulee8 @amjames @aakhundov

## Trigger

A compiled function copies uint16 values into a bfloat16 tensor viewed as uint16 while Triton compute-type upcasting is enabled.

## Relevant entities

source uint16 tensor; bfloat16 target storage; dtype view; Triton compute value; bitcast flag

## Implemented relation

tensor.view(dtype) reinterprets fixed-width storage bits; copy_ must preserve those source bits before storing through the view.

## Pre-state

The source contains uint16 bit pattern 4 and the target view has the same two-byte element width.

## Binding transition

Inductor lowers the view copy through to_dtype_bitcast after values may have been promoted to a wider Triton compute type.

## Post-state

The old code compares promoted widths or bitcasts the promoted value directly, converting the numerical value rather than preserving the uint16 bits.

## Validity obligation

A same-width dtype view performs a bit reinterpretation at the declared source width, independent of the compiler's compute type.

## Invalid decision

to_dtype_bitcast derived the bitcast decision from upcast compute types instead of first restoring the declared source dtype.

## Physical effect

The generated Triton store writes the bfloat16 encoding of numeric 4 rather than uint16 bits 0x0004.

## Observed consequence

Compiled execution prints 4.0 where eager bit-level copy yields approximately 3.67e-40.

## Repair site

torch/_inductor/codegen/triton.py::TritonOverrides.to_dtype_bitcast

## Repair

PR 154121 asserts equal item sizes, casts a promoted value back to src_dtype, performs an unconditional bitcast to dtype, and then reapplies any compute upcast; it adds test_uint_view_copy.

## Why the repair works

The reinterpretation occurs between equal-width declared dtypes before any later compute representation is chosen.

## Causal basis

The patch states that it undoes upcasting before the dtype bitcast and its regression reproduces the report's uint16-to-bfloat16 view copy.

## Analysis implication

A view check needs storage width, declared source and destination dtypes, current register dtype, conversion kind, and final store dtype.

## Fix patches

- [`fixes/pytorch-151156-pr154121.patch`](fixes/pytorch-151156-pr154121.patch) — https://github.com/pytorch/pytorch/pull/154121
