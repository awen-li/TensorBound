# [inductor] Missing barrier on same-kernel global buffer read-back (RAW) causes accuracy failure

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/190517](https://github.com/pytorch/pytorch/issues/190517)
- Opened: 2026-07-20T08:07:39Z
- Closed: 2026-07-27T23:35:02Z
- Labels: `high priority`, `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

### Summary

When an inductor Triton kernel stores a buffer and then reads it back later in the same kernel, no barrier is emitted between the store and the load. The store and the read-back can end up on different warps, so a warp may read the location before another warp's write is visible - a read-after-write race through global memory.

This is the read-after-write counterpart to the write-after-read guard in `store()` (triton issue #1615). That direction is guarded; store-then-read-back is not.

### Repro
pit_b_224 (timm), inductor + AMP fp16 inference on Intel XPU:

      TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python benchmarks/dynamo/timm_models.py \
        -d xpu --accuracy --amp --inference --backend=inductor \
        --cold-start-latency -n10 --disable-cudagraphs --only pit_b_224

Result: fail_accuracy (RMSE ~0.4 vs tol ~0.001).

The generated kernel stores `out_ptr0` and reads it back in the reduction loop with nothing in between:

      tl.store(out_ptr0 + (r0_2 + 256*x3), tmp26, r0_mask & xmask)
      ...
      tmp34 = tl.load(out_ptr0 + (r0_2 + 256*x3), r0_mask & xmask, ...)

### Versions

Collecting environment information...

XPU used to build PyTorch: 20260000
Intel GPU driver version:
* libze1:       1.28.2-1~24.04~ppa1
* intel-opencl-icd:     26.18.38308.1-1~24.04~ppa1
Intel GPU models onboard:
N/A
Intel GPU models detected:
* [0] _XpuDeviceProperties(name='Intel(R) Data Center GPU Max 1100', platform_name='Intel(R) oneAPI Unified Runtime over Level-Zero', type='gpu', device_id=0xBDA, uuid=8680da0b-2f00-0000-ad00-000000000000, driver_version='1.15.38308+1', total_memory=49136MB, local_mem_size=128KB, last_level_cache_size=196608KB, max_compute_units=448, memory_clock_rate=3200MHz, memory_bus_width=64-bit, gpu_eu_count=448, gpu_subslice_count=56, max_work_group_size=1024, max_num_sub_groups=64, sub_group_sizes=[16 32], has_fp16=1, has_fp64=1, has_atomic64=1, is_integrated_gpu=0)
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False
Caching allocator config: N/A

CPU:
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           52 bits physical, 57 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  128
On-line CPU(s) list:                     0-127
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Xeon(R) Gold 6438Y+
CPU family:                              6
Model:                                   143
Thread(s) per core:                      2
Core(s) per socket:                      32
Socket(s):                               2
Stepping:                                8
CPU(s) scaling MHz:                      79%
CPU max MHz:                             4000.0000
CPU min MHz:                             800.0000
BogoMIPS:                                4000.00
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cat_l2 cdp_l3 invpcid_single cdp_l2 ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect avx_vnni avx512_bf16 wbnoinvd dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req avx512vbmi umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid bus_lock_detect cldemote movdiri movdir64b enqcmd fsrm md_clear serialize tsxldtrk pconfig arch_lbr amx_bf16 avx512_fp16 amx_tile amx_int8 flush_l1d arch_capabilities
Virtualization:                          VT-x
L1d cache:                               3 MiB (64 instances)
L1i cache:                               2 MiB (64 instances)
L2 cache:                                128 MiB (64 instances)
L3 cache:                                120 MiB (2 instances)
NUMA node(s):                            2
NUMA node0 CPU(s):                       0-31,64-95
NUMA node1 CPU(s):                       32-63,96-127
Vulnerability Gather data sampling:      Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; PBRSB-eIBRS SW sequence; BHI BHI_DIS_S
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Not affected

Versions of relevant libraries:
[pip3] numpy==2.2.6
[pip3] torch==2.14.0a0+git23b1588
[pip3] torchaudio==2.11.0a0+c0cbdb9
[pip3] torchvision==0.29.0a0+0bc41e6
[pip3] triton==3.8.0+gitf3b38653
[pip3] triton-kernels-benchmark==3.7.0+git693eb3fd
[conda] numpy                     2.2.6                    pypi_0    pypi
[conda] torch                     2.14.0a0+git23b1588          pypi_0    pypi
[conda] torchaudio                2.11.0a0+c0cbdb9          pypi_0    pypi
[conda] torchvision               0.29.0a0+0bc41e6          pypi_0    pypi
[conda] triton                    3.8.0+gitf3b38653          pypi_0    pypi
[conda] triton-kernels-benchmark  3.7.1+git15410d29          pypi_0    pypi

cc @ezyang @gchanan @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @muchulee8 @amjames @aakhundov @coconutruben @jataylo

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

- [`fixes/pytorch-190517-commit-62a4f51.patch`](fixes/pytorch-190517-commit-62a4f51.patch) — https://github.com/pytorch/pytorch/commit/62a4f51cf34e32503ae138c8f977b434bb90a325
