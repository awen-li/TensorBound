# Segmentation fault when calling exported choleksy on CPU

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/29610](https://github.com/jax-ml/jax/issues/29610)
- Opened: 2025-06-20T13:57:01Z
- Closed: 2026-04-01T20:54:17Z
- Labels: `bug`

## Reported behavior

### Description

Hello,

I am trying to call an exported JAX function.
Unfortunately, on my CPU jax installations it only seems to work when I have exported the function in the same Python process, otherwise I get a segmentation fault.
I have tested this on two machines (M2 Pro, Linux).
This only seems to affect the CPU jax installations, as everything works as expected on my Linux cuda installation.

I have included a script to reproduce the bug below.

Run following steps:
1. `python3 test_advi_export.py -export` works on all machines
2. `python3 test_advi_export.py` **run after step 1.** causes seg-fault on CPU installation, works on CUDA installation

Best,
Markus

UPDATE: I managed to reduce the minmal reproducible example, see comments.

### System info (python version, jaxlib version, accelerator, etc.)

Macbook M2 Pro CPU jax installation
```
jax:    0.6.2
jaxlib: 0.6.2
numpy:  2.2.2
python: 3.13.5 (main, Jun 11 2025, 15:36:57) [Clang 17.0.0 (clang-1700.0.13.3)]
device info: cpu-1, 1 local devices"
process_count: 1
platform: uname_result(system='Darwin', node='Markuss-MacBook-Pro-14.local', release='24.2.0', version='Darwin Kernel Version 24.2.0: Fri Dec  6 18:56:34 PST 2024; root:xnu-11215.61.5~2/RELEASE_ARM64_T6020', machine='arm64')
```

Desktop Ryzen 5950x CPU jax installation
```
WARNING:2025-06-20 16:04:50,510:jax._src.xla_bridge:794: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
jax:    0.6.2
jaxlib: 0.6.2
numpy:  2.3.0
python: 3.13.1 (main, Dec  9 2024, 00:00:00) [GCC 14.2.1 20240912 (Red Hat 14.2.1-3)]
device info: cpu-1, 1 local devices"
process_count: 1
platform: uname_result(system='Linux', node='fedora', release='6.12.10-200.fc41.x86_64', version='#1 SMP PREEMPT_DYNAMIC Fri Jan 17 18:05:24 UTC 2025', machine='x86_64')
```
Desktop GPU cuda jax installation
```
jax:    0.6.2
jaxlib: 0.6.2
numpy:  2.2.4
python: 3.13.1 (main, Dec  9 2024, 00:00:00) [GCC 14.2.1 20240912 (Red Hat 14.2.1-3)]
device info: NVIDIA GeForce GTX 1070-1, 1 local devices"
process_count: 1
platform: uname_result(system='Linux', node='fedora', release='6.12.10-200.fc41.x86_64', version='#1 SMP PREEMPT_DYNAMIC Fri Jan 17 18:05:24 UTC 2025', machine='x86_64')

$ nvidia-smi
Fri Jun 20 16:08:04 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 565.77                 Driver Version: 565.77         CUDA Version: 12.7     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce GTX 1070        Off |   00000000:2B:00.0  On |                  N/A |
|  0%   48C    P2             50W /  180W |     434MiB /   8192MiB |      2%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      2538      G   /usr/bin/gnome-shell                          211MiB |
|    0   N/A  N/A      3366    C+G   /usr/bin/ptyxis                                45MiB |
|    0   N/A  N/A      3433      G   /usr/bin/Xwayland                               3MiB |
|    0   N/A  N/A      3621      G   ...seed-version=20250619-180042.375000         63MiB |
|    0   N/A  N/A      7270      C   python3                                        88MiB |
+-----------------------------------------------------------------------------------------+

```

## Trigger

A process deserializes and calls an exported Cholesky computation without first tracing another LAPACK operation.

## Relevant entities

exported StableHLO; CPU platform; LAPACK FFI handler; kernel function pointer; deserialized callable

## Implemented relation

Every backend referenced by an exported computation is initialized before the deserialized callable dispatches its FFI kernels.

## Pre-state

The serialized program names a CPU LAPACK target, but the new process has not run the normal tracing path that initializes its handlers.

## Binding transition

The old deserialization call path reconstructs and executes the module without initializing CPU LAPACK.

## Post-state

The FFI handler resolves to a null kernel function pointer.

## Validity obligation

Deserialized execution initializes platform-specific runtime handlers independently of prior tracing in the process.

## Invalid decision

Export deserialization relied on incidental backend initialization performed by normal lowering.

## Physical effect

The CPU executable calls a null LAPACK kernel pointer.

## Observed consequence

Calling the rehydrated Cholesky function in a fresh process segfaults on CPU.

## Repair site

jax/_src/export/_export.py exported-call setup

## Repair

PR 33522 adds _ensure_backends_initialized, initializes jaxlib.cpu._lapack for CPU platforms, and invokes it before calling a deserialized export.

## Why the repair works

The required FFI kernel pointers are populated before the exported module can dispatch a LAPACK operation.

## Causal basis

Tracing first makes the crash disappear, and the linked fix identifies null CPU LAPACK handlers and initializes them in the deserialized-export path.

## Analysis implication

Portable-export checking needs serialized platform requirements, runtime handler registration, process initialization state, and dispatch targets.

## Fix patches

- [`fixes/jax-29610-pr33522.patch`](fixes/jax-29610-pr33522.patch) — https://github.com/jax-ml/jax/pull/33522
