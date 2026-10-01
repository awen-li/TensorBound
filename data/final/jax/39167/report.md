# Multi-GPU NCCL collectives fail with "corrupted comm object detected" when TensorFlow is imported in the same process (Kaggle T4 x2)

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/39167](https://github.com/jax-ml/jax/issues/39167)
- Opened: 2026-07-15T02:50:52Z
- Closed: 2026-07-15T16:48:09Z
- Labels: `bug`

## Reported behavior

### Description

a comment on https://github.com/jax-ml/jax/issues/34918

---

I hit the same failure on a different platform (Kaggle, 2x Tesla T4) and
can narrow it down to a one line reproduction, plus some evidence about
the mechanism.

## Description

On a 2x T4 machine, a minimal sharded all-reduce works with stock JAX, but
the identical program fails with `ncclAllReduce ... invalid argument` when
TensorFlow has been imported into the same process first. `import
tensorflow` alone is sufficient to flip the program from passing to
failing. With `NCCL_DEBUG=INFO`, NCCL reports
`Error: corrupted comm object detected` on every call against a
communicator that had just completed initialization successfully.

The evidence below points to more than one NCCL implementation being
active in the process (the jax CUDA plugin's dynamically loaded
`nvidia-nccl` wheel plus another copy bound by the TensorFlow side), such
that a communicator created by one copy is rejected by the other copy's
sanity check.

## Reproduction

Environment: Kaggle notebook, GPU T4 x2 runtime, Python 3.12, host driver
reports `cudaDriverVersion 13000`.

Case A (passes). Fresh session, no other imports:

```python
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
mesh = Mesh(jax.devices(), ("d",))
x = jax.device_put(jnp.ones((8, 128), jnp.float16), NamedSharding(mesh, P("d", None)))
print(jax.jit(lambda a: a.sum(), out_shardings=NamedSharding(mesh, P()))(x))
```

Output: `1024.0`

Case B (fails). Fresh session, same program, but run after a single
additional import:

```python
import tensorflow
```

Output:

```
XlaRuntimeError: INTERNAL: NCCL operation ncclAllReduce( send_buffer.opaque(),
recv_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
ToNcclReduction(reduction_kind), comm_, AsCudaStream(stream)) failed:
invalid argument
```

The same failure occurs when TensorFlow is imported indirectly, for
example via `import keras_hub`, which is how I originally hit this while
running Keras models on the JAX backend (KerasHub imports TensorFlow
eagerly, so any KerasHub user on multi-GPU JAX is affected).

## NCCL debug log

With `NCCL_DEBUG=INFO`, initialization completes and then every call on
the communicator fails the object check. Trimmed log from Case B:

```
NCCL INFO cudaDriverVersion 13000
NCCL INFO NCCL version 2.30.7+cuda13.3
NCCL INFO NCCL git version HEAD a5187fa2e
...
NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct
NCCL INFO Channel 01 : 0[0] -> 1[1] via SHM/direct
NCCL INFO Connected all rings, use ring PXN 0 GDR 1
NCCL INFO Connected all trees
NCCL INFO ncclCommInitRankConfig comm 0x7c0b40009ce0 rank 0 nranks 2
    cudaDev 0 nvmlDev 0 busId 40 commId 0x8670881a9fa8d8c7 - Init COMPLETE

external/nccl_archive/src/misc/argcheck.cc:39 NCCL WARN Error: corrupted comm object detected
NCCL INFO external/nccl_archive/src/init.cc:2213 -> 4
NCCL INFO init.cc:428 (ncclCommEnsureReady) -> 4
NCCL INFO dev_runtime.cc:1464 (ncclCommQueryProperties) -> 4

external/nccl_archive/src/misc/argcheck.cc:39 NCCL WARN Error: corrupted comm object detected
NCCL INFO external/nccl_archive/src/init.cc:2213 -> 4
NCCL INFO init.cc:428 (ncclCommEnsureReady) -> 4
NCCL INFO enqueue.cc:3141 (ncclEnqueueCheck) -> 4
NCCL INFO group.cc:110 (ncclGroupEnd) -> 4
```

Note the two different provenances in one log: the init banner is NCCL
2.30.7+cuda13.3 (the `nvidia-nccl-cu13` pip wheel), while the failing
checks log Bazel workspace paths (`external/nccl_archive/...`), which do
not exist in any jax wheel (verified below).

## What I ruled out

The failure is identical across all of the following, which suggests it is
not a version, transport, or allocator problem:

- NCCL 2.27.5 (cu12 wheel, the image's stock stack) and NCCL 2.30.7
  (cu13 wheel, clean `pip install "jax[cuda13]"` == jax/jaxlib/plugin/pjrt
  all 0.10.2)
- `NCCL_P2P_DISABLE=1`
- `NCCL_CUMEM_ENABLE=0`
- `--xla_gpu_enable_nccl_comm_splitting=false`
- `--xla_gpu_enable_command_buffer=` (disabled)
- `XLA_PYTHON_CLIENT_ALLOCATOR=platform` (instead of default bfc)

Also:

- PyTorch NCCL collectives work on the same machine in the same session:
  `torch.cuda.nccl.all_reduce` across `cuda:0`/`cuda:1` returns correct
  results. The platform itself is healthy.
- Case A above shows stock JAX collectives are healthy in a clean process.

## Binary forensics

- `jax_cuda13_pjrt-0.10.2` (`xla_cuda_plugin.so`, 401 MB): contains no
  NCCL implementation. Zero occurrences of `nccl_archive` or NCCL source
  paths in strings, no NCCL dynamic symbols defined, no `DT_NEEDED` on
  libnccl. It carries an RPATH entry `$ORIGIN/../../nvidia/nccl/lib`,
  i.e. NCCL is loaded at runtime from the `nvidia-nccl-cu13` wheel.
- `libtensorflow_cc.so.2` (TensorFlow pip wheel): declares the NCCL API
  (`ncclAllReduce`, `ncclCommInitRankConfig`, `ncclGroupEnd`, ...) as
  undefined dynamic symbols (`UND`), so TensorFlow binds an NCCL
  implementation at runtime in the same process.
- The system image additionally ships `/usr/lib/x86_64-linux-gnu/
  libnccl.so.2.25.1`. During the PyTorch control test, a third banner
  appeared: `NCCL version 2.25.1+cudaCUDA_MAJOR.CUDA_MINOR` (note the
  unsubstituted version macros, indicating a custom source build).

So at least two and possibly three NCCL implementations can be live in
one process, and NCCL communicators are not shareable across copies.
`ncclCommEnsureReady`'s magic check then fails exactly as observed.

## Expected behavior

Either JAX's NCCL usage is isolated from other frameworks' NCCL copies in
the same process (for example by binding symbols from its own dlopen
handle only), or the failure surfaces as an actionable error message
instead of `invalid argument` on the first collective.

## Workaround

Keep TensorFlow out of processes that run JAX multi-GPU collectives. In
this case that means not importing `keras_hub`/`tensorflow` in the same
process, or uninstalling TensorFlow from the environment.

## Environment summary

- Kaggle GPU T4 x2 runtime (2x Tesla T4, 15360 MiB each)
- NVIDIA driver 580.159.04, CUDA driver version 13.0
- Linux 6.12.90+, x86_64, Python 3.12.13, numpy 2.0.2
- Stock image stack: jax 0.7.2, jaxlib 0.7.2 (fails, NCCL 2.27.5+cuda12.9
  from the nvidia-nccl-cu12 wheel)
- Clean upgraded stack: jax == jaxlib == jax-cuda13-plugin ==
  jax-cuda13-pjrt == 0.10.2 with nvidia-nccl-cu13 2.30.7 (fails
  identically)
- `jax.print_environment_info()` from a failing session:

```
jax:    0.7.2
jaxlib: 0.7.2
numpy:  2.0.2
python: 3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0]
device info: Tesla T4-2, 2 local devices
process_count: 1
platform: uname_result(system='Linux', node='e417b27277b1', release='6.12.90+',
    version='#1 SMP Sat May 30 15:40:53 UTC 2026', machine='x86_64')
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises jAX NCCL communicator use when TensorFlow is imported in the same multi-GPU process.

## Observed implementation decision

The combined runtimes produce a communicator/device state that NCCL rejects as corrupted.

## Physical effect

Collective operations receive an invalid communicator object.

## Observed consequence

Multi-GPU collectives fail.

## Reported proposal or workaround

The reporter cites a downstream Flax workaround.

## Repair trace

The external workaround does not identify or repair a JAX/XLA binding; no landed upstream correction is linked.

## Archived evidence

- Raw report: `data/raw/github/jax/39167/report.json`
- SHA-256: `1d9cc498cd36a28cbae4e4019f8b04f525344b7fce45c8032f81c178decc9726`
