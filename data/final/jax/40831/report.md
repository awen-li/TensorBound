# jax.lax.psum_scatter on CPU ignores --xla_cpu_collective_timeout_seconds and fails after 30 s

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/40831](https://github.com/jax-ml/jax/issues/40831)
- Opened: 2026-09-21T06:16:12Z
- Closed: 2026-09-22T20:40:49Z
- Labels: `bug`

## Reported behavior

### Description

On CPU with `jax.distributed`, `jax.lax.psum_scatter` (reduce-scatter) fails after 30 s when the processes arrive more than about 30 s apart.
It ignores `--xla_cpu_collective_timeout_seconds`, while `jax.lax.all_gather` waits as configured.

The attached `repro.py` (about 40 s per run; HLO dumps attached) starts two processes with one CPU device each and sets both XLA collective timeout flags to 3600 s.
Each process runs one warm-up collective, then process 1 waits 40 s before the second one.

```
python repro.py psum_scatter   # bug: exit code 1
python repro.py all_gather     # control: exit code 0
```

Expected: process 0 waits and both calls succeed, as with `all_gather` (process 0 succeeded after 40.0 s).
Observed on jax and jaxlib 0.12.0.dev20260920: `psum_scatter` failed in 3 of 3 runs, like this:

```
process 0: second psum_scatter FAILED after 30.0 s: UNKNOWN: Gloo ReduceScatter failed: [external/xla++third_party_ext+gloo/gloo/transport/tcp/buffer.cc:72] Read timeout [::1]:38624
process 1: second psum_scatter FAILED after 0.0 s: UNKNOWN: Gloo ReduceScatter failed: [external/xla++third_party_ext+gloo/gloo/transport/tcp/pair.cc:547] Connection closed by peer [::1]:2475
```

Suspected cause: `GlooCollectives::CreateCommunicators` (`gloo_collectives.cc`) never calls `gloo::Context::setTimeout()`, so the gloo context keeps gloo's default of 30 s.
The other collectives in `gloo_communicator.cc` pass XLA's timeout on each call, but `GlooCommunicator::ReduceScatter` ignores the timeout in its `Executor` argument.
gloo's `Buffer::waitRecv()` (`gloo/transport/tcp/buffer.cc`), which reduce-scatter uses, waits at most the timeout of a pair (gloo's connection between two ranks), and a pair copies the context's timeout when it is created.

Potential fix: call `setTimeout()` with XLA's collective timeout before `connectFullMesh()` in `CreateCommunicators`, or implement reduce-scatter with `UnboundBuffer` send and receive and per-call timeouts, as `CollectivePermute` and `AllToAll` do.

[repro.py](https://github.com/user-attachments/files/32453192/repro.py)
[module_0000.jit_collective.cpu_after_optimizations.txt](https://github.com/user-attachments/files/32453194/module_0000.jit_collective.cpu_after_optimizations.txt)
[module_0000.jit_collective.before_optimizations.txt](https://github.com/user-attachments/files/32453193/module_0000.jit_collective.before_optimizations.txt)


### System info (python version, jaxlib version, accelerator, etc.)

Versions: jax and jaxlib 0.12.0.dev20260920 and 0.11.1.dev20260810 (both fail), Python 3.13.15, Linux 6.12 x86-64, 64 CPUs.

## Trigger

Two distributed CPU ranks enter psum_scatter more than Gloo's 30-second default apart while XLA's collective timeout is configured to 3600 seconds.

## Relevant entities

XLA collective timeout; CpuCollectives::Executor; Gloo context default timeout; reduce-scatter send/receive waits; participating ranks

## Implemented relation

The executor's timeout governs how long all communication waits for one collective may await peer progress.

## Pre-state

The caller supplies a 3600-second timeout through the collective executor.

## Binding transition

GlooCommunicator invokes the built-in reduce-scatter path without consuming the executor timeout.

## Post-state

Gloo transport waits use the context's 30-second default instead of the requested collective deadline.

## Validity obligation

Every blocking transfer used by a collective applies the timeout configured for that collective invocation.

## Invalid decision

The old GlooCommunicator::ReduceScatter ignored the timeout in its Executor argument.

## Physical effect

The early rank's receive wait expires at 30 seconds and closes the connection before the delayed rank arrives.

## Observed consequence

psum_scatter fails after 30 seconds in three of three runs, while all_gather respects the 3600-second setting.

## Repair site

xla/backends/cpu/collectives/gloo_communicator.cc and gloo_reduce_scatter.cc

## Repair

PR 49201 casts the executor, passes cpu_executor->timeout() into GlooReduceScatter, derives a deadline, and uses it for UnboundBuffer waitSend and waitRecv operations.

## Why the repair works

Every communication wait in the replacement collective shares a deadline derived from the caller's XLA timeout.

## Causal basis

The report contrasts reduce-scatter with all-gather, and the patch connects the previously ignored executor field to each replacement transfer wait.

## Analysis implication

A collective timeout check needs configuration provenance, executor translation, per-operation deadline, and every blocking wait site.

## Fix patches

- [`fixes/xla-40830-40831-pr49201.patch`](fixes/xla-40830-40831-pr49201.patch) — https://github.com/openxla/xla/pull/49201
- [`fixes/jax-40830-40831-pr40880.patch`](fixes/jax-40830-40831-pr40880.patch) — https://github.com/jax-ml/jax/pull/40880
