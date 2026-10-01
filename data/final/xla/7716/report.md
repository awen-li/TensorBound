# Nondeterminism in triton-autotuner and others leads to hangs in multi-process gpu execution

- Repository: `openxla/xla`
- Report: [https://github.com/openxla/xla/issues/7716](https://github.com/openxla/xla/issues/7716)
- Opened: 2023-12-12T17:50:48Z
- Closed: 2024-12-03T13:50:08Z
- Labels: `bug`, `GPU`

## Reported behavior

Opening an issue to help us keep track of this even though the discussion has already been ongoing elsewhere.

When using multi-process execution, XLA compiles each module indepedently for each process. Nondeterministic passes like `triton-autotuner` can produce different results per process, which can lead to different optimizations happening later down the pipeline. If any of those optimizations leads to a different ordering of collectives in the schedule, or a different `is_async` designation for collectives, then there will be deadlocks on the first execution.

The ideal solution would be to compile the module once and broadcast the result to all other processes. I believe @hawkinsp is currently planning to work on this.

## Trigger

Several hosts independently compile the same distributed module while Triton autotuning can choose different configurations on each host.

## Relevant entities

distributed hosts; HLO module; autotune configuration; optimized executable; collective order; asynchronous collective designation

## Implemented relation

All participants in a distributed execution compile collective-compatible schedules from a common autotuning decision set.

## Pre-state

Every host starts from the same logical module and must issue matching collective sequences.

## Binding transition

The old multi-process path runs nondeterministic autotuning independently on every host.

## Post-state

Later optimization can produce host-specific collective ordering or asynchronous designations.

## Validity obligation

Compilation choices that affect collective schedule are coordinated across every process participating in those collectives.

## Invalid decision

Autotune results were local process state even when their downstream schedules had a cross-process compatibility requirement.

## Physical effect

One host waits in a collective whose peer executes a different collective or ordering.

## Observed consequence

The first distributed execution can deadlock.

## Repair site

JAX distributed compilation/autotune sharing and XLA multi-process autotune sharding configuration

## Repair

JAX commit 9a098e9 adds coordinator sharing of compiled binaries or autotune configs; XLA commit 2500111 later enables multi-process autotune sharding by default.

## Why the repair works

Participating processes no longer derive schedule-affecting autotune choices as unrelated local decisions, so their collective programs remain compatible.

## Causal basis

The report identifies divergent autotuning as the source of incompatible collective schedules, records the JAX sharing implementation, and closes after XLA enables multi-process autotune sharding by default.

## Analysis implication

Distributed compilation analysis needs module identity, process group, autotune candidates and results, sharing/sharding mode, compiled collective sequence, and async flags per host.

## Fix patches

- [`fixes/xla-7716-jax-commit9a098e9.patch`](fixes/xla-7716-jax-commit9a098e9.patch) — https://github.com/jax-ml/jax/commit/9a098e922aff62a3b49bd673b9518d97ee599248
- [`fixes/xla-7716-commit2500111.patch`](fixes/xla-7716-commit2500111.patch) — https://github.com/openxla/xla/commit/2500111022fc9791bd9134134258c2606be50c4d
