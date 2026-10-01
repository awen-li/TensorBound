# jax.lax.psum_scatter on CPU intermittently returns wrong values on rank 0

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/40830](https://github.com/jax-ml/jax/issues/40830)
- Opened: 2026-09-21T06:05:52Z
- Closed: 2026-09-22T20:40:48Z
- Labels: `bug`

## Reported behavior

### Description

This might be an XLA:CPU bug but I've experienced it in JAX.

With `jax.distributed` initialized, XLA:CPU runs collectives through gloo, even between devices of one process.
`jax.lax.psum_scatter` (reduce-scatter) then intermittently returns wrong values at the end of rank 0's result chunk, with no error.

`repro.py` (attached, with HLO dumps) runs 24 CPU devices in one process as 12 groups of 2 and calls `psum_scatter` in 30 iterations.
Each device receives a 4 MiB float64 result chunk.

```
python3 repro.py                    # exits 1 if any result is wrong
python3 repro.py --no-distributed   # control: same program, no gloo
```

Expected: no wrong results, as with `--no-distributed` (0 of 60 iterations wrong).
Observed on jax and jaxlib 0.12.0.dev20260920: 148 of 150 iterations wrong (five runs), for example:

```
iteration 0: device 2 (rank 0 of group 1): 225979 of 524288 elements wrong, indices 298309..524287
```

All 608 wrong chunks were on rank 0 and ended at the chunk's last element.
With 256 KiB chunks (`--chunk-kib 256`), 0 of 60 iterations were wrong.

Suspected cause: `gloo::ReduceScatterHalvingDoubling::run()` (`gloo/reduce_scatter.h`), which `GlooCommunicator::ReduceScatter` calls, sends from the caller's buffer.
Every rank except rank 0 then overwrites offset 0 of that buffer (the slot of rank 0's chunk) with its final result, without waiting for its earlier sends from that region to complete.
gloo's TCP transport writes the part of a message that the socket does not accept immediately later, from its event-loop thread, reading the buffer at that time, so the receiver gets the overwritten bytes; this is why chunk size matters.

Potential fix (untested): call `waitSend()` before overwriting, as XLA's all-reduce, all-gather and all-to-all do, or implement reduce-scatter with `UnboundBuffer` send and receive plus a local reduction.

[module_0000.jit__lambda.before_optimizations.txt](https://github.com/user-attachments/files/32452836/module_0000.jit__lambda.before_optimizations.txt)
[module_0000.jit__lambda.cpu_after_optimizations.txt](https://github.com/user-attachments/files/32452837/module_0000.jit__lambda.cpu_after_optimizations.txt)

[repro.py](https://github.com/user-attachments/files/32452905/repro.py)


### System info (python version, jaxlib version, accelerator, etc.)

Versions: jax and jaxlib 0.12.0.dev20260920 and 0.11.1.dev20260810 (both fail), Python 3.13.15, Linux 6.12 x86-64, 64 CPUs, `net.core.wmem_max` = 212992 bytes.

## Trigger

Distributed CPU reduce-scatter operates on large buffers whose sends remain pending when a nonzero rank writes its final shard into the send buffer.

## Relevant entities

per-rank send buffer; pending TCP send; rank-zero shard region; final receive shard; Gloo event loop

## Implemented relation

Each pending reduce-scatter send reads a stable source region until the transport reports send completion.

## Pre-state

The TCP transport still references bytes in the caller's send buffer for an earlier phase.

## Binding transition

Gloo's built-in halving-doubling implementation copies the rank's final result over offset zero of that same buffer.

## Post-state

The event-loop thread can transmit overwritten bytes for the still-pending message.

## Validity obligation

A collective does not modify a send region until every transfer reading that region has completed.

## Invalid decision

GlooCommunicator delegated to ReduceScatterHalvingDoubling despite its unsynchronized reuse of the caller buffer.

## Physical effect

Large rank-zero result chunks receive corrupted suffixes read after another rank overwrote its send region.

## Observed consequence

psum_scatter intermittently returns hundreds of thousands of wrong elements on rank zero without an error.

## Repair site

xla/backends/cpu/collectives/gloo_communicator.cc and new gloo_reduce_scatter.cc

## Repair

PR 49201 replaces the built-in call with an XLA implementation using work storage, UnboundBuffer transfers, explicit waits, and broad reduce-scatter tests; PR 40880 adds a large-buffer JAX regression.

## Why the repair works

Source bytes reside in dedicated work storage and phase transfers are awaited before overlapping regions are reused or final shards are distributed.

## Causal basis

The report localizes corruption to delayed large-buffer sends, and the replacement algorithm removes the caller-buffer overwrite pattern and adds large-buffer coverage.

## Analysis implication

A collective check needs buffer-region identity, outstanding send/receive lifetimes, phase ordering, message size, and aliasing between send and receive buffers.

## Fix patches

- [`fixes/xla-40830-40831-pr49201.patch`](fixes/xla-40830-40831-pr49201.patch) — https://github.com/openxla/xla/pull/49201
- [`fixes/jax-40830-40831-pr40880.patch`](fixes/jax-40830-40831-pr40880.patch) — https://github.com/jax-ml/jax/pull/40880
