# jax_debug_nans interaction with buffer donation

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/12514](https://github.com/jax-ml/jax/issues/12514)
- Opened: 2022-09-26T18:21:59Z
- Closed: 2022-11-08T17:57:51Z
- Labels: `bug`, `P2 (eventual)`

## Reported behavior

When we set `jax_debug_nans`, we might try to re-run jitted computations. But if those jitted computations involve buffer donation, then we'll get a low-level runtime error about repeatedly donating the same input:

```
Invalid nan value encountered in the output of a C++-jit/pmap function. Calling the de-optimized version.
[10:13:36 pjrt_stream_executor_client.cc:2130] Execution of replica 0 failed: INVALID_ARGUMENT: Invalid buffer passed to Execute() as argument 0 to replica 0: INVALID_ARGUMENT: Donation requested for invalid buffer
```

TODO: make a repro

## Trigger

A donated-input compiled computation produces NaN while jax_debug_nans requests a de-optimized rerun.

## Relevant entities

input buffer; donation flag; compiled execution; NaN detector; de-optimized rerun

## Implemented relation

A diagnostic rerun reads the original logical arguments, so those buffers remain valid until the rerun decision is complete.

## Pre-state

The initial compiled call owns a valid input buffer that debug mode may need again.

## Binding transition

Compiled execution donates and invalidates the input, then NaN handling calls the function again with the same argument.

## Post-state

The rerun attempts to donate or read an already invalid buffer.

## Validity obligation

When a runtime diagnostic may rerun a computation, inputs needed by that rerun are not donated during the first execution.

## Invalid decision

JIT preparation honored donate_argnums even when debug_nans enabled the fallback rerun path.

## Physical effect

PJRT marks the input buffer deleted after the first execute.

## Observed consequence

NaN diagnosis fails with 'Donation requested for invalid buffer' instead of running the de-optimized function.

## Repair site

jax/_src/api.py::_prepare_jit and pmap/pjit donation-vector construction

## Repair

PR 13144 clears donated-invars whenever jax_debug_nans is enabled for jit, pmap, and pjit and adds regressions for all three paths.

## Why the repair works

The original buffers remain readable for the conditional diagnostic rerun.

## Causal basis

The report describes reuse after donation, and the patch suppresses donation under exactly that diagnostic mode with three reproductions.

## Analysis implication

A donation check needs diagnostic mode, possible rerun edges, input liveness, donation flags, and execute count.

## Fix patches

- [`fixes/jax-12514-pr13144.patch`](fixes/jax-12514-pr13144.patch) — https://github.com/jax-ml/jax/pull/13144
