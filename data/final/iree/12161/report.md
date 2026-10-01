# Caching allocator causes wrong computation result

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/12161](https://github.com/iree-org/iree/issues/12161)
- Opened: 2023-02-11T01:20:58Z
- Closed: 2023-03-16T20:16:33Z
- Labels: `bug 🐞`

## Reported behavior

### What happened?

Enabling the caching allocator causes numerical errors. In this [branch](https://github.com/nod-ai/SHARK-Runtime/tree/stateful-model-test) I have modified the python binding per the suggestion of @benvanik to enable quickly the caching allocator while waiting for the Python API to catch up.
There is a [test](https://github.com/nod-ai/SHARK-Runtime/tree/stateful-model-test/tests/e2e/models/stateful_model_test) there that reproduces the problem. I am not convinced that this test will reproduce reliably on other platforms.

### Steps to reproduce your issue

At the aforementioned branch run
```
python ./tests/e2e/models/stateful_model_test/stateful_model_test.py
```

### What component(s) does this issue relate to?

Compiler, Runtime

### Version information

_No response_

### Additional context

https://discord.com/channels/689900678990135345/706175572920762449/1073493760429789254

### Edit:
A more up-to-date version of the test can be found [here](https://github.com/nod-ai/SHARK-Runtime/tree/stateful-model-test-with-allocator-python-api).

## Trigger

Repeated parameter updates acquire a free-pool buffer that still contains bytes from an earlier logical tensor.

## Relevant entities

logical parameter; initial_data; free-pool buffer; cached content; published HAL buffer

## Implemented relation

A reused physical buffer contains the requested logical tensor value before the allocator returns it to the caller.

## Pre-state

The selected free-pool buffer has valid capacity and usage flags but still
contains bytes from an earlier parameter.

## Binding transition

The reuse branch returns existing_buffer before the initial_data map-write used by the new-allocation branch.

## Post-state

The new parameter is bound to storage whose bytes still represent the preceding parameter.

## Validity obligation

When the caller supplies `initial_data`, reuse writes those bytes successfully
before publishing the buffer.

## Invalid decision

The old caching allocator treated a matching reusable allocation as initialized even when initial_data was supplied.

## Physical effect

Execution reads stale parameter bytes through a current logical binding.

## Observed consequence

Repeated set_params calls exchange parameter values and produce wrong numerical output.

## Repair site

runtime/src/iree/hal/utils/caching_allocator.c::iree_hal_caching_allocator_pool_acquire

## Repair

Commit 7be84611 writes initial_data into an existing buffer, checks the write status, and returns the buffer only after successful initialization.

## Why the repair works

Both newly allocated and reused storage establish the caller-requested byte content before publication.

## Causal basis

The maintainer diagnosis, reporter retest, and exact fixing commit all identify the skipped existing-buffer initialization path.

## Analysis implication

Reuse analysis needs the requested initialization and its completion status at
the publication decision, alongside allocation compatibility and lifetime.

## Fix patches

- [`fixes/iree-12161-fix.patch`](fixes/iree-12161-fix.patch) — https://github.com/iree-org/iree/commit/7be84611ae0d998a9e13b9be2de03681c8697305
