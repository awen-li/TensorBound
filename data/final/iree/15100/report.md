# [ROCM] incorrect device selection using UUID for multiple rocm devices

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/15100](https://github.com/iree-org/iree/issues/15100)
- Opened: 2023-10-04T16:24:19Z
- Closed: 2023-11-06T06:51:19Z
- Labels: `bug 🐞`, `hal/hip`

## Reported behavior

### What happened?

Requirement : Ability to select rocm devices based on UUID

Steps:
1. add uuid to device info for rocm ([added here](https://github.com/nod-ai/SRT/commit/416ec59d940828a784bcb1ba2f740861efa99f91))
2. allow device selection logic to get appropriate device id from given UUID
3. Create hal device for the chosen device (2, 3 [added here](https://github.com/nod-ai/SRT/commit/476fb521fdeb47799636ec840037fb8a1989f2b2))

with the above mentioned changes it turns out that the device created by `iree_hal_rocm_device_create` is always the default device regardless of the input `device` id.

Observation:
with `hipCtxCreate` the device fetched from the created context turns out to be the default device
(could be a that we're not using this API correctly or getting incorrect results from it)
[debug patch here](https://github.com/nod-ai/SRT/commit/3dff64e401ae2066136989d6f3b7cf1a285b957f)

output from debug patch on a system with 4 rocm supported cards,
choosing `--device=rocm://<uuid for device#4>` in `--iree-run-module`
```
...
Context creating! with device - 3
Context created!
 Checking for device created in this context
got device from ctx - device# : 0
...
```
Changes can be found in [this branch](https://github.com/nod-ai/SRT/tree/add_rocm_device_path) on SRT

### Steps to reproduce your issue

1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error


### What component(s) does this issue relate to?

_No response_

### Version information

_No response_

### Additional context

_No response_

## Trigger

iree-run-module selects the fourth ROCm device through its GPU UUID path.

## Relevant entities

ROCm device UUID; HAL device path; HAL device ID; HIP device ordinal; HIP primary context; current context

## Implemented relation

The UUID selected by the user resolves to one HIP device ordinal, and the HAL device is created in that same device's context.

## Pre-state

Enumeration exposes four HIP devices and the requested UUID belongs to ordinal three.

## Binding transition

The old device-creation path passes a device value through an incomplete ID mapping and creates a HIP context that resolves to the default device.

## Post-state

The HAL device is bound to ordinal zero although the selected UUID identifies ordinal three.

## Validity obligation

Enumeration, path parsing, HAL IDs, HIP ordinals, and context creation preserve one device identity end to end.

## Invalid decision

The ROCm driver lacked a supported UUID path and a consistent conversion between the default HAL sentinel and zero-based HIP ordinals.

## Physical effect

Commands submitted through the selected HAL device execute in the default GPU's primary context.

## Observed consequence

Selecting the fourth GPU by UUID still creates and uses device zero.

## Repair site

experimental/rocm/rocm_driver.c and rocm_device.c

## Repair

PR 15234 publishes GPU UUID paths, parses UUID and index paths, introduces explicit HIP-to-HAL ID conversions, and retains then sets the selected device's primary context.

## Why the repair works

The path now resolves the requested UUID to a concrete HIP ordinal, and context creation consumes that same ordinal without colliding with the HAL default sentinel.

## Causal basis

The report logs a request for device three creating device zero and marks PR 15234 as the fix; that patch implements the UUID-to-ordinal-to-primary-context path.

## Analysis implication

Placement analysis needs external device paths, stable UUIDs, API-specific ID conventions, selected ordinals, and the context made current for execution.

## Fix patches

- [`fixes/iree-15100-pr15234.patch`](fixes/iree-15100-pr15234.patch) — https://github.com/iree-org/iree/pull/15234
