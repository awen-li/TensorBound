# [dlpack][runtime] Memory leak with dlpack capsules

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/20849](https://github.com/iree-org/iree/issues/20849)
- Opened: 2025-05-19T15:37:44Z
- Closed: 2025-06-04T16:37:48Z
- Labels: `bug 🐞`

## Reported behavior

### What happened?

I'm not entirely sure what is going on, but I noticed in `iree.turbine.kernel.wave.utils.run_utils`, they need to rename the dlpack capsule from `used_dltensor` back to `dltensor` for some reason.

I was running into some memory issues with larger models (with several iree kernels plugged in), and renaming the capsules seemed to resolve the OOM.

My rough guess is that renaming the capsule to `used_dltensor` somehow invalidates the external deleter (which I assume is provided by pytorch).



### Steps to reproduce your issue

install `iree-turbine` versioned earlier than today (5/19/2025)

run:

```python
import torch
from iree.turbine.ops._jinja_test_ops import test_add

for _ in range(10000):
    x = torch.randn([256,256,256], device="cuda:0")
    y = torch.randn([256,256,256], device="cuda:0")
    _ = test_add(x,y)

```

This will raise:

```
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacity of 191.98 GiB of which 40.00 MiB is free. Of the allocated memory 191.25 GiB is allocated by PyTorch, and 0 bytes is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_HIP_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

However, if we rename the capsule back to `dltensor` after buffer view import in `iree.turbine.runtime.device._device_import_torch_tensor_cuda_hip`, then the issue no longer persists.

### What component(s) does this issue relate to?

Python, Runtime

### Version information

_No response_

### Additional context

_No response_

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

[dlpack][runtime] Memory leak with dlpack capsules

## Observed implementation decision

The import path renames a consumed capsule to a name other than the protocol-defined used_dltensor marker, so capsule destruction follows the wrong ownership state.

## Physical effect

A misspelled DLPack capsule-consumption name breaks the ownership-handoff convention and prevents the external tensor deleter from reclaiming imported allocations.

## Observed consequence

Repeated Torch-to-IREE imports retain GPU allocations until the process exhausts device memory.

## Reported proposal or workaround

Renaming the capsule according to the protocol removes the leak; maintainers identify the typo, find the leak, confirm the reproducer no longer leaks, and close the issue.

## Repair trace

The discussion reports that a capsule-name typo was fixed and PR 893 removes a workaround after an IREE version bump, but the underlying runtime patch is absent; the repair site and exact source delta cannot be verified.

## Archived evidence

- Raw report: `data/raw/github/iree/20849/report.json`
- SHA-256: `49bbec18cddabe21c1b560facc65e7efcf7e353c46151423e9091a2d75fd0ea7`
