# jax.debug.breakpoint() give numpy array and with bfloat16, it comptue

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/17402](https://github.com/jax-ml/jax/issues/17402)
- Opened: 2023-09-01T15:47:07Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

jax.debug.breakpoint() give strange results.
Some of the inputs are tensors of 1s with shape of 2048 (bfloat16). But if we do `x.sum()` is return 256.
But if I do that in the python script, I get the good results (2048):
```
# python test.py
2048 <class 'jax.numpy.bfloat16'> <class 'jaxlib.xla_extension.ArrayImpl'>
Entering jdb:
(jdb) x.sum()
256
(jdb) c
2048.0 <class 'jax.numpy.float16'> <class 'jaxlib.xla_extension.ArrayImpl'>
Entering jdb:
(jdb) x.sum()
2048.0
(jdb) c
2048.0 <class 'jax.numpy.float32'> <class 'jaxlib.xla_extension.ArrayImpl'>
Entering jdb:
(jdb) x.sum()
2048.0
```

To prevent all that category of issues, breakpoint should give ArrayImpl object and not numpy object.
@sharadmv

### What jax/jaxlib version are you using?

upstream

### Which accelerator(s) are you using?

Independent of backend.

### Additional system info

_No response_

### NVIDIA GPU info

_No response_

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises jAX debug.breakpoint host transfer of bfloat16 values.

## Observed implementation decision

The breakpoint callback exposes NumPy values even when NumPy cannot preserve the device array's bfloat16 semantics.

## Physical effect

The host representation changes the type/operations available to the debugger.

## Observed consequence

Breakpoint-side computation behaves differently or fails.

## Reported proposal or workaround

A maintainer suggests returning a CPU JAX Array.

## Repair trace

The suggestion is not implemented in a linked landed patch and the issue remains open.

## Archived evidence

- Raw report: `data/raw/github/jax/17402/report.json`
- SHA-256: `7f25e347bff8883fd3a256e1f14320c194ff5e69afdd6ec532477abf3ca47650`
