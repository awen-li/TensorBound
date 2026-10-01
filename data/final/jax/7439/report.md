# id_tap on CPU crashes with exceptions

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/7439](https://github.com/jax-ml/jax/issues/7439)
- Opened: 2021-08-02T11:14:14Z
- Closed: 2024-11-14T00:35:06Z
- Labels: `bug`

## Reported behavior

This crashes a CPU kernel, but prints the exception on a GPU/TPU kernel:
```
from jax.experimental import host_callback
import jax

def callback(val, transform):
  raise RuntimeError('Error!')

try:
  host_callback.id_tap(callback, 1.2)
  host_callback.barrier_wait()
except jax.experimental.host_callback.CallbackException as e:
  print(e)
  ```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

A CPU host callback raises RuntimeError while using the CustomCall implementation rather than outfeed.

## Observed implementation decision

The callback exception escapes through pybind11 without a status-returning CustomCall boundary.

## Physical effect

C++ terminates the process instead of returning the Python exception to the invoking thread.

## Observed consequence

The CPU program aborts, while outfeed-backed GPU and TPU executions report a callback exception.

## Reported proposal or workaround

A later jaxlib version propagates RuntimeError, and id_tap was eventually removed.

## Repair trace

The issue attributes recovery to unspecified XLA CustomCall changes but does not identify or archive the source delta and regression. Later API removal also does not establish the historical repair relationship.

## Archived evidence

- Raw report: `data/raw/github/jax/7439/report.json`
- SHA-256: `f6530afdf5834b7355fbed9e6e9762e8725b32a16cb53c0c9caa30421aecd7c1`
