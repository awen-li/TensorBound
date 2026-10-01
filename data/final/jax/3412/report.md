# jnp.asarray does not copy when copy=True

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/3412](https://github.com/jax-ml/jax/issues/3412)
- Opened: 2020-06-11T20:59:29Z
- Closed: 2020-06-12T01:48:31Z
- Labels: `bug`

## Reported behavior

This can be a problem now that we have donate_argnums. It took me a while to realize this, before I smartened up and just used x.copy() in the example below.

```
import functools
import jax
import jax.numpy as jnp

@functools.partial(jax.jit, donate_argnums=(0,))
def _test(array: jnp.ndarray):
  return array.at[0].set(77)

x = jnp.asarray([0, 1])
x_copy = jnp.array(x, copy=True)
new_x = _test(x)

# Gives: RuntimeError: Invalid argument: CopyToHostAsync() called on invalid buffer.
print(x_copy)

```

@tomhennigan

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

jnp.asarray does not copy when copy=True

## Observed implementation decision

The array-construction path satisfies an explicit copy request by reusing the existing device buffer because the runtime lacked a device-buffer copy operation.

## Physical effect

jnp.array(copy=True) retains the input device buffer instead of allocating independent storage, so donation of the original invalidates the supposed copy.

## Observed consequence

After a jitted TPU computation donates the original, accessing the copy raises CopyToHostAsync called on invalid buffer.

## Reported proposal or workaround

The minimal reproducer combines explicit copy and donation, and a maintainer confirms the runtime lacked buffer copying and proposes a host-bounce implementation until native support exists.

## Repair trace

The report establishes that copy=True reuses the donated buffer and records a host-bounce proposal, but no implemented copy repair or fixed regression is linked.

## Archived evidence

- Raw report: `data/raw/github/jax/3412/report.json`
- SHA-256: `5f56cb2f24df239fdf6cef8b05bb05af5fff7c3e68e498c8dfc9265783015254`
