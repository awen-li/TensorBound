# Wrong behaviour of pallas.atomic_cas

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/18909](https://github.com/jax-ml/jax/issues/18909)
- Opened: 2023-12-10T19:46:06Z
- Closed: Open at collection
- Labels: `bug`, `NVIDIA GPU`, `pallas`

## Reported behavior

### Description

Hello Team,
I have found a wrong behaviour of pallas.atomic_cas in my specific case. In the example, I am creating 10 counters and 10 locks, one for each counter. The same code works well on Triton, but the implementation on Pallas goes into an infinite loop.
Example:
```
@triton.jit
def counter_kernel_triton(
    _,
    __,
    data,
    lock
):
    id_ = tl.program_id(0)
    offset = id_
    ptrs = data + offset + tl.arange(0, 1)
    while tl.atomic_cas(lock + offset, 0, 1) == 1:
        pass

    tl.store(ptrs, tl.load(ptrs) + 1.0)
    tl.atomic_xchg(lock + offset, 0)


def run_counter_triton():
    counter = jnp.zeros((10,), jnp.int32)
    lock = jnp.zeros((10,), jnp.int32)

    out_shape = (
        jax.ShapeDtypeStruct((10,), jnp.int32),
        jax.ShapeDtypeStruct((10,), jnp.int32),
    )
    return jt.triton_call(
        counter,
        lock,
        kernel=counter_kernel_triton,
        out_shape=out_shape,
        grid=(10, 10),
        input_output_aliases={
            0: 0,
            1: 1,
        },
    )


def counter_kernel_pallas(
    _,
    __,
    lock,
    counter,
):
    id_ = pl.program_id(0)
    def _cond(_):
        return pl.atomic_cas(lock, 0, 1) == 1
    jax.lax.while_loop(_cond, lambda a: a, 0)
    counter[...] += 1
    pl.atomic_xchg(lock, (), 0)


def run_counter_pallas():
    lock = jnp.zeros((10,), jnp.int32)
    counter = jnp.zeros((10,), jnp.int32)
    return pl.pallas_call(
        counter_kernel_pallas,
        grid=(10, 10),
        out_shape=(
            jax.ShapeDtypeStruct((10,), jnp.int32),
            jax.ShapeDtypeStruct((10,), jnp.int32),
        ),
        in_specs=[
            pl.BlockSpec(lambda c, i: (c), (None,)),
            pl.BlockSpec(lambda c, i: (c), (1,)),
        ],
        out_specs=[
            pl.BlockSpec(lambda c, i: (c), (None,)),
            pl.BlockSpec(lambda c, i: (c), (1,)),
        ],
        name="simple_example_kernel",
        input_output_aliases={
            0: 0,
            1: 1,
        },
    )(
        lock,
        counter,
    )
```
As I understand it, the problem lies in the offset of the reference from block specs not reaching atomic_cas.

### What jax/jaxlib version are you using?

jax v0.4.20, jaxlib 0.4.20

### Which accelerator(s) are you using?

GPU

### Additional system info?

_No response_

### NVIDIA GPU info

_No response_

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises pallas atomic_cas behavior for the reported value and memory update.

## Observed implementation decision

The observed compare-and-swap relation differs from the reporter's expected success/value semantics.

## Physical effect

The returned value or storage update is bound to the wrong CAS outcome.

## Observed consequence

The Pallas program produces unexpected atomic results.

## Reported proposal or workaround

No accepted workaround or implementation proposal is recorded.

## Repair trace

The issue remains open with no maintainer-confirmed root cause, landed patch, or fixed regression.

## Archived evidence

- Raw report: `data/raw/github/jax/18909/report.json`
- SHA-256: `9a3dfd26fc6b9c80b93ed46196c4a505616c55d04f4cff9f44409eebb4b581ea`
