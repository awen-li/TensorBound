# Multi-GPU Workflow Hangs

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/11637](https://github.com/jax-ml/jax/issues/11637)
- Opened: 2022-07-28T00:08:21Z
- Closed: 2022-08-01T15:29:38Z
- Labels: `bug`, `NVIDIA GPU`

## Reported behavior

Potentially related to #10969.

When running the following code on a single device, it will run as expected. However, if n_devices > 1, I'm observing all devices will allocate memory but an inconsistent number will perform computation (as displayed via nvtop / nvidia-smi), leading to an indefinite hang.

Environment:
* OS: ubuntu 20.04
* Num GPU: 4
* Python Version: 3.8.10
* Nvidia Driver: 515.48.07
* CUDA Version: 11.7
* Jax Version: 0.3.14


While the actual code in which I'm seeing this occur is far more complex, I've created a simpler repro.
```python
def func(num_rows_per_device, num_devices, num_features):
    num_entries = num_rows_per_device * num_devices * num_features

    # often hangs
    def fn(x_):
        return jax.lax.cond(
            x_.sum() > num_entries,
            lambda: 0.,
            lambda: jnp.sum(x_**2)
        )

    # doesn't seem to hang
    # def fn(x_):
    #     return x_.sum()

    data = jnp.arange(num_entries).reshape(num_rows_per_device, num_devices, num_features).astype(jnp.float32)
    return jax.lax.map(jax.pmap(fn), data)

print(jax.jit(func, static_argnames=["num_rows_per_device", "num_devices", "num_features"])(5, 4, 2))
# seems to hang quite frequently
```

## Trigger

Local GPU threads initialize one NCCL clique concurrently and one rank finishes ncclCommInitRank before another rank.

## Relevant entities

local GPU ranks; NCCL communicator; collective stream; device page tables; CUDA allocator lock; host-buffer copy

## Implemented relation

All local participants in an NCCL clique complete communicator initialization before any participant issues a collective using that clique.

## Pre-state

No rank in the clique has an initialized communicator.

## Binding transition

The old acquisition path returns each rank's communicator as soon as that rank completes ncclCommInitRank.

## Post-state

One rank can issue the collective and then block in page-table work while another rank is still allocating during communicator initialization.

## Validity obligation

Collective execution begins only after every local rank has completed ncclCommInitRank for the shared clique.

## Invalid decision

Communicator acquisition lacked a clique-wide post-initialization barrier.

## Physical effect

One rank holds the page-table lock while waiting for peer collective progress, and the peer waits for that lock inside cudaMalloc during initialization.

## Observed consequence

The reported four-GPU JAX collective intermittently deadlocks.

## Repair site

tensorflow/compiler/xla/service/gpu/nccl_utils.cc::AcquireNcclComm

## Repair

Upstream commit f9df012 stores communicators in clique state and waits until the number of initialized communicators equals the local participant count before notifying readiness.

## Why the repair works

No local rank can issue collective or page-table-dependent work until all peers have left communicator initialization.

## Causal basis

A maintainer reproduces the deadlock and gives the lock cycle in the report; the upstream commit names issue 11637, implements that barrier, and enters JAX through PR 11691's XLA update.

## Analysis implication

Collective analysis needs clique identity, local participant count, per-rank initialization state, readiness notification, stream progress, and allocator-lock interactions.

## Fix patches

- [`fixes/jax-11637-xla-fix.patch`](fixes/jax-11637-xla-fix.patch) — https://github.com/tensorflow/tensorflow/commit/f9df01236193494a18f05fdc0b12391552db0b00
