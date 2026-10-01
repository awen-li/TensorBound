# jax_threefry_partitionable + rematerialization doesn't seem to be working together in distributed training

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/17982](https://github.com/jax-ml/jax/issues/17982)
- Opened: 2023-10-06T14:22:33Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

I have a transformer model where each transformer block is rematerialized. The model is distributed over multiple devices using jit. Each transformer block has dropout enabled.

To prevent rng implementation from inserting synchronization operations I'm also enabling `jax_threefry_partitionable` as suggested in the doc.

Problem is, `jax_threefry_partitionable` doesn't seem to play nicely with rematerialization. As soon as I enable dropout, I get GPU OOM because JAX decides to preserve huge arrays containing rng key per activation tensor component for each transformer block, despite them being rematerialized. It should be possible for jax to reconstruct this key array from a single key during rematerialization, but it doesn't seem to do that.

I'm happy to provide a repoduction if you can confirm that this is unexpected behavior. If not, can you please suggest a workaround? Currently it doesn't seem possible to efficiently train large models with dropout.

A relevant discussion with OOM error message example here: https://github.com/google/flax/discussions/3090

### What jax/jaxlib version are you using?

0.4.14

### Which accelerator(s) are you using?

GPU

### Additional system info

python3.10

### NVIDIA GPU info

```
Fri Oct  6 14:22:06 2023
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.12             Driver Version: 535.104.12   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA H100 80GB HBM3          On  | 00000000:8A:00.0 Off |                    0 |
| N/A   35C    P0              72W / 700W |      4MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA H100 80GB HBM3          On  | 00000000:8B:00.0 Off |                    0 |
| N/A   31C    P0              71W / 700W |      4MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA H100 80GB HBM3          On  | 00000000:8C:00.0 Off |                    0 |
| N/A   31C    P0              72W / 700W |      4MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA H100 80GB HBM3          On  | 00000000:8D:00.0 Off |                    0 |
| N/A   35C    P0              74W / 700W |      4MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   4  NVIDIA H100 80GB HBM3          On  | 00000000:9C:00.0 Off |                    0 |
| N/A   36C    P0              76W / 700W |      4MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   5  NVIDIA H100 80GB HBM3          On  | 00000000:9D:00.0 Off |                    0 |
| N/A   31C    P0              72W / 700W |      4MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   6  NVIDIA H100 80GB HBM3          On  | 00000000:9E:00.0 Off |                    0 |
| N/A   31C    P0              71W / 700W |      4MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   7  NVIDIA H100 80GB HBM3          On  | 00000000:9F:00.0 Off |                    0 |
| N/A   35C    P0              73W / 700W |      4MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises partitionable Threefry random generation combined with rematerialization in distributed training.

## Observed implementation decision

Rematerialization and random-key partitioning do not retain the intended sharding/identity relation across the recomputed region.

## Physical effect

Recomputed random work is placed or replicated differently from the forward computation.

## Observed consequence

Distributed training fails or consumes excessive memory.

## Reported proposal or workaround

The issue discussion explores configuration changes.

## Repair trace

No agreed root decision, landed repair, and matching regression are available.

## Archived evidence

- Raw report: `data/raw/github/jax/17982/report.json`
- SHA-256: `a6c34b253471d16a0c50a3edd7710a98bdc2e1529b83083a87220c645e638410`
