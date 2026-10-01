# process can exit before jax.debug.prints, needs jax.effects_barrier()

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/19867](https://github.com/jax-ml/jax/issues/19867)
- Opened: 2024-02-18T03:58:01Z
- Closed: 2024-07-20T18:11:58Z
- Labels: `bug`, `P2 (eventual)`

## Reported behavior

### Description

This does not print anything:

```python
import jax
import jax.numpy as jnp

def scanner(carry, x):
  jax.debug.print("carry={carry} x={x}", carry=carry, x=x)
  return carry + x, x


out = jax.lax.scan(scanner, jnp.array(0.), jnp.arange(10.))
```

For comparison, this for-loop-based version gives the expected output:
```python
carry = 0.
for x in jnp.arange(10.):
  jax.debug.print("carry={carry} x={x}", carry=carry, x=x)
  carry = carry + x
```
```
carry=0.0 x=0.0
carry=0.0 x=1.0
carry=1.0 x=2.0
carry=3.0 x=3.0
carry=6.0 x=4.0
carry=10.0 x=5.0
carry=15.0 x=6.0
carry=21.0 x=7.0
carry=28.0 x=8.0
carry=36.0 x=9.0
```

### System info (python version, jaxlib version, accelerator, etc.)

```
>>> import jax; jax.print_environment_info()
jax:    0.4.23
jaxlib: 0.4.23
numpy:  1.26.1
python: 3.11.4 (main, Nov 20 2023, 15:01:59) [GCC 12.3.0]
jax.devices (2 total, 2 local): [cuda(id=0) cuda(id=1)]
process_count: 1

$ nvidia-smi
Sat Feb 17 22:57:36 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.06              Driver Version: 545.23.06    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA RTX A6000               On  | 00000000:2D:00.0 Off |                  Off |
| 30%   26C    P8              18W / 300W |  36793MiB / 49140MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA RTX A6000               On  | 00000000:41:00.0 Off |                  Off |
| 30%   29C    P8              20W / 300W |    271MiB / 49140MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A     25870      C   ...anifest2-ozEuPuop-py3.11/bin/python    36784MiB |
|    1   N/A  N/A     25870      C   ...anifest2-ozEuPuop-py3.11/bin/python      262MiB |
+---------------------------------------------------------------------------------------+
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

A program dispatches debug prints inside lax.scan and then reaches process exit without an explicit effects barrier.

## Observed implementation decision

The host process can terminate before asynchronous callback effects complete.

## Physical effect

Pending print callbacks are abandoned before their output is emitted.

## Observed consequence

The scan produces no visible debug output although the equivalent eager loop prints all iterations.

## Reported proposal or workaround

jax.effects_barrier completes pending effects; another user reports that 0.4.26 prints without it.

## Repair trace

The 0.4.23-to-0.4.26 history contains no identified process-exit or pending-callback repair tied to this report, and no regression records automatic exit draining. A single later-version observation does not establish the repair delta.

## Archived evidence

- Raw report: `data/raw/github/jax/19867/report.json`
- SHA-256: `a27ac63b86cb129fd5f1925c88ddadf34a7676505443adaaa5eac403406fbedc`
