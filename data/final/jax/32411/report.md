# vjp3 + array refs produce incorrect gradient

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/32411](https://github.com/jax-ml/jax/issues/32411)
- Opened: 2025-10-07T15:23:36Z
- Closed: 2025-10-08T19:38:26Z
- Labels: `bug`

## Reported behavior

### Description

Hey, I'm very excited with introduction of vjp3 + array refs for zero-memory-overhead gradient accumulation in JAX. I've tried to implement it in my training codebase and noticed that loss curves on vjp3+refs doesn't match original baseline. After some digging into, I've noticed that the problem occurs only when using vjp3 in conjunction with array refs, using only vjp3 without mutable arrays perfectly matches the baseline.

I've come up with the following repro to demonstrate the issue:

```python
import jax
import jax.numpy as jnp
from jax._src.api import vjp3  # another prototype of a user-facing api for VJPs with refs
import jax.ref
import functools

NUM_LAYERS = 3
EMB_SIZE = 4*1024
NUM_MUBATCHES = 5
MUBATCH_SIZE = 7

USE_ARRAY_REFS = True


def mubatch_loss(Ws, xs, ys):
  # Inner loop: scan over layers
  act, _ = jax.lax.scan(lambda xs, W: (jnp.tanh(jnp.dot(xs, W)), None), xs, Ws)
  ys_pred = jnp.mean(act, axis=-1)
  return jnp.mean((ys - ys_pred) ** 2)

def process_batch(Ws, xs_batch, ys_batch, use_array_refs=USE_ARRAY_REFS):
  if use_array_refs:
    grad_acc = jax.array_ref(jnp.zeros_like(Ws))
    def process_mubatch(_, xs_ys):
      xs, ys = xs_ys
      loss, mubatch_vjp = vjp3(lambda Ws: mubatch_loss(Ws, xs, ys), Ws)
      mubatch_vjp.with_refs(grad_acc)(jnp.ones_like(loss))
      return (), loss
  else:
    def process_mubatch(grad_acc, xs_ys):
      xs, ys = xs_ys
      loss, mubatch_grad = jax.value_and_grad(mubatch_loss)(Ws, xs, ys)
      grad_acc = grad_acc + mubatch_grad
      return grad_acc, loss

  assert xs_batch.shape[0] == NUM_MUBATCHES * MUBATCH_SIZE
  xs_mubatches = xs_batch.reshape(NUM_MUBATCHES, MUBATCH_SIZE, *xs_batch.shape[1:])
  ys_mubatches = ys_batch.reshape(NUM_MUBATCHES, MUBATCH_SIZE, *ys_batch.shape[1:])

  # Outer loop: scan over microbatches
  if use_array_refs:
    batch_grad, _losses = jax.lax.scan(process_mubatch, (), (xs_mubatches, ys_mubatches))
    batch_grad = jax.ref.freeze(grad_acc)
  else:
    batch_grad = jnp.zeros_like(Ws)
    batch_grad, _losses = jax.lax.scan(process_mubatch, batch_grad, (xs_mubatches, ys_mubatches))

  batch_grad = batch_grad / NUM_MUBATCHES

  return jnp.sum(_losses), Ws - batch_grad * 0.5


Ws_ref = jax.random.normal(jax.random.PRNGKey(0), shape=(NUM_LAYERS, EMB_SIZE, EMB_SIZE)) / jnp.sqrt(EMB_SIZE)
Ws_no_ref = jax.random.normal(jax.random.PRNGKey(0), shape=(NUM_LAYERS, EMB_SIZE, EMB_SIZE)) / jnp.sqrt(EMB_SIZE)
xs_batch = jax.random.normal(jax.random.PRNGKey(0), shape=(NUM_MUBATCHES * MUBATCH_SIZE, EMB_SIZE))
ys_batch = jnp.ones(shape=(NUM_MUBATCHES * MUBATCH_SIZE,)) * 0.5


process_batch_ref_jit = jax.jit(functools.partial(process_batch, use_array_refs=True), donate_argnums=(0,)).lower(Ws_ref, xs_batch, ys_batch).compile()
process_batch_no_ref_jit = jax.jit(functools.partial(process_batch, use_array_refs=False), donate_argnums=(0,)).lower(Ws_no_ref, xs_batch, ys_batch).compile()

with open("compiled_ref.txt", "w") as f:
  f.write(process_batch_ref_jit.as_text())

with open("compiled_no_ref.txt", "w") as f:
  f.write(process_batch_no_ref_jit.as_text())

for i in range(250):
  loss_ref, Ws_ref = process_batch_ref_jit(Ws_ref, xs_batch, ys_batch)
  loss_no_ref, Ws_no_ref = process_batch_no_ref_jit(Ws_no_ref, xs_batch, ys_batch)
  print(f"Step {i}, ref loss: {loss_ref}, no ref loss: {loss_no_ref}")

```

The repro ends up with the following output:
```
Step 246, ref loss: 0.2064920961856842, no ref loss: 3.473728249492325e-10
Step 247, ref loss: 0.20546941459178925, no ref loss: 3.269319259757708e-10
Step 248, ref loss: 0.20445892214775085, no ref loss: 3.097700151943883e-10
Step 249, ref loss: 0.20345085859298706, no ref loss: 2.8735352985975737e-10
```

It is also interesting that setting the NUM_MUBATCHES to 1 (effectively disabling the grad accum) produces similar results where the vjp3 + refs converges way more slowly:
```
Step 245, ref loss: 0.006776727735996246, no ref loss: 4.745410878484457e-13
Step 246, ref loss: 0.006740346550941467, no ref loss: 4.745410878484457e-13
Step 247, ref loss: 0.006704122293740511, no ref loss: 4.685775964290151e-13
Step 248, ref loss: 0.006668666377663612, no ref loss: 4.1592128504318315e-13
Step 249, ref loss: 0.006633440963923931, no ref loss: 4.1173415046315276e-13
```

I've tried it both on GPU and CPU platforms and results are the same so it doesn't seem like a platform-related issue.

### System info (python version, jaxlib version, accelerator, etc.)

```
root@computeinstance-e00p9411jfafcfq06n:/papyrax# python3 -c 'import jax; jax.print_environment_info()'
jax:    0.7.2
jaxlib: 0.7.2
numpy:  2.1.3
python: 3.11.13 (main, Jun  4 2025, 08:57:30) [GCC 13.3.0]
device info: NVIDIA H100 80GB HBM3-8, 8 local devices"
process_count: 1
platform: uname_result(system='Linux', node='computeinstance-e00p9411jfafcfq06n', release='5.15.0-152-generic', version='#162-Ubuntu SMP Wed Jul 23 09:48:42 UTC 2025', machine='x86_64')
XLA_PYTHON_CLIENT_MEM_FRACTION=.85
JAX_COMPILATION_CACHE_DIR=/mnt/llm/cache/jax_compilation_cache

$ nvidia-smi
Tue Oct  7 15:18:00 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.163.01             Driver Version: 550.163.01     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:8D:00.0 Off |                    0 |
| N/A   36C    P0            120W /  700W |     558MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA H100 80GB HBM3          On  |   00000000:91:00.0 Off |                    0 |
| N/A   32C    P0            122W /  700W |     538MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA H100 80GB HBM3          On  |   00000000:95:00.0 Off |                    0 |
| N/A   31C    P0            120W /  700W |     538MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA H100 80GB HBM3          On  |   00000000:99:00.0 Off |                    0 |
| N/A   36C    P0            123W /  700W |     538MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA H100 80GB HBM3          On  |   00000000:AB:00.0 Off |                    0 |
| N/A   37C    P0            123W /  700W |     538MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA H100 80GB HBM3          On  |   00000000:AF:00.0 Off |                    0 |
| N/A   32C    P0            120W /  700W |     538MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA H100 80GB HBM3          On  |   00000000:B3:00.0 Off |                    0 |
| N/A   31C    P0            121W /  700W |     538MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA H100 80GB HBM3          On  |   00000000:B7:00.0 Off |                    0 |
| N/A   37C    P0            124W /  700W |     538MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+

```

## Trigger

A reverse backward scan accumulates gradients through mutable array references.

## Relevant entities

physical loop counter; logical reverse index; mutable array ref; gradient update; scan length

## Implemented relation

Each reverse-scan iteration reads and updates the mutable-ref slot for the logical element processed in that iteration.

## Pre-state

The backward scan visits logical element length minus one first and element zero last.

## Binding transition

_scan_state_partial_discharge_rule uses its forward-increasing loop counter directly as the ref-array index.

## Post-state

Gradient updates are stored in forward slot order while their values correspond to reverse traversal order.

## Validity obligation

State discharge maps the physical counter to length - counter - 1 whenever scan reverse is true.

## Invalid decision

The old discharge rule ignored the reverse flag when indexing mutable-ref inputs and updates.

## Physical effect

The accumulated gradient tensor is reversed along the scanned layer dimension.

## Observed consequence

vjp3 with refs follows a different loss curve and converges much more slowly than the non-ref gradient baseline.

## Repair site

jax/_src/lax/control_flow/loops.py::_scan_state_partial_discharge_rule

## Repair

PR 32438 separates the increasing counter i_ from logical index i, computes i = length - i_ - 1 for reverse scans, and adds a reverse-vjp3 mutable-ref regression.

## Why the repair works

Every read and update uses the logical reverse index while loop progress still increments monotonically.

## Causal basis

The reporter identifies an exactly reversed gradient, and the merged patch applies reverse indexing at the mutable-array discharge rule and tests the vjp3 path.

## Analysis implication

Loop-state checking needs traversal direction, physical iteration count, logical element index, and mutable-reference update coordinates.

## Fix patches

- [`fixes/jax-32411-pr32438.patch`](fixes/jax-32411-pr32438.patch) — https://github.com/jax-ml/jax/pull/32438
