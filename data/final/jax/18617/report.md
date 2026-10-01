# Buffer donation fails after loading from disk

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/18617](https://github.com/jax-ml/jax/issues/18617)
- Opened: 2023-11-21T06:16:27Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

I've found what I suspect to be a bug in donating buffers. Create an array that is small enough that it fits in memory, but large enough that it does *not* fit in memory twice. Calling a simple pmapped function with a correctly-set `donate_buffers` should work, since the large array is never in memory twice. This works if we create the array directly, but fails if we create the array via a load from disk using `pickle`.

The code to reproduce is below. Set the `GIANT_ARRAY_SIZE` to an appropriate size such that setting `WHICH=1` fails but `WHICH=2` succeeds, and you will see that `WHICH=3` fails (it should succeed).

```python
import pickle
import jax
import jax.numpy as jnp

# This is the function I want to call
def add1(s):
    return jax.tree_map(lambda x: x + 1, s)

# This is the input. Adjust the size based on your GPU.
# This number is for an H100 with XLA_PYTHON_CLIENT_MEM_FRACTION=1.
GIANT_ARRAY_SIZE = 9750078228
X = {'w': jnp.arange(GIANT_ARRAY_SIZE, dtype=jnp.float32)}
X = jax.device_put_replicated(X, jax.local_devices())
jax.block_until_ready(X)
print('Created X')

WHICH = 1
print('Running', WHICH)
# V1: call without donated buffer
if WHICH == 1:
    p_add1 = jax.pmap(add1, axis_name='data_parallel')
    X = p_add1(X) # crashes OOM

# V2: call with donated buffer
elif WHICH == 2:
    p_add1 = jax.pmap(add1, axis_name='data_parallel', donate_argnums=0)
    X = p_add1(X)  # succeeds

# V3: call with donated buffer after loading from pickle
else:
    path = '/tmp/test.pkl'
    with open(path, 'wb') as file:
        pickle.dump(X, file)
    print('Dumped X')
    del X
    print('Deleted X')
    with open(path, 'rb') as file:
        X = pickle.load(file)
    print('Loaded X')

    p_add1 = jax.pmap(add1, axis_name='data_parallel', donate_argnums=0)
    X = p_add1(X) # crashes OOM

jax.block_until_ready(X)
print('Finished')
```

### What jax/jaxlib version are you using?

jax v0.4.20, jaxlib v0.4.20

### Which accelerator(s) are you using?

GPU

### Additional system info

Python 311, Ubuntu 22.04

### NVIDIA GPU info

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA H100 80GB HBM3          On  | 00000000:04:00.0 Off |                  Off |
| N/A   38C    P0             113W / 700W |   1332MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
+---------------------------------------------------------------------------------------+
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises jAX buffer donation after arrays are restored from checkpoint storage.

## Observed implementation decision

Restored arrays carry ownership/commitment state that prevents the expected donation transition.

## Physical effect

Execution allocates rather than transferring the restored buffer to the donated result.

## Observed consequence

Donation fails after loading from disk.

## Reported proposal or workaround

An external checkpointing proposal discusses reconstructing suitable arrays.

## Repair trace

The external PR was closed and no upstream JAX repair or fixed donation regression is linked.

## Archived evidence

- Raw report: `data/raw/github/jax/18617/report.json`
- SHA-256: `9cbc6ebd54276e643a22a835fefdebe9b6e457a3c4e47c99bf43b2da33a37c2f`
