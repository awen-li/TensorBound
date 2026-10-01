# Pickling a JAX array does not preserve its original device

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/12253](https://github.com/jax-ml/jax/issues/12253)
- Opened: 2022-09-07T13:21:14Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

### What I hope to achieve

The JAX arrays are created in a subprocess, evoked by `multiprocessing.map`, on CPU. I hope that when the arrays are sent to the main process, they are still on the same device (i.e. CPU). However, after they are sent to the main process, they are placed on the default device (i.e. GPU) instead.

```python
import jax
import jax.numpy as np
import multiprocessing

def f():
    jax.config.update('jax_platforms', 'cpu')
    a = np.zeros((2,))
    print(a.device())  # TFRT_CPU_0
    return a

def main():
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(1) as p:
        a = p.starmap(f, ((),))[0]
    print(a.device())  # gpu:0, expected TFRT_CPU_0

if __name__ == '__main__':
    main()
```

### This also applies to the underlying pickle module

```python
import jax
import jax.numpy as np
import pickle

jax.config.update('jax_platforms', 'cpu')

a = np.zeros((2,))
print(a.device())  # TFRT_CPU_0

with open('1.dat', 'wb') as f:
    pickle.dump(a, f)
```

```python
import pickle

with open('1.dat', 'rb') as f:
    a = pickle.load(f)
print(a.device())  # gpu:0, expected TFRT_CPU_0
```

### What jax/jaxlib version are you using?

jax 0.3.17, jaxlib 0.3.15+cuda11.cudnn82

### Which accelerator(s) are you using?

GPU

### Additional System Info

Python 3.10.6, Arch Linux x86_64

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises jAX array pickling and restoration of original device placement.

## Observed implementation decision

Serialization preserves logical values but not the array's device binding.

## Physical effect

Unpickling reconstructs the array under the current/default placement.

## Observed consequence

The restored array resides on a different device.

## Reported proposal or workaround

Discussion explores storing device information with the pickle.

## Repair trace

The issue remains open without a landed serialization-format change or fixed regression.

## Archived evidence

- Raw report: `data/raw/github/jax/12253/report.json`
- SHA-256: `f982e69e587c2d7be3a220077a17e81f45a099781ae2444611f54ef594808dc9`
