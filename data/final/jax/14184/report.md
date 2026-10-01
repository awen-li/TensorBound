# `jax.device_put` not working under `multiprocessing`

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/14184](https://github.com/jax-ml/jax/issues/14184)
- Opened: 2023-01-27T15:27:02Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

```python
import jax.numpy as jnp
import jax
import multiprocessing as mp
import threading
import queue

def worker(queue):
    print(jax.devices())
    array = jax.device_put(jnp.ones(10), jax.devices()[1])
    print(array.unsafe_buffer_pointer())
    print(f"array before queue is on device {array.device()}")
    queue.put(array)

q = mp.Queue(maxsize=4)
mp.Process(target=worker, args=(q,)).start()
array = q.get()
print(array.unsafe_buffer_pointer())
print(f"multiprocessing: array after queue is on device {array.device()}")

q = queue.Queue(maxsize=4)
threading.Thread(target=worker, args=(q,)).start()
array = q.get()
print(array.unsafe_buffer_pointer())
print(f"threading: array after queue is on device {array.device()}")
```
```
[GpuDevice(id=0, process_index=0), GpuDevice(id=1, process_index=0)]
139659753553920
array before queue is on device gpu:1
139667471073280
multiprocessing: array after queue is on device gpu:0
[GpuDevice(id=0, process_index=0), GpuDevice(id=1, process_index=0)]
139659619336192
array before queue is on device gpu:1
139659619336192
threading: array after queue is on device gpu:1
```

Note that when using threading, the array was successfully put on GPU 1, but this fails with multiprocessing — the array was somehow placed back to GPU 0 when retrieved from the `queue`.

### What jax/jaxlib version are you using?

jax '0.3.17' jaxlib '0.3.15'

### Which accelerator(s) are you using?

GPU

### Additional system info

Ubuntu

### NVIDIA GPU info


Two RTX 3060 TIs
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:04:00.0 Off |                  N/A |
|  0%   37C    P8    17W / 200W |      8MiB /  7982MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:09:00.0 Off |                  N/A |
| 30%   39C    P8    20W / 200W |     18MiB /  7981MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1444      G   /usr/lib/xorg/Xorg                  4MiB |
|    1   N/A  N/A      1444      G   /usr/lib/xorg/Xorg                  9MiB |
|    1   N/A  N/A      1665      G   /usr/bin/gnome-shell                3MiB |
+-----------------------------------------------------------------------------+
```

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises jax.device_put execution from multiprocessing workers.

## Observed implementation decision

Forked/process-local runtime state does not provide a valid device/client binding for the transfer.

## Physical effect

The transfer targets runtime state that is unavailable or incompatible in the child.

## Observed consequence

device_put fails under multiprocessing.

## Reported proposal or workaround

The report contains no completed implementation proposal.

## Repair trace

No source-localized repair, landed patch, or fixed regression is linked.

## Archived evidence

- Raw report: `data/raw/github/jax/14184/report.json`
- SHA-256: `10e4587e3f64fa6d68b227c0b290ec854ab2a7f84a5d916f46e3807b746afd50`
