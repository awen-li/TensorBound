# No error message but failing zero-copy?

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/30228](https://github.com/jax-ml/jax/issues/30228)
- Opened: 2025-07-15T18:28:54Z
- Closed: 2025-07-16T03:24:30Z
- Labels: `bug`

## Reported behavior

### Description

Greetings.
I’m (still) trying to get JAX and PETSc to work together in a no-copy system using the DLPack tools in both. Unfortunately I can’t seem to get it to work right. Ideally, I’d like to create a PETSc vec object using petsc4py, pass it to to a JAX object without copying, make a change to it in a JAX jitted function and have that change reflected in the PETSc object. All of this without copying.

Of note: When I try to do this I get an error that the alignment is wrong and a copy must be made when I call the from-dlpack function, but changing the alignment in the PETSc ./config stage to 32 causes the error message to disappear, even so it still doesn’t function correctly. I’ve tried looking through the documentation, but I’m getting a little turned around.

I’ve included a code snippet below:

```
from petsc4py import PETSc as PETSc

import jax

from functools import partial

import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

@partial(jax.jit, donate_argnums=(0,))

def set_in_place(x):

    return [x.at](http://x.at/)[:].set(3.0)



print('\nTesting jax from_dlpack given a PETSc vector that was allocated by PETSc')

x = jnp.ones((1000,1))

y_petsc = PETSc.Vec().createSeq(x.shape[0])

y_petsc.set(0.0)

print(hex(y_petsc.handle))

y2_petsc = PETSc.Vec().createWithDLPack(y_petsc.toDLPack('rw'))

y2_petsc.set(-1.0)

assert y_petsc.getValue(0) == y2_petsc.getValue(0)

print('After creating a second PETSc vector via a DLPack of the first, modifying the memory of one affects the other.')

#y = jnp.from_dlpack(y_petsc.toDLPack('rw'), copy=False)

y = jnp.from_dlpack(y_petsc, copy=False)

orig_ptr = y.unsafe_buffer_pointer()

print(f'before: ptr at {hex(orig_ptr)}')

y = set_in_place(y)

print(f'after:  ptr at {hex(y.unsafe_buffer_pointer())}')

assert orig_ptr == y.unsafe_buffer_pointer()

#assert y_petsc.getValue(0) == y[0], f'The PETSc value {y_petsc.getValue(0)} did not match the JAX value {y[0]}, so modifying the JAX memory did not affect the PETSc memory.'



```
I’d like the bottom two asserts to pass, but I can only get one of them. I asked about it in discussion a few weeks ago but heard nothing so I thought it may ask through a different avenue, I hope that's alright. I looked into the documentation I couldn't see anything that mentioned why this might be. Furthermore no error is being thrown up either by PETSc or JAX so I'm unsure where the problem may lie.

Alberto

### System info (python version, jaxlib version, accelerator, etc.)

jax:    0.6.0
jaxlib: 0.6.0
numpy:  1.26.2
python: 3.11.4 (tags/v3.11.4:d2340ef, Jun  7 2023, 05:45:37) [MSC v.1934 64 bit (AMD64)]
device info: cpu-1, 1 local devices"
process_count: 1
platform: uname_result(system='Windows', node='Alberto-Grad', release='10', version='10.0.26100', machine='AMD64')


$ nvidia-smi
Tue Jul 15 14:27:47 2025
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 546.30                 Driver Version: 546.30       CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3050 ...  WDDM  | 00000000:01:00.0  On |                  N/A |
| N/A   38C    P8               4W /  40W |    590MiB /  4096MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      3492    C+G   ...CBS_cw5n1h2txyewy\TextInputHost.exe    N/A      |
|    0   N/A  N/A      5940    C+G   ...m Files\Mozilla Firefox\firefox.exe    N/A      |
|    0   N/A  N/A      6636    C+G   ...1.0_x64__8wekyb3d8bbwe\Video.UI.exe    N/A      |
|    0   N/A  N/A      7016    C+G   ...t.LockApp_cw5n1h2txyewy\LockApp.exe    N/A      |
|    0   N/A  N/A      8300    C+G   ...2txyewy\StartMenuExperienceHost.exe    N/A      |
|    0   N/A  N/A     10404    C+G   C:\Windows\explorer.exe                   N/A      |
|    0   N/A  N/A     11220    C+G   ...cw5n1h2txyewy\CrossDeviceResume.exe    N/A      |
|    0   N/A  N/A     11708    C+G   ...5.0_x64__w2gh52qy24etm\Nahimic3.exe    N/A      |
|    0   N/A  N/A     12048    C+G   ...on\138.0.3351.77\msedgewebview2.exe    N/A      |
|    0   N/A  N/A     12196    C+G   ...nt.CBS_cw5n1h2txyewy\SearchHost.exe    N/A      |
|    0   N/A  N/A     13148    C+G   ...on\138.0.3351.77\msedgewebview2.exe    N/A      |
|    0   N/A  N/A     14204    C+G   ...5n1h2txyewy\ShellExperienceHost.exe    N/A      |
|    0   N/A  N/A     15284    C+G   ...al\Discord\app-1.0.9198\Discord.exe    N/A      |
|    0   N/A  N/A     15740    C+G   ...m Files\Mozilla Firefox\firefox.exe    N/A      |
|    0   N/A  N/A     18280    C+G   ...oogle\Chrome\Application\chrome.exe    N/A      |
|    0   N/A  N/A     18600    C+G   ...__8wekyb3d8bbwe\Notepad\Notepad.exe    N/A      |
|    0   N/A  N/A     21080    C+G   ...oogle\Chrome\Application\chrome.exe    N/A      |
|    0   N/A  N/A     24208    C+G   ...\cef\cef.win7x64\steamwebhelper.exe    N/A      |
|    0   N/A  N/A     24260    C+G   ...crosoft\Edge\Application\msedge.exe    N/A      |
|    0   N/A  N/A     27912    C+G   ...Programs\Microsoft VS Code\Code.exe    N/A      |
|    0   N/A  N/A     30076    C+G   ...273.0_x64__dt26b99r8h8gj\RtkUWP.exe    N/A      |
|    0   N/A  N/A     34048    C+G   ...s\System32\ApplicationFrameHost.exe    N/A      |
|    0   N/A  N/A     43348    C+G   C:\Windows\System32\ShellHost.exe         N/A      |
+---------------------------------------------------------------------------------------+

## Trigger

jnp.from_dlpack imports a PETSc vector whose address does not satisfy JAX's zero-copy alignment requirement and explicitly requests copy=False.

## Relevant entities

PETSc allocation; DLPack capsule; copy parameter; alignment requirement; JAX buffer; donated result

## Implemented relation

copy=False either aliases the producer storage or reports that zero-copy import is impossible; it does not return independent storage.

## Pre-state

The caller requests a shared view so later donated JAX writes are observable through the PETSc object.

## Binding transition

The old C++ importer handles invalid CPU zero-copy conditions by creating a copy before the Python layer applies the copy parameter.

## Post-state

The JAX array can occupy a separate allocation even though import returns successfully under copy=False.

## Validity obligation

The copy policy reaches the component that chooses between reusing producer storage, allocating a copy, and rejecting the import.

## Invalid decision

dlpack_managed_tensor_to_buffer did not receive copy and silently used its CPU fallback when zero-copy buffer creation failed.

## Physical effect

JAX mutations update the copied buffer rather than the PETSc-owned allocation.

## Observed consequence

Pointer or value checks show that the PETSc view does not observe the JAX update despite copy=False and no corresponding error.

## Repair site

jax/_src/dlpack.py and jaxlib/dlpack.cc DLPackManagedTensorToBuffer

## Repair

PR 29963 passes copy into the native importer: False disables fallback and raises when copying is required, True performs a copy, and None retains the permitted fallback policy.

## Why the repair works

The storage-selection decision now consumes the caller's copy contract, preventing a successful copy=False return from binding the array to independent storage.

## Causal basis

Maintainers map the report to PR 29963 and later explain its alignment-specific behavior; the merged patch threads copy through the exact native fallback and adds aligned and misaligned tests.

## Analysis implication

Interop checking needs the requested copy policy, producer ownership, device, pointer alignment, native fallback result, destination pointer, and donation semantics.

## Fix patches

- [`fixes/jax-30228-pr29963.patch`](fixes/jax-30228-pr29963.patch) — https://github.com/jax-ml/jax/pull/29963
