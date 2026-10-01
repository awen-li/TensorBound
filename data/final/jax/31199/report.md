# Bug in `out_shardings`?

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/31199](https://github.com/jax-ml/jax/issues/31199)
- Opened: 2025-08-20T20:42:53Z
- Closed: 2025-08-20T22:07:48Z
- Labels: `bug`

## Reported behavior

### Description

This is adapted from here: https://github.com/jax-ml/jax/discussions/31127

I understand that `jax.lax.with_sharding_constraint` is not meant for eager regions but the following seems like a bug, perhaps lying with something unrelated to `jax.lax.with_sharding_constraint`. There @yashk2810 mentioned the eager behavior was implemented with `out_shardings`.

The following fails:

```python
import jax
import numpy as np
import jax.numpy as jnp

cpu = jax.devices()[0]
sharding = jax.sharding.SingleDeviceSharding(cpu)

x = jnp.arange(10)  # jax array
y_wsc = jax.lax.with_sharding_constraint(x, sharding)
y_put = jax.device_put(x, sharding)
assert y_wsc.committed == y_put.committed == True  # This fails; y_wsc.committed = False
```

While strangely, this passes:

```python
import jax
import jax.sharding as jshard
import numpy as np
import jax.numpy as jnp

jax.config.update("jax_num_cpu_devices", 2)

num_devices = len(jax.devices())
mesh = jax.make_mesh((num_devices,), ("batch",))
sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))

x = jnp.arange(10)  # jax array
y_wsc = jax.lax.with_sharding_constraint(x, sharding)
y_put = jax.device_put(x, sharding)
assert y_wsc.committed == y_put.committed == True  # This passes
```

Also strangely, if I set `x = np.arange(10)`, both cases pass.

### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.7.0
jaxlib: 0.7.0
numpy:  2.3.2
python: 3.11.13 | packaged by conda-forge | (main, Jun  4 2025, 14:52:34) [Clang 18.1.8 ]
device info: cpu-1, 1 local devices"
process_count: 1
platform: uname_result(system='Darwin', node='dhcp-10-250-233-140.harvard.edu', release='24.6.0', version='Darwin Kernel Version 24.6.0: Mon Jul 14 11:30:51 PDT 2025; root:xnu-11417.140.69~1/RELEASE_ARM64_T8112', machine='arm64')
```

## Trigger

Eager with_sharding_constraint receives an uncommitted JAX array and an equivalent SingleDeviceSharding.

## Relevant entities

input array; requested sharding; input committed flag; output array; identity jit

## Implemented relation

The eager constraint promises an output array committed to the requested sharding.

## Pre-state

The input has an equivalent device placement but is not committed.

## Binding transition

_sharding_constraint_impl returns the original input through its equivalent-sharding fast path.

## Post-state

The output retains the input object's uncommitted state although the constraint requested a committed result.

## Validity obligation

Applying an eager output-sharding constraint materializes the output semantics, including commitment, even when placement is already equivalent.

## Invalid decision

The fast path equated sharding equivalence with full output-semantic equivalence and returned x unchanged.

## Physical effect

No identity executable or device transfer creates a newly committed array object.

## Observed consequence

with_sharding_constraint returns committed=False while device_put with the same sharding returns committed=True.

## Repair site

jax/_src/pjit.py::_sharding_constraint_impl

## Repair

PR 31203 removes both equivalent-sharding forwarding branches and always runs the identity jit for eager constraints.

## Why the repair works

The identity jit constructs an output under out_shardings, so the returned array carries the requested committed state.

## Causal basis

The maintainer points to the forwarding shortcut, and the patch removes only that shortcut and asserts commitment in the regression.

## Analysis implication

An identity-elision check needs all observable array state, not only equivalent sharding and layout.

## Fix patches

- [`fixes/jax-31199-pr31203.patch`](fixes/jax-31199-pr31203.patch) — https://github.com/jax-ml/jax/pull/31203
