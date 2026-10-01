# `psum` fails with 1 process per GPU

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/12119](https://github.com/jax-ml/jax/issues/12119)
- Opened: 2022-08-26T05:06:03Z
- Closed: 2022-08-26T21:33:14Z
- Labels: `bug`, `XLA`, `P0 (urgent)`, `NVIDIA GPU`

## Reported behavior

### Description

After assigning GPU to a process using `jax.config.update("jax_cuda_visible_devices", localid)`, `psum` fails with the following error (localid = local process id) (I added additional debug print statement in XLA code to confirm):

```
jaxlib.xla_extension.XlaRuntimeError: INTERNAL: RET_CHECK failure (external/org_tensorflow/tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.cc:75) 0 <= local_device_ordinal && local_device_ordinal < gpu_global_device_ids->size()
local_device_ordinal: 3
gpu_global_device_ids size: 1
```

Note that such a failure doesn't occur if we use `os.environ["CUDA_VISIBLE_DEVICES"]=localid`

---

Here's the repro code:

```
import jax
import os
from absl import flags, app

flags.DEFINE_string('server_addr', 'localhost:54321', help='server ip addr')
flags.DEFINE_integer('num_hosts', 1, help='num of hosts' )
flags.DEFINE_integer('host_idx', 0, help='index of current host' )
FLAGS = flags.FLAGS

def main(argv):
    print(FLAGS.server_addr, FLAGS.num_hosts, FLAGS.host_idx)

    jax.config.update("jax_cuda_visible_devices", str(FLAGS.host_idx))

    jax.distributed.initialize(FLAGS.server_addr, FLAGS.num_hosts, FLAGS.host_idx)


    print(f"Device count: {jax.device_count()}")

    print(f"Local device count: {jax.local_device_count()}")

    xs = jax.numpy.ones(jax.local_device_count())
    y = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)

    print(f"output: {y}")

if __name__ == '__main__':
  app.run(main)
```

Launch 4 processes (or any number > 1) within shell as follows
```
python manual_dist.py --server_addr=127.0.0.1:54321 --num_hosts=4 --host_idx=0 &
python manual_dist.py --server_addr=127.0.0.1:54321 --num_hosts=4 --host_idx=1 &
python manual_dist.py --server_addr=127.0.0.1:54321 --num_hosts=4 --host_idx=2 &
python manual_dist.py --server_addr=127.0.0.1:54321 --num_hosts=4 --host_idx=3 &
```

### What jax/jaxlib version are you using?

jax/jaxlib 0.3.17 (built from head)

### Which accelerator(s) are you using?

GPU

### Additional System Info

Fails when launching multiple processes with 1.) SLURM (srun), 2.) MPI (mpirun) and 3.) shell

## Trigger

Four processes each expose one physical GPU whose local stream ordinal can be a nonzero physical ordinal, such as three.

## Relevant entities

local device ordinal; global device ID; process-local visible devices; GPU run options; NCCL participants; collective rank

## Implemented relation

Each local GPU ordinal resolves to the correct global device ID regardless of gaps in the process-visible ordinal set.

## Pre-state

A process has one local GPU entry but that entry may carry physical ordinal 1, 2, or 3.

## Binding transition

The old run options store global IDs in a compact vector and use the physical local_device_ordinal as a vector index.

## Post-state

Ordinal three indexes a one-element vector, so collective setup cannot identify the process's global participant.

## Validity obligation

The local-to-global identity mapping is keyed by the actual local ordinal rather than inferred from vector position.

## Invalid decision

BuildDistributedDevices discarded the ordinal keys from gpu_device_ids and GetGlobalDeviceId interpreted the remaining values as a dense ordinal-indexed vector.

## Physical effect

NCCL collective setup fails before communicator selection for processes assigned nonzero GPU ordinals.

## Observed consequence

psum raises a RET_CHECK failure because local_device_ordinal 3 is not less than gpu_global_device_ids size 1.

## Repair site

XLA PJRT GPU device construction and NcclExecuteParams::GetGlobalDeviceId

## Repair

Commit d8d5a36 stores the mapping as map<int, GlobalDeviceId>, preserves it through GPU run options, and resolves an ordinal with map::find; collective locality derives values from the same map.

## Why the repair works

A sparse or nonzero local ordinal now retrieves its associated global ID directly, so communicator membership no longer depends on dense vector positions.

## Causal basis

The landed XLA commit explicitly fixes JAX issue 12119 and replaces the exact dense-vector lookup named in the reported RET_CHECK with a keyed mapping.

## Analysis implication

Distributed collective analysis needs process identity, visible physical ordinals, local-to-global device mapping, participant groups, and the lookup operation used by communicator construction.

## Fix patches

- [`fixes/jax-12119-commit-d8d5a36.patch`](fixes/jax-12119-commit-d8d5a36.patch) — https://github.com/tensorflow/tensorflow/commit/d8d5a36dfec400918d51ffa08b1dc744149b7d42
