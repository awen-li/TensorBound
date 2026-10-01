# Nested pmap issue on GPU

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/4223](https://github.com/jax-ml/jax/issues/4223)
- Opened: 2020-09-08T15:29:21Z
- Closed: 2020-09-12T05:40:13Z
- Labels: `bug`

## Reported behavior

Running the code with nested pmap/vmap in the reproducer below works with CPU/TPU but fails on GPU. The explicitly specified `cpu` backend when running `pmap` seems to be ignored, as there are clearly enough devices available.

Changing

`updates_at_idxs = updates[..., None] * rlax.one_hot(indices, num_classes)`

to

`updates_at_idxs = updates[..., None]`

makes things work on GPU as well. Not entirely sure how this is related to the mapped axes though.

- [Colab reproducer](https://colab.research.google.com/drive/1C8i4xUbu8wDWJKy0EvGhu2Vn7ptAky6W?usp=sharing)

## Trigger

Two nested pmaps request backend='cpu' over four configured CPU devices on a GPU host.

## Relevant entities

pmap axis; explicit backend; CPU devices; default backend; replica device count

## Implemented relation

Replica count and device assignment derive from the backend selected for that pmap.

## Pre-state

The mapped functions specify CPU and enough CPU devices exist for both nested axes.

## Binding transition

pxla.replicate calls local_device_count without passing the selected backend.

## Post-state

Replica validation uses the default GPU device count and rejects or misplaces the CPU pmap.

## Validity obligation

Every device-count query in pmap replication uses the same explicit backend as device selection and execution.

## Invalid decision

The replication path dropped the backend parameter when deriving local device count.

## Physical effect

Logical pmap axes are bound against the wrong backend's device population.

## Observed consequence

The nested pmap reproducer fails on a GPU host although the explicitly selected CPU backend has enough devices.

## Repair site

jax/interpreters/pxla.py::replicate

## Repair

PR 4272 passes backend to local_device_count and adds the nested backend='cpu' regression.

## Why the repair works

Replica cardinality and physical devices now come from the same backend namespace.

## Causal basis

The failure depends on a GPU host despite explicit CPU pmaps, and the linked patch threads backend into the lone device-count query and adds the reproducer.

## Analysis implication

Placement analysis needs backend identity on both enumeration and assignment operations.

## Fix patches

- [`fixes/jax-4223-pr4272.patch`](fixes/jax-4223-pr4272.patch) — https://github.com/jax-ml/jax/pull/4272
