# INVALID_ARGUMENT when copying array in pinned_host memory with device_put(..., may_alias=False)

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/24521](https://github.com/jax-ml/jax/issues/24521)
- Opened: 2024-10-25T08:05:34Z
- Closed: 2024-11-07T23:51:57Z
- Labels: `bug`

## Reported behavior

### Description

The minimal reproducer below can be added to `memories_test.py` to reproduce this issue. The `block_until_ready` at the bottom of this example is expected to succeed, but fails with `INVALID_ARGUMENT: BlockHostUntilReady() called on deleted or donated buffer`. The donation of `inp_host` into `inp_host_donate` is necessary for the failure to occur.

```
  def test_disallow_alias_copies_arrays_with_donated_input(self):
    _, _, _, inp_host = _create_inputs(
        (8, 2), P("x", "y"), mem_kind="pinned_host")

    inp_host_donate = jax.device_put(inp_host, may_alias=False, donate=True)

    inp_host_donate_copy = jax.device_put(inp_host_donate, may_alias=False)

    for a in jax.tree_util.tree_leaves(inp_host_donate):
      a.delete()

    jax.block_until_ready(inp_host_donate_copy)
```




### System info (python version, jaxlib version, accelerator, etc.)

```
jax:    0.4.36
jaxlib: 0.4.36
numpy:  2.0.1
python: 3.11.8 (stable, redacted, redacted) [Clang google3-trunk (527cd117b2cb189f59ba64882b2bc5d15e35eefb)]
device info: TPU v4-4, 4 local devices"
process_count: 1
platform: uname_result(system='Linux', [redacted])
```

## Trigger

device_put copies a donated pinned-host array with may_alias=False and the source array is then deleted.

## Relevant entities

Python may_alias and donate options; JAX CopySemantics; C++ ArrayCopySemantics; source pinned-host buffer; destination array

## Implemented relation

The transfer copy policy determines whether the destination owns independent storage or may reuse/donate the source storage.

## Pre-state

The caller requests an independent destination while the input remains eligible for donation on another transfer.

## Binding transition

JAX shards and transfers the array through Python-to-C++ APIs without carrying the requested copy semantics.

## Post-state

The destination may reuse the source buffer despite may_alias=False and therefore shares the source's deletion or donation state.

## Validity obligation

A transfer marked may_alias=False produces destination storage whose validity does not depend on later deletion or donation of the source.

## Invalid decision

The sharding path omitted copy semantics when calling the C++ transfer API, which defaulted the operation to input reuse.

## Physical effect

The purported copy remains bound to storage that is donated and deleted through the source path.

## Observed consequence

block_until_ready on the destination fails with an INVALID_ARGUMENT for a deleted or donated buffer.

## Repair site

JAX dispatch/pxla sharding handlers and XLA Python/IFRT array transfer bindings

## Repair

JAX PR 24752 carries ALIAS, COPY, and DONATE through shard handlers; XLA PR 19134 exposes and consumes matching REUSE_INPUT, ALWAYS_COPY, and DONATE_INPUT semantics in the C++ transfer APIs.

## Why the repair works

The pinned-host transfer receives ALWAYS_COPY for may_alias=False, so the destination is created independently of the source buffer's later donation and deletion.

## Causal basis

The reproducer requires source donation after a disallowed-alias copy, and the paired merged patches explicitly propagate ALWAYS_COPY across the boundary where that request was lost.

## Analysis implication

A transfer check needs the user copy/alias request, donation state, translated C++ copy policy, source storage identity, and destination storage identity.

## Fix patches

- [`fixes/jax-24521-pr24752.patch`](fixes/jax-24521-pr24752.patch) — https://github.com/jax-ml/jax/pull/24752
- [`fixes/xla-24521-pr19134.patch`](fixes/xla-24521-pr19134.patch) — https://github.com/openxla/xla/pull/19134
