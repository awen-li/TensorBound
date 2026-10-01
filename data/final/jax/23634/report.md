# Corner-case sharding regression when replacing concrete mesh with abstract mesh

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/23634](https://github.com/jax-ml/jax/issues/23634)
- Opened: 2024-09-13T20:31:03Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Description

With no data dependency on inputs, abstract mesh and concrete mesh in `jax.lax.with_sharding_constraint` result in different behavior, with concrete mesh behaving as expected.

Sending a PR with this test, and will tag this issue, but reproduction is essentially:
```
def f(x):
  # This breaks the data dependency on x.
  x = with_sharding_constraint(
      jnp.ones(shape=x.shape, dtype=x.dtype),
      NamedSharding(mesh, P('x')),
  )
  return x * 2
  ```
When mesh is abstract, the resulting array is not sharded as expected; when mesh is concrete, sharding is WAI.

### System info (python version, jaxlib version, accelerator, etc.)

Internal Google test.

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises jAX sharding constraints using an AbstractMesh after the value's dependency on a concretely sharded input is removed.

## Observed implementation decision

The abstract-mesh constraint is not rebound to the active concrete mesh when a constant replaces the input-dependent value.

## Physical effect

The result loses the requested concrete sharding.

## Observed consequence

Abstract and concrete mesh variants return differently sharded arrays.

## Reported proposal or workaround

PR 23635 adds a failing test.

## Repair trace

The linked PR contains only the reproducing test and closed without a repair; no landed correction exists.

## Archived evidence

- Raw report: `data/raw/github/jax/23634/report.json`
- SHA-256: `cfa522191e557431d35af6d3cb7d2a1583c3508916f65d7cb20186910be92379`
