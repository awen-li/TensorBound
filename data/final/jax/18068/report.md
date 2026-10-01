# Cannot call custom primitive with ordered effect twice in JVP rule

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/18068](https://github.com/jax-ml/jax/issues/18068)
- Opened: 2023-10-11T20:36:17Z
- Closed: 2024-02-21T01:05:53Z
- Labels: `bug`

## Reported behavior

### Description

We are currently migrating mpi4jax from manual XLA tokens to the new ordered effects. The only remaining test failures are related to our JVP rules which call the underlying primitive twice (once on the primal and once on the tangent). Commenting out one of the `.bind` calls in the JVP rule works.

Reproducer:

```python
import functools

from jax.core import Primitive, ShapedArray
from jax.interpreters import ad, xla, mlir

from jax._src.effects import (
    Effect,
    lowerable_effects,
    ordered_effects,
    control_flow_allowed_effects,
    custom_derivatives_allowed_effects,
)

# Register an ordered effect
class EffectType(Effect):
    pass

lowerable_effects.add_type(EffectType)
ordered_effects.add_type(EffectType)
control_flow_allowed_effects.add_type(EffectType)
custom_derivatives_allowed_effects.add_type(EffectType)

ordered_effect = EffectType()

# Define custom Jax primitive
custom_p = Primitive("custom")
custom_impl = functools.partial(xla.apply_primitive, custom_p)


def custom(x):
    return custom_p.bind(x)


def custom_xla_encode_cpu(ctx, x):
    token = ctx.tokens_in.get(ordered_effect)[0]
    ctx.set_tokens_out(mlir.TokenSet({ordered_effect: (token,)}))
    return [x]


def custom_abstract_eval(xs):
    return ShapedArray(xs.shape, xs.dtype), {ordered_effect}


def custom_value_and_jvp(in_args, tan_args):
    (x,) = in_args
    (x_tan,) = tan_args

    val = custom_p.bind(x)
    jvp = custom_p.bind(x_tan)
    return val, jvp


def custom_transpose_rule(x_tan, *x_args):
    res = custom_p.bind(x_tan)
    return (res,)


custom_p.def_impl(custom_impl)
custom_p.def_effectful_abstract_eval(custom_abstract_eval)

ad.primitive_jvps[custom_p] = custom_value_and_jvp
ad.primitive_transposes[custom_p] = custom_transpose_rule

mlir.register_lowering(custom_p, custom_xla_encode_cpu, platform="cpu")


if __name__ == "__main__":
    import jax
    x = jax.numpy.zeros(100)

    def testfun(x):
        return custom(x).sum()

    jax.grad(testfun)(x)
```

Error:

```bash
$ python bug.py
Traceback (most recent call last):
  File "/home/dion/codes/mpi4jax/bug.py", line 74, in <module>
    jax.grad(testfun)(x)
  File "/home/dion/codes/mpi4jax/bug.py", line 72, in testfun
    return custom(x).sum()
  File "/home/dion/codes/mpi4jax/bug.py", line 31, in custom
    return custom_p.bind(x)
  File "/home/dion/codes/mpi4jax/bug.py", line 49, in custom_value_and_jvp
    jvp = custom_p.bind(x_tan)
jax._src.source_info_util.JaxStackTraceBeforeTransformation: ValueError: INVALID_ARGUMENT: Execution supplied 1 buffers but compiled program expected 2 buffers

The preceding stack trace is the source of the JAX operation that, once transformed by JAX, triggered the following exception.

--------------------

The above exception was the direct cause of the following exception:

jax.errors.SimplifiedTraceback: For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/dion/codes/mpi4jax/bug.py", line 74, in <module>
    jax.grad(testfun)(x)
  File "/home/dion/codes/mpi4jax/bug.py", line 54, in custom_transpose_rule
    res = custom_p.bind(x_tan)
ValueError: INVALID_ARGUMENT: Execution supplied 1 buffers but compiled program expected 2 buffers
```

### What jax/jaxlib version are you using?

0.4.18

### Which accelerator(s) are you using?

CPU

### Additional system info

Linux

### NVIDIA GPU info

_No response_

## Trigger

A JVP rule binds the same ordered-effect primitive for the primal and tangent values; the same failure also appears when an eager caller invokes the primitive twice with the same abstract values.

## Relevant entities

custom primitive; ordered effect; hidden input and output tokens; cached xla_primitive_callable; C++ apply_primitive fast-call wrapper; compiled executable input buffers

## Implemented relation

Every primitive invocation supplies the current ordered-effect token alongside its explicit data arguments and returns the successor token for the next invocation.

## Pre-state

The compiled primitive executable expects one explicit data buffer and one internally threaded ordered-effect token.

## Binding transition

The old op-by-op path cached a C++ fast-call wrapper produced by create_cpp_call_for_apply_primitive and reused it for a later bind with the same abstract values.

## Post-state

The reused wrapper supplied only the explicit data buffer on the later call although the compiled executable still expected the hidden token buffer as a second input.

## Validity obligation

Each execution builds its complete explicit and hidden input list from the current effect state, then carries the returned token to the next execution.

## Invalid decision

The dispatch cache treated a wrapper that inserts and advances an ordered-effect token as a reusable stateless callable.

## Physical effect

The second execution reached the compiled program with one buffer while its input signature required two.

## Observed consequence

Differentiating the custom primitive failed with `Execution supplied 1 buffers but compiled program expected 2 buffers`; adding a dummy explicit token only masked the missing hidden input.

## Repair site

jax/_src/dispatch.py::apply_primitive and xla_primitive_callable; jax/_src/interpreters/pxla.py::MeshExecutable.create_cpp_call_for_apply_primitive

## Repair

Commit e624610 removes the hand-built lowering and cached C++ apply_primitive wrapper, deletes create_cpp_call_for_apply_primitive, and implements op-by-op primitive calls as a cached jax.jit wrapper around prim.bind.

## Why the repair works

The regular jit invocation reconstructs the call inputs and threads JAX's ordered-effect tokens for every execution instead of reusing the consumed state of the former C++ fast-call wrapper.

## Causal basis

The issue reproducer fails on JAX 0.4.18. A second reporter narrows the failure to reuse of create_cpp_call_for_apply_primitive, observes the missing internally prepended token on the second call, and confirms that forcing unsafe_call avoids it. Commit e624610, merged between 0.4.18 and 0.4.23, deletes that exact wrapper and replaces it with jax.jit. A contributor and a JAX collaborator then run the original reproducer successfully on 0.4.23. The commit does not name issue 18068, so the repair attribution rests on the exact removed mechanism and the release-bounded reproduction evidence.

## Analysis implication

Analysis of repeated effectful primitive calls needs the explicit operands, effect set, hidden token inputs and outputs, cached-call identity, invocation order, and compiled executable signature together.

## Fix patches

- [`fixes/jax-18068-commit-e624610.patch`](fixes/jax-18068-commit-e624610.patch) — https://github.com/jax-ml/jax/commit/e624610e722769f9b2aad60e79a47c8c70c6631e
