# Pre-fix operation and repair site

Checked against JAX revision `c83a71903d8655511510aa0e6f79dfb5bcd6b8e4`
on 2026-10-25. These excerpts support a revision-time source-location
correction; they are not new human-review decisions or a diagnostic rerun.

## Required state

[`jaxlib/cpu/lapack_kernels.h`, lines 201–213](https://github.com/jax-ml/jax/blob/c83a71903d8655511510aa0e6f79dfb5bcd6b8e4/jaxlib/cpu/lapack_kernels.h#L201-L213)
declares `CholeskyFactorization<dtype>::fn` with a null initial value:

```cpp
inline static FnType* fn = nullptr;
```

## Existing consuming operation

[`jaxlib/cpu/lapack_kernels.cc`, lines 323–345](https://github.com/jax-ml/jax/blob/c83a71903d8655511510aa0e6f79dfb5bcd6b8e4/jaxlib/cpu/lapack_kernels.cc#L323-L345)
implements `CholeskyFactorization<dtype>::Kernel`. Its batch loop calls `fn`
at line 340:

```cpp
fn(&uplo_v, &x_order_v, x_out_data, &x_leading_dim_v, info_data);
```

This call consumes the pointer. The preceding buffer copy and dimension
checks neither initialize it nor require its value. Exported-IR parsing and
module merging also precede execution without dereferencing the pointer.
On the reported fresh-process path, the call is the first supported
operation requiring the missing initialization.

## Initialization and repair

[`jaxlib/cpu/lapack.cc`, lines 79–82](https://github.com/jax-ml/jax/blob/c83a71903d8655511510aa0e6f79dfb5bcd6b8e4/jaxlib/cpu/lapack.cc#L79-L82)
populates the Cholesky function pointers in `GetLapackKernelsFromScipy`.
The module exposes that function as `initialize` at line 201.

The [archived patch](fixes/jax-29610-pr33522.patch) adds
`_ensure_backends_initialized(exported.platforms)` at the start of
`_call_exported_lowering`. The new helper calls `_lapack.initialize()` for
CPU exports, before the executable can reach the existing pointer call.
The helper is a repair site, not a pre-fix operation.

The [case report](report.md) supplies the fresh-process trigger and observed
crash. Initialization is the primary obligation restored by the repair;
runtime capability agreement remains secondary. The correction preserves
the existing runtime-consumer classification.

Source excerpts retain the upstream Apache-2.0 license and copyright.
