# Fix unaligned memory accesses

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/9922](https://github.com/iree-org/iree/issues/9922)
- Opened: 2022-07-27T19:54:54Z
- Closed: 2025-07-27T01:35:46Z
- Labels: `bug 🐞`, `runtime`

## Reported behavior

### What happened?

When e2e tests are executed on Android, the Undefined Behavior Sanitizer complains about multiple unaligned memory accesses.
Setting the `IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED=1` cmake flag does not fix the ubsan alarms, as I'd expect. We need to either wrap such access with `iree_unalidnged_*` or fix the underlying issue that causes pointer to be misaligned.

Accessing unaligned memory is prohibited by C/C++, regardless of whether it is known to work in practice or not. It'd be better to not violate any alignment assumptions so that we can run CI with ubsan checks enabled at all times.

### Steps to reproduce your issue

Install Android NDK r24 and set `ANDROID_NDK`.

CMake script:
```shell
#! /bin/sh

set -eou pipefail
readonly IREE_DIR="$(realpath "$1")"
readonly BUILD_DIR="$(realpath "$2")"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$IREE_DIR" -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                        -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK?}/build/cmake/android.toolchain.cmake" \
			-DIREE_HOST_BINARY_ROOT="$(realpath "$BUILD_DIR/../relass/run")" \
			-DANDROID_ABI="arm64-v8a" \
                        -DANDROID_PLATFORM="android-29" \
			-DIREE_BUILD_COMPILER=0 \
			-DIREE_ENABLE_ASSERTIONS=1 \
			-DIREE_ENABLE_ASAN=1 \
			-DIREE_ENABLE_UBSAN=1 \
			-DIREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED=1 \
			-DIREE_BUILD_TESTS=1 \
                        -DIREE_BUILD_SAMPLES=0 \
		        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
			-DCMAKE_INSTALL_PREFIX=run
```

Build with `ninja all iree-test-deps`

Test command:
```shell
ctest --output-on-failure -R '.*e2e_matmul_mmt4d_i8_small_llvm-cpu_local-task_on_android_device.*' -V | ../iree/build_tools/scripts/android_symbolize.sh
```

(I'll open a PR adding the `IREE_ENABLE_UBSAN` option shortly)

### What component(s) does this issue relate to?

Runtime

### Version information

_No response_

### Additional context

Sample ubsan report attached: [ubsan.log](https://github.com/iree-org/iree/files/9203171/ubsan.log).

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

UBSan-enabled Android and Linux tests exercise bytecode dispatch, VM references, and local-task command buffers.

## Observed implementation decision

Several reported accesses use pointers whose observed addresses do not meet the C/C++ alignment requirement.

## Physical effect

UBSan reports distinct null, load, store, and member-access violations at multiple runtime sites.

## Observed consequence

The sanitizer emits multiple undefined-behavior diagnostics.

## Reported proposal or workaround

PR 9923 adds an IREE_ENABLE_UBSAN build option and documentation.

## Repair trace

The report groups multiple violations, while the linked patch only enables detection; no per-site root cause, corrective change, or fixed regression closes an implemented-relation-to-repair chain.

## Archived evidence

- Raw report: `data/raw/github/iree/9922/report.json`
- SHA-256: `d2ce7ead917acdb0496db3e027a0ff88fc99dc79082ec2714f9acdf2b6988b22`
