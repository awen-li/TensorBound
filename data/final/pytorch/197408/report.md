# [Inductor] dtype-view custom op followed by diagonal_scatter returns wrong values and aliases input

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/197408](https://github.com/pytorch/pytorch/issues/197408)
- Opened: 2026-09-17T09:02:56Z
- Closed: Open at collection
- Labels: `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `module: pt2-dispatcher`, `module: reinplacing`, `bot-triaged`, `release triage`

## Reported behavior

Compiled execution produces wrong values and aliasing.

> **AI-assisted technical report**
>
> **Proposed title:** [Inductor] dtype-view custom op followed by diagonal_scatter returns wrong values and aliases input
>
> ### Summary
>
> This CPU failure was reproduced on a fresh matching Python/native build of main `c5f7fdf7bfcedb8de0a32e324f22ab37690588b9`, the latest upstream main confirmed at verification start on 2026-09-17. Two separate executions of the same unchanged CPU scratch test reproduced the behavior below. The second execution was independently assessed before its reviewer read the builder's probe result or verdict; both executions used the same fresh build.
>
> The reproducer uses `torch.compile(fullgraph=True)` to mutate a float32 view of an int32 input through a custom op, apply `diagonal_scatter`, copy the result back to the int32 input, and return the scatter result. Compiled execution differs from eager in both values and aliasing. The compiled result shares storage with the input, and mutating that result after the call changes the input, whereas the eager result is independent.
>
> This is a separate follow-up to the dtype-view/custom-op behavior discussed in https://github.com/pytorch/pytorch/pull/195484#issuecomment-5556135152. Main still has the generic scatter-output escape tracked by #195451, and #195484 is not merged into this tested main revision. These fresh-main runs establish the observed failure but do not isolate a dtype-specific root cause or establish causation by PR #195484 or PR #173177.
>
> ### Expected and observed behavior
>
> Expected: compiled output and input mutation match eager; the returned `diagonal_scatter` result does not alias the input; changing the returned tensor after the call leaves the input unchanged.
>
> Both fresh-main CPU executions recorded:
>
> ```text
> clone_before: [0]
> clone_after: []
> aliases_input: true
> output_values_match_before_mutation: false
> input_values_match_before_mutation: false
> input_unchanged_after_output_mutation: false
> observed_clone_removal: true
> ```
>
> These are two executions of one test, not two distinct tests. Each execution ran one CPU test with one failure, zero errors, and zero skips. The builder's run took 11.340 seconds; the independent run took 11.191 seconds.
>
> Both executions stopped at the first output-numerics assertion, which reported 4/16 mismatched elements (25%), including a NaN difference at `(0, 0)`. Input-numerics, storage-alias, and post-return mutation facts were computed and printed before that assertion; the later input/storage/mutation assertions were not reached and are not being reported as passed assertions. The precise causes of the numerical mismatch and escaping alias have not been isolated from each other or from the generic #195451 escape.
>
> ### Reproducer
>
> Standalone CPU example, without graph instrumentation or an internal test runner:
>
> ```python
> import torch
> from torch._inductor import config
>
>
> def mutate(x, y):
>     x.copy_(y)
>
>
> def fn(cache, data, diag):
>     view = cache.view(torch.float32)
>     torch.ops.dtype_scatter_repro.mutate(view, data)
>     updated = torch.diagonal_scatter(view, diag)
>     cache.copy_(updated.view(torch.int32))
>     return updated
>
>
> with torch.library._scoped_library("dtype_scatter_repro", "FRAGMENT") as lib:
>     lib.define("mutate(Tensor(a!) x, Tensor y) -> ()")
>     lib.impl("mutate", mutate, "CompositeExplicitAutograd")
>     torch.library.register_fake(
>         "dtype_scatter_repro::mutate", lambda x, y: None, lib=lib
>     )
>
>     with torch.no_grad(), config.patch(
>         enable_auto_functionalized_v2=True, implicit_fallbacks=True
>     ):
>         for name, call in (
>             ("eager", fn),
>             ("compiled", torch.compile(fn, fullgraph=True)),
>         ):
>             cache = torch.zeros((4, 4), dtype=torch.int32)
>             data = torch.arange(16.0).reshape(4, 4)
>             diag = torch.full((4,), -1.0)
>             out = call(cache, data, diag)
>             snapshot = cache.clone()
>
>             print(f"\n{name}")
>             print("Output:", out)
>             print("Input viewed as float32:", cache.view(torch.float32))
>             print(
>                 "Output aliases input:",
>                 out.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr(),
>             )
>             out.add_(100)
>             print("Input unchanged after out.add_(100):", torch.equal(cache, snapshot))
>             print("Input afterward, viewed as float32:", cache.view(torch.float32))
> ```
>
> Observed on the matching CPU build at `c5f7fdf7bfcedb8de0a32e324f22ab37690588b9`, including an independent execution:
>
> | Observation | Eager | Compiled |
> | --- | --- | --- |
> | Diagonal values before output mutation | `-1` | `NaN` |
> | Output aliases input | `False` | `True` |
> | Input unchanged after `out.add_(100)` | `True` | `False` |
>
> The clone-transition observations, probe hash, and unittest counts elsewhere in this report refer to the original instrumented test. This shorter script was verified separately; it prints the mismatch and exits successfully rather than raising an assertion. CUDA execution of this example has not been verified.
>
> ### Suspected interaction
>
> `fix_auto_functionalized_dtype_views` runs after reinplacing. In this graph, a float32 base shares storage with an int32 graph input, and the later pass removes clone index 0 from `only_clone_these_tensors`. The observed graph already represents the scatter as a view plus in-place copy on the auto-functionalized result, returns that result, and copies its int32 view back into the input.
>
> A hypothesis is that the late clone removal invalidates a freshness assumption from the earlier scatter-reinplacing decision. The recorded transition and incorrect behavior do not by themselves establish that causal chain. Main still has the generic #195451 scatter-output escape; no refinement-disabled control or full historical bisect has isolated an additional dtype-specific cause or the source of every numerical mismatch.
>
> PR #173177 introduced the dtype-view clone-removal pass to address the unnecessary-copy performance issue #173781. That history is context, not evidence that #173177 introduced this failure. Neither the fresh-main runs nor the historical module comparison below establishes an introducing commit or causation by #195484.
>
> ### Fresh matching-main build and provenance
>
> - Source HEAD, `torch.version.git_version`, and the native compiled `COMMIT_SHA` reported by `torch.__config__.show()` all match `c5f7fdf7bfcedb8de0a32e324f22ab37690588b9`.
> - Linux x86_64; Python 3.12.3; GCC 13.3; PyTorch `2.15.0a0+gitc5f7fdf`; Release CPU-only build, `USE_CUDA=0`, `torch.version.cuda is None`, CUDA unavailable, CPU capability AVX2.
> - A fresh isolated native build and editable installation succeeded on the first attempt in 93 minutes 17 seconds, using the approved command:
>
>   ```bash
>   USE_CUDA=0 MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e . -v --no-build-isolation
>   ```
>
> - The observed backend invocation was `ninja -j 8`. Other build features remained at their defaults. Recursive submodules matched their pinned revisions; tracked source remained clean through build and probe execution.
> - The normal editable installation points to this new main checkout. Python compiler/test modules resolve there; native libraries resolve to the new isolated venv. No Python-source overlay, replacement native library, or old-installation header injection was used. PYTHONPATH, PYTHONHOME, LD_LIBRARY_PATH, and LIBRARY_PATH were unset, and user site-packages were disabled. Compilation caches were disabled/isolated.
> - The independent execution checked all seven mapped PyTorch native libraries in the new venv, with matching hashes before and after its run. Its inspected compiler/test module paths contained no unexpected source.
> - Key native SHA256 hashes, also independently matched:
>   - `_C.cpython-312-x86_64-linux-gnu.so`: `7ad15d8dbb27d4ca3256b6ef79a81a283c07a6fa25eb4861324ead943ea15e4e`
>   - `libtorch_cpu.so`: `2cf048fe3b7eb4012350952c3536238929e3e3022bf849cd22fcf4c6067abbad`
>   - `libtorch_python.so`: `ea65e19b32e57fe0957e2a2b57886bc9cc6b9c46d3932427ebee76a8867ba881`
> - The unchanged probe SHA256 is `64045ddb5014dc09ff10e11464873d955f6473ae488a65ee2c4eca12c7109c0f`.
>
> ### Independent harness chronology
>
> The independent worker launched its harness twice but executed the probe only once. Its first launch failed before invoking the probe: a scratch cache directory named `triton` shadowed the expected module as a namespace package. Renaming only that directory to `triton-cache` resolved the harness setup problem. The error log was retained; the runner, probe, installation, and PyTorch source were unchanged. This pre-test namespace-shadow error is separate from the reproduced PyTorch correctness failure and is excluded from the two fresh-main test executions reported above.
>
> ### Historical overlay evidence and attribution limits
>
> The following earlier CPU evidence is retained as historical context, separate from the fresh matching-main build and its two executions:
>
> - Python compiler/test sources were `dabf26f7f4c957548a3b66a00a4c5b9b1e51cc0c` from the PR #195484 follow-up.
> - Existing native libraries were historically built at `489d8f2add83bb6e4ff894d7e12213cdcbb0cf12` with CUDA 12.8.1. The native `_C` SHA256 was `a7c8f62b6886a4aba6b7cf1894875ae34ffe055091845d6c60b2c0180c620c12`.
> - This was a Python-source overlay over that existing native build, not a complete native build of the source HEAD. The script's `binary_revision` field prints generated `torch.version.git_version` metadata; that field alone does not establish native provenance in the historical overlay.
> - A second historical CPU execution replaced only `reinplace.py` with its merge-base version from `73db2e4f6cefd6f13de332c7b195f94165c73d60`, keeping other source files and native libraries fixed. It showed the same clone removal, aliasing, input mutation, and numerical mismatch. This was a controlled one-module comparison, not a full older-version checkout/build or proof of an introducing commit.
> - Each historical run executed one CPU test and failed. An independent reviewer inspected that reproducer, both logs, and relevant pass code; that historical inspection was not a third execution.
> - The fresh matching-main runs supersede the earlier lack of main/native verification for this reproducer. CUDA execution of the dtype-view probe, a full matching historical-version comparison, and an introducing-commit bisect remain unverified. The tested main SHA was latest at verification start, not a claim that it remains latest indefinitely.


cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo @bdhirsh @bobrenjc93 @aorenste

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises torchInductor reinplacement of a dtype-view custom-op result followed by diagonal_scatter and copy-back.

## Observed implementation decision

The report observes the scatter result escaping while reinplacement binds it to input storage, but does not isolate a dtype-specific cause beyond the unmerged generalized-scatter work.

## Physical effect

The returned value aliases the mutated input and contains values different from eager execution.

## Observed consequence

Mutating the returned tensor also changes the input.

## Reported proposal or workaround

PR 195484 proposes blocking reinplacement for escaping generalized-scatter results.

## Repair trace

PR 195484 is unmerged and the report explicitly does not establish causation by that proposal; no landed direct repair or fixed regression exists.

## Archived evidence

- Raw report: `data/raw/github/pytorch/197408/report.json`
- SHA-256: `39e53745a9e71054da2ce67c59b115c16cdb552655dd76caf09042874b22d684`
