# [Inductor] Direct generalized_scatter copy-back changes returned output aliasing

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/195451](https://github.com/pytorch/pytorch/issues/195451)
- Opened: 2026-08-31T15:02:25Z
- Closed: Open at collection
- Labels: `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: aotdispatch`, `module: inductor`, `module: pt2-dispatcher`, `module: reinplacing`, `bot-triaged`

## Reported behavior

I found a correctness issue in Inductor’s direct generalized-scatter copy-back path. The values are correct, but the aliasing behavior differs from eager mode: the compiled function returns the mutated input rather than a fresh tensor.

> **AI-assisted technical report**
>
> I reproduced a correctness issue in the direct generalized-scatter copy-back
> path. It preserves initial values but changes the returned tensor's alias
> contract: eager returns a fresh result, while Inductor returns the mutated
> input itself.
>
> ### Minimal reproducer
>
> ```python
> import torch
>
>
> def fn(x, src):
>     updated = torch.slice_scatter(x, src, 0, 0, 1)
>     x.copy_(updated)
>     return updated
>
>
> def run(callable):
>     x = torch.tensor([1.0, 2.0])
>     out = callable(x, torch.tensor([10.0]))
>     torch.testing.assert_close(x, torch.tensor([10.0, 2.0]))
>     torch.testing.assert_close(out, torch.tensor([10.0, 2.0]))
>     return x, out
>
>
> eager_x, eager_out = run(fn)
> print("eager aliases input:", torch._C._is_alias_of(eager_x, eager_out))
> eager_out.add_(100.0)
> print("eager input after output mutation:", eager_x)
>
> compiled_x, compiled_out = run(torch.compile(fn, fullgraph=True))
> print("compiled aliases input:", torch._C._is_alias_of(compiled_x, compiled_out))
> compiled_out.add_(100.0)
> print("compiled input after output mutation:", compiled_x)
> ```
>
> Expected eager behavior:
>
> ```text
> eager aliases input: False
> eager input after output mutation: tensor([10.,  2.])
> ```
>
> Actual compiled behavior:
>
> ```text
> compiled aliases input: True
> compiled input after output mutation: tensor([110., 102.])
> ```
>
> The same behavior reproduced with both `torch.compile(fn)` and
> `torch.compile(fn, fullgraph=True)` on CPU.
>
> ### Diagnosis and scope
>
> Functionalization creates this shape:
>
> ```text
> generalized_scatter(input, ...) -> copy_(input, scatter) -> return scatter
> ```
>
> `should_reinplace_scatter()` treats the direct copy-back as profitable and
> reinplaces the scatter. That is valid only when the functional scatter result
> is dead after copy-back. Here it is returned, so reinplacing changes a
> non-aliasing functional result into an alias of the input.
>
> A narrow direction is to apply a post-copy data-use/alias-liveness check to
> the direct copy-back case as well, including view and multi-output-view
> (`getitem`) aliases. The exact guard needs separate validation; this report
> does not propose an unconditional profitability change.
>
> ### Relation to existing work
>
> - #195285 and draft PR #195292 concern the indirect
>   `generalized_scatter -> index_put/_unsafe_index_put -> copy_(input, ...)`
>   traversal, motivated by a full-buffer clone. This report has no indexed
>   update and concerns return-alias correctness.
> - The fork reference implementation at
>   `6092aca85937d19257a8a1569c7d434b11313f1b` addresses liveness across the
>   indexed-update traversal, but intentionally leaves the pre-existing direct
>   copy-back fast path unchanged.
> - #195329 is a performance issue where the returned view intentionally aliases
>   the input. This reproducer is the opposite: eager returns an independent
>   result and compiled execution incorrectly creates an alias.
>
> ### Local verification
>
> Reproduced on CPU using the current Python compiler source checkout with an
> ABI-compatible local editable runtime. CUDA was unavailable in this
> environment.


cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo @bdhirsh @bobrenjc93 @aorenste

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

A functional scatter result is copied back into its input and is also returned.

## Observed implementation decision

Reinplacement allows the returned functional result to share the input's storage.

## Physical effect

Post-return mutation of the result also mutates the input.

## Observed consequence

Compiled execution changes the eager output-alias contract.

## Reported proposal or workaround

None recorded.

## Repair trace

PR 198096 blocks reinplacement when an unrelated later copy overwrites a live result, but explicitly preserves reinplacement when copy_ writes this result back. It therefore does not demonstrate a repair for report 195451's same-result copy-back alias escape; the direct repair remains unarchived.

## Archived evidence

- Raw report: `data/raw/github/pytorch/195451/report.json`
- SHA-256: `756d79755b7d6b39a553163408d92f8ed3bc23b0827f8c49d0fd46e159dfdf23`
