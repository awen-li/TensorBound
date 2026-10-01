# [inductor][while_loop] cond_fn-only mutation of a pre-mutated captured tensor is not replayed

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/195327](https://github.com/pytorch/pytorch/issues/195327)
- Opened: 2026-08-30T07:49:09Z
- Closed: 2026-09-04T02:51:25Z
- Labels: `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `module: higher order operators`, `module: pt2-dispatcher`, `bot-triaged`

## Reported behavior

I am investigating PyTorch correctness issues relevant to stateful simulation loops. I asked Codex to re-evaluate this behavior against current `main` and existing unmerged PRs. After reviewing the narrowed scope and its relationship to #184164, I decided to report it because the first compiled invocation can appear correct while leaving persistent state stale, after which later invocations silently return wrong results.

AI assistance disclosure: Codex prepared and executed the reproducer and drafted the technical analysis quoted below. I reviewed the narrowed analysis before submitting it.

> ## Bug
>
> In inference, `torch.while_loop` permits `cond_fn` and `body_fn` to mutate captured tensors that are not carried inputs. With Python sources at current `main`, Inductor fails to replay a `cond_fn`-only mutation to external storage when the captured tensor was already updated before entering the loop.
>
> https://github.com/pytorch/pytorch/blob/e4d9b6187e6ef2635cc2b648fbb409d25d6a9d9a/torch/_higher_order_ops/while_loop.py#L136-L209
>
> Eager and `aot_eager` preserve the mutation. Inductor updates the internal functional tensor, so the first call's returned clone is correct, but leaves the external tensor at the value immediately before the HOP (`0.8` on the first call). A second call then starts from that stale value and silently produces wrong numerical results.
>
> ## Reproducer
>
> ```python
> import torch
>
>
> def f(state):
>     state.mul_(0.8)
>
>     def cond(i, acc):
>         state.mul_(0.5)
>         return i < 2
>
>     def body(i, acc):
>         return i + 1, acc + state
>
>     i, acc = torch.while_loop(
>         cond,
>         body,
>         (
>             torch.zeros((), dtype=torch.int64),
>             torch.zeros_like(state),
>         ),
>     )
>     return i, acc, state.clone()
>
>
> def run(backend):
>     torch._dynamo.reset()
>     state = torch.ones((), dtype=torch.float64)
>     fn = f if backend == "eager" else torch.compile(
>         f, backend=backend, fullgraph=True
>     )
>
>     results = []
>     with torch.inference_mode():
>         for _ in range(2):
>             output = fn(state)
>             results.append(
>                 (
>                     tuple(item.item() for item in output),
>                     state.item(),
>                 )
>             )
>     print(backend, results)
>
>
> for backend in ("eager", "aot_eager", "inductor"):
>     run(backend)
> ```
>
> Each result is `(returned_values, external_state_after_call)`:
>
> ```text
> eager [
>   ((2, 0.6000000000000001, 0.1), 0.1),
>   ((2, 0.06000000000000001, 0.010000000000000002), 0.010000000000000002)
> ]
> aot_eager [
>   ((2, 0.6000000000000001, 0.1), 0.1),
>   ((2, 0.06000000000000001, 0.010000000000000002), 0.010000000000000002)
> ]
> inductor [
>   ((2, 0.6000000000000001, 0.1), 0.8),
>   ((2, 0.4800000000000001, 0.08000000000000002), 0.6400000000000001)
> ]
> ```
>
> The same failure occurs under `torch.no_grad()`. It also reproduces when the condition is false on its first evaluation: eager persists the mutation made by that one `cond_fn` call, whereas Inductor leaves external state at the pre-loop value even though `body_fn` never runs.
>
> ## Expected behavior
>
> Inductor should match eager and `aot_eager`: mutations of non-carried captured tensors made by `cond_fn` should be replayed to their external storage, including when the loop executes zero body iterations and across repeated compiled calls.
>
> ## Likely cause
>
> Dynamo already unions the storage mutations discovered in `cond_fn` and `body_fn` when constructing the higher-order operator:
>
> https://github.com/pytorch/pytorch/blob/e4d9b6187e6ef2635cc2b648fbb409d25d6a9d9a/torch/_dynamo/variables/higher_order_ops.py#L932-L987
>
> In contrast, `torch._inductor.ir.WhileLoop.create` obtains mutation indices only from `body_fn`:
>
> ```python
> mutated_idxs = check_input_alias_and_mutation(
>     body_fn.graph.module, fake_all_inputs
> )[3]
> ```
>
> https://github.com/pytorch/pytorch/blob/e4d9b6187e6ef2635cc2b648fbb409d25d6a9d9a/torch/_inductor/ir.py#L11812-L11863
>
> Consequently, a captured additional input mutated only by `cond_fn` receives no `MutationOutput`/scheduler write dependency. A local prototype that unions the `cond_fn` and `body_fn` mutation indices, on top of the additional-input handling from #184164, fixes the nonzero-iteration, zero-iteration, and repeated-call cases above.
>
> ## Relationship to #184164
>
> #184164 is a closely related partial fix, not an exact duplicate. It fixes mutation replay for captured additional inputs mutated by `body_fn`. Applying that PR locally fixes the body-mutation reproducer but does not fix this `cond_fn`-only reproducer.
>
> The PR test's cond-mutated counter is lowered from an unchanged graph-input base, while a separate buffer is mutated in `body_fn`; it therefore does not exercise a computed functionalization base mutated only by `cond_fn`. This issue can likely be handled by extending #184164 to collect mutation indices from both subgraphs and adding the two regression boundaries above.
>
> Related issue: #182036
>
> ## Versions
>
> - Python sources: current `main` at `e4d9b6187e6ef2635cc2b648fbb409d25d6a9d9a`
> - Native extension build used with that source tree: `2.15.0a0+gitb3632e5` (`b3632e56519f39da86e50fb0aa27235d626e8306`)
> - Python: 3.12.3
> - OS: Ubuntu 24.04.4 LTS, x86_64
> - CPU: Intel Core i7-6700
> - `fullgraph=True`, CPU, `torch.inference_mode()`; CUDA was not involved


cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo @ydwu4 @bdhirsh @bobrenjc93 @aorenste

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

[inductor][while_loop] cond_fn-only mutation of a pre-mutated captured tensor is not replayed

## Observed implementation decision

WhileLoop.create builds MutationOutput and scheduler dependencies from body_fn alone rather than unioning condition and body mutation sets.

## Physical effect

While-loop lowering records mutation indices only from the body subgraph and omits a captured tensor mutated solely by the condition subgraph.

## Observed consequence

The condition's internal functional tensor changes and returned clones look correct, but external state is not updated; repeated compiled calls start from stale state and silently diverge.

## Reported proposal or workaround

Eager/AOT/Inductor comparison covers nonzero and zero iterations; a local union prototype on top of #184164 fixes both cases and repeated calls.

## Repair trace

A local prototype unions condition and body mutations, but no accepted repair, maintainer adjudication, or fixed repository revision is available.

## Archived evidence

- Raw report: `data/raw/github/pytorch/195327/report.json`
- SHA-256: `f43456a6e608b0fd4fd8667a1ebda4445eff33ff9288d7539f4ad47a18b2bbd0`
