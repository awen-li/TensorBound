# inductor `full_like` decompositions give incorrect strides

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/144699](https://github.com/pytorch/pytorch/issues/144699)
- Opened: 2025-01-13T18:54:30Z
- Closed: 2025-07-28T15:32:49Z
- Labels: `high priority`, `triaged`, `actionable`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `pt2: ubn`

## Reported behavior

min repro:
```
import torch

def f(x):
    return torch.full_like(x, 3)

x = torch.randn(4, 5, 6).transpose(1, -1)
out = f(x)
out_compiled = torch.compile(f, backend="aot_eager_decomp_partition")(x)
print(out.stride())
print(out_compiled.stride())
# prints
# (30, 1, 6)
# (30, 5, 1)
```

This seems like the root cause of an NJT compile crash that @jbschlosser was running into (see his [repro](https://www.internalfb.com/intern/paste/P1710266970), [njt_patch](https://www.internalfb.com/phabricator/paste/view/P1710266748) and [error](https://www.internalfb.com/phabricator/paste/view/P1710267237))

cc @ezyang @gchanan @zou3519 @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @yf225 @chenyang78 @muchulee8 @ColinPeppler @amjames @desertfire @aakhundov

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

full_like receives a transposed dense tensor under AOT decomposition.

## Observed implementation decision

The decomposition reconstructs a memory format that cannot express every input stride permutation.

## Physical effect

The compiled output has a different stride order from eager full_like.

## Observed consequence

The report prints eager stride (30,1,6) and compiled stride (30,5,1), with a downstream nested-jagged-tensor crash.

## Reported proposal or workaround

PR 144765 explored a general full_like stride repair but was not the merged closure; PR 153641 explicitly creates strides only for a FlexAttention grad_q path.

## Repair trace

The general proposal did not establish a durable merged repair, and the merged FlexAttention patch fixes one consumer rather than the reported full_like decomposition; no fixed general regression is linked to the closure.

## Archived evidence

- Raw report: `data/raw/github/pytorch/144699/report.json`
- SHA-256: `703a28ca791b297f52a546a11feba4deb0df7e3b71ce237644ac2b080a9f493d`
