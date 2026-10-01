# Unwanted removal of LDS memory reads when using subgroup specialization

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/22495](https://github.com/iree-org/iree/issues/22495)
- Opened: 2025-10-31T14:11:23Z
- Closed: Open at collection
- Labels: `bug 🐞`

## Reported behavior

### What happened?

In this following example, we read from LDS without writing to it in the same subgroup. IREE gets rid of these reads leading to incorrect results.
Adding a "Fake" if scf.if on the LDS for these subgroup fixes the correctness issue.
Quick investigation shows the iree-codegen-optimize-vector-transfer pass behaves differently between the 2 cases.



### Steps to reproduce your issue

MLIR to reproduce the issue :
https://gist.github.com/sebvince/ef2731b39e25e8bc8339ea757ee159b2
Command to use:
```
iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-optimize-vector-transfer))" file.mlir
```

Same thing with a "fake" scf.if to avoid this issue:
https://gist.github.com/sebvince/f5c17d727ebe5431477f813f52ad50a6
Command to use:
```
iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-optimize-vector-transfer))" file.mlir
```


### What component(s) does this issue relate to?

_No response_

### Version information

_No response_

### Additional context

_No response_

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises iREE vector-transfer optimization of LDS reads under subgroup specialization.

## Observed implementation decision

The pass reasons locally about a subgroup that does not write a shared-memory value and moves or removes reads whose producer belongs to another subgroup.

## Physical effect

Consumer subgroups lose the cross-subgroup LDS values required inside the loop.

## Observed consequence

The generated optimized Wave kernel returns incorrect data.

## Reported proposal or workaround

The report supplies before/after IR and identifies the moved reads.

## Repair trace

No correcting dependence analysis, landed patch, or fixed regression is linked.

## Archived evidence

- Raw report: `data/raw/github/iree/22495/report.json`
- SHA-256: `a3bd28561183e881a70b21291a14d1737a7f7431c8a295d33077f2e15424219a`
