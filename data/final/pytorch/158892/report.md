# Inductor can be silently incorrect on custom ops with float8_e8m0fnu input

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/158892](https://github.com/pytorch/pytorch/issues/158892)
- Opened: 2025-07-23T02:13:55Z
- Closed: 2025-07-29T00:31:59Z
- Labels: `triage review`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `pt2: ubn`

## Reported behavior

Initial repro from @gau-nernst, I distilled it down to the following.

The problem is:
- Let's say a custom operator accepts float8_e8m0fnu input and has strict requirements on contiguity.
- We will get to this branch https://github.com/pytorch/pytorch/blob/9df0f565972a8a034fd77d65aff2c53e6e9856d1/torch/_inductor/graph.py#L1561-L1575
- Unfortunately, this branch (call this the "bad branch") doesn't coerce the inputs to the operator to be contiguous. [this other branch](https://github.com/pytorch/pytorch/blob/9df0f565972a8a034fd77d65aff2c53e6e9856d1/torch/_inductor/graph.py#L1178) handles that logic.
- The reason why we got inside the bad branch is because we got here: https://github.com/pytorch/pytorch/blob/9df0f565972a8a034fd77d65aff2c53e6e9856d1/torch/_inductor/lowering.py#L1908-L1924

We need some sort of a refactor. Either:
- all fallback operations should go through GraphLowering.call_function too!
- the bad branch needs input stride coercion logic added to it.

cc @ezyang @gchanan @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @chenyang78 @muchulee8 @amjames @aakhundov @coconutruben

## Trigger

A custom operator receives a transposed float8 tensor and requires that tensor and a second argument to be contiguous in their declared logical order.

## Relevant entities

FX custom-op node; float8 input; unsupported-type predicate; built-in fallback branch; GraphLowering.call_function; exact input strides; custom kernel

## Implemented relation

A custom operator's logical arguments are materialized with the exact physical strides expected at its opaque call boundary.

## Pre-state

The FX graph records a transposed input and the custom operator's call boundary requires a contiguous realization of that logical value.

## Binding transition

fallback_node_due_to_unsupported_type routes the float8 custom-op node through a special fallback branch intended for unsupported built-in aten and prim operators.

## Post-state

The call bypasses GraphLowering.call_function and its input-stride coercion, so the custom implementation receives a physical layout that does not match its boundary contract.

## Validity obligation

Opaque custom operators pass through the exact-stride call path regardless of whether their input dtype is supported by ordinary Inductor lowering.

## Invalid decision

The unsupported-type branch classified all operator overloads as built-ins and captured custom-library operations.

## Physical effect

The custom implementation indexes a noncontiguous or differently ordered buffer under its contiguous-input assumption.

## Observed consequence

Compiled custom-operator execution with float8_e8m0fnu inputs can silently produce incorrect values instead of receiving the contiguous arguments required by the operator.

## Repair site

torch/_inductor/graph.py::GraphLowering.run_node fallback dispatch

## Repair

Commit 657e5e9 restricts the unsupported-type shortcut to built-in operator overloads, sending all custom operators through GraphLowering.call_function, and adds float and float8 tests whose custom implementation asserts the required contiguous layouts.

## Why the repair works

Custom calls now use the existing path that reconciles FX view semantics with the opaque operator's exact physical stride requirements before invoking it.

## Causal basis

The landed commit explicitly fixes report 158892, identifies the custom-versus-built-in routing error, changes that predicate, and adds a custom operator that asserts the transposed and ordinary inputs arrive contiguous for both float and float8 cases.

## Analysis implication

Opaque-call analysis needs operator provenance, built-in versus custom classification, dtype support, FX strides, coercion path, physical strides, and callee layout contract together.

## Fix patches

- [`fixes/pytorch-158892-commit-657e5e9.patch`](fixes/pytorch-158892-commit-657e5e9.patch) — https://github.com/pytorch/pytorch/commit/657e5e9aa6600cbb40e42432e2e3b6b50678de63
