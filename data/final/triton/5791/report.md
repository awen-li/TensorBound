# Triton interpreter cannot handle parameters that alias

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/5791](https://github.com/triton-lang/triton/issues/5791)
- Opened: 2025-02-02T01:56:42Z
- Closed: 2025-02-12T02:36:23Z
- Labels: `bug`

## Reported behavior

### Describe the bug

When invoking the interpreter, Triton will make a copy of the tensors passed in so that they can be operated on directly (e.g. by copying to the host). Unfortunately the straightforward way to do this where each input tensor gets a new host copy, running the kernel, then copying out is subtly incorrect. In particular, a kernel like this:

```python3
@triton.jit
def aliasing_test(buffer, buffer2):
    triton.language.store(buffer, 1)

if __name__ == "__main__":
    buffer = torch.zeros(1, device="cuda")
    aliasing_test[(1,)](buffer, buffer)
    print(buffer)
```

should print "1" but it prints "0". This is because buffer is copied in from the kernel and then buffer2 (which isn't written to) overwrites it with the original value.

### Environment details

Triton: built from main
GPU: H100

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

Triton interpreter cannot handle parameters that alias

## Observed implementation decision

Argument staging is performed per parameter without grouping identical or overlapping storage ranges into one shared host backing allocation.

## Physical effect

The Triton interpreter creates independent host copies for aliased kernel parameters and copies each parameter back separately, losing their shared physical-storage relation.

## Observed consequence

A store through the first alias is overwritten during copy-back by the untouched second alias, so a kernel that writes one returns the original zero.

## Reported proposal or workaround

The one-element identical-argument reproducer isolates staging; maintainers acknowledge the bug and outline parent-storage grouping with best-effort overlap support.

## Repair trace

The maintainers outline storage-range grouping, but no implemented interpreter repair or fixed test is available.

## Archived evidence

- Raw report: `data/raw/github/triton/5791/report.json`
- SHA-256: `a148a4cda0218eff63cbc4976a4ce9837613b73d188acc3d44a1e020a89ca3b0`
