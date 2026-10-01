# [Inductor] uniform_ on a non-contiguous view returns invalid values

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/191193](https://github.com/pytorch/pytorch/issues/191193)
- Opened: 2026-07-27T11:36:09Z
- Closed: 2026-08-02T02:54:04Z
- Labels: `high priority`, `triaged`, `module: random`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `bot-triaged`

## Reported behavior

### 🐛 Describe the bug

Calling uniform_ on a non-contiguous view works correctly in eager mode, but the Inductor-compiled version produces invalid values, including NaN and values outside (0,1).

```python
import torch

def f(x):
    y = x[:, ::2, :]
    y.uniform_(0.0, 1.0)
    return y


x = torch.randn(3, 5, 6)

print("eager:")
print(f(x.clone()))

print("compiled:")
compiled_f = torch.compile(f, backend="inductor", fullgraph=True)
print(compiled_f(x.clone()))
```


### Log

```text
eager:
tensor([[[0.2098, 0.0668, 0.7135, 0.6084, 0.8372, 0.3190],
         [0.6879, 0.2051, 0.2626, 0.6813, 0.8424, 0.0320],
         [0.8212, 0.6882, 0.7847, 0.0551, 0.1949, 0.9001]],

        [[0.2279, 0.3357, 0.8571, 0.4241, 0.7035, 0.7001],
         [0.1578, 0.5551, 0.2524, 0.6783, 0.8121, 0.2102],
         [0.4324, 0.3899, 0.5189, 0.8966, 0.5486, 0.0250]],

        [[0.0571, 0.4071, 0.2865, 0.2589, 0.3404, 0.0634],
         [0.9750, 0.6381, 0.5279, 0.4490, 0.9692, 0.6973],
         [0.6509, 0.1065, 0.0394, 0.4110, 0.4979, 0.0284]]])
compiled:
tensor([[[ 3.8167e-01,  9.4777e-01,  1.8339e-01,  8.9167e-01,  9.1508e-01,
           9.1606e-01],
         [ 8.2016e-01,  5.9327e-01,  8.5354e-01,  4.8089e-01,  4.3562e-01,
           5.8247e-03],
         [ 5.3227e-01,  4.1677e-01,  6.8199e-01,  7.1345e-01,  8.3278e-01,
           5.8679e-01]],

        [[ 7.8604e-01,  5.3039e-01,  8.8662e-01,  1.8709e-01,  8.1824e-01,
           2.7351e-01],
         [ 4.0210e-01,  2.6702e-01,  9.1031e-01,  2.4682e-01,  1.8269e-01,
           8.7642e-01],
         [ 1.1351e-43,  0.0000e+00, -1.9490e-22,  3.0876e-41,  8.7512e-25,
           4.5587e-41]],

        [[ 2.7273e-31,  4.5586e-41,  7.2293e-25,  4.5587e-41,  2.7130e-31,
           4.5586e-41],
         [ 1.1210e-43,  0.0000e+00,  3.3631e-43,  0.0000e+00, -2.6234e+35,
           3.0883e-41],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
           0.0000e+00]]])
```

### Versions

PyTorch version: 2.14.0.dev20260723+cu130
OS: Ubuntu 24.04.4 LTS (x86_64)

cc @ezyang @gchanan @kadeng @msaroufim @pbelevich @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Trigger

uniform_ fills the view x[:, ::2, :] with logical shape (3,3,6) and strides inherited from slicing a contiguous (3,5,6) tensor.

## Relevant entities

logical output shape; requested strides; random sample buffer; required storage length; as_strided view; uniform values

## Implemented relation

The sample allocation covers every storage offset addressable by the requested output size-and-stride layout.

## Pre-state

The sliced view has fewer logical elements than its maximum reachable storage offset because adjacent selected rows are separated by skipped rows.

## Binding transition

The old decomposition allocates torch.rand(shape) as a compact logical tensor and then applies the noncontiguous output strides with as_strided.

## Post-state

The reconstructed view names offsets beyond the compact random-sample allocation.

## Validity obligation

Before constructing a strided view, the backing allocation length is at least the maximum reachable offset plus one for the requested shape, stride, and storage offset.

## Invalid decision

Allocation size was derived from the product of logical dimensions rather than from the physical span of the target stride mapping.

## Physical effect

Some output positions read memory outside the initialized random sample buffer and expose stale or unrelated values.

## Observed consequence

Compiled uniform_ returns NaNs and values outside [0,1] in later rows, while eager execution returns valid uniform samples at every view position.

## Repair site

torch/_inductor/decomposition.py::uniform

## Repair

Commit 1005a82 computes the required storage length from output shape and strides, generates a one-dimensional random buffer covering that span, and always reconstructs the requested layout with as_strided. It adds the report's sliced-view range and stride regression.

## Why the repair works

Every address produced by the noncontiguous view now refers to an initialized random sample within the allocated storage, while the final tensor retains the requested logical shape and strides.

## Causal basis

The issue closes on commit 1005a82, whose message states that it fixes 191193. The patch changes uniform allocation from the logical shape to the required storage length and adds the exact sliced noncontiguous uniform_ regression.

## Analysis implication

Random-fill decomposition analysis needs logical shape, physical strides, storage offset, maximum reachable offset, allocation length, and reconstructed-view bounds together.

## Fix patches

- [`fixes/pytorch-191193-commit-1005a82.patch`](fixes/pytorch-191193-commit-1005a82.patch) — https://github.com/pytorch/pytorch/commit/1005a82c255c8bd9f2cfcdbdb64468da49e0a3ab
