# Silent correctness issue if input tensors have existing storage offsets

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/108472](https://github.com/pytorch/pytorch/issues/108472)
- Opened: 2023-09-02T02:24:40Z
- Closed: 2023-09-06T15:31:28Z
- Labels: `high priority`, `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`

## Reported behavior

### 🐛 Describe the bug

This is already fixed on main by https://github.com/pytorch/pytorch/pull/108168

```
    def test_adding_tensor_offsets(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return x[16:32]

        with torch.no_grad():
            x = torch.randn(1024, device=self.device)
            self.assertEqual(fn(x[0:]), x[16:][:16])
            self.assertEqual(fn(x[128:]), x[128 + 16 :][:16])
```

Fails with
```
Traceback (most recent call last):
  File "/home/jansel/conda/envs/pytorch/lib/python3.10/unittest/case.py", line 59, in testPartExecutor
    yield
  File "/home/jansel/conda/envs/pytorch/lib/python3.10/unittest/case.py", line 591, in run
    self._callTestMethod(testMethod)
  File "/home/jansel/conda/envs/pytorch/lib/python3.10/unittest/case.py", line 549, in _callTestMethod
    method()
  File "/home/jansel/pytorch/torch/testing/_internal/common_utils.py", line 2388, in wrapper
    method(*args, **kwargs)
  File "/home/jansel/pytorch/test/inductor/test_torchinductor.py", line 6997, in new_test
    return value(self)
  File "/home/jansel/pytorch/test/inductor/test_torchinductor.py", line 4682, in test_adding_tensor_offsets
    self.assertEqual(fn(x[128:]), x[128 + 16 :][:16])
  File "/home/jansel/pytorch/torch/testing/_internal/common_utils.py", line 3285, in assertEqual
    raise error_metas.pop()[0].to_error(
AssertionError: Tensor-likes are not close!

Mismatched elements: 16 / 16 (100.0%)
Greatest absolute difference: 2.224123239517212 at index (13,) (up to 1e-05 allowed)
Greatest relative difference: 12.916085243225098 at index (2,) (up to 1.3e-06 allowed)
```

The existing storage_offset is discarded by inductor.

### Versions

release/2.1 branch

cc @ezyang @gchanan @zou3519 @kadeng @msaroufim @wconstab @bdhirsh @anijain2305 @voznesenskym @penguinwu @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @Xia-Weiwen @wenzhe-nrv @jiayisunx @peterbell10 @ipiszy @ngimel @yf225 @chenyang78 @muchulee8 @aakhundov

## Trigger

A compiled slice takes elements 16 through 31 from an input view beginning at storage offset 128.

## Relevant entities

input tensor; backing storage; existing storage offset; view offset increment; generated output view

## Implemented relation

A derived view's storage offset equals the input tensor's existing offset plus the new logical view offset.

## Pre-state

The graph input denotes storage beginning at offset 128 within its backing allocation.

## Binding transition

Generated as_strided constructs the derived view with offset 16 as an absolute storage offset.

## Post-state

The compiled result reads storage positions 16 through 31 instead of 144 through 159.

## Validity obligation

View lowering composes a new relative offset with the input's established storage offset.

## Invalid decision

The old Inductor reference code discarded an input tensor's existing storage offset when materializing a derived view.

## Physical effect

The generated view binds the logical slice to the wrong physical elements of the backing storage.

## Observed consequence

All 16 output elements mismatch the eager result for the nonzero-offset input.

## Repair site

torch/_inductor/ir.py::ReinterpretView.codegen_reference and torch/csrc/inductor/inductor_ops.cpp

## Repair

PR 108259 introduces _reinterpret_tensor, sets storage_offset to self.storage_offset plus offset_increment, and uses it for generated view and buffer-reuse references.

## Why the repair works

The generated wrapper preserves the input's physical starting point and applies the new slice displacement relative to it.

## Causal basis

The report states that the existing offset is discarded, and the linked patch replaces absolute as_strided offset construction with explicit additive composition and the reported regression.

## Analysis implication

View analysis needs backing-storage identity, inherited storage offset, relative offset, size, and strides at each derivation.

## Fix patches

- [`fixes/pytorch-108472-pr108259.patch`](fixes/pytorch-108472-pr108259.patch) — https://github.com/pytorch/pytorch/pull/108259
