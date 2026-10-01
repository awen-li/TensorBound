# AOTInductor drops the storage offset of a mutable view argument under dynamic shapes (missing lowering for aten.sym_storage_offset)

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/198068](https://github.com/pytorch/pytorch/issues/198068)
- Opened: 2026-09-22T00:38:05Z
- Closed: Open at collection
- Labels: `high priority`, `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: dynamic shapes`, `module: inductor`, `oncall: export`, `module: aotinductor`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug

When a custom operator declares a mutable argument (`mutates_args=...`, i.e. a `Tensor(a!)` in the schema) and that argument is a **view of a larger buffer**, AOTInductor drops the view's storage offset under **dynamic shapes**. Every write lands at the start of the base buffer instead of at the view's offset.

There is no error or warning. The compiled model just returns wrong numbers, so it looks like a numerical problem in the model rather than a compiler issue.

Static export is unaffected, and so is `torch.compile` with the default Python wrapper. Only dynamic-shape AOTInductor is wrong.

#### Repro

Reproduces on **CPU and CUDA**, so no GPU is needed.

```python
"""AOTInductor drops the storage offset of a mutable (Tensor(a!)) view argument
under dynamic shapes.

    python repro.py         # cuda
    python repro.py cpu     # cpu
"""
import sys
import torch
from torch import nn
from torch.export import Dim
from torch._inductor import aoti_compile_and_package
from torch._inductor.package import load_package

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda"


@torch.library.custom_op("repro::write", mutates_args=("out",))
def write(out: torch.Tensor, x: torch.Tensor) -> None:
    out.copy_(x)


@write.register_fake
def _(out, x):
    return None


class M(nn.Module):
    """Two mutable writes into two disjoint slices of one zeroed buffer."""

    def forward(self, x0, x1):
        n0, n1 = x0.shape[0], x1.shape[0]
        buf = torch.zeros(n0 * 8 + n1 * 8, dtype=x0.dtype, device=x0.device)
        v0 = buf[: n0 * 8].view(n0, 8)      # storage_offset 0
        v1 = buf[n0 * 8 :].view(n1, 8)      # storage_offset n0*8, symbolic
        torch.ops.repro.write(v0, x0)
        torch.ops.repro.write(v1, x1)
        return v0.sum(), v1.sum()


def main():
    m = M().eval()
    xs = (torch.ones(10, 8, device=DEV), torch.full((5, 8), 2.0, device=DEV))
    expected = tuple(t.item() for t in m(*xs))          # (80.0, 80.0)

    for label, dynamic_shapes in (
        ("static ", None),
        ("dynamic", ({0: Dim("n0", min=2, max=64)}, {0: Dim("n1", min=2, max=64)})),
    ):
        ep = torch.export.export(m, xs, dynamic_shapes=dynamic_shapes)
        pkg = aoti_compile_and_package(ep, package_path=f"/tmp/repro_{label.strip()}.pt2")
        got = tuple(t.item() for t in load_package(pkg)(*xs))
        print(f"{label}  expected={expected}  aoti={got}  {'OK' if got == expected else 'WRONG'}")


if __name__ == "__main__":
    print(f"torch {torch.__version__}  device {DEV}")
    main()
```

#### Output

```
torch 2.14.0+cu130  device cuda
static   expected=(80.0, 80.0)  aoti=(80.0, 80.0)  OK
dynamic  expected=(80.0, 80.0)  aoti=(120.0, 0.0)  WRONG

torch 2.14.0+cu130  device cpu
static   expected=(80.0, 80.0)  aoti=(80.0, 80.0)  OK
dynamic  expected=(80.0, 80.0)  aoti=(120.0, 0.0)  WRONG
```

Both writes went to the first slice: it sums to 120 instead of 80, and the second slice was never written, so it sums to 0.

#### Root cause

`aten.sym_storage_offset` has no Inductor lowering.

1. Functionalization turns the mutating call into `auto_functionalized_v2`, recording the mutated view as a base plus its size, stride and storage offset.
2. The re-inplacing pass rebuilds the view as `as_strided(base, size, stride, sym_storage_offset(view))`. The post-grad FX graph is correct at this point, e.g.

   ```
   sym_storage_offset_default: "Sym(16384*(((s4 + 255)//256)))" = torch.ops.aten.sym_storage_offset.default(slice_2)
   as_strided_default = torch.ops.aten.as_strided.default(full, [...], [64, 1], sym_storage_offset_default)
   ```
3. `aten.sym_storage_offset.default` is not in `torch/_inductor/lowering.py`, while its siblings `sym_size.int`, `sym_stride.int` and `sym_numel` all are. Those exist precisely so the value is read from the node's metadata instead of being recomputed.
4. So it takes the implicit fallback, which re-executes the op on tensors rebuilt by `ir_node_to_tensor()`. That helper allocates with `torch.empty_strided(size, stride, ...)`, copying size and stride from the layout but never the offset, so every rebuilt tensor reports `storage_offset() == 0`.
5. The node lowers to the integer `0`, `as_strided` builds a `ReinterpretView` at offset 0, and the generated C++ passes `0L`:

   ```cpp
   // mutable argument, dynamic build
   reinterpret_tensor_wrapper(buf0, 2, int_array_2, int_array_1, 0L)
   // read-only argument in the same call, correct
   reinterpret_tensor_wrapper(buf0, 2, int_array_2, int_array_1,
                              16384L*c10::div_floor_integer(255L + s33, 256L))
   ```

Static export escapes this because the offset is a constant that never becomes a node. The default Python wrapper escapes it because the mutation stays in place and no `as_strided` is emitted.

Instrumenting the lowering shows the node's metadata is right and the lowered value is not:

```
[run_node] sym_storage_offset_default: meta_val=16384*(((s4 + 255)//256))  ->  lowered=0 (int)
```

#### Suggested fix

Register the missing lowering next to its siblings in `torch/_inductor/lowering.py`:

```python
@register_lowering(aten.sym_storage_offset.default)
def sym_storage_offset(a):
    val = V.graph.current_node.meta["val"]
    if isinstance(val, torch.SymInt):
        return val.node.expr
    else:
        return int(val)
```

With that patch applied, the repro above prints `OK` for both static and dynamic on CPU and CUDA. It also fixes a real model where several accumulator buffers are packed into one allocation and each consumer is handed its own slice.

A PR with this fix and a regression test in `test/inductor/test_aot_inductor.py` follows.

### Versions

Reproduced on 2.11.0+cu130, 2.13.0+cu130 and 2.14.0+cu130, on both CPU and CUDA (B200, CUDA 13.0, Python 3.12 and 3.14).

`torch/_inductor/lowering.py` on `main` registers `sym_size.int`, `sym_stride.int` and `sym_numel`, and contains no occurrence of `sym_storage_offset`, so `main` looks affected as well.

cc @ezyang @gchanan @kadeng @msaroufim @chauhang @penguinwu @bobrenjc93 @aditvenk @laithsakka @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @muchulee8 @amjames @aakhundov @coconutruben @jataylo @avikchaudhuri @zhxchen17 @tugsbayasgalan @angelayi @ydwu4 @desertfire @yushangdi @iupaikov-amd @chenyang78 @eellison

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises dynamic-shape AOTInductor handling of mutable view arguments whose storage offset is symbolic.

## Observed implementation decision

Missing aten.sym_storage_offset lowering causes the offset to be recomputed from a tensor rebuilt at offset zero.

## Physical effect

Writes intended for a disjoint view land at the beginning of the base buffer.

## Observed consequence

Two mutable slices overwrite the wrong physical region and return wrong sums.

## Reported proposal or workaround

PR 198069 proposes reading the symbolic offset from node metadata.

## Repair trace

PR 198069 was not merged, so the diagnosed missing lowering has no landed repair or fixed regression.

## Archived evidence

- Raw report: `data/raw/github/pytorch/198068/report.json`
- SHA-256: `e359741e784b8eb7e3b817bfe944b46eb134dbc89e90f1dc1d1db0737b40e86a`
