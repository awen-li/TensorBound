# [inductor] FX-graph cache key ignores tensor pinned-ness

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/197519](https://github.com/pytorch/pytorch/issues/197519)
- Opened: 2026-09-18T07:27:28Z
- Closed: Open at collection
- Labels: `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `module: dynamo`, `bot-triaged`, `release triage`

## Reported behavior

### 🐛 Describe the bug
The FX-graph cache key doesn't account for tensor pinned-ness. `FxGraphCachePickler._reduce_tensor` (`torch/_inductor/codecache.py`) hashes `TensorMetadata` (+ values for inlined constants), and `TensorMetadata` has no `is_pinned` field and `pin` appears nowhere in the key logic. This hits both the inlined and the frozen-param metadata-only paths: compile the same values once pinned and once pageable, and the second compile silently reuses the first compile's artifact, with the wrong memory kind.

Spotted while fixing #197211,  whose warm-cache masking is a symptom of this.

### Repro
Requires CUDA. Both compiles share one fresh `TORCHINDUCTOR_CACHE_DIR`:

```python
import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/fxcache_shared"  # fresh dir
import torch

def make(pin):
    def fn():
        return torch.tensor([1, 2, 3], pin_memory=pin, device="cpu")
    torch._dynamo.reset()
    return torch.compile(fn, backend="inductor")()

o1 = make(True); o2 = make(False)   # expect o1 pinned, o2 pageable
assert (o1.is_pinned(), o2.is_pinned()) == (True, False)
```

### Error logs

On main @ 0e99909, the second compile always reuses the first compile's artifact:
- pin-first: `got pinned=[True, True]` (`AssertionError: second compile wrong: True`)
- reversed, fresh dir: `got pinned=[False, False]` (`AssertionError: second compile wrong: False`)


Related: #197211, #173939
cc @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @kadeng @muchulee8 @amjames @aakhundov @coconutruben @jataylo @azahed98 @kshitij12345 @bobrenjc93

### Versions

PyTorch: 2.15.0a0+git0e99909 (CUDA 12.8 build), GPU: NVIDIA H200

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises torchInductor FX-graph caching for tensor inputs whose pinned-memory state affects generated behavior.

## Observed implementation decision

The cache key omits pinned-ness and may reuse a graph specialized for a different host-memory transfer contract.

## Physical effect

A cached compilation applies the wrong memory-property assumptions to a later input.

## Observed consequence

The report observes configuration-dependent incorrect cache reuse.

## Reported proposal or workaround

PR 197557 proposes adding pinned-ness to the cache key.

## Repair trace

PR 197557 was not merged and no alternative landed repair or fixed regression is linked.

## Archived evidence

- Raw report: `data/raw/github/pytorch/197519/report.json`
- SHA-256: `5248e820db70301cce5a48c70190d4525594b5ddf467682e7667cf452f6f7889`
