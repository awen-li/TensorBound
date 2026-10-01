# [Inductor] Missing host-side synchronization after non-blocking D2H copies

- Repository: `pytorch/pytorch`
- Report: [https://github.com/pytorch/pytorch/issues/174695](https://github.com/pytorch/pytorch/issues/174695)
- Opened: 2026-02-10T17:36:05Z
- Closed: 2026-03-28T04:02:21Z
- Labels: `high priority`, `triaged`, `module: correctness (silent)`, `oncall: pt2`, `module: inductor`, `module: compiled streams`

## Reported behavior

### 🐛 Describe the bug

When compiling a function that performs non‑blocking device→host copies followed by a CPU kernel (here: fused cat), Inductor preserves non_blocking=True on the copies but does not insert any wait/barrier before the CPU kernel consumes those pinned CPU buffers. As a result, the CPU kernel can read from buffers whose D2H DMA hasn’t completed yet. The Python‑level torch.Event().record()/synchronize() calls that enforce ordering in eager mode are dropped by Inductor and no alternative synchronization is emitted.

```python
# repro.py
import torch

def f(a_list):
    a_cpu_list = []
    a_to_cpu_event_list = []
    for a in a_list:
        # non-blocking D2H
        a_cpu = a.to(device="cpu", non_blocking=True)
        # explicit event ordering in eager
        e = torch.Event()
        e.record()
        a_cpu_list.append(a_cpu)
        a_to_cpu_event_list.append(e)

    # ensure all D2H have finished before CPU consumption
    for e in a_to_cpu_event_list:
        e.synchronize()

    return torch.cat(a_cpu_list)

def main():
    f_compiled = torch.compile(f)  # default Inductor backend
    inputs = [torch.rand(1000, dtype=torch.float16, device="cuda") for _ in range(100)]
    out = f_compiled(inputs)

if __name__ == "__main__":
    main()
```

Enabling TORCH_COMPILE_DEBUG=1 and running the script produces code (shortened) like:
```python
buf0 = empty_strided_cpu_pinned((1000,), (1,), torch.float16)
buf0.copy_(arg0_1, True)   # non_blocking=True is preserved

# ... repeated for buf1..buf99 ...

# fused CPU kernel immediately consumes buf0..buf99
cpp_fused_cat_0(buf0, buf1, ..., buf99, buf100, ..., buf199)
```

Preserving the non_blocking attribute was added by https://github.com/pytorch/pytorch/pull/136513.

The reproducer is taken from the pytorch unit test [here](https://github.com/pytorch/pytorch/blob/947729b8be8d33d45fe613e0f14885a55fbc2dba/test/inductor/test_torchinductor.py#L14852-L14896).  The test is flaky, and was skipped on ROCm Navi because it was frequently observed to fail there.  But this is a generic issue.

### Versions

CC @eellison @NiklasZ

cc @ezyang @gchanan @kadeng @msaroufim @chauhang @penguinwu @voznesenskym @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @wenzhe-nrv @jiayisunx @ipiszy @muchulee8 @amjames @aakhundov @coconutruben @jataylo

## Trigger

Compiled code issues nonblocking device-to-host copies, records an event, synchronizes it, and then consumes the pinned host tensors.

## Relevant entities

asynchronous device_put copies; destination CPU tensors; recorded event; Event.synchronize; host consumer

## Implemented relation

The event synchronization orders host consumption after completion of the device-to-host copies.

## Pre-state

The copies may still be in flight and their CPU destinations are not ready for host reads.

## Binding transition

Dynamo captures Event.synchronize and Inductor reorders nodes for locality.

## Post-state

The compiled graph lacks a preserved ordered synchronization between the cross-device copies and the host consumer.

## Validity obligation

Every host read of a nonblocking device-to-host destination occurs after a completion event for the producing copy.

## Invalid decision

Event.synchronize was not represented as an ordered effect, and locality reordering could move cross-device device_put operations across synchronization.

## Physical effect

The CPU consumer can execute while DMA is still writing its input buffers.

## Observed consequence

The compiled host computation can read stale or partially copied values; the report later confirms correct generated output on main.

## Repair site

torch/_dynamo/variables/streams.py, torch/_inductor/fx_passes/post_grad.py, and torch/_inductor/lowering.py

## Repair

PR 176878 introduces an ordered synchronize_event operation, lowers it through Inductor, and prevents cross-device nonblocking device_put from moving across synchronization during locality reordering.

## Why the repair works

The completion event becomes an ordered graph operation and the producing cross-device copies retain their position before that event and the host consumer.

## Causal basis

The patch implements both missing pieces identified in the issue discussion, and the reporter confirms that main emits the correct ordered output code.

## Analysis implication

An asynchronous-transfer check needs the destination device, nonblocking flag, completion event, and ordering of the first host consumer.

## Fix patches

- [`fixes/pytorch-174695-pr176878.patch`](fixes/pytorch-174695-pr176878.patch) — https://github.com/pytorch/pytorch/pull/176878
