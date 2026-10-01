# Scalar vs. Tensor atomic, Sychornization & Broadcasting Behavior

- Repository: `triton-lang/triton`
- Report: [https://github.com/triton-lang/triton/issues/7125](https://github.com/triton-lang/triton/issues/7125)
- Opened: 2025-06-10T02:33:41Z
- Closed: Open at collection
- Labels: `bug`

## Reported behavior

### Describe the bug

TL;DR The synchronization behavior for `atomic` ops is inconsistent between scalar vs. tensor, and release pattern vs. acquire pattern.


I'm looking for a nice way to implement locks & tickets for synchronization between blocks. Similar to the backward pass of this example: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html.

## Acquire Pattern with Atomics.

### Spin and acquire a Scalar Lock
Spin until a lock is acquired, force a fence, and then access data.
The desired ptx pattern:
```
acquire_lock:
@%p0 ld.global.gpu.acquire.b32 %r0, [ %rd ];
setp.eq.s32 %p, %r0, <const>;
@%p bra acquire_lock

bar.sync 0; # _sync_thread();
```
However, the example in the tutorial generates many extra barriers and shared memory broadcasting within the busy spinning loop, causing a significant performance drop:
```python
while tl.atomic_add(addr, 0, sem="acquire"):
   pass
```
becomes
```
acquire_lock:
bar.sync 0
@%p0 ld.global.gpu.acquire.b32 %r0, [ %rd ];
@%p0 st.shared.b32 %r0 ... # stores the register to shared memory for broadcast
bar.sync 0
ld.shared.u32 %r0 ...
setp.eq.s32 %p1, %r0, <const>
@%p1 bra acquire_lock;
```

### Spin and acquire a lock in a tensor
Now, if the lock is represented as a tensor, the implicit synchronize and broadcast no longer happens, and predicate for tensor length is not properly propagated, resulting a deadlock.

```python

@triton.jit
def test_barrier(addrs, expect, target, sem: tl.constexpr):
    barrier_value = tl.atomic_cas(addrs, expect, target, sem=sem, scope="sys")
    passed = barrier_value == expect
    # if some failed: [0, 0, 0, 0, 1, 0, 0, 0]
    # if all passed: [0, 0, 0, 0, 0, 0, 0, 0]

    all_passed = tl.min(passed.to(tl.int1)).to(tl.int1)

    return all_passed

addrs = tl.arange(0, 1) # vector of length 1
ones = tl.full(1, (1, ), dtype=tl.uint32)
zeros = tl.zeros((1, ), dtype=tl.uint32)
while not test_barrier(addrs, ones, zeros, sem="acquire"):
 pass
tl.debug_barrier()
```

compiled to ptx:
```
setp.eq.s32 	%p1, %r1, 0;
mov.b32 	%r11, 1;
mov.b32 	%r12, 0;
acquire_lock:
mov.u32 %r10, 0x0;
@%p1 atom.global.acquire.sys.cas.b32 %r10, [ %rd5 + 0 ], %r11, %r12;
setp.ne.s32 	%p2, %r10, 1;
@%p2 bra 	$acquire_lock;
bar.sync 0
```
This would deadlock even when the lock is set to 1:
register r10 is initialized to 0, and for tid > 0, %p1 is false and r10 is never updated. This means `setp.ne` always evaluates to true, and bra is always taken, causing a deadlock.

I've also tested when the vector length is 4, where the first 4 threads each waits on a different lock, then `tl.min` for reduction to check if all locks are acquired. It turns out that intra-warp reduction doesn't force a fence/synchronization / broadcasting behavior, and the deadlock still exists.

**Desired Behavior**: First N thread (N=length of the lock tensor) each waits on a different lock address in a spinning while loop. Then `__syncthreads()` before accessing data.
```
acquire_lock:
    @%p1 atom.global.acquire.sys.cas.b32 %r10, [ %rd5 + 0 ], %r11, %r12;
    setp.ne.s32 	%p2, %r10, 1;
    add  %p3 %p1 %p2 # propagate the predicate and only activate first N threads.
    @%p3 bra 	$acquire_lock;
bar.sync 0
```

```python
ones = tl.full(1, (N, ), dtype=tl.uint32)
zeros = tl.zeros((N, ), dtype=tl.uint32)
while not test_barrier(addrs, ones, zeros, sem="acquire"):
 pass
tl.debug_barrier()
```

## Release Pattern with Atomic
On the other hand, releasing a scalar lock needs an explicit barrier: https://github.com/triton-lang/triton/pull/6307

```
# need a barrier to ensure all threads finished before
    # releasing the lock
    tl.debug_barrier()

    # Release the lock
    tl.atomic_xchg(Lock, 0)
```


# A Consistent Broadcasting / Synchronization Behavior?
Opened this issue to discuss what would be a more consistent behavior for `atomics`. And if there is a way to enable locks without a significant performance overhead in Triton.



### Environment details

GPU: H100

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises triton lowering of scalar and tensor atomic acquire loops with inactive lanes.

## Observed implementation decision

Inactive lanes retain a default result that fails the loop-exit comparison, while scalar lowering uses shared-memory broadcast and barriers.

## Physical effect

Inactive lanes keep the acquire loop active after participating lanes succeed.

## Observed consequence

The tensor lock implementation deadlocks.

## Reported proposal or workaround

A commenter proposes predicate-select fixes for AtomicCAS and AtomicRMW.

## Repair trace

No linked pull request, landed commit, or fixed regression supports the proposed change; the issue remains open.

## Archived evidence

- Raw report: `data/raw/github/triton/7125/report.json`
- SHA-256: `94d9a4bc48b6d88de415af1816002e7fa525f0e67fdf3243d39c37ef9e28a8bd`
