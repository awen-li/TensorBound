# TSAN race in //tests:lax_numpy_einsum_test_cpu

- Repository: `jax-ml/jax`
- Report: [https://github.com/jax-ml/jax/issues/26305](https://github.com/jax-ml/jax/issues/26305)
- Opened: 2025-02-04T13:43:38Z
- Closed: 2025-02-24T10:39:37Z
- Labels: `bug`, `free threading`

## Reported behavior

### Description

In this TSAN run https://github.com/jax-ml/jax/actions/runs/13134558166/job/36646914845?pr=26300 we see the following race, which I've seen a few times.

It appears to be a race between the bf_getbuffer handler on JAX arrays and the allocator. The JAX code is just zeroing the buffer view object right at the start of exporting it, so I doubt it's a JAX bug.

My guess: perhaps there's a race in memoryview or NumPy, where the consumer of the exported buffer doesn't have adequate synchronization?

https://github.com/python/cpython/issues/127716 implies that memoryview isn't thread safe yet.

```
WARNING: ThreadSanitizer: data race (pid=411609)
  Read of size 8 at 0x7fffc21f42b8 by thread T71 (mutexes: read M0):
    #0 mbuf_traverse /__w/jax/jax/cpython/Objects/memoryobject.c:134:5 (python3.13+0x27ef50) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #1 update_refs /__w/jax/jax/cpython/Python/gc_free_threading.c:441:5 (python3.13+0x44818a) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #2 _mi_heap_area_visit_blocks /__w/jax/jax/cpython/Objects/mimalloc/heap.c:630:14 (python3.13+0x2a9da4) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #3 mi_heap_area_visitor /__w/jax/jax/cpython/Objects/mimalloc/heap.c:681:12 (python3.13+0x2aa307) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #4 mi_heap_visit_areas_page /__w/jax/jax/cpython/Objects/mimalloc/heap.c:661:10 (python3.13+0x2aa307)
    #5 mi_heap_visit_pages /__w/jax/jax/cpython/Objects/mimalloc/heap.c:46:12 (python3.13+0x2aa307)
    #6 mi_heap_visit_areas /__w/jax/jax/cpython/Objects/mimalloc/heap.c:667:10 (python3.13+0x2aa307)
    #7 mi_heap_visit_blocks /__w/jax/jax/cpython/Objects/mimalloc/heap.c:692:10 (python3.13+0x2aa307)
    #8 gc_visit_heaps_lock_held /__w/jax/jax/cpython/Python/gc_free_threading.c:267:14 (python3.13+0x44435f) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #9 gc_visit_heaps /__w/jax/jax/cpython/Python/gc_free_threading.c:306:11 (python3.13+0x44435f)
    #10 deduce_unreachable_heap /__w/jax/jax/cpython/Python/gc_free_threading.c:614:5 (python3.13+0x44554b) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #11 gc_collect_internal /__w/jax/jax/cpython/Python/gc_free_threading.c:1125:15 (python3.13+0x44554b)
    #12 gc_collect_main /__w/jax/jax/cpython/Python/gc_free_threading.c:1238:5 (python3.13+0x44554b)
    #13 _Py_RunGC /__w/jax/jax/cpython/Python/gc_free_threading.c:1688:5 (python3.13+0x446f5b) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #14 _Py_HandlePending /__w/jax/jax/cpython/Python/ceval_gil.c:1296:9 (python3.13+0x453590) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #15 _PyEval_EvalFrameDefault /__w/jax/jax/cpython/Python/generated_cases.c.h:1364:13 (python3.13+0x3e4a3f) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #16 _PyEval_EvalFrame /__w/jax/jax/cpython/./Include/internal/pycore_ceval.h:119:16 (python3.13+0x3de77a) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #17 _PyEval_Vector /__w/jax/jax/cpython/Python/ceval.c:1812:12 (python3.13+0x3de77a)
    #18 _PyFunction_Vectorcall /__w/jax/jax/cpython/Objects/call.c (python3.13+0x1eb3bf) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #19 _PyObject_VectorcallDictTstate /__w/jax/jax/cpython/Objects/call.c:135:15 (python3.13+0x1e9f3d) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)

  Previous write of size 8 at 0x7fffc21f42b8 by thread T69 (mutexes: read M0):
    #0 __tsan_memset <null> (python3.13+0xda21d) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #1 memset /usr/include/x86_64-linux-gnu/bits/string_fortified.h:59:10 (xla_extension.so+0xa45c77c) (BuildId: 3f4154db7f97f151aeb111dc3712d107501fa2db)
    #2 xla::(anonymous namespace)::PyArray_bf_getbuffer(_object*, Py_buffer*, int)::$_0::operator()() const /proc/self/cwd/external/xla/xla/python/py_array.cc:1559:5 (xla_extension.so+0xa45c77c)
    #3 xla::(anonymous namespace)::PyArray_bf_getbuffer(_object*, Py_buffer*, int) /proc/self/cwd/external/xla/xla/python/py_array.cc:1476:25 (xla_extension.so+0xa45c77c)
    #4 PyObject_GetBuffer /__w/jax/jax/cpython/Objects/abstract.c:442:15 (python3.13+0x1ba6dd) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #5 _PyManagedBuffer_FromObject /__w/jax/jax/cpython/Objects/memoryobject.c:97:9 (python3.13+0x27fbb5) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #6 PyMemoryView_FromObjectAndFlags /__w/jax/jax/cpython/Objects/memoryobject.c:813:42 (python3.13+0x27fbb5)
    #7 PyMemoryView_FromObject /__w/jax/jax/cpython/Objects/memoryobject.c:856:12 (python3.13+0x27fada) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #8 _array_from_array_like <null> (_multiarray_umath.cpython-313t-x86_64-linux-gnu.so+0x152531) (BuildId: 438c93a6ad49fa70852ddd4604c7d065ff197b82)
    #9 _PyObject_VectorcallTstate /__w/jax/jax/cpython/./Include/internal/pycore_call.h:168:11 (python3.13+0x1ead4a) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #10 PyObject_Vectorcall /__w/jax/jax/cpython/Objects/call.c:327:12 (python3.13+0x1ead4a)
    #11 _PyEval_EvalFrameDefault /__w/jax/jax/cpython/Python/generated_cases.c.h:813:23 (python3.13+0x3e264b) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #12 _PyEval_EvalFrame /__w/jax/jax/cpython/./Include/internal/pycore_ceval.h:119:16 (python3.13+0x3de77a) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #13 _PyEval_Vector /__w/jax/jax/cpython/Python/ceval.c:1812:12 (python3.13+0x3de77a)
    #14 _PyFunction_Vectorcall /__w/jax/jax/cpython/Objects/call.c (python3.13+0x1eb3bf) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #15 _PyObject_VectorcallTstate /__w/jax/jax/cpython/./Include/internal/pycore_call.h:168:11 (python3.13+0x1ef440) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #16 method_vectorcall /__w/jax/jax/cpython/Objects/classobject.c:92:18 (python3.13+0x1ef440)
    #17 _PyVectorcall_Call /__w/jax/jax/cpython/Objects/call.c:273:16 (python3.13+0x1eb033) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #18 _PyObject_Call /__w/jax/jax/cpython/Objects/call.c:348:16 (python3.13+0x1eb033)
    #19 PyObject_Call /__w/jax/jax/cpython/Objects/call.c:373:12 (python3.13+0x1eb0b5) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
    #20 _PyEval_EvalFrameDefault /__w/jax/jax/cpython/Python/generated_cases.c.h:1355:26 (python3.13+0x3e4832) (BuildId: 88c4dbf7314b62867b98656bbedcd8e585dde43c)
```

### System info (python version, jaxlib version, accelerator, etc.)

Python 3.13t

## Trigger

A free-threaded Python build exports a JAX array through bf_getbuffer while garbage collection traverses the memoryview object.

## Relevant entities

JAX array buffer; Py_buffer view; bf_getbuffer writer; Python garbage collector; GIL

## Implemented relation

The exported Py_buffer metadata binds a Python memoryview to the JAX buffer and its external reference.

## Pre-state

The view object is visible to Python runtime traversal while its metadata is about to be initialized.

## Binding transition

PyArray_bf_getbuffer releases the GIL and writes or clears fields of the Py_buffer view.

## Post-state

Garbage collection can read the view concurrently with JAX writing the same fields.

## Validity obligation

Publication of buffer-view metadata is synchronized with Python runtime traversal and retains the backing buffer reference.

## Invalid decision

The handler assumed the POD Py_buffer could be written safely after releasing the GIL.

## Physical effect

The GC traversal and memset of the Py_buffer view race on the same memory.

## Observed consequence

ThreadSanitizer reports a data race between mbuf_traverse and PyArray_bf_getbuffer.

## Repair site

xla/python/py_array.cc::PyArray_bf_getbuffer

## Repair

PR 22946 limits the GIL-released region to blocking buffer operations, then initializes and publishes the view while the GIL is held.

## Why the repair works

Python GC cannot concurrently traverse the view while the handler writes its published fields.

## Causal basis

The TSAN stacks identify the exact read/write conflict, and the linked patch moves the conflicting writes outside the GIL-released region.

## Analysis implication

A buffer-export check needs object-publication state, backing-buffer lifetime, lock state, and concurrent runtime traversals.

## Fix patches

- [`fixes/xla-26305-pr22946.patch`](fixes/xla-26305-pr22946.patch) — https://github.com/openxla/xla/pull/22946
