# `iree_hal_allocator_deallocate_buffer` segfault with minimal script

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/9033](https://github.com/iree-org/iree/issues/9033)
- Opened: 2022-05-03T00:29:02Z
- Closed: 2022-05-10T03:51:32Z
- Labels: `bug 🐞`, `help wanted`

## Reported behavior

This script

```python
import numpy as np

import iree.runtime as ireert

def test_cpu():
    driver = ireert.HalDriver.create("dylib-sync")
    device = driver.create_default_device()
    init_ary = np.zeros([3, 4], dtype=np.int32) + 2
    ary = ireert.asdevicearray(device, init_ary)
    print(ary)


test_cpu()
```

segfaults like this

```
* thread #1, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0x0)
  * frame #0: 0x00007fffa6434ea4 binding.cpython-39-x86_64-linux-gnu.so`___lldb_unnamed_symbol859$$binding.cpython-39-x86_64-linux-gnu.so + 4
    frame #1: 0x00007fffa6436fa1 binding.cpython-39-x86_64-linux-gnu.so`___lldb_unnamed_symbol903$$binding.cpython-39-x86_64-linux-gnu.so + 49
    frame #2: 0x00007fffa641b6b0 binding.cpython-39-x86_64-linux-gnu.so`___lldb_unnamed_symbol584$$binding.cpython-39-x86_64-linux-gnu.so + 80
    frame #3: 0x00007fffa6401974 binding.cpython-39-x86_64-linux-gnu.so`___lldb_unnamed_symbol428$$binding.cpython-39-x86_64-linux-gnu.so + 484
    frame #4: 0x000055555567655b python`dict_dealloc at object.c:2209
    frame #5: 0x00005555556fd333 python`subtype_dealloc at object.c:2209
    frame #6: 0x000055555567683c python`frame_dealloc at object.c:2209
    frame #7: 0x00005555556db8f6 python`_PyFunction_Vectorcall at object.c:2209
    frame #8: 0x00005555556533be python`_PyEval_EvalFrameDefault.cold.2979 at abstract.h:118
    frame #9: 0x00005555556dab32 python`_PyEval_EvalCode at pycore_ceval.h:40
    frame #10: 0x000055555578addc python`PyEval_EvalCodeEx at ceval.c:4359
    frame #11: 0x00005555556dbb5b python`PyEval_EvalCode(co=<unavailable>, globals=<unavailable>, locals=<unavailable>) at ceval.c:826
    frame #12: 0x000055555578ae8b python`run_eval_code_obj at pythonrun.c:1219
    frame #13: 0x00005555557bb215 python`run_mod at pythonrun.c:1240
    frame #14: 0x0000555555666677 python`pyrun_file.cold.3071 at pythonrun.c:1138
    frame #15: 0x00005555557bfcef python`PyRun_SimpleFileExFlags at pythonrun.c:449
    frame #16: 0x00005555557c03b8 python`Py_RunMain at main.c:379
    frame #17: 0x00005555557c05b9 python`Py_BytesMain at main.c:1129
    frame #18: 0x00007ffff7c550b3 libc.so.6`__libc_start_main + 243
    frame #19: 0x0000555555748a64 python`_start + 41
```

Building IREE from source gets me

```
* thread #1, name = 'python', stop reason = signal SIGSEGV: invalid address (fault address: 0x0)
  * frame #0: 0x00007fffa646de34 binding.cpython-39-x86_64-linux-gnu.so`iree_hal_allocator_deallocate_buffer(allocator=0x0000555555d9ea60, buffer=0x0000555555d7a400) at allocator.c:139:3
    frame #1: 0x00007fffa64708b4 binding.cpython-39-x86_64-linux-gnu.so`iree_hal_buffer_view_release [inlined] iree_hal_buffer_view_destroy(buffer_view=0x0000555555d74c90) at buffer_view.c:89:3
    frame #2: 0x00007fffa64708a3 binding.cpython-39-x86_64-linux-gnu.so`iree_hal_buffer_view_release(buffer_view=0x0000555555d74c90) at buffer_view.c:81
    frame #3: 0x00007fffa6458979 binding.cpython-39-x86_64-linux-gnu.so`pybind11::class_<iree::python::HalBufferView>::dealloc(pybind11::detail::value_and_holder&) [inlined] iree::python::ApiPtrAdapter<iree_hal_buffer_view_t>::Release(bv=<unavailable>) at hal.h:55:5
    frame #4: 0x00007fffa6458974 binding.cpython-39-x86_64-linux-gnu.so`pybind11::class_<iree::python::HalBufferView>::dealloc(pybind11::detail::value_and_holder&) [inlined] iree::python::ApiRefCounted<iree::python::HalBufferView, iree_hal_buffer_view_t>::Release(this=<unavailable>) at binding.h:88
    frame #5: 0x00007fffa645896c binding.cpython-39-x86_64-linux-gnu.so`pybind11::class_<iree::python::HalBufferView>::dealloc(pybind11::detail::value_and_holder&) [inlined] iree::python::ApiRefCounted<iree::python::HalBufferView, iree_hal_buffer_view_t>::~ApiRefCounted(this=<unavailable>) at binding.h:40
    frame #6: 0x00007fffa645896c binding.cpython-39-x86_64-linux-gnu.so`pybind11::class_<iree::python::HalBufferView>::dealloc(pybind11::detail::value_and_holder&) [inlined] std::default_delete<iree::python::HalBufferView>::operator(__ptr=0x0000555555d74ce0)(iree::python::HalBufferView*) const at unique_ptr.h:85
    frame #7: 0x00007fffa645896c binding.cpython-39-x86_64-linux-gnu.so`pybind11::class_<iree::python::HalBufferView>::dealloc(pybind11::detail::value_and_holder&) at unique_ptr.h:361
    frame #8: 0x00007fffa6458963 binding.cpython-39-x86_64-linux-gnu.so`pybind11::class_<iree::python::HalBufferView>::dealloc(v_h=0x00007fffffffc948) at pybind11.h:1651
    frame #9: 0x00007fffa64254ab binding.cpython-39-x86_64-linux-gnu.so`pybind11::detail::clear_instance(self=0x00007fffa616bb30) at class.h:399:17
    frame #10: 0x00007fffa64251ac binding.cpython-39-x86_64-linux-gnu.so`::pybind11_object_dealloc(self=0x00007fffa616bb30) at class.h:419:5
    frame #11: 0x000055555567655b python`dict_dealloc at object.c:2209
    frame #12: 0x00005555556fd333 python`subtype_dealloc at object.c:2209
    frame #13: 0x000055555567683c python`frame_dealloc at object.c:2209
    frame #14: 0x00005555556db8f6 python`_PyFunction_Vectorcall at object.c:2209
    frame #15: 0x00005555556533be python`_PyEval_EvalFrameDefault.cold.2979 at abstract.h:118
    frame #16: 0x00005555556dab32 python`_PyEval_EvalCode at pycore_ceval.h:40
    frame #17: 0x000055555578addc python`PyEval_EvalCodeEx at ceval.c:4359
    frame #18: 0x00005555556dbb5b python`PyEval_EvalCode(co=<unavailable>, globals=<unavailable>, locals=<unavailable>) at ceval.c:826
    frame #19: 0x000055555578ae8b python`run_eval_code_obj at pythonrun.c:1219
    frame #20: 0x00005555557bb215 python`run_mod at pythonrun.c:1240
    frame #21: 0x0000555555666677 python`pyrun_file.cold.3071 at pythonrun.c:1138
    frame #22: 0x00005555557bfcef python`PyRun_SimpleFileExFlags at pythonrun.c:449
    frame #23: 0x00005555557c03b8 python`Py_RunMain at main.c:379
    frame #24: 0x00005555557c05b9 python`Py_BytesMain at main.c:1129
    frame #25: 0x00007ffff7c550b3 libc.so.6`__libc_start_main + 243
    frame #26: 0x0000555555748a64 python`_start + 41
(lldb) ^C

```

I say "possibly on conda" because @ScottTodd was able to run [without issue in colab](https://colab.research.google.com/gist/ScottTodd/a3d6c66c26a7291063bdd67650414a64/ireert-sanity-check.ipynb) but I ran on 4 different machines and got the same segfault.

In order to prove I'm not insane, I put together a Dockerfile that repros (that I checked on two machines):

```Dockerfile
FROM ubuntu:18.04

SHELL ["/bin/bash", "-c"]
RUN apt-get update

RUN apt-get install -y wget lldb && rm -rf /var/lib/apt/lists/*

# install conda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH="~/miniconda3/bin:$PATH"
RUN conda --version
RUN conda create -n iree_sanity_check python=3.7
RUN conda run -n iree_sanity_check python -m pip install iree-runtime -f \
  https://github.com/google/iree/releases

# create script
RUN echo -e '\n\
import numpy as np\n\
import iree.runtime as ireert\n\
def test_cpu():\n\t\
 driver = ireert.HalDriver.create("dylib-sync")\n\t\
 device = driver.create_default_device()\n\t\
 init_ary = np.zeros([3, 4], dtype=np.int32) + 2\n\t\
 ary = ireert.asdevicearray(device, init_ary)\n\t\
 print(ary)\n\
test_cpu()\n\
' >> /iree_sanity_check.py

RUN cat /iree_sanity_check.py

#CMD /root/miniconda3/envs/iree_sanity_check/bin/python /iree_sanity_check.py && echo $?
CMD lldb -o run /root/miniconda3/envs/iree_sanity_check/bin/python -- /iree_sanity_check.py
```

which can be built and run with (if you want to drop into LLDB)

```bash
docker build . -t iree_sanity_check
docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined iree_sanity_check
```

or with

```bash
docker build . -t iree_sanity_check
docker run iree_sanity_check
```

if you don't want to do `cap-add` (but in this case you need to comment out the second `CMD` in the Dockerfile and uncomment the first).

cc @powderluv

## Trigger

A Python function returns a device array after its local driver and device references leave scope.

## Relevant entities

device array buffer; iree_hal_buffer_t; backing iree_hal_allocator_t; Python driver/device wrappers

## Implemented relation

A HAL buffer stores the allocator used later to deallocate its storage.

## Pre-state

The allocator and buffer are live, and the buffer depends on the allocator for destruction.

## Binding transition

Python teardown releases the driver/device while the returned array still owns the buffer.

## Post-state

The buffer remains live but its backing allocator has been released.

## Validity obligation

The backing allocator remains alive until every buffer that uses it has completed destruction.

## Invalid decision

Buffer initialization stored the allocator pointer without retaining the allocator.

## Physical effect

Later buffer destruction dereferences allocator state that has already been freed.

## Observed consequence

ASan reports a heap use-after-free and the minimal script segfaults during teardown across multiple drivers.

## Repair site

runtime/src/iree/hal/buffer.c

## Repair

Buffer initialization retains the device allocator, and buffer destruction releases it after deallocation.

## Why the repair works

The buffer now owns an allocator reference for exactly the interval in which destruction can call the allocator.

## Causal basis

The retain/release delta and regression test directly restore the lifetime relation identified in the issue.

## Analysis implication

A lifetime check needs the ownership edge from a buffer to the allocator used by its destructor.

## Fix patches

- [`fixes/iree-9033-fix.patch`](fixes/iree-9033-fix.patch) — https://github.com/iree-org/iree/pull/9068
