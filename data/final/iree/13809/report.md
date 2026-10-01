# Memory/alignment errors in Emscripten/Wasm/WebGPU

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/13809](https://github.com/iree-org/iree/issues/13809)
- Opened: 2023-05-26T18:51:58Z
- Closed: open at collection time
- Labels: `bug 🐞`, `runtime`, `platform/web 🌐`

## Reported behavior

I'm trying to get more programs running through WebGPU on the [`webgpu` branch](https://github.com/openxla/iree/tree/webgpu/) with the application code at [`experimental/web/sample_webgpu`](https://github.com/openxla/iree/tree/webgpu/experimental/web/sample_webgpu) and the latest errors appear to be related to memory-corruption.

## Symptoms

The specific symptom I'm seeing is with programs that have more than one output, buffer handles are incorrect during `iree_hal_create_transfer_command_buffer`, triggering an assert:
![image](https://github.com/openxla/iree/assets/4010439/5eb49f38-87e9-441b-ba05-9a2088b83cfa)

<details><summary>Assert callstack</summary>

```
iree_api_webgpu.js:76 (C) Aborted(Assertion failed)
printErr                                        @ iree_api_webgpu.js:76
abort                                           @ web-sample-webgpu.js:899
assert                                          @ web-sample-webgpu.js:465
Manager.get                                     @ web-sample-webgpu.js:1425
_wgpuCommandEncoderCopyBufferToBuffer           @ web-sample-webgpu.js:1774
imports.<computed>                              @ web-sample-webgpu.js:2743
$legalfunc$wgpuCommandEncoderCopyBufferToBuffer @ web-sample-webgpu.wasm:0x94c71
$iree_hal_webgpu_command_buffer_copy_buffer     @ command_buffer.c:735
$iree_hal_create_transfer_command_buffer        @ command_buffer.c:465
$call_function                                  @ main.c:453
ret.<computed>                                  @ web-sample-webgpu.js:2777
(anonymous)                                     @ web-sample-webgpu.js:982
ccall                                           @ web-sample-webgpu.js:2671
(anonymous)                                     @ web-sample-webgpu.js:2711
_ireeCallFunction                               @ iree_api_webgpu.js:210
ireeCallFunction                                @ iree_api_webgpu.js:53
callFunctionWithFormInputs                      @ ?function=multiple_r…lts_webgpu.vmfb:300
onclick                                         @ ?function=multiple_r…ults_webgpu.vmfb:95
```
</details>

This program works as expected:
```mlir
func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
  %result = math.absf %input : tensor<f32>
  return %result : tensor<f32>
}
```

This program crashes there:
```mlir
func.func @multiple_results(
    %input_0 : tensor<f32>,
    %input_1 : tensor<f32>
) -> (tensor<f32>, tensor<f32>) {
  %result_0 = math.absf %input_0 : tensor<f32>
  %result_1 = math.absf %input_1 : tensor<f32>
  return %result_0, %result_1 : tensor<f32>, tensor<f32>
}
```

## Debugging

The WebGPU HAL and web/Emscripten runtime code are both still works in progress, so I'm not ruling out some configuration or code issue there. The WebGPU application code (`sample_webgpu/main.c`) is new too, and I've been iterating on it.

Sanitizers can be useful for these sorts of issues, so I'd like to use them if possible: https://emscripten.org/docs/debugging/Sanitizers.html

### ASan

`heap-buffer-overflow` at `iree_hal_create_transfer_command_buffer` (same location as the Emscripten assert):

<details><summary>ASan logs</summary>

```
(C) =================================================================
iree_api_webgpu.js:76 (C) ==42==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x14c00ec4 at pc 0x001cf7b8 bp 0x12e66510 sp 0x12e6651c
iree_api_webgpu.js:76 (C) READ of size 4 at 0x14c00ec4 thread T0
iree_api_webgpu.js:76 (C)     #0 0x1cf7b8 in __asan_report_load4+0x1cf7b8 (http://localhost:8000/web-sample-webgpu.wasm+0x1cf7b8)
iree_api_webgpu.js:76 (C)
iree_api_webgpu.js:76 (C) 0x14c00ec4 is located 4 bytes to the right of 48-byte region [0x14c00e90,0x14c00ec0)
iree_api_webgpu.js:76 (C) allocated by thread T0 here:
iree_api_webgpu.js:76 (C)     #0 0x1f6e1e in __sanitizer::StackTrace::GetCurrentPc()+0x1f6e1e (http://localhost:8000/web-sample-webgpu.wasm+0x1f6e1e)
iree_api_webgpu.js:76 (C)     #1 0x1c77df in calloc+0x1c77df (http://localhost:8000/web-sample-webgpu.wasm+0x1c77df)
iree_api_webgpu.js:76 (C)     #2 0x16f349 in iree_allocator_system_ctl+0x16f349 (http://localhost:8000/web-sample-webgpu.wasm+0x16f349)
iree_api_webgpu.js:76 (C)     #3 0x16e208 in iree_allocator_malloc+0x16e208 (http://localhost:8000/web-sample-webgpu.wasm+0x16e208)
iree_api_webgpu.js:76 (C)     #4 0x141b53 in iree_hal_buffer_subspan+0x141b53 (http://localhost:8000/web-sample-webgpu.wasm+0x141b53)
iree_api_webgpu.js:76 (C)     #5 0x37729 in iree_hal_module_buffer_view_create+0x37729 (http://localhost:8000/web-sample-webgpu.wasm+0x37729)
iree_api_webgpu.js:76 (C)     #6 0x1218c1 in iree_vm_shim_rIIiiCID_r+0x1218c1 (http://localhost:8000/web-sample-webgpu.wasm+0x1218c1)
iree_api_webgpu.js:76 (C)
iree_api_webgpu.js:76 (C) SUMMARY: AddressSanitizer: heap-buffer-overflow (http://localhost:8000/web-sample-webgpu.wasm+0x1cf7b7) in __asan_report_load4+0x1cf7b7
iree_api_webgpu.js:76 (C) Shadow bytes around the buggy address:
iree_api_webgpu.js:76 (C)   0x02980180: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
iree_api_webgpu.js:76 (C)   0x02980190: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
iree_api_webgpu.js:76 (C)   0x029801a0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
iree_api_webgpu.js:76 (C)   0x029801b0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
iree_api_webgpu.js:76 (C)   0x029801c0: fa fa fa fa fa fa fa fa fa fa 00 00 00 00 00 00
iree_api_webgpu.js:76 (C) =>0x029801d0: fa fa 00 00 00 00 00 00[fa]fa 00 00 00 00 04 fa
iree_api_webgpu.js:76 (C)   0x029801e0: fa fa 00 00 00 00 04 fa fa fa 00 00 00 00 00 04
iree_api_webgpu.js:76 (C)   0x029801f0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
iree_api_webgpu.js:76 (C)   0x02980200: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
iree_api_webgpu.js:76 (C)   0x02980210: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
iree_api_webgpu.js:76 (C)   0x02980220: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
iree_api_webgpu.js:76 (C) Shadow byte legend (one shadow byte represents 8 application bytes):
iree_api_webgpu.js:76 (C)   Addressable:           00
iree_api_webgpu.js:76 (C)   Partially addressable: 01 02 03 04 05 06 07
iree_api_webgpu.js:76 (C)   Heap left redzone:       fa
iree_api_webgpu.js:76 (C)   Freed heap region:       fd
iree_api_webgpu.js:76 (C)   Stack left redzone:      f1
iree_api_webgpu.js:76 (C)   Stack mid redzone:       f2
iree_api_webgpu.js:76 (C)   Stack right redzone:     f3
iree_api_webgpu.js:76 (C)   Stack after return:      f5
iree_api_webgpu.js:76 (C)   Stack use after scope:   f8
iree_api_webgpu.js:76 (C)   Global redzone:          f9
iree_api_webgpu.js:76 (C)   Global init order:       f6
iree_api_webgpu.js:76 (C)   Poisoned by user:        f7
iree_api_webgpu.js:76 (C)   Container overflow:      fc
iree_api_webgpu.js:76 (C)   Array cookie:            ac
iree_api_webgpu.js:76 (C)   Intra object redzone:    bb
iree_api_webgpu.js:76 (C)   ASan internal:           fe
iree_api_webgpu.js:76 (C)   Left alloca redzone:     ca
iree_api_webgpu.js:76 (C)   Right alloca redzone:    cb
iree_api_webgpu.js:76 (C) ==42==ABORTING
?function=multiple_r…lts_webgpu.vmfb:315 Function call error: '[object Object]'
```
</details>

### UBSan and `-sSAFE_HEAP`

```
Aborted(alignment fault)
    at SAFE_HEAP_LOAD_i32_2_2 (web-sample-webgpu.wasm:0x196919)
    at iree_vm_bytecode_function_verify_bytecode_op (verifier.c:1141)
```

<details><summary>More logs</summary>

```
iree_api_webgpu.js:76 (C) D:/dev/projects/iree/runtime/src/iree/hal/drivers/webgpu/webgpu_device.c:136:34: runtime error: member access within misaligned address 0x005a24e8 for type 'iree_hal_webgpu_device_t' (aka 'struct iree_hal_webgpu_device_t'), which requires 16 byte alignment
iree_api_webgpu.js:76 (C) 0x005a24e8: note: pointer points here
iree_api_webgpu.js:76 (C)  4b 12 02 00  01 00 00 00 f4 7f 02 00  00 00 00 00 05 00 00 00  18 37 5c 00 11 00 00 00  00 00 00 00
iree_api_webgpu.js:76 (C)               ^
iree_api_webgpu.js:76 (C) D:/dev/projects/iree/runtime/src/iree/hal/drivers/webgpu/webgpu_device.c:136:59: runtime error: member access within misaligned address 0x005a24e8 for type 'iree_hal_webgpu_device_t' (aka 'struct iree_hal_webgpu_device_t'), which requires 16 byte alignment
iree_api_webgpu.js:76 (C) 0x005a24e8: note: pointer points here
iree_api_webgpu.js:76 (C)  4b 12 02 00  01 00 00 00 f4 7f 02 00  00 00 00 00 05 00 00 00  18 37 5c 00 11 00 00 00  00 00 00 00
iree_api_webgpu.js:76 (C)               ^
?function=multiple_r…lts_webgpu.vmfb:187 IREE initialized, ready to load programs.
iree_api_webgpu.js:76 (C) D:/dev/projects/iree/runtime/src/iree/hal/drivers/webgpu/webgpu_device.c:270:15: runtime error: member access within misaligned address 0x005a24e8 for type 'iree_hal_webgpu_device_t' (aka 'struct iree_hal_webgpu_device_t'), which requires 16 byte alignment
iree_api_webgpu.js:76 (C) 0x005a24e8: note: pointer points here
iree_api_webgpu.js:76 (C)  4b 12 02 00  03 00 00 00 f4 7f 02 00  00 00 00 00 05 00 00 00  18 37 5c 00 11 00 00 00  00 00 00 00
iree_api_webgpu.js:76 (C)               ^
iree_api_webgpu.js:76 (C) D:/dev/projects/iree/runtime/src/iree/hal/drivers/webgpu/webgpu_device.c:270:49: runtime error: member access within misaligned address 0x005a24e8 for type 'iree_hal_webgpu_device_t' (aka 'struct iree_hal_webgpu_device_t'), which requires 16 byte alignment
iree_api_webgpu.js:76 (C) 0x005a24e8: note: pointer points here
iree_api_webgpu.js:76 (C)  4b 12 02 00  03 00 00 00 f4 7f 02 00  00 00 00 00 05 00 00 00  18 37 5c 00 11 00 00 00  00 00 00 00
iree_api_webgpu.js:76 (C)               ^
iree_api_webgpu.js:76 (C) D:/dev/projects/iree/runtime/src/iree/vm/bytecode/verifier.c:1127:5: runtime error: load of misaligned address 0x005c4472 for type 'const uint32_t' (aka 'const unsigned int'), which requires 4 byte alignment
iree_api_webgpu.js:76 (C) 0x005c4472: note: pointer points here
iree_api_webgpu.js:76 (C)  00 00  79 0d 02 00 00 00 00 00  10 02 80 0d 0d 00 00 00  01 00 0d 1c 00 00 00 02  00 0d 03 00 00 00
iree_api_webgpu.js:76 (C)               ^
iree_api_webgpu.js:76 (C) D:/dev/projects/iree/runtime/src/iree/vm/bytecode/verifier.c:1141:5: runtime error: load of misaligned address 0x005c4479 for type 'const uint16_t' (aka 'const unsigned short'), which requires 2 byte alignment
iree_api_webgpu.js:76 (C) 0x005c4479: note: pointer points here
iree_api_webgpu.js:76 (C)  00 00 00  10 02 80 0d 0d 00 00 00  01 00 0d 1c 00 00 00 02  00 0d 03 00 00 00 03 00  0d 11 00 00 00
iree_api_webgpu.js:76 (C)               ^
iree_api_webgpu.js:76 (C) Aborted(alignment fault)
web-sample-webgpu.js:750 Uncaught RuntimeError: Aborted(alignment fault)
    at abort (web-sample-webgpu.js:750:10)
    at alignfault (web-sample-webgpu.js:429:2)
    at imports.<computed> (web-sample-webgpu.js:5243:24)
    at SAFE_HEAP_LOAD_i32_2_2 (web-sample-webgpu.wasm:0x196919)
    at iree_vm_bytecode_function_verify_bytecode_op (verifier.c:1141)
    at __flatbuffers_uoffset_read_from_pe (verifier.c:383)
    at iree_vm_RodataSegmentDef_vec_len (bytecode_module_def_reader.h:547)
    at iree_vm_bytecode_function_verify (verifier.c:323)
    at __flatbuffers_uoffset_read_from_pe (module.c:923)
    at iree_vm_FunctionDescriptor_vec_len (bytecode_module_def_reader.h:414)
    at iree_vm_bytecode_module_create (module.c:845)
    at iree_allocator_system (main.c:135)
    at load_program (main.c:119)
    at ret.<computed> (web-sample-webgpu.js:5265:24)
    at web-sample-webgpu.js:775:20
```
</details>

### Fixes attempted

```c
// base/target_platform.h
#if defined(__EMSCRIPTEN__)
#define IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED 1
#endif

// Still seeing this error:
//   iree_api_webgpu.js:76 (C) D:/dev/projects/iree/runtime/src/iree/vm/stack.c:584:17:
//   runtime error: store to misaligned address 0x005d00ac for type 'iree_vm_source_offset_t'
//   (aka 'long long'), which requires 8 byte alignment
```

```c
// base/config.h
#define IREE_VM_BYTECODE_VERIFICATION_ENABLE 0

// Still seeing this error:
//   iree_api_webgpu.js:76 (C) D:/dev/projects/iree/runtime/src/iree/vm/bytecode/dispatch.c:864:5:
//   runtime error: load of misaligned address 0x005a848d for type 'const uint16_t'
//   (aka 'const unsigned short'), which requires 2 byte alignment
```

## Trigger

A WebGPU function returns multiple buffer views and the sample constructs one batched device-to-host transfer for their subspan buffers.

## Relevant entities

HAL buffer view; HAL subspan buffer; allocated WebGPU buffer; subspan byte offset; WGPUBuffer handle; transfer command

## Implemented relation

A logical HAL subspan is bound to the native handle of its allocated WebGPU backing buffer together with the subspan's byte offset.

## Pre-state

Each output buffer view refers to a HAL buffer that may be a subspan of a concrete WebGPU allocation.

## Binding transition

The readback path casts the view's subspan object directly to iree_hal_webgpu_buffer_t and also supplies that subspan as the transfer source.

## Post-state

WebGPU handle lookup reads fields from the subspan object's layout as though it were the allocated-buffer layout, and the transfer loses the separation between backing-buffer identity and subspan offset.

## Validity obligation

Native-handle access and transfer submission use the allocated buffer object, while logical range selection preserves the subspan byte offset and length.

## Invalid decision

iree_hal_webgpu_buffer_handle and process_call_outputs used the logical subspan object where the concrete allocated buffer was required.

## Physical effect

The runtime obtains an invalid WGPUBuffer handle and submits the output copy against an object whose concrete representation does not match the WebGPU buffer cast.

## Observed consequence

The multi-output program reaches a WebGPU handle assertion; AddressSanitizer reports a four-byte out-of-bounds read during transfer-command creation.

## Repair site

experimental/webgpu/buffer.c::iree_hal_webgpu_buffer_handle and experimental/web/sample_webgpu/main.c::process_call_outputs

## Repair

PR 14163 resolves a HAL buffer through iree_hal_buffer_allocated_buffer before casting it or placing it in a transfer command, retains the original subspan offset, resets the mapped readback offset to zero, and propagates asynchronous mapping errors.

## Why the repair works

Handle lookup and command submission now use the concrete WebGPU allocation, while the saved offset selects the same logical bytes within that allocation.

## Causal basis

The report identifies incorrect output buffer handles at transfer construction; the linked merged patch states that handle access bypassed iree_hal_buffer_allocated_buffer and changes both handle lookup and the failing transfer source accordingly.

## Analysis implication

Buffer tracing needs logical view identity, subspan identity, allocated-buffer identity, byte offset, byte length, native handle, and transfer endpoint together.

## Fix patches

- [`fixes/iree-13809-pr14163.patch`](fixes/iree-13809-pr14163.patch) — https://github.com/iree-org/iree/pull/14163
