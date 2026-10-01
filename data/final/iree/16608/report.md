# ubsan load/store misaligned address errors from memory_stream_test and gguf_parser_test

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/16608](https://github.com/iree-org/iree/issues/16608)
- Opened: 2024-02-28T20:54:42Z
- Closed: Open at collection
- Labels: `bug 🐞`

## Reported behavior

### What happened?

Was running asan tests memory_stream_test and gguf_parser_test downstream, and got illegal instructions errors.  Turned on ubsan and got more detailed errors.

Then tried it with a copy of github/openxla/iree @ head, compiled with ubsan.  ctest don't fail but the LastLog shows the runtime error messages.  Also see it in raw test output when test binaries are run directly.

[ RUN      ] MemoryStreamTest.Fill
/opt/work/iree/github-head/openxla/iree/runtime/src/iree/io/memory_stream.c:267:9: runtime error: store to misaligned address 0x7ffdba454351 for type 'uint16_t' (aka 'unsigned short'), which requires 2 byte alignment
0x7ffdba454351: note: pointer points here
 56 00 00  dd dd dd dd dd dd dd dd  dd dd dd dd dd dd dd dd  50 81 45 8d 54 56 00 00  a8 53 45 ba fd
              ^
SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior /opt/work/iree/github-head/openxla/iree/runtime/src/iree/io/memory_stream.c:267:9 in

[ RUN      ] GgufFormatTest.SingleTensor
/opt/work/iree/github-head/openxla/iree/runtime/src/iree/io/formats/gguf/gguf_parser.c:417:22: runtime error: load of misaligned address 0x558d6855052e for type 'const uint64_t' (aka 'const unsigned long'), which requires 8 byte alignment
0x558d6855052e: note: pointer points here
 02 00 00 00 02 00  00 00 00 00 00 00 02 00  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  00 00
             ^
SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior /opt/work/iree/github-head/openxla/iree/runtime/src/iree/io/formats/gguf/gguf_parser.c:417:22 in

More detailed output at memory_stream_test: https://gist.github.com/yuennancy/e0ad695cc2d5097ccbd4a16872d13382
and gguf_parser_test: https://gist.github.com/yuennancy/77d268f08721b5dc72fe3bce8f7525ae

### Steps to reproduce your issue

1. Configure cmake with ubsan:
```cmake -G Ninja -B ../iree-build/ -S .     -DCMAKE_BUILD_TYPE=Debug     -DIREE_ENABLE_ASSERTIONS=ON     -DIREE_ENABLE_SPLIT_DWARF=ON     -DIREE_ENABLE_THIN_ARCHIVES=ON     -DCMAKE_C_COMPILER=clang     -DCMAKE_CXX_COMPILER=clang++     -DIREE_ENABLE_LLD=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DIREE_ENABLE_UBSAN=ON```

2. Build memory_stream_test and gguf_parser_test:
```cmake --build ../iree-build/ --target runtime/src/iree/io/memory_stream_test --target runtime/src/iree/io/formats/gguf/gguf_parser_test```

3. Run ctest and view Lastlog output.  ctest passes, but errors will show up in Lastlog:
```ctest --test-dir ../iree-build/  -R memory_stream_test; cat ../iree-build/Testing/Temporary/LastTest.log```

```ctest --test-dir ../iree-build/  -R gguf_parser_test; cat ../iree-build/Testing/Temporary/LastTest.log```

4. Run tests directly.  Tests pass, but errors are in the test output:
```../iree-build/runtime/src/iree/io/memory_stream_test```

```../iree-build/runtime/src/iree/io/gguf_parser_test```


### What component(s) does this issue relate to?

Runtime

### Version information

Checked out IREE @ head

### Additional context

_No response_

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

The report's reproducer exercises iREE runtime unaligned memory accesses in memory_stream fill and GGUF parsing.

## Observed implementation decision

The runtime performs typed loads/stores at addresses that do not satisfy the C/C++ alignment requirement.

## Physical effect

UBSan observes misaligned uint16 and uint64 accesses; architecture-dependent execution may trap or be optimized unpredictably.

## Observed consequence

Sanitizer diagnostics appear during the two runtime tests.

## Reported proposal or workaround

Discussion suggests alignment-required access helpers but does not resolve the GGUF path.

## Repair trace

The issue remains open and contains no accepted source correction or fixed sanitizer regression.

## Archived evidence

- Raw report: `data/raw/github/iree/16608/report.json`
- SHA-256: `c7f0979de05459db2288bc631e40014c82d0611635d2b46c3bb209dfbf160a46`
