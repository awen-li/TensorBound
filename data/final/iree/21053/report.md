# Wrong result due to resource reuse in stream dialect

- Repository: `iree-org/iree`
- Report: [https://github.com/iree-org/iree/issues/21053](https://github.com/iree-org/iree/issues/21053)
- Opened: 2025-06-09T16:41:51Z
- Closed: Open at collection
- Labels: `bug 🐞`

## Reported behavior

### What happened?

During the compilation of a bidirectional LSTM model, the compiler generates two independent loops (one for each direction) that reuse a resource, which results in the computation of the first loop being overwritten with the result of the second loop. I have found that `stream` is the first phase where I can "fix" the issue by editing the intermediate MLIR. Below is the generated code for each phase.

[LstmTestCat.1.input.mlir.txt](https://github.com/user-attachments/files/20657759/LstmTestCat.1.input.mlir.txt)
[LstmTestCat.2.abi.mlir.txt](https://github.com/user-attachments/files/20657767/LstmTestCat.2.abi.mlir.txt)
[LstmTestCat.3.preprocessing.mlir.txt](https://github.com/user-attachments/files/20657762/LstmTestCat.3.preprocessing.mlir.txt)
[LstmTestCat.4.global-optimization.mlir.txt](https://github.com/user-attachments/files/20657765/LstmTestCat.4.global-optimization.mlir.txt)
[LstmTestCat.5.dispatch-creation.mlir.txt](https://github.com/user-attachments/files/20657763/LstmTestCat.5.dispatch-creation.mlir.txt)
[LstmTestCat.6.flow.mlir.txt](https://github.com/user-attachments/files/20657764/LstmTestCat.6.flow.mlir.txt)
[LstmTestCat.7.stream.mlir.txt](https://github.com/user-attachments/files/20657760/LstmTestCat.7.stream.mlir.txt)
[LstmTestCat.8.executable-sources.mlir.txt](https://github.com/user-attachments/files/20657768/LstmTestCat.8.executable-sources.mlir.txt)
[LstmTestCat.9.executable-configurations.mlir.txt](https://github.com/user-attachments/files/20657766/LstmTestCat.9.executable-configurations.mlir.txt)
[LstmTestCat.10.executable-targets.mlir.txt](https://github.com/user-attachments/files/20657770/LstmTestCat.10.executable-targets.mlir.txt)
[LstmTestCat.11.hal.mlir.txt](https://github.com/user-attachments/files/20657761/LstmTestCat.11.hal.mlir.txt)
[LstmTestCat.12.vm.mlir.txt](https://github.com/user-attachments/files/20657769/LstmTestCat.12.vm.mlir.txt)

And below is the edited `stream` phase that produces the correct result:

[LstmTestCat.7.stream_fix.mlir.txt](https://github.com/user-attachments/files/20657813/LstmTestCat.7.stream_fix.mlir.txt)

The change affects the following section in the original generated code:

```mlir
  // ...
  %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<transient>{%c128} => !stream.timepoint
  %2 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepoint) => with(%result as %arg3: !stream.resource<transient>{%c128}) {
    stream.cmd.concurrent {
      stream.cmd.fill %c0_i8, %arg3[%c0 for %c24] : i8 -> !stream.resource<transient>{%c128}
      stream.cmd.fill %c0_i8, %arg3[%c64 for %c8] : i8 -> !stream.resource<transient>{%c128}
    }
  } => !stream.timepoint
  %3 = stream.timepoint.await %2 => %result : !stream.resource<transient>{%c128}
  %4 = stream.resource.subview %3[%c0] : !stream.resource<transient>{%c128} -> !stream.resource<transient>{%c24}
  %5 = stream.resource.subview %3[%c64] : !stream.resource<transient>{%c128} -> !stream.resource<transient>{%c8}
  cf.br ^bb1(%c0, %4, %5, %5 : index, !stream.resource<transient>, !stream.resource<transient>, !stream.resource<transient>)
^bb1(%6: index, %7: !stream.resource<transient>, %8: !stream.resource<transient>, %9: !stream.resource<transient>):  // 2 preds: ^bb0, ^bb2
  %10 = arith.cmpi slt, %6, %c3 : index
  cf.cond_br %10, ^bb2, ^bb3(%c0, %4, %5, %5 : index, !stream.resource<transient>, !stream.resource<transient>, !stream.resource<transient>)
^bb2:  // pred: ^bb1
  // ...
```

Here `^bb1` and `^bb2` are the predicate and body of the first loop, while `^bb3` is the predicate of the second loop (the loops run one after the other and do more or less the same computation). The fixed version changes as follows:

```mlir
  // ...
  %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<transient>{%c128} => !stream.timepoint
  %2 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepoint) => with(%result as %arg3: !stream.resource<transient>{%c128}) {
    stream.cmd.concurrent {
      stream.cmd.fill %c0_i8, %arg3[%c0 for %c24] : i8 -> !stream.resource<transient>{%c128}
      stream.cmd.fill %c0_i8, %arg3[%c64 for %c8] : i8 -> !stream.resource<transient>{%c128}
    }
  } => !stream.timepoint
  %3 = stream.timepoint.await %2 => %result : !stream.resource<transient>{%c128}
  %4 = stream.resource.subview %3[%c0] : !stream.resource<transient>{%c128} -> !stream.resource<transient>{%c24}
  %5 = stream.resource.subview %3[%c64] : !stream.resource<transient>{%c128} -> !stream.resource<transient>{%c8}
  // Make new resources for second loop
  %result000, %result000_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<transient>{%c128} => !stream.timepoint
  %2000 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result000_timepoint) => with(%result000 as %arg3: !stream.resource<transient>{%c128}) {
    stream.cmd.concurrent {
      stream.cmd.fill %c0_i8, %arg3[%c0 for %c24] : i8 -> !stream.resource<transient>{%c128}
      stream.cmd.fill %c0_i8, %arg3[%c64 for %c8] : i8 -> !stream.resource<transient>{%c128}
    }
  } => !stream.timepoint
  %3000 = stream.timepoint.await %2000 => %result000 : !stream.resource<transient>{%c128}
  %4000 = stream.resource.subview %3000[%c0] : !stream.resource<transient>{%c128} -> !stream.resource<transient>{%c24}
  %5000 = stream.resource.subview %3000[%c64] : !stream.resource<transient>{%c128} -> !stream.resource<transient>{%c8}
  cf.br ^bb1(%c0, %4, %5, %5 : index, !stream.resource<transient>, !stream.resource<transient>, !stream.resource<transient>)
^bb1(%6: index, %7: !stream.resource<transient>, %8: !stream.resource<transient>, %9: !stream.resource<transient>):  // 2 preds: ^bb0, ^bb2
  %10 = arith.cmpi slt, %6, %c3 : index
  cf.cond_br %10, ^bb2, ^bb3(%c0, %4000, %5000, %5000 : index, !stream.resource<transient>, !stream.resource<transient>, !stream.resource<transient>)
^bb2:  // pred: ^bb1
  // ...
```

Essentially it duplicates the resource creation and passes the new resources to the second loop (`^bb3`), instead of reusing the same as the first loop.

I attempted a similar fix on `flow` and earlier phases (namely avoiding reuse) but it did not have any effect (though I may have done something wrong as my knowledge about IREE is still very limited).

### Steps to reproduce your issue

1. Compile the given module with the following command:
  ```sh
   > iree-compile LstmTestCat.7.stream.mlir.txt --compile-from=stream -o .\LstmTestCat.vmfb
   ```
2. Run the module with some example input as follows:
  ```sh
  iree-run-module.exe --module=LstmTestCat.vmfb --function=main_graph --input="1x3x2xf32=[[[0. 1.][2. 3.][4. 5.]]]" --output=-
  ```
3. The first two columns in the result are equal to the last two columns but vertically flipped. See the formatted output below:
  ```none
  [[0.635237 0.635237 0.655356 0.655356]
   [0.762444 0.762444 0.762444 0.762444]
   [0.655356 0.655356 0.635237 0.635237]]
  ```
4. Repeat the process with `LstmTestCat.7.stream_fix.mlir.txt`. The result (below) is now correct:
  ```none
  [[0.0952412 0.0952412 0.655356 0.655356]
   [0.3247900 0.3247900 0.762444 0.762444]
   [0.5894990 0.5894990 0.635237 0.635237]]
  ```

### What component(s) does this issue relate to?

Compiler

### Version information

Git commit b65764401ad52b8787a22da92c215ea100082397

### Additional context

The generated code corresponds to a minimal bidirectional LSTM example. In practice, the result for the forward direction is lost, an instead replaced by the result for the backwards direction flipped along the "time" dimension. This seems to be because the backwards result is computed in reverse and then flipped, and the intermediate non-flipped result is written to the same location as the result for the forward direction.

## Review outcome

This report is a confirmed TensorBound cohort case. The archived public
artifacts establish the reported binding failure and its observable effect.
They do not identify a traceable landed repair for this report.

## Trigger

Wrong result due to resource reuse in stream dialect

## Observed implementation decision

Stream resource planning passes the first loop's resource subviews into the second loop instead of allocating an independent resource for the second live value.

## Physical effect

Two sequential bidirectional-LSTM loops are assigned subviews of the same transient resource, allowing the backward computation to overwrite the forward result before both outputs are consumed.

## Observed consequence

The forward-direction columns are replaced by the reversed backward result, yielding a reproducible incorrect tensor.

## Reported proposal or workaround

The report provides full phase IR and a controlled stream-level edit that duplicates the resource; recompiling that edit restores the expected output.

## Repair trace

A controlled IR edit restores output, but no repository repair, maintainer adjudication, or fixed regression connects that edit to an implemented correction.

## Archived evidence

- Raw report: `data/raw/github/iree/21053/report.json`
- SHA-256: `8122e84da16ac96103d95005fc71d80f2519d3a3768ed74f930ca8ce0d77aaab`
