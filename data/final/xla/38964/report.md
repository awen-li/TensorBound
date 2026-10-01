# `bitcast_convert` of packed constant `U8` (uint8) to "packed" types (`S4`, `S2`, `U4`, `U2`) fails

- Repository: `openxla/xla`
- Report: [https://github.com/openxla/xla/issues/38964](https://github.com/openxla/xla/issues/38964)
- Opened: 2026-03-10T09:03:19Z
- Closed: 2026-08-13T20:05:19Z
- Labels: `bug`, `stat:awaiting openxla-eng`

## Reported behavior

This following minimal `StableHLO` code exemplifies it:

- The `U8` value `0xE1` if fed as an input parameter and as a constant --> `0x1` nibble is 1 in `S4`, and `0xE` becomes -2 in `S4`.
- Both values (presumably the same) are `%stablehlo.bitcast_convert()`'ed to `Int4`
- And then converted to `Int8` for outputting/printing.

But the outputs are different!?

```
module @TestSubByteDTypes_Int4_AsLiteral {
  func.func @main(%x: tensor<1xui8>) -> (tensor<1x2xi8>, tensor<1x2xi8>, tensor<1xi1>) {
    %0 = "stablehlo.constant"() { value = dense<[225]> : tensor<1xui8> } : () -> tensor<1xui8>
    %1 = "stablehlo.compare"(%0, %x) {
      compare_type = #stablehlo<comparison_type UNSIGNED>,
      comparison_direction = #stablehlo<comparison_direction EQ>
    } : (tensor<1xui8>, tensor<1xui8>) -> tensor<1xi1>
    %2 = "stablehlo.bitcast_convert"(%0) : (tensor<1xui8>) -> tensor<1x2xi4>
    %3 = "stablehlo.convert"(%2) : (tensor<1x2xi4>) -> tensor<1x2xi8>
    %4 = "stablehlo.bitcast_convert"(%x) : (tensor<1xui8>) -> tensor<1x2xi4>
    %5 = "stablehlo.convert"(%4) : (tensor<1x2xi4>) -> tensor<1x2xi8>
    "stablehlo.return"(%3, %5, %1) : (tensor<1x2xi8>, tensor<1x2xi8>, tensor<1xi1>) -> ()
  }
}
```

If one feeds this function the same value as the constant `0xE1` (225 in decimal) one would expect them to be bitcast to `Int4` values [1, -2]. But the outputs of the `StableHLO` program above are:

1. `[[1, 0]]`  --> Fed as a constant, it ignores the upper nibble (upper 4 bits).
2. `[[1, -2]]` --> Same value fed as an input parameter, makes the correct `stablehlo.bitcast_convert()`.
3. `[true]` --> A (in-)sanity check: the input parameter and the constant, both set to `[0xE1]`, compare to true, even though they `bitcast_convert` to different values.

ps.: Notice PJRT `PJRT_Client_BufferFromHostBuffer()` method doesn't seem to accept buffers with packed `S4` as inputs, so it requires feeding them as `U8` and then `%stablehlo.bitcast_convert` them to `S4`.

## Trigger

Constant folding bitcast-converts U8 literal 0xE1 to two S4 values.

## Relevant entities

unpacked host Literal; packed device representation; source/destination bit width; constant folder; parameter path

## Implemented relation

Bitcast-convert operates on the packed bitstream defined by tensor element widths, independent of whether the value is a literal or runtime parameter.

## Pre-state

The constant and parameter contain the same eight bits 11100001.

## Binding transition

LiteralBase::BitcastConvert checks layout-dependent byte sizes and memcpy-copies unpacked one-byte-per-subbyte-element storage.

## Post-state

Only the lower nibble is interpreted correctly; the upper nibble becomes zero or valid conversions are rejected.

## Validity obligation

Literal evaluation packs and unpacks sub-byte elements according to their storage bit width before bit reinterpretation.

## Invalid decision

LiteralBase assumed its host buffer already used the packed device representation.

## Physical effect

Constant folding yields [1,0] while the device parameter path yields [1,-2].

## Observed consequence

Equivalent constant and input values produce different StableHLO results.

## Repair site

xla/literal.cc::LiteralBase::BitcastConvert

## Repair

PR 47178 replaces layout-dependent byte-size checks with bit-accurate element-width checks and converts sub-byte literals through PackIntN/UnpackIntN; it adds conversions in both directions and boundary regressions.

## Why the repair works

Both literal and device paths now reinterpret the same packed bitstream.

## Causal basis

The patch names #38964, explains the constant/parameter divergence, and tests the reported U8-to-S4 bytes plus reverse and boundary cases.

## Analysis implication

A constant-folding check needs logical element count, storage bit width, host literal encoding, device packing, layout metadata, and operation bit semantics.

## Fix patches

- [`fixes/xla-38964-pr47178.patch`](fixes/xla-38964-pr47178.patch) — https://github.com/openxla/xla/pull/47178
