<!-- SPDX-License-Identifier: MPL-2.0 -->

# Cross-width serialization fixtures

`tools/cross_width_fixture.zig` produces and verifies deterministic portable-serialization fixtures
for `RoaringBitmap` and `Roaring64Bitmap`. The same file can be produced on one pointer width and
verified on another. Verification checks the corpus hash, set equality after safe deserialization,
and byte-identical reserialization.

The corpus uses fixed-width integer generation and a checked-in FNV-1a hash. It includes empty and
single-container bitmaps, a mixed bitmap with array, bitset, and run containers across chunk
boundaries, and 64-bit bitmaps spanning multiple high-32-bit buckets.

## File format

All integers are little-endian. The file begins with:

| Field | Type | Value |
| --- | --- | --- |
| magic | 8 bytes | `RAWRXW01` |
| format version | `u32` | `1` |
| corpus hash | `u64` | checked-in by the tool |
| case count | `u32` | `6` |

Each case then contains a `u8` case identifier, a `u8` 64-bit-bitmap flag, a reserved zero `u16`, a
`u64` payload length, and the portable serialized payload. Cases have a fixed order enforced by the
consumer.

## Commands

Build a fixture tool for the selected target:

```sh
zig build cross-width-fixture
zig-out/bin/cross_width_fixture produce fixture.bin
zig-out/bin/cross_width_fixture verify fixture.bin
```

The 64-bit host round trip used by spec 40-00 is:

```sh
zig build check-cross-width-64
```

For cross-width exchange, produce on one target, transfer the file without modification, and verify
it with the other target. Repeat in the opposite direction. A corpus-hash mismatch indicates corpus
generation drift; a set or byte mismatch indicates serialization incompatibility.
