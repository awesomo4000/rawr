<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 00-01: Serialization I/O API Migration

## Goal

Make the known first Zig 0.16.0 failure compile by migrating
serialization/deserialization away from removed `std.io.fixedBufferStream`.

This is intentionally narrow: it should unblock the next compiler errors
without mixing in unrelated library changes.

## Scope

Primary files:

- `src/serialize.zig`
- Any closely related wrapper methods in `src/bitmap.zig`, only if required

## Implementation Notes

Replace fixed buffer stream usage with Zig 0.16.0 fixed I/O interfaces:

```zig
var writer = std.Io.Writer.fixed(buf);
try serializeToWriter(bm, &writer);

var reader = std.Io.Reader.fixed(data);
return deserializeFromReader(allocator, &reader, data.len);
```

Update reader call sites:

- `reader.readInt(T, endian)` -> `reader.takeInt(T, endian)`
- `reader.readAll(buf)` -> `reader.readSliceAll(buf)` and handle `error.EndOfStream` as invalid format where needed
- `reader.skipBytes(n, .{})` -> `reader.discardAll(n)`

Keep `serializeToWriter` and `deserializeFromReader` generic enough for callers
that pass a compatible writer/reader. If pointer receivers are required by Zig
0.16.0 APIs, adjust calls consistently.

Do not change the wire format. The serialized byte output must remain
RoaringFormatSpec-compatible.

## Validation

```bash
zig build test
```

It is acceptable if this command reveals later, unrelated Zig 0.16.0 failures.
The chunk is complete when the serialization fixed-buffer API failure is gone.

## Checklist

- [x] Replace `std.io.fixedBufferStream` in serialization
- [x] Update reader method names for Zig 0.16.0
- [x] Keep `serializeToWriter` and `deserializeFromReader` usable by wrappers
- [x] `zig build test` progresses past serialization I/O failures
- [x] Serialization round-trip tests still pass once the library compiles
