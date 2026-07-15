<!-- SPDX-License-Identifier: MPL-2.0 -->

# Optimization: Bulk I/O for Serialize/Deserialize

## Problem

Serialize is 2.49x slower than CRoaring. Deserialize is 3.43x slower. The cause
is per-element `writeInt` / `readInt` calls through `fixedBufferStream`. For a
bitset container, that's 1024 individual `writeInt(u64, ...)` calls. For a 4000-
element array container, 4000 `writeInt(u16, ...)` calls.

CRoaring does a single `memcpy` per container. On little-endian machines (ARM,
x86), the in-memory byte order already matches the wire format, so bulk copy is
correct without byte-swapping.

## Fix

Replace per-element read/write loops with bulk `writeAll` / `readAll` using
`std.mem.sliceAsBytes`. Add a comptime endianness assert since the serialization
format is little-endian.

### File: `src/serialize.zig`

#### Add endianness guard (top of file)

```zig
comptime {
    if (@import("builtin").cpu.arch.endian() != .little) {
        @compileError("rawr serialization assumes little-endian byte order");
    }
}
```

#### serializeToWriter — container data section

Replace the per-element loops (currently around lines 141-165):

```zig
// BEFORE:
.array => |ac| {
    for (ac.values[0..ac.cardinality]) |v| {
        try writer.writeInt(u16, v, .little);
    }
},
.bitset => |bc| {
    for (bc.words) |word| {
        try writer.writeInt(u64, word, .little);
    }
},
.run => |rc| {
    try writer.writeInt(u16, rc.n_runs, .little);
    for (rc.runs[0..rc.n_runs]) |run| {
        try writer.writeInt(u16, run.start, .little);
        try writer.writeInt(u16, run.length, .little);
    }
},
```

```zig
// AFTER:
.array => |ac| {
    try writer.writeAll(std.mem.sliceAsBytes(ac.values[0..ac.cardinality]));
},
.bitset => |bc| {
    try writer.writeAll(std.mem.sliceAsBytes(bc.words));
},
.run => |rc| {
    try writer.writeInt(u16, rc.n_runs, .little);
    try writer.writeAll(std.mem.sliceAsBytes(rc.runs[0..rc.n_runs]));
},
```

`RunPair` is `packed struct { start: u16, length: u16 }` — 4 bytes, no padding,
so `sliceAsBytes` produces the exact wire format.

#### Also bulk-write the descriptive header

The descriptive header loop (currently around lines 105-110) writes key+card
pairs one at a time. This can't be bulk-written as easily since keys and
cardinalities are in separate arrays (`bm.keys` and `bm.containers`). Leave
this as-is — it's only `size * 2` writes, not a hot path.

#### deserializeFromReader — container data section

Replace the per-element read loops (currently around lines 233-276):

```zig
// BEFORE (bitset path):
for (0..BitsetContainer.NUM_WORDS) |w| {
    bc.words[w] = try reader.readInt(u64, .little);
}

// AFTER:
const bytes_read = try reader.readAll(std.mem.sliceAsBytes(bc.words));
if (bytes_read != BitsetContainer.SIZE_BYTES) return error.InvalidFormat;
```

```zig
// BEFORE (array path):
for (0..card) |v| {
    ac.values[v] = try reader.readInt(u16, .little);
}

// AFTER:
const bytes_needed = card * 2;
const bytes_read = try reader.readAll(std.mem.sliceAsBytes(ac.values[0..card]));
if (bytes_read != bytes_needed) return error.InvalidFormat;
```

```zig
// BEFORE (run path):
for (0..n_runs) |r| {
    rc.runs[r].start = try reader.readInt(u16, .little);
    rc.runs[r].length = try reader.readInt(u16, .little);
}

// AFTER:
const bytes_needed = @as(usize, n_runs) * 4;
const bytes_read = try reader.readAll(std.mem.sliceAsBytes(rc.runs[0..n_runs]));
if (bytes_read != bytes_needed) return error.InvalidFormat;
```

#### Also bulk-read the offset header skip

The offset header skip (currently around lines 227-231) reads and discards
one u32 at a time. Replace with a bulk seek:

```zig
// BEFORE:
for (0..size) |_| {
    _ = try reader.readInt(u32, .little);
}

// AFTER:
// Skip offset header (size * 4 bytes)
try reader.skipBytes(size * 4, .{});
```

Note: `fixedBufferStream.reader()` supports `skipBytes`. If using a generic
reader type that doesn't, seek forward by reading into a discard buffer.

## Verification

1. `zig build test` — existing serialize/deserialize round-trip tests cover
   correctness. The byte-level CRoaring validation (`zig build validate`)
   confirms wire-format identity.

2. Re-run `zig build bench-compare` — serialize and deserialize should both
   drop to ~1.0x vs CRoaring, since the inner loop is now the same operation
   (memcpy).

## Expected impact

Serialize: 2.49x → ~1.0x (eliminate ~1024 function calls per bitset container)
Deserialize: 3.43x → ~1.0x (same reasoning)

These are pure I/O path changes. No algorithm, data structure, or format changes.
