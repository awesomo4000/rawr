<!-- SPDX-License-Identifier: MPL-2.0 -->

# OwnedBitmap: Switch deserializeOwned to FBA

**Based on:** Allocator matrix experiment (`bench_allocators.zig --matrix`, 5 runs)

## Background

We benchmarked 4 allocators for allocation-heavy operations:

```
deserialize output allocator (ms, median of 5 runs):
  c_allocator:    25.54
  smp_allocator:   1.92
  arena(page):     0.92
  FixedBuffer:     0.65   ← 30% faster than arena
```

Arena is good. FixedBufferAllocator is better — no page list, no alignment
bookkeeping, pure pointer bump. For `deserializeOwned` we can use FBA because
we have the full byte slice and can compute a safe buffer size before
allocating anything.

## Change 1: `deserializeOwned` uses FBA (src/bitmap.zig)

Current code (~line 1223):

```zig
pub fn deserializeOwned(backing: std.mem.Allocator, data: []const u8) !OwnedBitmap {
    var arena = std.heap.ArenaAllocator.init(backing);
    errdefer arena.deinit();
    const bm = try Self.deserialize(arena.allocator(), data);
    return .{ .bitmap = bm, .arena = arena };
}
```

New code:

```zig
pub fn deserializeOwned(backing: std.mem.Allocator, data: []const u8) !OwnedBitmap {
    // FBA over a single allocation — faster than arena (0.65ms vs 0.92ms).
    // 2x serialized size is a safe upper bound: serialized format is roughly
    // 1:1 with in-memory representation (array values are identical, bitsets
    // are identical, run pairs are slightly smaller). The extra 1x covers
    // container struct overhead, alignment padding, and the bitmap's own
    // keys/containers arrays.
    const buf_size = @max(data.len * 2, 64 * 1024);
    const buf = try backing.alloc(u8, buf_size);
    errdefer backing.free(buf);

    var fba = std.heap.FixedBufferAllocator.init(buf);
    const bm = try Self.deserialize(fba.allocator(), data);

    return .{
        .bitmap = bm,
        .backing_buf = buf,
        .backing_alloc = backing,
    };
}
```

If FBA returns `OutOfMemory` (buffer too small for unusual bitmaps), fall back
to arena:

```zig
pub fn deserializeOwned(backing: std.mem.Allocator, data: []const u8) !OwnedBitmap {
    const buf_size = @max(data.len * 2, 64 * 1024);
    const buf = try backing.alloc(u8, buf_size);

    var fba = std.heap.FixedBufferAllocator.init(buf);
    if (Self.deserialize(fba.allocator(), data)) |bm| {
        return .{
            .bitmap = bm,
            .backing_buf = buf,
            .backing_alloc = backing,
        };
    } else |err| switch (err) {
        error.OutOfMemory => {
            // FBA too small — fall back to arena
            backing.free(buf);
            var arena = std.heap.ArenaAllocator.init(backing);
            errdefer arena.deinit();
            const bm = try Self.deserialize(arena.allocator(), data);
            return .{ .bitmap = bm, .arena = arena };
        },
        else => {
            backing.free(buf);
            return err;
        },
    }
}
```

Pick whichever approach you prefer — simple (first version, trust the 2x bound)
or defensive (second version, arena fallback). The 2x bound should hold for all
real-world bitmaps but I haven't proven it exhaustively.

## Change 2: Update OwnedBitmap struct (src/bitmap.zig)

Current (~line 1262):

```zig
pub const OwnedBitmap = struct {
    bitmap: RoaringBitmap,
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *OwnedBitmap) void {
        self.arena.deinit();
    }
    // ...
};
```

New:

```zig
pub const OwnedBitmap = struct {
    bitmap: RoaringBitmap,

    // One of these is active, not both:
    arena: ?std.heap.ArenaAllocator = null,
    backing_buf: ?[]u8 = null,
    backing_alloc: ?std.mem.Allocator = null,

    pub fn deinit(self: *OwnedBitmap) void {
        // Don't call bitmap.deinit() — we own all memory.
        if (self.backing_buf) |buf| {
            self.backing_alloc.?.free(buf);
        }
        if (self.arena) |*a| {
            a.deinit();
        }
    }

    // contains, cardinality, iterator, serialize — unchanged
};
```

The set operation `Owned` variants (`bitwiseAndOwned`, `bitwiseOrOwned`,
`bitwiseDifferenceOwned`) keep using arena — we can't predict output size
for those.

## Change 3: Doc comment on allocator choice (src/bitmap.zig)

Add to the `RoaringBitmap` struct doc comment or top of file:

```zig
/// ## Allocator guidance
///
/// Avoid `std.heap.c_allocator` — it is 10-40x slower than alternatives
/// for rawr's allocation patterns (many small containers).
///
/// Recommended:
/// - `OwnedBitmap` API: fastest (uses optimized allocation internally)
/// - `std.heap.smp_allocator`: fast general-purpose, supports mutation
/// - `std.heap.ArenaAllocator`: fast batch alloc, bulk free only
```

## Testing

Existing tests should pass unchanged — the FBA path produces identical bitmaps.

- `zig build test`
- `zig build validate` (CRoaring byte-identity)
- Run `bench_croaring` to confirm `deserialize (arena)` line improves
  (it uses `deserializeOwned` internally... actually check this — if bench
  calls `deserialize` directly with an arena, the bench number won't change.
  Add a `deserialize (owned)` line to see the FBA improvement.)

### New benchmark line in bench_croaring.zig

After the existing `deserialize (arena)` benchmark, add:

```zig
fn benchRawrDeserializeOwned() void {
    var owned = RoaringBitmap.deserializeOwned(
        std.heap.smp_allocator, rawr_serialized.?
    ) catch unreachable;
    defer owned.deinit();
    std.mem.doNotOptimizeAway(&owned);
}
```

And in main:

```zig
r = benchmark(benchRawrDeserializeOwned, .{});
printResult("deserialize (owned)", r.median_ns, cr.median_ns);
```

Expected: ~0.65ms (down from arena's ~1.0ms).

## Platform note

All numbers are from macOS on Apple M4. Relative performance between allocators
may differ on other platforms — in particular, the c_allocator 25x penalty might
be macOS-specific (its malloc through Zig's aligned_alloc wrapper). The FBA vs
arena gap (30%) should be stable everywhere since both are pure Zig with no
syscalls in the hot path. If we ever need to validate on another platform,
`bench_allocators --matrix` is the tool.

## What NOT to change

- Don't change `bitwiseAndOwned`/`bitwiseOrOwned` — keep arena for those
- Don't change the `deserialize()` or `bitwiseAnd()` functions that take
  a bare allocator — those are the power-user path
- Don't remove arena support from OwnedBitmap (set ops still need it)
- Don't add FBA to `deserializeFromReader` — reader can't peek ahead
