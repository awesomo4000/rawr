# API: OwnedBitmap for Arena-Backed Operations

**Applies to:** `src/bitmap.zig`
**Depends on:** Nothing. Uses existing arena benchmark results as motivation.

## Problem

Arena allocation makes deserialize, bitwiseAnd, and bitwiseOr dramatically
faster (2-5x), but the caller must know to create and manage an arena. The
default API uses `std.mem.Allocator`, and naive callers get the slow path.

```zig
// Slow (2.9x slower than CRoaring):
var result = try a.bitwiseAnd(allocator, &b);
defer result.deinit();

// Fast (0.92x, faster than CRoaring) but requires arena knowledge:
var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
defer arena.deinit();
var result = try a.bitwiseAnd(arena.allocator(), &b);
```

## Solution: OwnedBitmap type + convenience methods

### New type (src/bitmap.zig)

Add after the `RoaringBitmap` struct definition:

```zig
/// A RoaringBitmap that owns its memory via an arena allocator.
/// All internal allocations use bump-pointer allocation for speed.
/// Call `deinit()` to free everything in one operation.
///
/// Returned by `deserializeOwned`, `bitwiseAndOwned`, `bitwiseOrOwned`.
pub const OwnedBitmap = struct {
    bitmap: RoaringBitmap,
    arena: std.heap.ArenaAllocator,

    /// Free all memory in one bulk operation.
    pub fn deinit(self: *OwnedBitmap) void {
        // Don't call bitmap.deinit() — arena owns all allocations.
        self.arena.deinit();
    }

    /// Check if a value is in the bitmap.
    pub fn contains(self: *const OwnedBitmap, value: u32) bool {
        return self.bitmap.contains(value);
    }

    /// Return the number of values in the bitmap.
    pub fn cardinality(self: *const OwnedBitmap) u32 {
        return self.bitmap.cardinality();
    }

    /// Iterate over all values in sorted order.
    pub fn iterator(self: *const OwnedBitmap) RoaringBitmap.Iterator {
        return self.bitmap.iterator();
    }

    /// Serialize to bytes. The output is allocated with the provided
    /// allocator (NOT the internal arena), so the caller owns it.
    pub fn serialize(self: *const OwnedBitmap, out_allocator: std.mem.Allocator) ![]u8 {
        return self.bitmap.serialize(out_allocator);
    }
};
```

The key design decisions:
- `deinit` does NOT call `bitmap.deinit()` — the arena owns everything.
- `serialize` takes a separate `out_allocator` because the serialized bytes
  should outlive the OwnedBitmap (you serialize then deinit).
- Read-only methods forward directly. No `add`/`remove` — the arena makes
  individual frees impossible, so mutation after creation is not supported.
  Users who need to mutate the result should use the standard `bitwiseAnd`
  with their own allocator.

### Convenience methods on RoaringBitmap

Add to the `RoaringBitmap` struct:

```zig
/// Deserialize a bitmap using arena allocation (recommended).
/// ~2x faster than CRoaring. Returns an OwnedBitmap that frees all
/// memory in one operation via deinit().
pub fn deserializeOwned(backing: std.mem.Allocator, data: []const u8) !OwnedBitmap {
    var arena = std.heap.ArenaAllocator.init(backing);
    errdefer arena.deinit();
    const bm = try Self.deserialize(arena.allocator(), data);
    return .{ .bitmap = bm, .arena = arena };
}

/// Compute intersection using arena allocation (recommended).
/// ~8% faster than CRoaring. Returns an OwnedBitmap.
pub fn bitwiseAndOwned(self: *const Self, backing: std.mem.Allocator, other: *const Self) !OwnedBitmap {
    var arena = std.heap.ArenaAllocator.init(backing);
    errdefer arena.deinit();
    const result = try self.bitwiseAnd(arena.allocator(), other);
    return .{ .bitmap = result, .arena = arena };
}

/// Compute union using arena allocation (recommended).
/// ~1.7x faster than CRoaring. Returns an OwnedBitmap.
pub fn bitwiseOrOwned(self: *const Self, backing: std.mem.Allocator, other: *const Self) !OwnedBitmap {
    var arena = std.heap.ArenaAllocator.init(backing);
    errdefer arena.deinit();
    const result = try self.bitwiseOr(arena.allocator(), other);
    return .{ .bitmap = result, .arena = arena };
}
```

Also add `bitwiseDifferenceOwned` for completeness (same pattern):

```zig
/// Compute difference (self \ other) using arena allocation.
pub fn bitwiseDifferenceOwned(self: *const Self, backing: std.mem.Allocator, other: *const Self) !OwnedBitmap {
    var arena = std.heap.ArenaAllocator.init(backing);
    errdefer arena.deinit();
    const result = try self.bitwiseDifference(arena.allocator(), other);
    return .{ .bitmap = result, .arena = arena };
}
```

### Error handling

The `errdefer arena.deinit()` ensures the arena is freed if the inner
operation fails. On success, ownership transfers to the returned OwnedBitmap.

## What NOT to change

- Existing `bitwiseAnd`, `bitwiseOr`, `bitwiseDifference`, `deserialize`
  signatures stay as-is. They're the "advanced" API for callers who manage
  their own allocator (e.g., kb's evaluation loop sharing one arena across
  many operations).
- `bitwiseAndInPlace`, `bitwiseOrInPlace` are unaffected — they mutate in
  place, no result allocation.

## Tests

Add tests for OwnedBitmap:

```zig
test "OwnedBitmap bitwiseAndOwned" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);
    _ = try b.add(2);
    _ = try b.add(3);
    _ = try b.add(4);

    var result = try a.bitwiseAndOwned(allocator, &b);
    defer result.deinit();

    try std.testing.expect(result.contains(2));
    try std.testing.expect(result.contains(3));
    try std.testing.expect(!result.contains(1));
    try std.testing.expect(!result.contains(4));
    try std.testing.expectEqual(@as(u32, 2), result.cardinality());
}

test "OwnedBitmap bitwiseOrOwned" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    _ = try a.add(1);
    _ = try a.add(2);
    _ = try b.add(3);
    _ = try b.add(4);

    var result = try a.bitwiseOrOwned(allocator, &b);
    defer result.deinit();

    try std.testing.expectEqual(@as(u32, 4), result.cardinality());
    try std.testing.expect(result.contains(1));
    try std.testing.expect(result.contains(4));
}

test "OwnedBitmap deserializeOwned" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(42);
    _ = try bm.add(1000);

    const data = try bm.serialize(allocator);
    defer allocator.free(data);

    var owned = try RoaringBitmap.deserializeOwned(allocator, data);
    defer owned.deinit();

    try std.testing.expect(owned.contains(42));
    try std.testing.expect(owned.contains(1000));
    try std.testing.expectEqual(@as(u32, 2), owned.cardinality());
}
```

## Usage examples (for README / doc comments)

```zig
// Recommended: fast path with OwnedBitmap
var result = try a.bitwiseAndOwned(allocator, &b);
defer result.deinit();
if (result.contains(42)) { ... }

// Recommended: fast deserialization
var bm = try RoaringBitmap.deserializeOwned(allocator, data);
defer bm.deinit();

// Advanced: manual arena for batch operations (e.g., evaluation loops)
var arena = std.heap.ArenaAllocator.init(allocator);
defer arena.deinit();
var r1 = try a.bitwiseAnd(arena.allocator(), &b);
var r2 = try r1.bitwiseOr(arena.allocator(), &c);
// All results freed at once by arena.deinit()
```

## Verification

```bash
zig build test          # new OwnedBitmap tests
zig build validate      # CRoaring byte-identity (serialize path unchanged)
zig build bench-compare # existing arena benchmarks still work
```
