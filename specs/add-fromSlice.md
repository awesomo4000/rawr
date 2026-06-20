# Add `fromSlice` — Safe Bulk Construction

## Summary

Add `fromSlice(allocator, values)` as the safe counterpart to `fromSorted`. Sorts and deduplicates the input, then delegates to `fromSorted`. Update `fromSorted` doc comment to cross-reference.

## Why

`fromSorted` requires strictly ascending, deduplicated input. Violation is UB in release. Real-world callers (pentesting data imports, LMDB bulk loads, user-facing APIs) often have unsorted or duplicated data. They need a path that Just Works without thinking about preconditions.

## API

```zig
/// Build from an arbitrary slice of values. O(n log n).
/// Sorts in-place and deduplicates. Mutates the input slice.
/// If input may already be sorted and unique, prefer `fromSorted` (O(n)).
pub fn fromSlice(allocator: std.mem.Allocator, values: []u32) !Self
```

Takes `[]u32` (mutable), not `[]const u32` — the signature itself communicates "I will modify your data." No hidden allocations for a copy.

## Implementation

Add to `bitmap.zig`, right after `fromSorted`:

```zig
pub fn fromSlice(allocator: std.mem.Allocator, values: []u32) !Self {
    if (values.len == 0) return Self.init(allocator);

    // Sort in-place
    std.mem.sortUnstable(u32, values, {}, std.sort.asc(u32));

    // Deduplicate in-place
    var write: usize = 1;
    for (values[1..]) |v| {
        if (v != values[write - 1]) {
            values[write] = v;
            write += 1;
        }
    }

    return fromSorted(allocator, values[0..write]);
}
```

Also add to `OwnedBitmap`:

```zig
pub fn fromSlice(backing: std.mem.Allocator, values: []u32) !OwnedBitmap {
    var inner = try RoaringBitmap.fromSlice(backing, values);
    return .{ .bitmap = inner, .arena = null };
}
```

## Update `fromSorted` doc comment

```zig
/// Build from pre-sorted, deduplicated values. O(n), no binary searches.
/// Caller must ensure values are in strictly ascending order with no duplicates.
/// Debug builds assert this precondition. In release, duplicates cause undefined
/// behavior (incorrect cardinality, corrupt containers).
/// If input may be unsorted or contain duplicates, use `fromSlice` instead.
pub fn fromSorted(allocator: std.mem.Allocator, values: []const u32) !Self
```

## Tests

**Test 1: fromSlice with unsorted duplicates**

```zig
test "fromSlice sorts and deduplicates" {
    const allocator = std.testing.allocator;
    var values = [_]u32{ 10, 3, 3, 7, 1, 10, 7, 1 };

    var bm = try RoaringBitmap.fromSlice(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 4), bm.cardinality());
    try std.testing.expect(bm.contains(1));
    try std.testing.expect(bm.contains(3));
    try std.testing.expect(bm.contains(7));
    try std.testing.expect(bm.contains(10));
    try std.testing.expect(!bm.contains(2));
}
```

**Test 2: fromSlice matches add oracle**

```zig
test "fromSlice matches incremental add" {
    const allocator = std.testing.allocator;
    var values = [_]u32{ 100, 1, 65536, 1, 200, 65536, 50 };

    var from_slice = try RoaringBitmap.fromSlice(allocator, &values);
    defer from_slice.deinit();

    var from_add = try RoaringBitmap.init(allocator);
    defer from_add.deinit();
    for ([_]u32{ 100, 1, 65536, 200, 50 }) |v| {
        _ = try from_add.add(v);
    }

    try std.testing.expect(from_slice.equals(&from_add));
}
```

**Test 3: fromSlice empty input**

```zig
test "fromSlice empty" {
    const allocator = std.testing.allocator;
    var values = [_]u32{};

    var bm = try RoaringBitmap.fromSlice(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 0), bm.cardinality());
    try std.testing.expect(bm.isEmpty());
}
```

**Test 4: fromSlice all duplicates**

```zig
test "fromSlice all duplicates" {
    const allocator = std.testing.allocator;
    var values = [_]u32{ 42, 42, 42, 42 };

    var bm = try RoaringBitmap.fromSlice(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 1), bm.cardinality());
    try std.testing.expect(bm.contains(42));
}
```

**Test 5: fromSlice cross-container with dupes**

```zig
test "fromSlice cross-container with duplicates" {
    const allocator = std.testing.allocator;
    var values = [_]u32{ 131072, 0, 65536, 0, 131072, 1, 65537 };

    var bm = try RoaringBitmap.fromSlice(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 5), bm.cardinality());
    try std.testing.expectEqual(@as(u32, 3), bm.size); // 3 containers
}
```

## Checklist

- [ ] `fromSlice` added to `RoaringBitmap`
- [ ] `fromSlice` added to `OwnedBitmap`
- [ ] `fromSorted` doc comment updated with cross-reference
- [ ] Tests 1-5 written and passing
- [ ] All existing tests still pass
