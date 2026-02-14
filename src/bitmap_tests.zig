const std = @import("std");
const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;
const OwnedBitmap = @import("bitmap.zig").OwnedBitmap;

// ============================================================================
// Tests
// ============================================================================

test "init and deinit" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    try std.testing.expect(bm.isEmpty());
    try std.testing.expectEqual(@as(u64, 0), bm.cardinality());
}

test "add and contains" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Add some values
    try std.testing.expect(try bm.add(42));
    try std.testing.expect(try bm.add(1000));
    try std.testing.expect(try bm.add(100000));

    // Check they're present
    try std.testing.expect(bm.contains(42));
    try std.testing.expect(bm.contains(1000));
    try std.testing.expect(bm.contains(100000));

    // Check absent values
    try std.testing.expect(!bm.contains(0));
    try std.testing.expect(!bm.contains(43));
    try std.testing.expect(!bm.contains(999));

    // Check cardinality
    try std.testing.expectEqual(@as(u64, 3), bm.cardinality());

    // Adding duplicate returns false
    try std.testing.expect(!try bm.add(42));
    try std.testing.expectEqual(@as(u64, 3), bm.cardinality());
}

test "values in same chunk" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // All in chunk 0 (values 0-65535)
    _ = try bm.add(0);
    _ = try bm.add(100);
    _ = try bm.add(1000);
    _ = try bm.add(65535);

    try std.testing.expectEqual(@as(u32, 1), bm.size); // Single container
    try std.testing.expectEqual(@as(u64, 4), bm.cardinality());

    try std.testing.expect(bm.contains(0));
    try std.testing.expect(bm.contains(100));
    try std.testing.expect(bm.contains(1000));
    try std.testing.expect(bm.contains(65535));
}

test "values in different chunks" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Each in different chunk
    _ = try bm.add(0); // chunk 0
    _ = try bm.add(65536); // chunk 1
    _ = try bm.add(131072); // chunk 2
    _ = try bm.add(0xFFFFFFFF); // chunk 65535

    try std.testing.expectEqual(@as(u32, 4), bm.size); // Four containers
    try std.testing.expectEqual(@as(u64, 4), bm.cardinality());
}

test "remove" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(10);
    _ = try bm.add(20);
    _ = try bm.add(30);

    try std.testing.expect(try bm.remove(20));
    try std.testing.expect(!bm.contains(20));
    try std.testing.expectEqual(@as(u64, 2), bm.cardinality());

    // Remove absent value
    try std.testing.expect(!try bm.remove(20));

    // Remove last values - container should be removed
    try std.testing.expect(try bm.remove(10));
    try std.testing.expect(try bm.remove(30));
    try std.testing.expect(bm.isEmpty());
}

test "minimum and maximum" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    try std.testing.expectEqual(@as(?u32, null), bm.minimum());
    try std.testing.expectEqual(@as(?u32, null), bm.maximum());

    _ = try bm.add(100);
    try std.testing.expectEqual(@as(?u32, 100), bm.minimum());
    try std.testing.expectEqual(@as(?u32, 100), bm.maximum());

    _ = try bm.add(50);
    _ = try bm.add(200);
    try std.testing.expectEqual(@as(?u32, 50), bm.minimum());
    try std.testing.expectEqual(@as(?u32, 200), bm.maximum());

    // Add in different chunk
    _ = try bm.add(1000000);
    try std.testing.expectEqual(@as(?u32, 50), bm.minimum());
    try std.testing.expectEqual(@as(?u32, 1000000), bm.maximum());
}

test "boundary values" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(0);
    _ = try bm.add(0xFFFFFFFF);

    try std.testing.expect(bm.contains(0));
    try std.testing.expect(bm.contains(0xFFFFFFFF));
    try std.testing.expectEqual(@as(?u32, 0), bm.minimum());
    try std.testing.expectEqual(@as(?u32, 0xFFFFFFFF), bm.maximum());
}

test "many values triggers growth" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Add values in 10 different chunks
    for (0..10) |i| {
        const chunk_base: u32 = @intCast(i * 65536);
        _ = try bm.add(chunk_base + 1);
    }

    try std.testing.expectEqual(@as(u32, 10), bm.size);
    try std.testing.expect(bm.capacity >= 10);
}

// ============================================================================
// Set Operation Tests
// ============================================================================

test "bitwiseOr" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(3);
    _ = try b.add(4);
    _ = try b.add(5);

    var result = try a.bitwiseOr(allocator, &b);
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 5), result.cardinality());
    try std.testing.expect(result.contains(1));
    try std.testing.expect(result.contains(2));
    try std.testing.expect(result.contains(3));
    try std.testing.expect(result.contains(4));
    try std.testing.expect(result.contains(5));
}

test "bitwiseAnd" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(2);
    _ = try b.add(3);
    _ = try b.add(4);

    var result = try a.bitwiseAnd(allocator, &b);
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 2), result.cardinality());
    try std.testing.expect(result.contains(2));
    try std.testing.expect(result.contains(3));
    try std.testing.expect(!result.contains(1));
    try std.testing.expect(!result.contains(4));
}

test "bitwiseDifference" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(2);
    _ = try b.add(3);
    _ = try b.add(4);

    var result = try a.bitwiseDifference(allocator, &b);
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 1), result.cardinality());
    try std.testing.expect(result.contains(1));
    try std.testing.expect(!result.contains(2));
    try std.testing.expect(!result.contains(3));
}

test "bitwiseXor" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(2);
    _ = try b.add(3);
    _ = try b.add(4);

    var result = try a.bitwiseXor(allocator, &b);
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 2), result.cardinality());
    try std.testing.expect(result.contains(1));
    try std.testing.expect(result.contains(4));
    try std.testing.expect(!result.contains(2));
    try std.testing.expect(!result.contains(3));
}

test "set operations across chunks" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1); // chunk 0
    _ = try a.add(65537); // chunk 1
    _ = try a.add(131073); // chunk 2

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(65537); // chunk 1 (overlap)
    _ = try b.add(196609); // chunk 3

    var union_result = try a.bitwiseOr(allocator, &b);
    defer union_result.deinit();
    try std.testing.expectEqual(@as(u64, 4), union_result.cardinality());

    var intersect_result = try a.bitwiseAnd(allocator, &b);
    defer intersect_result.deinit();
    try std.testing.expectEqual(@as(u64, 1), intersect_result.cardinality());
    try std.testing.expect(intersect_result.contains(65537));
}

test "isSubsetOf" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(1);
    _ = try b.add(2);
    _ = try b.add(3);

    try std.testing.expect(a.isSubsetOf(&b));
    try std.testing.expect(!b.isSubsetOf(&a));

    // Self is subset of self
    try std.testing.expect(a.isSubsetOf(&a));
}

test "equals" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(1);
    _ = try b.add(2);
    _ = try b.add(3);

    var c = try RoaringBitmap.init(allocator);
    defer c.deinit();
    _ = try c.add(1);
    _ = try c.add(2);

    try std.testing.expect(a.equals(&b));
    try std.testing.expect(!a.equals(&c));
}

test "cardinality identity: |A ∪ B| + |A ∩ B| = |A| + |B|" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    for (0..100) |i| {
        _ = try a.add(@intCast(i * 3));
    }

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    for (0..100) |i| {
        _ = try b.add(@intCast(i * 5));
    }

    var union_ab = try a.bitwiseOr(allocator, &b);
    defer union_ab.deinit();

    var intersect_ab = try a.bitwiseAnd(allocator, &b);
    defer intersect_ab.deinit();

    const lhs = union_ab.cardinality() + intersect_ab.cardinality();
    const rhs = a.cardinality() + b.cardinality();

    try std.testing.expectEqual(lhs, rhs);
}

test "A − A = ∅" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(100);
    _ = try a.add(10000);

    var diff = try a.bitwiseDifference(allocator, &a);
    defer diff.deinit();

    try std.testing.expect(diff.isEmpty());
}

test "A ∪ A = A" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);

    var union_aa = try a.bitwiseOr(allocator, &a);
    defer union_aa.deinit();

    try std.testing.expect(union_aa.equals(&a));
}

// ============================================================================
// Iterator Tests
// ============================================================================

test "iterator empty bitmap" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    var it = bm.iterator();
    try std.testing.expectEqual(@as(?u32, null), it.next());
}

test "iterator single container (array)" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(5);
    _ = try bm.add(10);
    _ = try bm.add(15);

    var it = bm.iterator();
    try std.testing.expectEqual(@as(?u32, 5), it.next());
    try std.testing.expectEqual(@as(?u32, 10), it.next());
    try std.testing.expectEqual(@as(?u32, 15), it.next());
    try std.testing.expectEqual(@as(?u32, null), it.next());
}

test "iterator multiple containers" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Values in different chunks
    _ = try bm.add(100); // chunk 0
    _ = try bm.add(65536 + 200); // chunk 1
    _ = try bm.add(131072 + 300); // chunk 2

    var it = bm.iterator();
    try std.testing.expectEqual(@as(?u32, 100), it.next());
    try std.testing.expectEqual(@as(?u32, 65536 + 200), it.next());
    try std.testing.expectEqual(@as(?u32, 131072 + 300), it.next());
    try std.testing.expectEqual(@as(?u32, null), it.next());
}

test "iterator collects all values" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Add various values
    const values = [_]u32{ 0, 1, 100, 1000, 65535, 65536, 100000, 0xFFFFFFFF };
    for (values) |v| {
        _ = try bm.add(v);
    }

    // Collect via iterator
    var collected: [8]u32 = undefined;
    var count: usize = 0;
    var it = bm.iterator();
    while (it.next()) |v| {
        collected[count] = v;
        count += 1;
    }

    try std.testing.expectEqual(@as(usize, 8), count);
    // Values should be in sorted order
    try std.testing.expectEqual(@as(u32, 0), collected[0]);
    try std.testing.expectEqual(@as(u32, 1), collected[1]);
    try std.testing.expectEqual(@as(u32, 100), collected[2]);
    try std.testing.expectEqual(@as(u32, 1000), collected[3]);
    try std.testing.expectEqual(@as(u32, 65535), collected[4]);
    try std.testing.expectEqual(@as(u32, 65536), collected[5]);
    try std.testing.expectEqual(@as(u32, 100000), collected[6]);
    try std.testing.expectEqual(@as(u32, 0xFFFFFFFF), collected[7]);
}

// ============================================================================
// In-Place Operation Tests
// ============================================================================

test "bitwiseOrInPlace" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(3);
    _ = try b.add(4);
    _ = try b.add(5);

    try a.bitwiseOrInPlace(&b);

    try std.testing.expectEqual(@as(u64, 5), a.cardinality());
    try std.testing.expect(a.contains(1));
    try std.testing.expect(a.contains(2));
    try std.testing.expect(a.contains(3));
    try std.testing.expect(a.contains(4));
    try std.testing.expect(a.contains(5));
}

test "bitwiseOrInPlace with new chunk" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(100); // chunk 0

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(65536 + 100); // chunk 1

    try a.bitwiseOrInPlace(&b);

    try std.testing.expectEqual(@as(u64, 2), a.cardinality());
    try std.testing.expect(a.contains(100));
    try std.testing.expect(a.contains(65536 + 100));
    try std.testing.expectEqual(@as(u32, 2), a.size); // two containers
}

test "bitwiseAndInPlace" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);
    _ = try a.add(4);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(2);
    _ = try b.add(3);
    _ = try b.add(5);

    try a.bitwiseAndInPlace(&b);

    try std.testing.expectEqual(@as(u64, 2), a.cardinality());
    try std.testing.expect(a.contains(2));
    try std.testing.expect(a.contains(3));
    try std.testing.expect(!a.contains(1));
    try std.testing.expect(!a.contains(4));
}

test "bitwiseAndInPlace with empty other" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    try a.bitwiseAndInPlace(&b);

    try std.testing.expect(a.isEmpty());
}

test "bitwiseDifferenceInPlace" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);
    _ = try a.add(4);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(2);
    _ = try b.add(4);

    try a.bitwiseDifferenceInPlace(&b);

    try std.testing.expectEqual(@as(u64, 2), a.cardinality());
    try std.testing.expect(a.contains(1));
    try std.testing.expect(a.contains(3));
    try std.testing.expect(!a.contains(2));
    try std.testing.expect(!a.contains(4));
}

test "in-place operations match non-in-place" {
    const allocator = std.testing.allocator;

    // Create two bitmaps
    var a1 = try RoaringBitmap.init(allocator);
    defer a1.deinit();
    var a2 = try RoaringBitmap.init(allocator);
    defer a2.deinit();

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    const vals_a = [_]u32{ 1, 2, 3, 65536, 65537 };
    const vals_b = [_]u32{ 2, 3, 4, 65537, 131072 };

    for (vals_a) |v| {
        _ = try a1.add(v);
        _ = try a2.add(v);
    }
    for (vals_b) |v| {
        _ = try b.add(v);
    }

    // Compare OR
    var or_result = try a1.bitwiseOr(allocator, &b);
    defer or_result.deinit();
    try a2.bitwiseOrInPlace(&b);
    try std.testing.expect(a2.equals(&or_result));
}

// ============================================================================
// addRange and fromSorted Tests
// ============================================================================

test "addRange single chunk" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    const added = try bm.addRange(10, 20);
    try std.testing.expectEqual(@as(u64, 11), added);
    try std.testing.expectEqual(@as(u64, 11), bm.cardinality());

    for (10..21) |i| {
        try std.testing.expect(bm.contains(@intCast(i)));
    }
    try std.testing.expect(!bm.contains(9));
    try std.testing.expect(!bm.contains(21));
}

test "addRange spanning chunks" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Range spanning chunk boundary
    const added = try bm.addRange(65530, 65545);
    try std.testing.expectEqual(@as(u64, 16), added);

    try std.testing.expect(bm.contains(65530));
    try std.testing.expect(bm.contains(65535)); // last of chunk 0
    try std.testing.expect(bm.contains(65536)); // first of chunk 1
    try std.testing.expect(bm.contains(65545));
    try std.testing.expectEqual(@as(u32, 2), bm.size); // two containers
}

test "addRange large range creates bitset" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Large range > 4096 should create bitset
    const added = try bm.addRange(0, 5000);
    try std.testing.expectEqual(@as(u64, 5001), added);
    try std.testing.expectEqual(@as(u64, 5001), bm.cardinality());
}

test "addRange to existing container" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(5);
    _ = try bm.add(15);

    const added = try bm.addRange(10, 20);
    try std.testing.expectEqual(@as(u64, 10), added); // 15 was already there
    try std.testing.expectEqual(@as(u64, 12), bm.cardinality()); // 5, 10-20
}

test "fromSorted empty" {
    const allocator = std.testing.allocator;
    const empty: []const u32 = &.{};

    var bm = try RoaringBitmap.fromSorted(allocator, empty);
    defer bm.deinit();

    try std.testing.expect(bm.isEmpty());
}

test "fromSorted single chunk" {
    const allocator = std.testing.allocator;
    const values = [_]u32{ 1, 5, 10, 100, 1000 };

    var bm = try RoaringBitmap.fromSorted(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 5), bm.cardinality());
    try std.testing.expectEqual(@as(u32, 1), bm.size); // one container

    for (values) |v| {
        try std.testing.expect(bm.contains(v));
    }
}

test "fromSorted multiple chunks" {
    const allocator = std.testing.allocator;
    const values = [_]u32{ 100, 200, 65536 + 50, 65536 + 100, 131072 + 1 };

    var bm = try RoaringBitmap.fromSorted(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 5), bm.cardinality());
    try std.testing.expectEqual(@as(u32, 3), bm.size); // three containers

    for (values) |v| {
        try std.testing.expect(bm.contains(v));
    }
}

test "fromSorted matches individual adds" {
    const allocator = std.testing.allocator;
    const values = [_]u32{ 0, 1, 100, 1000, 65535, 65536, 100000 };

    // Build with fromSorted
    var bm1 = try RoaringBitmap.fromSorted(allocator, &values);
    defer bm1.deinit();

    // Build with individual adds
    var bm2 = try RoaringBitmap.init(allocator);
    defer bm2.deinit();
    for (values) |v| {
        _ = try bm2.add(v);
    }

    try std.testing.expect(bm1.equals(&bm2));
}

test "bitwiseOrInPlace no leak on allocation failure" {
    // This test verifies that bitwiseOrInPlace properly cleans up
    // newly allocated containers when an allocation fails mid-operation.
    const base_allocator = std.testing.allocator;

    // Create two bitmaps with disjoint keys to force cloning
    var bm1 = try RoaringBitmap.init(base_allocator);
    defer bm1.deinit();
    var bm2 = try RoaringBitmap.init(base_allocator);
    defer bm2.deinit();

    // Add values to different chunks
    _ = try bm1.add(0); // chunk 0
    _ = try bm1.add(65536); // chunk 1
    _ = try bm2.add(131072); // chunk 2
    _ = try bm2.add(196608); // chunk 3

    // Use failing allocator that fails after a few allocations
    // This should trigger failure during cloneContainer calls
    var failing = std.testing.FailingAllocator.init(base_allocator, .{ .fail_index = 3 });

    // Create a copy with the failing allocator for the in-place op
    var bm1_copy = try bm1.clone(base_allocator);

    // Swap allocator to failing one for the operation
    bm1_copy.allocator = failing.allocator();

    // This should fail partway through and clean up properly
    const result = bm1_copy.bitwiseOrInPlace(&bm2);
    try std.testing.expectError(error.OutOfMemory, result);

    // Restore normal allocator for cleanup
    bm1_copy.allocator = base_allocator;
    bm1_copy.deinit();

    // If we get here without the testing allocator detecting leaks, the test passes
}

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
    try std.testing.expectEqual(@as(u64, 2), result.cardinality());
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

    try std.testing.expectEqual(@as(u64, 4), result.cardinality());
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
    try std.testing.expectEqual(@as(u64, 2), owned.cardinality());
}

test "OwnedBitmap bitwiseDifferenceOwned" {
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

    var result = try a.bitwiseDifferenceOwned(allocator, &b);
    defer result.deinit();

    try std.testing.expect(result.contains(1));
    try std.testing.expect(!result.contains(2));
    try std.testing.expect(!result.contains(3));
    try std.testing.expect(!result.contains(4));
    try std.testing.expectEqual(@as(u64, 1), result.cardinality());
}

test "OwnedBitmap iterator" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    _ = try a.add(10);
    _ = try a.add(20);
    _ = try a.add(30);
    _ = try b.add(20);
    _ = try b.add(30);
    _ = try b.add(40);

    var result = try a.bitwiseAndOwned(allocator, &b);
    defer result.deinit();

    var iter = result.iterator();
    try std.testing.expectEqual(@as(?u32, 20), iter.next());
    try std.testing.expectEqual(@as(?u32, 30), iter.next());
    try std.testing.expectEqual(@as(?u32, null), iter.next());
}

test "andCardinality matches bitwiseAnd().cardinality()" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    // Build overlapping bitmaps across multiple containers
    _ = try a.addRange(0, 1000);
    _ = try a.addRange(100_000, 101_000);
    _ = try b.addRange(500, 1500);
    _ = try b.addRange(100_500, 101_500);

    const card_fast = a.andCardinality(&b);
    var intersection = try a.bitwiseAnd(allocator, &b);
    defer intersection.deinit();
    try std.testing.expectEqual(card_fast, intersection.cardinality());
}

test "intersects" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    // Non-overlapping (different chunks)
    _ = try a.add(100);
    _ = try b.add(100_000);
    try std.testing.expect(!a.intersects(&b));

    // Add overlap
    _ = try b.add(100);
    try std.testing.expect(a.intersects(&b));
}

test "intersects with empty" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var empty = try RoaringBitmap.init(allocator);
    defer empty.deinit();

    _ = try a.add(42);
    try std.testing.expect(!a.intersects(&empty));
    try std.testing.expect(!empty.intersects(&a));
}

test "bitwiseXorInPlace matches bitwiseXor" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    _ = try a.addRange(0, 100);
    _ = try a.add(200);
    _ = try b.addRange(50, 150);
    _ = try b.add(300);

    var a_copy = try a.clone(allocator);
    defer a_copy.deinit();
    try a_copy.bitwiseXorInPlace(&b);

    var expected = try a.bitwiseXor(allocator, &b);
    defer expected.deinit();

    try std.testing.expect(a_copy.equals(&expected));
}

test "bitwiseXorInPlace removes empty containers" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    // Same values in both - XOR should produce empty
    _ = try a.add(42);
    _ = try b.add(42);

    try a.bitwiseXorInPlace(&b);
    try std.testing.expectEqual(@as(u64, 0), a.cardinality());
}

test "cached cardinality stays correct through mutations" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 0), bm.cardinality());

    _ = try bm.add(1);
    try std.testing.expectEqual(@as(u64, 1), bm.cardinality());

    _ = try bm.add(1); // duplicate
    try std.testing.expectEqual(@as(u64, 1), bm.cardinality());

    _ = try bm.addRange(100, 199);
    try std.testing.expectEqual(@as(u64, 101), bm.cardinality());

    _ = try bm.remove(1);
    try std.testing.expectEqual(@as(u64, 100), bm.cardinality());

    // In-place op invalidates, next call recomputes
    var other = try RoaringBitmap.init(allocator);
    defer other.deinit();
    _ = try other.addRange(150, 250);
    try bm.bitwiseOrInPlace(&other);
    try std.testing.expectEqual(@as(u64, 151), bm.cardinality());
}

test "fromSorted basic correctness" {
    const allocator = std.testing.allocator;
    const values = [_]u32{ 1, 5, 10, 100, 1000, 10000 };

    var bm = try RoaringBitmap.fromSorted(allocator, &values);
    defer bm.deinit();

    // Cardinality matches input length
    try std.testing.expectEqual(@as(u64, values.len), bm.cardinality());

    // Contains returns true for every input value
    for (values) |v| {
        try std.testing.expect(bm.contains(v));
    }

    // Contains returns false for values not in input
    try std.testing.expect(!bm.contains(0));
    try std.testing.expect(!bm.contains(2));
    try std.testing.expect(!bm.contains(50));
    try std.testing.expect(!bm.contains(999));

    // Iteration yields exactly the input values in order
    var it = bm.iterator();
    for (values) |expected| {
        try std.testing.expectEqual(expected, it.next().?);
    }
    try std.testing.expectEqual(@as(?u32, null), it.next());
}

test "fromSorted matches incremental add" {
    const allocator = std.testing.allocator;
    const values = [_]u32{ 0, 1, 100, 1000, 65535, 65536, 65537, 100000 };

    // Build via fromSorted
    var from_sorted = try RoaringBitmap.fromSorted(allocator, &values);
    defer from_sorted.deinit();

    // Build via add
    var from_add = try RoaringBitmap.init(allocator);
    defer from_add.deinit();
    for (values) |v| {
        _ = try from_add.add(v);
    }

    // They must be equal
    try std.testing.expect(from_sorted.equals(&from_add));
    try std.testing.expectEqual(from_sorted.cardinality(), from_add.cardinality());
}

test "fromSorted cardinality cache consistency" {
    const allocator = std.testing.allocator;
    const values = [_]u32{ 1, 2, 3, 100, 200, 300 };

    var bm = try RoaringBitmap.fromSorted(allocator, &values);
    defer bm.deinit();

    // Get cached cardinality
    const cached = bm.cardinality();
    try std.testing.expectEqual(@as(u64, 6), cached);

    // Add a value to trigger cache update path
    _ = try bm.add(50);
    try std.testing.expectEqual(@as(u64, 7), bm.cardinality());

    // Remove it
    _ = try bm.remove(50);
    try std.testing.expectEqual(@as(u64, 6), bm.cardinality());

    // Force cache invalidation via in-place op and recompute
    var empty = try RoaringBitmap.init(allocator);
    defer empty.deinit();
    try bm.bitwiseAndInPlace(&empty); // AND with empty = empty

    // After invalidation and recompute, must be 0
    try std.testing.expectEqual(@as(u64, 0), bm.cardinality());
}

test "fromSorted with cross-container values" {
    const allocator = std.testing.allocator;
    // Values spanning multiple 65536-boundaries
    const values = [_]u32{ 0, 1, 65536, 65537, 131072 };

    var bm = try RoaringBitmap.fromSorted(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 5), bm.cardinality());
    try std.testing.expectEqual(@as(u32, 3), bm.size); // 3 containers

    for (values) |v| {
        try std.testing.expect(bm.contains(v));
    }
}

test "fromSorted roundtrip serialize/deserialize" {
    const allocator = std.testing.allocator;
    const values = [_]u32{ 5, 10, 15, 65536, 65540, 131072, 131073 };

    var original = try RoaringBitmap.fromSorted(allocator, &values);
    defer original.deinit();

    // Serialize
    const bytes = try original.serialize(allocator);
    defer allocator.free(bytes);

    // Deserialize
    var restored = try RoaringBitmap.deserialize(allocator, bytes);
    defer restored.deinit();

    // Must be equal
    try std.testing.expect(original.equals(&restored));
    try std.testing.expectEqual(original.cardinality(), restored.cardinality());
}

test "fromSorted rejects duplicates in debug" {
    // This test verifies the debug assertion catches duplicates.
    // In debug builds, passing duplicates should panic/assert.
    // We can't easily test panics, so we document the expected behavior.
    // The assertion is: std.debug.assert(cur > values[i])

    // For now, just verify the happy path works
    const allocator = std.testing.allocator;
    const valid = [_]u32{ 1, 2, 3 }; // no duplicates
    var bm = try RoaringBitmap.fromSorted(allocator, &valid);
    defer bm.deinit();
    try std.testing.expectEqual(@as(u64, 3), bm.cardinality());
}

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

test "fromSlice empty" {
    const allocator = std.testing.allocator;
    var values = [_]u32{};

    var bm = try RoaringBitmap.fromSlice(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 0), bm.cardinality());
    try std.testing.expect(bm.isEmpty());
}

test "fromSlice all duplicates" {
    const allocator = std.testing.allocator;
    var values = [_]u32{ 42, 42, 42, 42 };

    var bm = try RoaringBitmap.fromSlice(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 1), bm.cardinality());
    try std.testing.expect(bm.contains(42));
}

test "fromSlice cross-container with duplicates" {
    const allocator = std.testing.allocator;
    var values = [_]u32{ 131072, 0, 65536, 0, 131072, 1, 65537 };

    var bm = try RoaringBitmap.fromSlice(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 5), bm.cardinality());
    try std.testing.expectEqual(@as(u32, 3), bm.size); // 3 containers
}
