// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;
const OwnedBitmap = @import("bitmap.zig").OwnedBitmap;
const FrozenBitmap = @import("frozen.zig").FrozenBitmap;
const ArrayContainer = @import("array_container.zig").ArrayContainer;
const BitsetContainer = @import("bitset_container.zig").BitsetContainer;
const RunContainer = @import("run_container.zig").RunContainer;
const TaggedPtr = @import("container.zig").TaggedPtr;
const fmt = @import("format.zig");
const test_gen = @import("test_gen.zig");

const MALFORMED_SMOKE_SEED: u64 = 0xBAD5_EED0_1609;

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

test "initCapacity reserves exact container capacity" {
    const allocator = std.testing.allocator;
    const requested: u32 = 8;
    var bm = try RoaringBitmap.initCapacity(allocator, requested);
    defer bm.deinit();

    try std.testing.expectEqual(requested, bm.capacity);
    try std.testing.expectEqual(@as(u32, 0), bm.size);
    const keys_ptr = bm.keys.ptr;
    const containers_ptr = bm.containers.ptr;

    for (0..requested) |chunk| {
        _ = try bm.add(@as(u32, @intCast(chunk)) << 16);
    }

    try std.testing.expectEqual(requested, bm.size);
    try std.testing.expectEqual(keys_ptr, bm.keys.ptr);
    try std.testing.expectEqual(containers_ptr, bm.containers.ptr);
}

test "initCapacity zero grows on first insert" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.initCapacity(allocator, 0);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u32, 0), bm.capacity);
    try std.testing.expect(bm.isEmpty());
    try std.testing.expect(try bm.add(42));
    try std.testing.expect(bm.contains(42));
    try std.testing.expect(bm.capacity >= 1);
}

test "clone handles empty singleton and multi-container bitmaps" {
    const allocator = std.testing.allocator;

    var empty = try RoaringBitmap.init(allocator);
    defer empty.deinit();
    var empty_clone = try empty.clone(allocator);
    defer empty_clone.deinit();
    try std.testing.expect(try empty_clone.add(42));
    try std.testing.expect(empty_clone.contains(42));
    try std.testing.expect(empty.isEmpty());

    var singleton = try RoaringBitmap.init(allocator);
    defer singleton.deinit();
    _ = try singleton.add(1234);
    var singleton_clone = try singleton.clone(allocator);
    defer singleton_clone.deinit();
    try expectPortableClone(&singleton, &singleton_clone);

    var multi = try RoaringBitmap.init(allocator);
    defer multi.deinit();
    _ = try multi.addRange(0, 499_999);
    var multi_clone = try multi.clone(allocator);
    defer multi_clone.deinit();
    try expectPortableClone(&multi, &multi_clone);
}

test "clone is leak-free and preserves source across allocation failures" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        cloneAllocationFailureCase,
        .{},
    );
}

fn cloneAllocationFailureCase(allocator: std.mem.Allocator) !void {
    var source = try RoaringBitmap.init(std.testing.allocator);
    defer source.deinit();
    _ = try source.addRange(0, 499_999);

    const before = try source.serialize(std.testing.allocator);
    defer std.testing.allocator.free(before);

    var cloned = source.clone(allocator) catch |err| {
        const after = try source.serialize(std.testing.allocator);
        defer std.testing.allocator.free(after);
        try std.testing.expectEqualSlices(u8, before, after);
        return err;
    };
    defer cloned.deinit();
    try expectPortableClone(&source, &cloned);
}

fn expectPortableClone(source: *const RoaringBitmap, cloned: *const RoaringBitmap) !void {
    const source_bytes = try source.serialize(std.testing.allocator);
    defer std.testing.allocator.free(source_bytes);
    const clone_bytes = try cloned.serialize(std.testing.allocator);
    defer std.testing.allocator.free(clone_bytes);
    try std.testing.expectEqualSlices(u8, source_bytes, clone_bytes);
}

test "ensureTotalCapacity grows once and preserves contents" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.initCapacity(allocator, 1);
    defer bm.deinit();

    _ = try bm.add(42);
    try bm.ensureTotalCapacity(12);
    try std.testing.expect(bm.capacity >= 12);
    try std.testing.expect(bm.contains(42));

    const keys_ptr = bm.keys.ptr;
    const containers_ptr = bm.containers.ptr;
    const capacity = bm.capacity;
    try bm.ensureTotalCapacity(6);
    try std.testing.expectEqual(capacity, bm.capacity);
    try std.testing.expectEqual(keys_ptr, bm.keys.ptr);
    try std.testing.expectEqual(containers_ptr, bm.containers.ptr);

    try bm.ensureTotalCapacity(13);
    try std.testing.expect(bm.capacity >= 13);
    try std.testing.expect(bm.contains(42));
}

test "ensureTotalCapacity leaves bitmap unchanged when second allocation fails" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.initCapacity(allocator, 2);
    defer bm.deinit();

    _ = try bm.add(1);
    _ = try bm.add((@as(u32, 1) << 16) | 2);
    const keys_ptr = bm.keys.ptr;
    const containers_ptr = bm.containers.ptr;
    const capacity = bm.capacity;

    var failing = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 1 });
    bm.allocator = failing.allocator();
    try std.testing.expectError(error.OutOfMemory, bm.ensureTotalCapacity(8));
    bm.allocator = allocator;

    try std.testing.expectEqual(capacity, bm.capacity);
    try std.testing.expectEqual(keys_ptr, bm.keys.ptr);
    try std.testing.expectEqual(containers_ptr, bm.containers.ptr);
    try std.testing.expectEqual(@as(u64, 2), bm.cardinality());
    try std.testing.expect(bm.contains(1));
    try std.testing.expect(bm.contains((@as(u32, 1) << 16) | 2));

    _ = try bm.add((@as(u32, 2) << 16) | 3);
    try std.testing.expectEqual(@as(u64, 3), bm.cardinality());
}

test "clearRetainingCapacity retains index and shrinkToFit releases it" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.initCapacity(allocator, 8);
    defer bm.deinit();

    _ = try bm.add(1);
    _ = try bm.add((@as(u32, 3) << 16) | 2);
    const keys_ptr = bm.keys.ptr;
    const containers_ptr = bm.containers.ptr;

    bm.clearRetainingCapacity();
    try std.testing.expect(bm.isEmpty());
    try std.testing.expectEqual(@as(u64, 0), bm.cardinality());
    try std.testing.expectEqual(@as(u32, 8), bm.capacity);
    try std.testing.expectEqual(keys_ptr, bm.keys.ptr);
    try std.testing.expectEqual(containers_ptr, bm.containers.ptr);

    _ = try bm.add(99);
    try std.testing.expect(bm.contains(99));
    bm.clearRetainingCapacity();
    try std.testing.expect((try bm.shrinkToFit()) > 0);
    try std.testing.expectEqual(@as(u32, 0), bm.capacity);

    _ = try bm.add(100);
    try std.testing.expect(bm.contains(100));
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

test "shrinkToFit trims bitmap and array container capacity" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    for (0..9) |i| {
        _ = try bm.add(@intCast(i * 2));
    }
    for (5..9) |i| {
        try std.testing.expect(try bm.remove(@intCast(i * 2)));
    }
    for (1..10) |chunk| {
        _ = try bm.add(@as(u32, @intCast(chunk)) << 16);
    }

    try std.testing.expect(bm.capacity > bm.size);
    const ac_before = bm.containers[0].getArray();
    try std.testing.expect(ac_before.capacity > ac_before.cardinality);
    const old_array_cap = ac_before.capacity;

    const freed = try bm.shrinkToFit();
    try std.testing.expect(freed > 0);
    try std.testing.expectEqual(bm.size, bm.capacity);

    const ac_after = bm.containers[0].getArray();
    try std.testing.expect(ac_after.capacity < old_array_cap);
    try std.testing.expect(ac_after.capacity >= ac_after.cardinality);
    try std.testing.expectEqual(@as(u64, 14), bm.cardinality());
    try std.testing.expect(bm.contains(8));
    try std.testing.expect(bm.contains(9 << 16));
    try std.testing.expectEqual(@as(usize, 0), try bm.shrinkToFit());
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

test "dense set operations handle full-run identities and zero-capacity results" {
    const allocator = std.testing.allocator;

    var full = try RoaringBitmap.init(allocator);
    defer full.deinit();
    _ = try full.addRange(0, std.math.maxInt(u16));
    var partial = try RoaringBitmap.init(allocator);
    defer partial.deinit();
    _ = try partial.addRange(10, 20);

    var union_result = try full.bitwiseOr(allocator, &partial);
    defer union_result.deinit();
    try expectPortableClone(&full, &union_result);
    try std.testing.expectEqual(TaggedPtr.ContainerType.run, union_result.containers[0].getType());

    var sparse = try RoaringBitmap.init(allocator);
    defer sparse.deinit();
    _ = try sparse.add(1);
    _ = try sparse.add(1000);
    var run_array_union = try full.bitwiseOr(allocator, &sparse);
    defer run_array_union.deinit();
    try expectPortableClone(&full, &run_array_union);
    try std.testing.expectEqual(TaggedPtr.ContainerType.run, run_array_union.containers[0].getType());

    var bitset = try RoaringBitmap.init(allocator);
    defer bitset.deinit();
    for (0..5000) |value| _ = try bitset.add(@intCast(value * 2));
    try std.testing.expectEqual(TaggedPtr.ContainerType.bitset, bitset.containers[0].getType());
    var run_bitset_union = try full.bitwiseOr(allocator, &bitset);
    defer run_bitset_union.deinit();
    try expectPortableClone(&full, &run_bitset_union);
    try std.testing.expectEqual(TaggedPtr.ContainerType.run, run_bitset_union.containers[0].getType());

    var empty = try RoaringBitmap.init(allocator);
    defer empty.deinit();
    var intersection = try full.bitwiseAnd(allocator, &empty);
    defer intersection.deinit();
    try std.testing.expectEqual(@as(u32, 0), intersection.capacity);
    try std.testing.expect(try intersection.add(42));
    try std.testing.expect(intersection.contains(42));

    var empty_union = try empty.bitwiseOr(allocator, &empty);
    defer empty_union.deinit();
    try std.testing.expect(try empty_union.add(99));
    try std.testing.expect(empty_union.contains(99));
}

test "dense set-operation construction is leak-free across allocation failures" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        denseSetOperationAllocationFailureCase,
        .{},
    );
}

fn denseSetOperationAllocationFailureCase(result_allocator: std.mem.Allocator) !void {
    const source_allocator = std.testing.allocator;
    var left = try RoaringBitmap.init(source_allocator);
    defer left.deinit();
    var right = try RoaringBitmap.init(source_allocator);
    defer right.deinit();
    _ = try left.addRange(0, 499_999);
    _ = try right.addRange(250_000, 749_999);

    var intersection = try left.bitwiseAnd(result_allocator, &right);
    intersection.deinit();
    var union_result = try left.bitwiseOr(result_allocator, &right);
    union_result.deinit();

    try std.testing.expectEqual(@as(u64, 500_000), left.cardinality());
    try std.testing.expectEqual(@as(u64, 500_000), right.cardinality());
}

test "orMany fused bitset path preserves values representations and inputs" {
    const allocator = std.testing.allocator;

    var array = try RoaringBitmap.init(allocator);
    defer array.deinit();
    for (0..128) |index| _ = try array.add(@intCast(index * 31));

    var bitset_a = try RoaringBitmap.init(allocator);
    defer bitset_a.deinit();
    var bitset_b = try RoaringBitmap.init(allocator);
    defer bitset_b.deinit();
    for (0..5000) |index| {
        _ = try bitset_a.add(@intCast(index * 2));
        _ = try bitset_b.add(@intCast(index * 2 + 1));
    }

    var run = try RoaringBitmap.init(allocator);
    defer run.deinit();
    _ = try run.addRange(1000, 20_000);

    const inputs = [_]*const RoaringBitmap{ &array, &bitset_a, &bitset_b, &run, &bitset_a };
    var expected = try array.bitwiseOr(allocator, &bitset_a);
    defer expected.deinit();
    try expected.bitwiseOrInPlace(&bitset_b);
    try expected.bitwiseOrInPlace(&run);

    const before = [_][]u8{
        try array.serialize(allocator),
        try bitset_a.serialize(allocator),
        try bitset_b.serialize(allocator),
        try run.serialize(allocator),
    };
    defer for (before) |bytes| allocator.free(bytes);

    var result = try RoaringBitmap.orMany(allocator, &inputs);
    defer result.deinit();
    try std.testing.expect(expected.equals(&result));
    try std.testing.expectEqual(expected.cardinality(), result.cardinality());
    try std.testing.expectEqual(@as(u32, 1), result.size);
    try std.testing.expectEqual(TaggedPtr.ContainerType.bitset, result.containers[0].getType());
    try result.validate();

    const sources = [_]*const RoaringBitmap{ &array, &bitset_a, &bitset_b, &run };
    for (sources, before) |source, bytes| {
        const after = try source.serialize(allocator);
        defer allocator.free(after);
        try std.testing.expectEqualSlices(u8, bytes, after);
    }

    var one_bitset = try RoaringBitmap.orMany(allocator, &.{ &bitset_a, &array });
    defer one_bitset.deinit();
    var one_bitset_expected = try bitset_a.bitwiseOr(allocator, &array);
    defer one_bitset_expected.deinit();
    try expectPortableClone(&one_bitset_expected, &one_bitset);
    _ = try one_bitset.add(std.math.maxInt(u32));
    try std.testing.expect(!bitset_a.contains(std.math.maxInt(u32)));
}

test "orMany fused path preserves the array to bitset boundary" {
    const allocator = std.testing.allocator;
    var even = try RoaringBitmap.init(allocator);
    defer even.deinit();
    var odd = try RoaringBitmap.init(allocator);
    defer odd.deinit();
    for (0..2048) |index| {
        _ = try even.add(@intCast(index * 2));
        _ = try odd.add(@intCast(index * 2 + 1));
    }

    var at_boundary = try RoaringBitmap.orMany(allocator, &.{ &even, &odd });
    defer at_boundary.deinit();
    try std.testing.expectEqual(@as(u64, 4096), at_boundary.cardinality());
    try std.testing.expectEqual(TaggedPtr.ContainerType.array, at_boundary.containers[0].getType());

    _ = try even.add(4096);
    var above_boundary = try RoaringBitmap.orMany(allocator, &.{ &even, &odd });
    defer above_boundary.deinit();
    try std.testing.expectEqual(@as(u64, 4097), above_boundary.cardinality());
    try std.testing.expectEqual(TaggedPtr.ContainerType.bitset, above_boundary.containers[0].getType());
    try above_boundary.validate();
}

test "orMany fused path handles empty single and allocation failures" {
    const allocator = std.testing.allocator;

    var empty = try RoaringBitmap.orMany(allocator, &.{});
    defer empty.deinit();
    try std.testing.expect(empty.isEmpty());

    var source = try makeOrManyFailureInput(0);
    defer source.deinit();
    var single = try RoaringBitmap.orMany(allocator, &.{&source});
    defer single.deinit();
    try expectPortableClone(&source, &single);
    _ = try single.add(std.math.maxInt(u32));
    try std.testing.expect(!source.contains(std.math.maxInt(u32)));

    try std.testing.checkAllAllocationFailures(
        allocator,
        orManyAllocationFailureCase,
        .{},
    );
}

fn orManyAllocationFailureCase(result_allocator: std.mem.Allocator) !void {
    var a = try makeOrManyFailureInput(0);
    defer a.deinit();
    var b = try makeOrManyFailureInput(1);
    defer b.deinit();
    const inputs = [_]*const RoaringBitmap{ &a, &b };

    const before_a = try a.serialize(std.testing.allocator);
    defer std.testing.allocator.free(before_a);
    const before_b = try b.serialize(std.testing.allocator);
    defer std.testing.allocator.free(before_b);

    var result = RoaringBitmap.orMany(result_allocator, &inputs) catch |err| {
        try expectBitmapBytes(&a, before_a);
        try expectBitmapBytes(&b, before_b);
        return err;
    };
    defer result.deinit();
    try expectBitmapBytes(&a, before_a);
    try expectBitmapBytes(&b, before_b);
}

fn makeOrManyFailureInput(offset: u32) !RoaringBitmap {
    var bitmap = try RoaringBitmap.init(std.testing.allocator);
    errdefer bitmap.deinit();
    for (0..5000) |index| _ = try bitmap.add(@intCast(index + offset));
    for (0..128) |index| {
        _ = try bitmap.add((@as(u32, 1) << 16) | @as(u32, @intCast(index * 31 + offset)));
    }
    return bitmap;
}

fn expectBitmapBytes(bitmap: *const RoaringBitmap, expected: []const u8) !void {
    const actual = try bitmap.serialize(std.testing.allocator);
    defer std.testing.allocator.free(actual);
    try std.testing.expectEqualSlices(u8, expected, actual);
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

const ConsumeFixture = struct {
    left: RoaringBitmap,
    right: RoaringBitmap,

    fn deinit(self: *ConsumeFixture) void {
        self.right.deinit();
        self.left.deinit();
    }
};

fn consumeTestArray(allocator: std.mem.Allocator, start: u16, count: u16) !TaggedPtr {
    const array = try ArrayContainer.init(allocator, count);
    for (0..count) |idx| array.values[idx] = start + @as(u16, @intCast(idx));
    array.cardinality = count;
    return TaggedPtr.initArray(array);
}

fn consumeTestBitset(allocator: std.mem.Allocator, start: u16, count: u16) !TaggedPtr {
    const bitset = try BitsetContainer.init(allocator);
    var value: u32 = start;
    const end = value + count;
    while (value < end) : (value += 1) _ = bitset.add(@intCast(value));
    return TaggedPtr.initBitset(bitset);
}

fn consumeTestRun(allocator: std.mem.Allocator, start: u16, end: u16) !TaggedPtr {
    const run = try RunContainer.init(allocator, 1);
    errdefer run.deinit(allocator);
    _ = try run.addRange(allocator, start, end);
    return TaggedPtr.initRun(run);
}

fn appendConsumeTestContainer(bitmap: *RoaringBitmap, key: u16, container: TaggedPtr) void {
    std.debug.assert(bitmap.size < bitmap.capacity);
    bitmap.keys[bitmap.size] = key;
    bitmap.containers[bitmap.size] = container;
    bitmap.size += 1;
    bitmap.cached_cardinality = -1;
}

fn makeConsumeFixture(allocator: std.mem.Allocator) !ConsumeFixture {
    var left = try RoaringBitmap.initCapacity(allocator, 3);
    errdefer left.deinit();
    appendConsumeTestContainer(&left, 0, try consumeTestArray(allocator, 0, 64));
    appendConsumeTestContainer(&left, 1, try consumeTestArray(allocator, 0, 64));
    appendConsumeTestContainer(&left, 2, try consumeTestRun(allocator, 0, 99));

    var right = try RoaringBitmap.initCapacity(allocator, 4);
    errdefer right.deinit();
    appendConsumeTestContainer(&right, 0, try consumeTestArray(allocator, 64, 64));
    appendConsumeTestContainer(&right, 1, try consumeTestBitset(allocator, 0, 5000));
    appendConsumeTestContainer(&right, 2, try consumeTestRun(allocator, 100, 199));
    appendConsumeTestContainer(&right, 3, try consumeTestArray(allocator, 0, 64));

    try left.validate();
    try right.validate();
    return .{ .left = left, .right = right };
}

fn iteratorCardinality(bitmap: *const RoaringBitmap) u64 {
    var count: u64 = 0;
    var iterator = bitmap.iterator();
    while (iterator.next() != null) count += 1;
    return count;
}

fn consumeOrAllocationFailureCase(allocator: std.mem.Allocator) !void {
    var fixture = try makeConsumeFixture(allocator);
    defer fixture.deinit();

    var expected_left = try fixture.left.bitwiseOr(std.testing.allocator, &fixture.right);
    defer expected_left.deinit();
    var expected_right = try fixture.right.clone(std.testing.allocator);
    defer expected_right.deinit();

    fixture.left.bitwiseOrInPlaceConsume(&fixture.right) catch |err| switch (err) {
        error.OutOfMemory => {
            try fixture.left.validate();
            try fixture.right.validate();
            try std.testing.expect(fixture.right.equals(&expected_right));
            try std.testing.expectEqual(iteratorCardinality(&fixture.left), fixture.left.cardinality());
            try std.testing.expectEqual(iteratorCardinality(&fixture.right), fixture.right.cardinality());
            return error.OutOfMemory;
        },
        else => return err,
    };

    try fixture.left.validate();
    try fixture.right.validate();
    try std.testing.expect(fixture.left.equals(&expected_left));
    try std.testing.expectEqual(@as(u64, 0), fixture.right.cardinality());
}

test "bitwiseOrInPlaceConsume handles all allocation failures" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        consumeOrAllocationFailureCase,
        .{},
    );
}

test "bitwiseOrInPlaceConsume rejects allocator mismatch and aliasing" {
    var left_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer left_arena.deinit();
    var right_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer right_arena.deinit();

    var left = try RoaringBitmap.init(left_arena.allocator());
    var right = try RoaringBitmap.init(right_arena.allocator());
    _ = try left.add(1);
    _ = try right.add(2);
    const left_cardinality = left.cardinality();
    const right_cardinality = right.cardinality();
    try std.testing.expect(left.allocator.vtable == right.allocator.vtable);
    try std.testing.expect(left.allocator.ptr != right.allocator.ptr);
    try std.testing.expectError(error.AllocatorMismatch, left.bitwiseOrInPlaceConsume(&right));
    try std.testing.expectEqual(left_cardinality, left.cardinality());
    try std.testing.expectEqual(right_cardinality, right.cardinality());
    try left.validate();
    try right.validate();

    var aliased = try RoaringBitmap.init(std.testing.allocator);
    defer aliased.deinit();
    _ = try aliased.add(7);
    var expected = try aliased.clone(std.testing.allocator);
    defer expected.deinit();
    try std.testing.expectError(error.AliasedOperands, aliased.bitwiseOrInPlaceConsume(&aliased));
    try std.testing.expect(aliased.equals(&expected));
    try aliased.validate();
}

test "bitwiseOrInPlaceConsume empties and permits reuse of other" {
    const allocator = std.testing.allocator;
    var left = try RoaringBitmap.init(allocator);
    defer left.deinit();
    var right = try RoaringBitmap.initCapacity(allocator, 8);
    defer right.deinit();
    _ = try left.add(1);
    _ = try left.add((@as(u32, 1) << 16) | 2);
    _ = try right.add(3);
    _ = try right.add((@as(u32, 2) << 16) | 4);

    var expected = try left.bitwiseOr(allocator, &right);
    defer expected.deinit();
    const right_capacity = right.capacity;
    try left.bitwiseOrInPlaceConsume(&right);
    try std.testing.expect(left.equals(&expected));
    try left.validate();
    try right.validate();
    try std.testing.expectEqual(@as(u64, 0), right.cardinality());
    try std.testing.expectEqual(right_capacity, right.capacity);

    _ = try right.add(99);
    var further = try RoaringBitmap.init(allocator);
    defer further.deinit();
    _ = try further.add(100);
    try right.bitwiseOrInPlace(&further);
    try right.validate();
    try std.testing.expect(right.contains(99));
    try std.testing.expect(right.contains(100));
}

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

test "duplicate-heavy array union remains a valid array" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    for (0..3000) |value| {
        _ = try a.add(@intCast(value));
        _ = try b.add(@intCast(value));
    }

    var result = try a.bitwiseOr(allocator, &b);
    defer result.deinit();
    try std.testing.expectEqual(@as(u64, 3000), result.cardinality());
    try std.testing.expectEqual(.array, result.containers[0].getType());
    try result.validate();

    const bytes = try result.serialize(allocator);
    defer allocator.free(bytes);
    var restored = try RoaringBitmap.deserializeSafe(allocator, bytes);
    defer restored.deinit();
    try std.testing.expect(result.equals(&restored));
}

test "overlapping array union remains a valid array" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    for (0..3000) |value| _ = try a.add(@intCast(value));
    for (1000..4000) |value| _ = try b.add(@intCast(value));

    var result = try a.bitwiseOr(allocator, &b);
    defer result.deinit();
    try std.testing.expectEqual(@as(u64, 4000), result.cardinality());
    try std.testing.expectEqual(.array, result.containers[0].getType());
    try result.validate();

    const bytes = try result.serialize(allocator);
    defer allocator.free(bytes);
    var restored = try RoaringBitmap.deserializeSafe(allocator, bytes);
    defer restored.deinit();
    try std.testing.expect(result.equals(&restored));
}

test "duplicate-heavy array union in place remains a valid array" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    for (0..3000) |value| {
        _ = try a.add(@intCast(value));
        _ = try b.add(@intCast(value));
    }

    try a.bitwiseOrInPlace(&b);
    try std.testing.expectEqual(@as(u64, 3000), a.cardinality());
    try std.testing.expectEqual(.array, a.containers[0].getType());
    try a.validate();

    const bytes = try a.serialize(allocator);
    defer allocator.free(bytes);
    var restored = try RoaringBitmap.deserializeSafe(allocator, bytes);
    defer restored.deinit();
    try std.testing.expect(a.equals(&restored));
}

test "overlapping array union in place remains a valid array" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    for (0..3000) |value| _ = try a.add(@intCast(value));
    for (1000..4000) |value| _ = try b.add(@intCast(value));

    try a.bitwiseOrInPlace(&b);
    try std.testing.expectEqual(@as(u64, 4000), a.cardinality());
    try std.testing.expectEqual(.array, a.containers[0].getType());
    try a.validate();

    const bytes = try a.serialize(allocator);
    defer allocator.free(bytes);
    var restored = try RoaringBitmap.deserializeSafe(allocator, bytes);
    defer restored.deinit();
    try std.testing.expect(a.equals(&restored));
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

test "bitwiseAndInPlace owns fallback result after scratch exhaustion" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    for (0..5000) |value| _ = try a.add(@intCast(value));

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    for (3000..8000) |value| _ = try b.add(@intCast(value));

    try std.testing.expectEqual(.bitset, a.containers[0].getType());
    try std.testing.expectEqual(.bitset, b.containers[0].getType());

    try a.bitwiseAndInPlace(&b);

    try std.testing.expectEqual(@as(u64, 2000), a.cardinality());
    try std.testing.expectEqual(.array, a.containers[0].getType());
    try std.testing.expect(a.contains(3000));
    try std.testing.expect(a.contains(4999));
    try a.validate();
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

fn buildMixedSerializedBitmap(allocator: std.mem.Allocator) ![]u8 {
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    for (0..128) |i| {
        _ = try bm.add(@intCast(i * 17));
    }
    for (0..5000) |i| {
        _ = try bm.add(65_536 + @as(u32, @intCast(i)) * 13);
    }
    _ = try bm.addRange(131_072 + 100, 131_072 + 2_000);
    _ = try bm.runOptimize();

    return bm.serialize(allocator);
}

fn tryDeserializeCorrupted(allocator: std.mem.Allocator, data: []const u8) !void {
    var restored = RoaringBitmap.deserialize(allocator, data) catch return;
    defer restored.deinit();

    _ = restored.cardinality();
    _ = restored.contains(0);
    var iter = restored.iterator();
    _ = iter.next();
}

fn tryFrozenCorrupted(data: []const u8) !void {
    var frozen = FrozenBitmap.init(data) catch return;
    defer frozen.deinit();

    _ = frozen.cardinality();
    const probes = [_]u32{
        0,
        10,
        65_536,
        65_536 + 13,
        131_072 + 100,
        131_072 + 2_000,
        0xFFFF_FFFF,
    };
    for (probes) |probe| {
        _ = frozen.contains(probe);
    }

    var iter = frozen.iterator();
    while (iter.next()) |_| {}
}

fn serializedDescStart(data: []const u8) ?usize {
    if (data.len < 4) return null;

    const cookie = std.mem.readInt(u32, data[0..4], .little);
    if ((cookie & 0xFFFF) == fmt.SERIAL_COOKIE) {
        const size = ((cookie >> 16) & 0xFFFF) + 1;
        return 4 + ((@as(usize, size) + 7) / 8);
    }
    if (cookie == fmt.SERIAL_COOKIE_NO_RUNCONTAINER) {
        return 8;
    }
    return null;
}

fn serializedSize(data: []const u8) u32 {
    const cookie = std.mem.readInt(u32, data[0..4], .little);
    if ((cookie & 0xFFFF) == fmt.SERIAL_COOKIE) {
        return ((cookie >> 16) & 0xFFFF) + 1;
    }
    return std.mem.readInt(u32, data[4..8], .little);
}

fn serializedOffsetTableStart(data: []const u8) usize {
    return serializedDescStart(data).? + @as(usize, serializedSize(data)) * 4;
}

fn serializedHasOffsetTable(data: []const u8) bool {
    const cookie = std.mem.readInt(u32, data[0..4], .little);
    const has_runs = (cookie & 0xFFFF) == fmt.SERIAL_COOKIE;
    const size = serializedSize(data);
    return !has_runs or size >= fmt.NO_OFFSET_THRESHOLD;
}

fn serializedDataStart(data: []const u8) usize {
    if (serializedHasOffsetTable(data)) {
        return std.mem.readInt(u32, data[serializedOffsetTableStart(data)..][0..4], .little);
    }
    return serializedDescStart(data).? + @as(usize, serializedSize(data)) * 4;
}

fn writeU16LE(data: []u8, offset: usize, value: u16) void {
    data[offset] = @truncate(value);
    data[offset + 1] = @truncate(value >> 8);
}

fn writeU32LE(data: []u8, offset: usize, value: u32) void {
    data[offset] = @truncate(value);
    data[offset + 1] = @truncate(value >> 8);
    data[offset + 2] = @truncate(value >> 16);
    data[offset + 3] = @truncate(value >> 24);
}

fn expectValidateError(allocator: std.mem.Allocator, expected: RoaringBitmap.ValidateError, data: []const u8) !void {
    var bm = try RoaringBitmap.deserialize(allocator, data);
    defer bm.deinit();
    try std.testing.expectError(expected, bm.validate());
}

fn buildTwoContainerSerialized(allocator: std.mem.Allocator) ![]u8 {
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(65_536 + 1);
    _ = try bm.add(131_072 + 1);

    return bm.serialize(allocator);
}

fn buildArraySerialized(allocator: std.mem.Allocator) ![]u8 {
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(10);
    _ = try bm.add(20);
    _ = try bm.add(30);

    return bm.serialize(allocator);
}

fn buildBitsetSerialized(allocator: std.mem.Allocator) ![]u8 {
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    for (0..5000) |i| {
        _ = try bm.add(@as(u32, @intCast(i)) * 13);
    }

    return bm.serialize(allocator);
}

fn buildRunSerialized(allocator: std.mem.Allocator) ![]u8 {
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.addRange(100, 200);
    _ = try bm.addRange(300, 400);
    _ = try bm.runOptimize();

    return bm.serialize(allocator);
}

test "deserialize malformed input smoke" {
    const allocator = std.testing.allocator;

    const bytes = try buildMixedSerializedBitmap(allocator);
    defer allocator.free(bytes);
    try std.testing.expect(bytes.len > 0);

    var prng = std.Random.DefaultPrng.init(MALFORMED_SMOKE_SEED);
    const rng = prng.random();

    for (0..16) |_| {
        var corrupted = try allocator.dupe(u8, bytes);
        defer allocator.free(corrupted);

        const idx = rng.uintLessThan(usize, corrupted.len);
        corrupted[idx] ^= @as(u8, 1) << @intCast(rng.uintLessThan(u4, 8));
        try tryDeserializeCorrupted(allocator, corrupted);
        try tryFrozenCorrupted(corrupted);
    }

    for (0..16) |_| {
        const new_len = rng.uintLessThan(usize, bytes.len);
        try tryDeserializeCorrupted(allocator, bytes[0..new_len]);
        try tryFrozenCorrupted(bytes[0..new_len]);
    }

    if (serializedDescStart(bytes)) |desc_start| {
        if (desc_start + 4 <= bytes.len) {
            const zero_cardinality = try allocator.dupe(u8, bytes);
            defer allocator.free(zero_cardinality);
            writeU16LE(zero_cardinality, desc_start + 2, 0);
            try tryDeserializeCorrupted(allocator, zero_cardinality);
            try tryFrozenCorrupted(zero_cardinality);

            const max_cardinality = try allocator.dupe(u8, bytes);
            defer allocator.free(max_cardinality);
            writeU16LE(max_cardinality, desc_start + 2, 0xFFFF);
            try tryDeserializeCorrupted(allocator, max_cardinality);
            try tryFrozenCorrupted(max_cardinality);
        }
    }
}

test "validate accepts empty bitmap and empty roundtrips" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    try bm.validate();

    const serialized = try bm.serialize(allocator);
    defer allocator.free(serialized);

    var restored = try RoaringBitmap.deserialize(allocator, serialized);
    defer restored.deinit();
    try restored.validate();

    var safe = try RoaringBitmap.deserializeSafe(allocator, serialized);
    defer safe.deinit();
    try safe.validate();

    var safe_owned = try RoaringBitmap.deserializeSafeOwned(allocator, serialized);
    defer safe_owned.deinit();
    try safe_owned.bitmap.validate();
}

test "validate rejects bitmap size beyond allocation" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer {
        bm.size = 0;
        bm.deinit();
    }

    bm.size = @intCast(bm.keys.len + 1);
    try std.testing.expectError(error.BitmapSizeRange, bm.validate());
}

test "validate rejects empty container" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    const ac = try ArrayContainer.init(allocator, 0);
    bm.keys[0] = 0;
    bm.containers[0] = TaggedPtr.initArray(ac);
    bm.size = 1;

    try std.testing.expectError(error.EmptyContainer, bm.validate());
}

test "validate rejects array cardinality range" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    const ac = try ArrayContainer.init(allocator, 0);
    ac.cardinality = ArrayContainer.MAX_CARDINALITY + 1;
    bm.keys[0] = 0;
    bm.containers[0] = TaggedPtr.initArray(ac);
    bm.size = 1;

    try std.testing.expectError(error.ArrayCardinalityRange, bm.validate());
}

test "validate rejects bitset cardinality range" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    const bc = try BitsetContainer.init(allocator);
    bm.keys[0] = 0;
    bm.containers[0] = TaggedPtr.initBitset(bc);
    bm.size = 1;

    try std.testing.expectError(error.BitsetCardinalityRange, bm.validate());
}

test "validate rejects run count beyond allocation" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    const rc = try RunContainer.init(allocator, 0);
    rc.n_runs = rc.capacity + 1;
    bm.keys[0] = 0;
    bm.containers[0] = TaggedPtr.initRun(rc);
    bm.size = 1;

    try std.testing.expectError(error.BitmapSizeRange, bm.validate());
}

test "validate rejects unsorted keys" {
    const allocator = std.testing.allocator;

    const serialized = try buildTwoContainerSerialized(allocator);
    defer allocator.free(serialized);

    const corrupted = try allocator.dupe(u8, serialized);
    defer allocator.free(corrupted);

    writeU16LE(corrupted, serializedDescStart(corrupted).? + 4, 0);
    try expectValidateError(allocator, error.UnsortedKeys, corrupted);
}

test "validate rejects duplicate keys" {
    const allocator = std.testing.allocator;

    const serialized = try buildTwoContainerSerialized(allocator);
    defer allocator.free(serialized);

    const corrupted = try allocator.dupe(u8, serialized);
    defer allocator.free(corrupted);

    writeU16LE(corrupted, serializedDescStart(corrupted).? + 4, 1);
    try expectValidateError(allocator, error.DuplicateKeys, corrupted);
}

test "validate rejects unsorted array values" {
    const allocator = std.testing.allocator;

    const serialized = try buildArraySerialized(allocator);
    defer allocator.free(serialized);

    const corrupted = try allocator.dupe(u8, serialized);
    defer allocator.free(corrupted);

    writeU16LE(corrupted, serializedDataStart(corrupted) + 2, 5);
    try expectValidateError(allocator, error.UnsortedArray, corrupted);
}

test "validate rejects bitset cardinality mismatch" {
    const allocator = std.testing.allocator;

    const serialized = try buildBitsetSerialized(allocator);
    defer allocator.free(serialized);

    const corrupted = try allocator.dupe(u8, serialized);
    defer allocator.free(corrupted);

    corrupted[serializedDataStart(corrupted)] ^= 1;
    try expectValidateError(allocator, error.BitsetCardinalityMismatch, corrupted);
}

test "validate rejects adjacent runs" {
    const allocator = std.testing.allocator;

    const serialized = try buildRunSerialized(allocator);
    defer allocator.free(serialized);

    const corrupted = try allocator.dupe(u8, serialized);
    defer allocator.free(corrupted);

    const data_start = serializedDataStart(corrupted);
    writeU16LE(corrupted, data_start + 2 + 4, 201);
    try expectValidateError(allocator, error.RunOrdering, corrupted);
}

test "validate rejects run cardinality mismatch" {
    const allocator = std.testing.allocator;

    const serialized = try buildRunSerialized(allocator);
    defer allocator.free(serialized);

    const corrupted = try allocator.dupe(u8, serialized);
    defer allocator.free(corrupted);

    writeU16LE(corrupted, serializedDescStart(corrupted).? + 2, 999 - 1);
    try expectValidateError(allocator, error.RunCardinalityMismatch, corrupted);
}

test "validate accepts generated profile roundtrips" {
    const allocator = std.testing.allocator;
    const profiles = [_]test_gen.Profile{ .sparse, .dense, .full, .runs, .single, .boundary };

    for (profiles, 0..) |profile, i| {
        for (&[_]bool{ false, true }) |run_optimize| {
            const optimize_seed: u64 = if (run_optimize) 0x100 else 0;
            var prng = std.Random.DefaultPrng.init(0x5AFE_0000 + @as(u64, @intCast(i)) + optimize_seed);
            const chunks = [_]test_gen.ChunkProfile{.{ .key = 0, .profile = profile }};
            var generated = try test_gen.build(allocator, prng.random(), &chunks, run_optimize);
            defer generated.deinit();

            const serialized = try generated.bm.serialize(allocator);
            defer allocator.free(serialized);

            var restored = try RoaringBitmap.deserialize(allocator, serialized);
            defer restored.deinit();

            try restored.validate();
        }
    }
}

test "deserializeSafe rejects invalid bitmap without leaking" {
    const allocator = std.testing.allocator;

    const serialized = try buildBitsetSerialized(allocator);
    defer allocator.free(serialized);

    const corrupted = try allocator.dupe(u8, serialized);
    defer allocator.free(corrupted);

    corrupted[serializedDataStart(corrupted)] ^= 1;
    try std.testing.expectError(error.BitsetCardinalityMismatch, RoaringBitmap.deserializeSafe(allocator, corrupted));
    try std.testing.expectError(error.BitsetCardinalityMismatch, RoaringBitmap.deserializeSafeOwned(allocator, corrupted));
}

const lazy_repair_test_container_count = 7;

const LazyRepairAuditAllocator = struct {
    backing: std.mem.Allocator,
    fail_index: ?usize = null,
    alloc_index: usize = 0,
    failed: bool = false,
    expected_words: [lazy_repair_test_container_count]usize = [_]usize{0} ** lazy_repair_test_container_count,
    freed_words: [lazy_repair_test_container_count]bool = [_]bool{false} ** lazy_repair_test_container_count,
    free_order: [lazy_repair_test_container_count]usize = [_]usize{0} ** lazy_repair_test_container_count,
    free_order_count: usize = 0,
    expected_count: usize = 0,
    duplicate_free: bool = false,
    unexpected_free: bool = false,

    const Self = @This();

    fn allocator(self: *Self) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn expectBitsets(self: *Self, bitmap: *const RoaringBitmap) !void {
        for (bitmap.containers[0..bitmap.size]) |tagged| {
            if (tagged.getType() != .bitset) continue;
            if (self.expected_count == self.expected_words.len) return error.TooManyBitsets;
            self.expected_words[self.expected_count] = @intFromPtr(tagged.getBitset().words);
            self.expected_count += 1;
        }
    }

    fn expectFreedPrefix(self: *const Self, count: usize) !void {
        try std.testing.expect(!self.duplicate_free);
        try std.testing.expect(!self.unexpected_free);
        for (self.freed_words[0..self.expected_count], 0..) |freed, index| {
            try std.testing.expectEqual(index < count, freed);
        }
    }

    const vtable: std.mem.Allocator.VTable = .{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    };

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        if (!self.failed and self.fail_index == self.alloc_index) {
            self.failed = true;
            return null;
        }
        const result = self.backing.rawAlloc(len, alignment, ret_addr) orelse return null;
        self.alloc_index += 1;
        return result;
    }

    fn resize(
        ctx: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        ret_addr: usize,
    ) bool {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.backing.rawResize(memory, alignment, new_len, ret_addr);
    }

    fn remap(
        ctx: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        ret_addr: usize,
    ) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.backing.rawRemap(memory, alignment, new_len, ret_addr);
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        if (memory.len == BitsetContainer.SIZE_BYTES) {
            const address = @intFromPtr(memory.ptr);
            var found = false;
            for (self.expected_words[0..self.expected_count], 0..) |expected, index| {
                if (address != expected) continue;
                found = true;
                if (self.freed_words[index]) {
                    self.duplicate_free = true;
                } else {
                    self.freed_words[index] = true;
                    if (self.free_order_count == self.free_order.len) {
                        self.duplicate_free = true;
                    } else {
                        self.free_order[self.free_order_count] = index;
                        self.free_order_count += 1;
                    }
                }
                break;
            }
            if (!found) self.unexpected_free = true;
        }
        self.backing.rawFree(memory, alignment, ret_addr);
    }
};

fn makeLazyRepairTestInputs() !struct { left: RoaringBitmap, right: RoaringBitmap } {
    const allocator = std.testing.allocator;
    var left = try RoaringBitmap.init(allocator);
    errdefer left.deinit();
    var right = try RoaringBitmap.init(allocator);
    errdefer right.deinit();

    for (0..lazy_repair_test_container_count) |chunk| {
        const high = @as(u32, @intCast(chunk)) << 16;
        _ = try left.add(high | 1);
        _ = try left.add(high | 3);
        _ = try right.add(high | 2);
        _ = try right.add(high | 4);
    }
    return .{ .left = left, .right = right };
}

fn expectLazyRepairResult(expected: *const RoaringBitmap, actual: *RoaringBitmap) !void {
    try actual.validate();
    try std.testing.expect(actual.equals(expected));
    try std.testing.expectEqual(expected.cardinality(), actual.cardinality());
    for (actual.containers[0..actual.size]) |tagged| {
        try std.testing.expectEqual(TaggedPtr.ContainerType.array, tagged.getType());
    }
}

test "repairAfterLazyWithOptions frees transient bitsets in reverse key order" {
    var inputs = try makeLazyRepairTestInputs();
    defer inputs.left.deinit();
    defer inputs.right.deinit();

    var expected = try inputs.left.lazyOr(std.testing.allocator, &inputs.right, true);
    defer expected.deinit();
    try expected.repairAfterLazy();

    var actual = try inputs.left.lazyOr(std.testing.allocator, &inputs.right, true);
    var audit = LazyRepairAuditAllocator{ .backing = std.testing.allocator };
    try audit.expectBitsets(&actual);
    actual.allocator = audit.allocator();
    defer actual.deinit();

    try actual.repairAfterLazyWithOptions(.{
        .allocator_benefits_from_descending_free_order = true,
    });
    try audit.expectFreedPrefix(lazy_repair_test_container_count);
    try std.testing.expectEqual(lazy_repair_test_container_count, audit.free_order_count);
    for (audit.free_order[0..audit.free_order_count], 0..) |freed_index, index| {
        try std.testing.expectEqual(lazy_repair_test_container_count - index - 1, freed_index);
    }
    try expectLazyRepairResult(&expected, &actual);
}

test "repairAfterLazyWithOptions falls back when scratch allocation fails" {
    var inputs = try makeLazyRepairTestInputs();
    defer inputs.left.deinit();
    defer inputs.right.deinit();

    var expected = try inputs.left.lazyOr(std.testing.allocator, &inputs.right, true);
    defer expected.deinit();
    try expected.repairAfterLazy();

    var actual = try inputs.left.lazyOr(std.testing.allocator, &inputs.right, true);
    var audit = LazyRepairAuditAllocator{
        .backing = std.testing.allocator,
        .fail_index = 0,
    };
    try audit.expectBitsets(&actual);
    try std.testing.expectEqual(lazy_repair_test_container_count, audit.expected_count);
    actual.allocator = audit.allocator();
    defer actual.deinit();

    try actual.repairAfterLazyWithOptions(.{
        .allocator_benefits_from_descending_free_order = true,
    });
    try std.testing.expect(audit.failed);
    try audit.expectFreedPrefix(lazy_repair_test_container_count);
    try expectLazyRepairResult(&expected, &actual);
}

test "repairAfterLazyWithOptions commits a retryable partial repair" {
    const failure_positions = [_]usize{ 0, lazy_repair_test_container_count / 2, lazy_repair_test_container_count - 1 };

    for (failure_positions) |failure_position| {
        var inputs = try makeLazyRepairTestInputs();
        defer inputs.left.deinit();
        defer inputs.right.deinit();

        var expected = try inputs.left.lazyOr(std.testing.allocator, &inputs.right, true);
        defer expected.deinit();
        try expected.repairAfterLazy();

        var actual = try inputs.left.lazyOr(std.testing.allocator, &inputs.right, true);
        var audit = LazyRepairAuditAllocator{
            .backing = std.testing.allocator,
            // Scratch is allocation zero; each completed array conversion uses two allocations.
            .fail_index = 1 + failure_position * 2,
        };
        try audit.expectBitsets(&actual);
        try std.testing.expectEqual(lazy_repair_test_container_count, audit.expected_count);
        actual.allocator = audit.allocator();
        defer actual.deinit();

        try std.testing.expectError(
            error.OutOfMemory,
            actual.repairAfterLazyWithOptions(.{
                .allocator_benefits_from_descending_free_order = true,
            }),
        );
        try std.testing.expect(audit.failed);
        try std.testing.expectEqual(lazy_repair_test_container_count, actual.size);
        try std.testing.expectEqual(@as(i64, -1), actual.cached_cardinality);
        for (actual.keys[0..actual.size], 0..) |key, index| {
            try std.testing.expectEqual(@as(u16, @intCast(index)), key);
            const expected_type: TaggedPtr.ContainerType = if (index < failure_position) .array else .bitset;
            try std.testing.expectEqual(expected_type, actual.containers[index].getType());
        }
        try audit.expectFreedPrefix(failure_position);

        try actual.repairAfterLazyWithOptions(.{
            .allocator_benefits_from_descending_free_order = true,
        });
        try audit.expectFreedPrefix(lazy_repair_test_container_count);
        try expectLazyRepairResult(&expected, &actual);
    }
}
