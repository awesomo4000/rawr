const std = @import("std");
const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;
const Roaring64Bitmap = @import("roaring64.zig").Roaring64Bitmap;
const gen64 = @import("roaring64_test_gen.zig");

const PROPERTY_ITERS: usize = 120;
const PROPERTY_COMPLEX_ITERS: usize = 80;
const PROPERTY_MAX_BUCKETS: usize = 6;

fn randomBitmap(allocator: std.mem.Allocator, rng: std.Random) !gen64.Generated {
    return gen64.randomMixed(allocator, rng, PROPERTY_MAX_BUCKETS);
}

fn toBitmap(allocator: std.mem.Allocator, generated: *const gen64.Generated) !Roaring64Bitmap {
    return gen64.toBitmap(Roaring64Bitmap, allocator, generated);
}

fn expectBitmapEqual(a: *const Roaring64Bitmap, b: *const Roaring64Bitmap) !void {
    if (!a.equals(b)) return error.BitmapsNotEqual;
}

fn expectEmptyPruned(bm: *const Roaring64Bitmap) !void {
    try std.testing.expect(bm.isEmpty());
    try std.testing.expectEqual(@as(u32, 0), bm.size);
    try std.testing.expectEqual(@as(u64, 0), bm.cardinality());
}

test "Roaring64 property laws: binary set algebra" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x6405_a16e_b1);
    const rng = prng.random();

    for (0..PROPERTY_ITERS) |_| {
        try checkBinaryAlgebra(allocator, rng);
    }
}

fn checkBinaryAlgebra(allocator: std.mem.Allocator, rng: std.Random) !void {
    var gen_a = try randomBitmap(allocator, rng);
    defer gen_a.deinit();
    var gen_b = try randomBitmap(allocator, rng);
    defer gen_b.deinit();

    var a = try toBitmap(allocator, &gen_a);
    defer a.deinit();
    var b = try toBitmap(allocator, &gen_b);
    defer b.deinit();

    var empty = try Roaring64Bitmap.init(allocator);
    defer empty.deinit();

    {
        var ab = try a.bitwiseOr(allocator, &b);
        defer ab.deinit();
        var ba = try b.bitwiseOr(allocator, &a);
        defer ba.deinit();
        try expectBitmapEqual(&ab, &ba);

        try std.testing.expectEqual(a.cardinality() + b.cardinality(), ab.cardinality() + a.andCardinality(&b));
    }
    {
        var ab = try a.bitwiseAnd(allocator, &b);
        defer ab.deinit();
        var ba = try b.bitwiseAnd(allocator, &a);
        defer ba.deinit();
        try expectBitmapEqual(&ab, &ba);
        try std.testing.expect(ab.isSubsetOf(&a));
    }
    {
        var ab = try a.bitwiseXor(allocator, &b);
        defer ab.deinit();
        var ba = try b.bitwiseXor(allocator, &a);
        defer ba.deinit();
        try expectBitmapEqual(&ab, &ba);

        var a_minus_b = try a.bitwiseDifference(allocator, &b);
        defer a_minus_b.deinit();
        var b_minus_a = try b.bitwiseDifference(allocator, &a);
        defer b_minus_a.deinit();
        var decomposition = try a_minus_b.bitwiseOr(allocator, &b_minus_a);
        defer decomposition.deinit();
        try expectBitmapEqual(&ab, &decomposition);
    }
    {
        var result = try a.bitwiseOr(allocator, &empty);
        defer result.deinit();
        try expectBitmapEqual(&a, &result);
    }
    {
        var result = try a.bitwiseOr(allocator, &a);
        defer result.deinit();
        try expectBitmapEqual(&a, &result);
    }
    {
        var result = try a.bitwiseAnd(allocator, &a);
        defer result.deinit();
        try expectBitmapEqual(&a, &result);
    }
    {
        var result = try a.bitwiseDifference(allocator, &a);
        defer result.deinit();
        try expectEmptyPruned(&result);
    }
    {
        var result = try a.bitwiseXor(allocator, &a);
        defer result.deinit();
        try expectEmptyPruned(&result);
    }
    {
        var diff = try a.bitwiseDifference(allocator, &b);
        defer diff.deinit();
        try std.testing.expect(diff.isSubsetOf(&a));
    }
    {
        var intersection = try a.bitwiseAnd(allocator, &b);
        defer intersection.deinit();
        var absorbed = try a.bitwiseOr(allocator, &intersection);
        defer absorbed.deinit();
        try expectBitmapEqual(&a, &absorbed);
    }
}

test "Roaring64 property laws: associativity and distributivity" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x6405_a16e_c3);
    const rng = prng.random();

    for (0..PROPERTY_COMPLEX_ITERS) |_| {
        try checkTripleAlgebra(allocator, rng);
    }
}

fn checkTripleAlgebra(allocator: std.mem.Allocator, rng: std.Random) !void {
    var gen_a = try randomBitmap(allocator, rng);
    defer gen_a.deinit();
    var gen_b = try randomBitmap(allocator, rng);
    defer gen_b.deinit();
    var gen_c = try randomBitmap(allocator, rng);
    defer gen_c.deinit();

    var a = try toBitmap(allocator, &gen_a);
    defer a.deinit();
    var b = try toBitmap(allocator, &gen_b);
    defer b.deinit();
    var c = try toBitmap(allocator, &gen_c);
    defer c.deinit();

    {
        var ab = try a.bitwiseOr(allocator, &b);
        defer ab.deinit();
        var ab_c = try ab.bitwiseOr(allocator, &c);
        defer ab_c.deinit();

        var bc = try b.bitwiseOr(allocator, &c);
        defer bc.deinit();
        var a_bc = try a.bitwiseOr(allocator, &bc);
        defer a_bc.deinit();

        try expectBitmapEqual(&ab_c, &a_bc);
    }
    {
        var ab = try a.bitwiseAnd(allocator, &b);
        defer ab.deinit();
        var ab_c = try ab.bitwiseAnd(allocator, &c);
        defer ab_c.deinit();

        var bc = try b.bitwiseAnd(allocator, &c);
        defer bc.deinit();
        var a_bc = try a.bitwiseAnd(allocator, &bc);
        defer a_bc.deinit();

        try expectBitmapEqual(&ab_c, &a_bc);
    }
    {
        var b_or_c = try b.bitwiseOr(allocator, &c);
        defer b_or_c.deinit();
        var lhs = try a.bitwiseAnd(allocator, &b_or_c);
        defer lhs.deinit();

        var a_and_b = try a.bitwiseAnd(allocator, &b);
        defer a_and_b.deinit();
        var a_and_c = try a.bitwiseAnd(allocator, &c);
        defer a_and_c.deinit();
        var rhs = try a_and_b.bitwiseOr(allocator, &a_and_c);
        defer rhs.deinit();

        try expectBitmapEqual(&lhs, &rhs);
    }
}

test "Roaring64 property laws: positional and serialization round-trips" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x6405_5011);
    const rng = prng.random();

    for (0..PROPERTY_ITERS) |_| {
        var generated = try randomBitmap(allocator, rng);
        defer generated.deinit();

        var bm = try toBitmap(allocator, &generated);
        defer bm.deinit();

        const values = try bm.toArrayAlloc(allocator);
        defer allocator.free(values);

        for (values, 0..) |value, index| {
            const rawr_index = bm.getIndex(value) orelse return error.GetIndexMissing;
            try std.testing.expectEqual(@as(u64, @intCast(index)), rawr_index);
            try std.testing.expectEqual(@as(?u64, value), bm.select(rawr_index));
            try std.testing.expectEqual(rawr_index + 1, bm.rank(value));
        }
        try std.testing.expectEqual(@as(?u64, null), bm.select(@intCast(values.len)));

        const bytes = try bm.serialize(allocator);
        defer allocator.free(bytes);
        try std.testing.expectEqual(bytes.len, try bm.serializedSizeInBytes());

        var restored = try Roaring64Bitmap.deserialize(allocator, bytes);
        defer restored.deinit();
        try expectBitmapEqual(&bm, &restored);

        var safe = try Roaring64Bitmap.deserializeSafe(allocator, bytes);
        defer safe.deinit();
        try expectBitmapEqual(&bm, &safe);
    }
}

test "Roaring64 deserializeSafe malformed frame smoke" {
    const allocator = std.testing.allocator;

    try std.testing.expectError(error.InvalidFormat, Roaring64Bitmap.deserializeSafe(allocator, ""));

    var bm = try Roaring64Bitmap.init(allocator);
    defer bm.deinit();
    _ = try bm.add(1);
    _ = try bm.add((@as(u64, 2) << 32) | 3);

    const bytes = try bm.serialize(allocator);
    defer allocator.free(bytes);
    for (0..bytes.len) |len| {
        try std.testing.expectError(error.InvalidFormat, Roaring64Bitmap.deserializeSafe(allocator, bytes[0..len]));
    }

    var overrun_count: [8]u8 = undefined;
    {
        var writer = std.Io.Writer.fixed(&overrun_count);
        try writer.writeInt(u64, @as(u64, std.math.maxInt(u32)) + 1, .little);
    }
    try std.testing.expectError(error.InvalidFormat, Roaring64Bitmap.deserializeSafe(allocator, &overrun_count));

    var sub = try RoaringBitmap.init(allocator);
    defer sub.deinit();
    _ = try sub.add(1);
    const sub_bytes = try sub.serialize(allocator);
    defer allocator.free(sub_bytes);

    const non_ascending = try buildFrame(allocator, &[_]u32{ 2, 1 }, sub_bytes);
    defer allocator.free(non_ascending);
    try std.testing.expectError(error.InvalidFormat, Roaring64Bitmap.deserializeSafe(allocator, non_ascending));

    var empty_sub = try RoaringBitmap.init(allocator);
    defer empty_sub.deinit();
    const empty_bytes = try empty_sub.serialize(allocator);
    defer allocator.free(empty_bytes);

    const empty_bucket = try buildFrame(allocator, &[_]u32{0}, empty_bytes);
    defer allocator.free(empty_bucket);
    try std.testing.expectError(error.InvalidFormat, Roaring64Bitmap.deserializeSafe(allocator, empty_bucket));
}

fn buildFrame(allocator: std.mem.Allocator, keys: []const u32, sub_bitmap: []const u8) ![]u8 {
    var size = try std.math.add(usize, 8, try std.math.mul(usize, keys.len, 4));
    size = try std.math.add(usize, size, try std.math.mul(usize, keys.len, sub_bitmap.len));

    const bytes = try allocator.alloc(u8, size);
    errdefer allocator.free(bytes);

    var writer = std.Io.Writer.fixed(bytes);
    try writer.writeInt(u64, keys.len, .little);
    for (keys) |key| {
        try writer.writeInt(u32, key, .little);
        try writer.writeAll(sub_bitmap);
    }
    return bytes;
}
