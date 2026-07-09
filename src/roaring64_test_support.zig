const std = @import("std");
const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;

pub fn hasRunContainers(bm: anytype) bool {
    for (bm.buckets[0..bm.size]) |*bucket| {
        for (bucket.bm.containers[0..bucket.bm.size]) |container| {
            if (container.getType() == .run) return true;
        }
    }
    return false;
}

pub fn fromValues(comptime Bitmap: type, allocator: std.mem.Allocator, values: []const u64) !Bitmap {
    var bm = try Bitmap.init(allocator);
    errdefer bm.deinit();
    try bm.addMany(values);
    return bm;
}

pub fn expectSerializationRoundTrip(allocator: std.mem.Allocator, bm: anytype) !void {
    const Bitmap = @TypeOf(bm.*);
    const bytes = try bm.serialize(allocator);
    defer allocator.free(bytes);

    try std.testing.expectEqual(bytes.len, try bm.serializedSizeInBytes());

    var restored = try Bitmap.deserialize(allocator, bytes);
    defer restored.deinit();
    try std.testing.expect(bm.equals(&restored));

    var restored_safe = try Bitmap.deserializeSafe(allocator, bytes);
    defer restored_safe.deinit();
    try std.testing.expect(bm.equals(&restored_safe));
}

pub fn expectMalformedFramesRejected(allocator: std.mem.Allocator, bm: anytype) !void {
    const Bitmap = @TypeOf(bm.*);

    try std.testing.expectError(error.InvalidFormat, Bitmap.deserializeSafe(allocator, ""));

    const bytes = try bm.serialize(allocator);
    defer allocator.free(bytes);

    for (0..bytes.len) |len| {
        try std.testing.expectError(error.InvalidFormat, Bitmap.deserializeSafe(allocator, bytes[0..len]));
    }

    var overrun_count: [8]u8 = undefined;
    {
        var writer = std.Io.Writer.fixed(&overrun_count);
        try writer.writeInt(u64, @as(u64, std.math.maxInt(u32)) + 1, .little);
    }
    try std.testing.expectError(error.InvalidFormat, Bitmap.deserializeSafe(allocator, &overrun_count));

    var sub = try RoaringBitmap.init(allocator);
    defer sub.deinit();
    _ = try sub.add(1);
    const sub_bytes = try sub.serialize(allocator);
    defer allocator.free(sub_bytes);

    const non_ascending = try buildFrame(allocator, &[_]u32{ 2, 1 }, sub_bytes);
    defer allocator.free(non_ascending);
    try std.testing.expectError(error.InvalidFormat, Bitmap.deserializeSafe(allocator, non_ascending));

    var empty_sub = try RoaringBitmap.init(allocator);
    defer empty_sub.deinit();
    const empty_sub_bytes = try empty_sub.serialize(allocator);
    defer allocator.free(empty_sub_bytes);

    const empty_bucket = try buildFrame(allocator, &[_]u32{0}, empty_sub_bytes);
    defer allocator.free(empty_bucket);
    try std.testing.expectError(error.InvalidFormat, Bitmap.deserializeSafe(allocator, empty_bucket));
}

pub fn buildFrame(allocator: std.mem.Allocator, keys: []const u32, sub_bitmap: []const u8) ![]u8 {
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
