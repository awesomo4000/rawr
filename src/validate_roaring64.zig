const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");

const Roaring64Bitmap = rawr.Roaring64Bitmap;
const gen64 = rawr.roaring64_test_gen;
const test_support = rawr.roaring64_test_support;
const oracle = @import("roaring64_oracle.zig");

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();

    {
        const empty = [_]u64{};
        try validateValuesCase(allocator, "empty", &empty);

        const boundary = [_]u64{
            0,
            1,
            std.math.maxInt(u32),
            @as(u64, 1) << 32,
            (@as(u64, 1) << 32) | 1,
            (@as(u64, 7) << 32) | 99,
            (@as(u64, 7) << 32) | std.math.maxInt(u32),
            std.math.maxInt(u64),
        };
        try validateValuesCase(allocator, "boundary", &boundary);

        try validateGeneratedCase(allocator);
        try validateRangeOps(allocator);
    }

    if (gpa.deinit() != .ok) return error.MemoryLeak;
    std.debug.print("validate64: OK\n", .{});
}

fn validateValuesCase(allocator: std.mem.Allocator, name: []const u8, values: []const u64) !void {
    var rbm = try test_support.fromValues(Roaring64Bitmap, allocator, values);
    defer rbm.deinit();

    try validateBitmapCase(allocator, name, &rbm, values);
}

fn validateGeneratedCase(allocator: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(0x6400_1006);
    const rng = prng.random();
    const specs = [_]gen64.BucketProfile{
        .{ .hi = 0, .profile = .boundary },
        .{ .hi = 1, .profile = .boundary },
        .{ .hi = 2, .profile = .sparse },
        .{ .hi = 17, .profile = .runs },
        .{ .hi = 0x0001_0000, .profile = .sparse },
        .{ .hi = 0x7fff_ffff, .profile = .boundary },
        .{ .hi = 0x8000_0000, .profile = .dense },
        .{ .hi = 0xffff_ffff, .profile = .boundary },
    };

    var generated = try gen64.build(allocator, rng, &specs);
    defer generated.deinit();

    var rbm = try gen64.toBitmap(Roaring64Bitmap, allocator, &generated);
    defer rbm.deinit();

    try validateBitmapCase(allocator, "generated", &rbm, generated.values);
}

fn validateBitmapCase(
    allocator: std.mem.Allocator,
    name: []const u8,
    rbm: *const Roaring64Bitmap,
    values: []const u64,
) !void {
    _ = name;

    const cr = try oracle.buildCRoaring(values);
    defer c.roaring64_bitmap_free(cr);

    const cr_by_one = c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr_by_one);
    for (values) |value| {
        c.roaring64_bitmap_add(cr_by_one, value);
    }
    if (!c.roaring64_bitmap_equals(cr, cr_by_one)) return error.CRoaringBuildMismatch;

    try oracle.assertAgreement(allocator, rbm, cr, values);
    try oracle.assertPositionalAgreement(rbm, cr, values);
}

fn validateRangeOps(allocator: std.mem.Allocator) !void {
    var rbm = try Roaring64Bitmap.init(allocator);
    defer rbm.deinit();

    const cr = c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr);

    try oracle.applyAddRange(allocator, &rbm, cr, (@as(u64, 3) << 32) | 10, (@as(u64, 3) << 32) | 20);
    try oracle.applyAddRange(allocator, &rbm, cr, (@as(u64, 4) << 32) | 0xffff_fffe, (@as(u64, 5) << 32) | 2);
    try oracle.applyAddRange(allocator, &rbm, cr, std.math.maxInt(u64), std.math.maxInt(u64));

    try oracle.assertRangeAgreement(&rbm, cr, (@as(u64, 3) << 32) | 10, (@as(u64, 3) << 32) | 20);
    try oracle.assertRangeAgreement(&rbm, cr, (@as(u64, 4) << 32) | 0xffff_fffe, (@as(u64, 5) << 32) | 2);
    try oracle.assertRangeAgreement(&rbm, cr, std.math.maxInt(u64), std.math.maxInt(u64));
    try oracle.assertRangeAgreement(&rbm, cr, (@as(u64, 6) << 32), (@as(u64, 6) << 32) | 10);
    try oracle.assertSerializationAgreement(allocator, &rbm, cr);

    try oracle.applyRemoveRange(allocator, &rbm, cr, (@as(u64, 3) << 32) | 12, (@as(u64, 3) << 32) | 18);
    try oracle.assertRangeAgreement(&rbm, cr, (@as(u64, 3) << 32) | 10, (@as(u64, 3) << 32) | 20);

    try oracle.applyRemoveRange(allocator, &rbm, cr, std.math.maxInt(u64), std.math.maxInt(u64));
    try oracle.assertRangeAgreement(&rbm, cr, std.math.maxInt(u64), std.math.maxInt(u64));
}
