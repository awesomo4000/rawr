const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");

const RoaringBitmap = rawr.RoaringBitmap;
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
        try validateConstructorOps(allocator);
        try validateConversionOps(allocator);
        try validateBulkOps(allocator);
        try validateCompactionOps(allocator);
        try validateClearOps(allocator);
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
    try oracle.assertJaccardAgreement(rbm, rbm, cr, cr);
    try oracle.assertFrozenAgreement(allocator, rbm);
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
    try oracle.assertFlipAgreement(allocator, &rbm, cr, (@as(u64, 3) << 32) | 11, (@as(u64, 3) << 32) | 21);
    try oracle.assertFlipAgreement(allocator, &rbm, cr, (@as(u64, 4) << 32) | 0xffff_fffd, (@as(u64, 5) << 32) | 3);
    try oracle.assertFlipAgreement(allocator, &rbm, cr, std.math.maxInt(u64), std.math.maxInt(u64));
    try oracle.assertSerializationAgreement(allocator, &rbm, cr);

    try oracle.applyRemoveRange(allocator, &rbm, cr, (@as(u64, 3) << 32) | 12, (@as(u64, 3) << 32) | 18);
    try oracle.assertRangeAgreement(&rbm, cr, (@as(u64, 3) << 32) | 10, (@as(u64, 3) << 32) | 20);

    try oracle.applyRemoveRange(allocator, &rbm, cr, std.math.maxInt(u64), std.math.maxInt(u64));
    try oracle.assertRangeAgreement(&rbm, cr, std.math.maxInt(u64), std.math.maxInt(u64));
}

fn validateConstructorOps(allocator: std.mem.Allocator) !void {
    const RangeCase = struct {
        min: u64,
        max: u64,
        step: u64,
    };
    const range_cases = [_]RangeCase{
        .{ .min = 10, .max = 10, .step = 1 },
        .{ .min = 10, .max = 20, .step = 0 },
        .{ .min = (@as(u64, 1) << 32) - 1, .max = (@as(u64, 1) << 32) + 2, .step = 1 },
        .{ .min = (@as(u64, 2) << 32) | 5, .max = (@as(u64, 2) << 32) | 40, .step = 7 },
        .{ .min = std.math.maxInt(u64) - 7, .max = std.math.maxInt(u64), .step = 4 },
    };

    for (range_cases) |case| {
        var rbm = try Roaring64Bitmap.fromRange(allocator, case.min, case.max, case.step);
        defer rbm.deinit();
        const cr = try oracle.buildCRoaringFromRange(case.min, case.max, case.step);
        defer c.roaring64_bitmap_free(cr);
        try oracle.assertAgreement(allocator, &rbm, cr, &.{ case.min, case.max });
    }

    const sorted_values = [_]u64{
        0,
        0,
        1,
        (@as(u64, 5) << 32) | 7,
        (@as(u64, 5) << 32) | 7,
        (@as(u64, 5) << 32) | 8,
        std.math.maxInt(u64),
    };
    var from_sorted = try Roaring64Bitmap.fromSortedSlice(allocator, &sorted_values);
    defer from_sorted.deinit();
    const cr_sorted = try oracle.buildCRoaringOfPtr(&sorted_values);
    defer c.roaring64_bitmap_free(cr_sorted);
    try oracle.assertAgreement(allocator, &from_sorted, cr_sorted, &sorted_values);
    try oracle.assertStatisticsAgreement(&from_sorted, cr_sorted);

    const unsorted_values = [_]u64{
        (@as(u64, 3) << 32) | 9,
        1,
        std.math.maxInt(u64),
        1,
        (@as(u64, 3) << 32) | 9,
    };
    var mutable_values = unsorted_values;
    var from_slice = try Roaring64Bitmap.fromSlice(allocator, &mutable_values);
    defer from_slice.deinit();
    const cr_slice = try oracle.buildCRoaringOfPtr(&unsorted_values);
    defer c.roaring64_bitmap_free(cr_slice);
    try oracle.assertAgreement(allocator, &from_slice, cr_slice, &unsorted_values);
}

fn validateConversionOps(allocator: std.mem.Allocator) !void {
    var values = [_]u32{ 0, 1, 65_535, 65_536, std.math.maxInt(u32) };
    var r32 = try RoaringBitmap.fromSlice(allocator, &values);
    defer r32.deinit();

    var rbm = try Roaring64Bitmap.fromRoaring32(allocator, &r32);
    defer rbm.deinit();
    const cr = try oracle.buildCRoaring64From32(&values);
    defer c.roaring64_bitmap_free(cr);

    try oracle.assertAgreement(allocator, &rbm, cr, &.{ 0, std.math.maxInt(u32) });
    var back = (try rbm.toRoaring32(allocator)) orelse return error.ToRoaring32Mismatch;
    defer back.deinit();
    if (!back.equals(&r32)) return error.ToRoaring32Mismatch;
    if (!r32.contains(std.math.maxInt(u32))) return error.FromRoaring32ConsumedSource;

    _ = try rbm.add(@as(u64, 1) << 32);
    if (try rbm.toRoaring32(allocator) != null) return error.ToRoaring32Mismatch;
}

fn validateBulkOps(allocator: std.mem.Allocator) !void {
    var rbm = try Roaring64Bitmap.init(allocator);
    defer rbm.deinit();
    const cr = c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr);

    var ctx = Roaring64Bitmap.BulkContext.init();
    var cr_ctx: c.roaring64_bulk_context_t = std.mem.zeroes(c.roaring64_bulk_context_t);
    const values = [_]u64{
        0,
        1,
        (@as(u64, 2) << 32) | 1,
        (@as(u64, 2) << 32) | 2,
        std.math.maxInt(u64),
    };
    for (values) |value| {
        try rbm.addBulk(&ctx, value);
        c.roaring64_bitmap_add_bulk(cr, &cr_ctx, value);
    }

    try rbm.removeBulk(&ctx, (@as(u64, 2) << 32) | 1);
    c.roaring64_bitmap_remove_bulk(cr, &cr_ctx, (@as(u64, 2) << 32) | 1);

    try oracle.assertAgreement(allocator, &rbm, cr, &values);
}

fn validateCompactionOps(allocator: std.mem.Allocator) !void {
    var rbm = try Roaring64Bitmap.init(allocator);
    defer rbm.deinit();

    const cr = c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr);

    try oracle.applyAddRange(allocator, &rbm, cr, (@as(u64, 11) << 32) | 10, (@as(u64, 11) << 32) | 9000);
    try oracle.applyAddRange(allocator, &rbm, cr, (@as(u64, 12) << 32) | 100, (@as(u64, 12) << 32) | 7000);

    const rawr_has_runs = try rbm.runOptimize();
    const cr_has_runs = c.roaring64_bitmap_run_optimize(cr);
    if (rawr_has_runs != cr_has_runs) return error.RunOptimizeMismatch;
    if (rawr_has_runs != test_support.hasRunContainers(&rbm)) return error.RunContainerMismatch;
    try oracle.assertAgreement(allocator, &rbm, cr, &.{});
    try oracle.assertStatisticsAgreement(&rbm, cr);

    const freed = try rbm.shrinkToFit();
    _ = c.roaring64_bitmap_shrink_to_fit(cr);
    if (freed == 0) return error.ShrinkToFitNoop;
    try oracle.assertAgreement(allocator, &rbm, cr, &.{});

    const second_freed = try rbm.shrinkToFit();
    if (second_freed != 0) return error.ShrinkToFitNotIdempotent;
}

fn validateClearOps(allocator: std.mem.Allocator) !void {
    const values = [_]u64{
        0,
        (@as(u64, 1) << 32) | 1,
        (@as(u64, 3) << 32) | 9,
        std.math.maxInt(u64),
    };

    var rbm = try test_support.fromValues(Roaring64Bitmap, allocator, values[0..]);
    defer rbm.deinit();

    const cr = try oracle.buildCRoaring(values[0..]);
    defer c.roaring64_bitmap_free(cr);

    rbm.clearRetainingCapacity();
    c.roaring64_bitmap_clear(cr);
    try oracle.assertAgreement(allocator, &rbm, cr, values[0..]);

    const value = (@as(u64, 50) << 32) | 99;
    if (!(try rbm.add(value))) return error.ClearReuseMismatch;
    c.roaring64_bitmap_add(cr, value);
    try oracle.assertAgreement(allocator, &rbm, cr, &.{value});
}
