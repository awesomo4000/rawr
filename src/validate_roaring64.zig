const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");

const Roaring64Bitmap = rawr.Roaring64Bitmap;

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();

    {
        const empty = [_]u64{};
        try validateCase(allocator, "empty", &empty);

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
        try validateCase(allocator, "boundary", &boundary);

        var generated: [512]u64 = undefined;
        fillGeneratedCorpus(&generated);
        try validateCase(allocator, "generated", &generated);

        try validateRangeOps(allocator);
    }

    if (gpa.deinit() != .ok) return error.MemoryLeak;
    std.debug.print("validate64: OK\n", .{});
}

fn validateCase(allocator: std.mem.Allocator, name: []const u8, values: []const u64) !void {
    var rbm = try Roaring64Bitmap.init(allocator);
    defer rbm.deinit();
    try rbm.addMany(values);

    const cr = try buildCRoaring(values);
    defer c.roaring64_bitmap_free(cr);

    const cr_by_one = c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr_by_one);
    for (values) |value| {
        c.roaring64_bitmap_add(cr_by_one, value);
    }
    if (!c.roaring64_bitmap_equals(cr, cr_by_one)) return error.CRoaringBuildMismatch;

    try assertAgreement(allocator, name, &rbm, cr, values);
    try assertPositionalAgreement(&rbm, cr, values);
}

fn buildCRoaring(values: []const u64) !*c.roaring64_bitmap_t {
    const cr = c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
    errdefer c.roaring64_bitmap_free(cr);

    if (values.len != 0) {
        c.roaring64_bitmap_add_many(cr, values.len, @ptrCast(values.ptr));
    }
    return cr;
}

fn assertAgreement(
    allocator: std.mem.Allocator,
    name: []const u8,
    rbm: *const Roaring64Bitmap,
    cr: *const c.roaring64_bitmap_t,
    probes: []const u64,
) !void {
    _ = name;

    const cr_card = c.roaring64_bitmap_get_cardinality(cr);
    if (rbm.cardinality() != cr_card) return error.CardinalityMismatch;
    if (rbm.isEmpty() != c.roaring64_bitmap_is_empty(cr)) return error.EmptyMismatch;

    for (probes) |value| {
        if (rbm.contains(value) != c.roaring64_bitmap_contains(cr, value)) {
            return error.ContainsMismatch;
        }
    }

    if (rbm.isEmpty()) {
        if (rbm.minimum() != null or rbm.maximum() != null) return error.EmptyMinMaxMismatch;
    } else {
        if (rbm.minimum() != c.roaring64_bitmap_minimum(cr)) return error.MinimumMismatch;
        if (rbm.maximum() != c.roaring64_bitmap_maximum(cr)) return error.MaximumMismatch;
    }

    const rawr_values = try rbm.toArrayAlloc(allocator);
    defer allocator.free(rawr_values);
    if (rawr_values.len != cr_card) return error.ArrayCardinalityMismatch;

    const cr_values = try allocator.alloc(u64, rawr_values.len);
    defer allocator.free(cr_values);
    if (cr_values.len != 0) {
        c.roaring64_bitmap_to_uint64_array(cr, @ptrCast(cr_values.ptr));
    }
    if (!std.mem.eql(u64, rawr_values, cr_values)) return error.ArrayMismatch;
}

fn assertPositionalAgreement(
    rbm: *const Roaring64Bitmap,
    cr: *const c.roaring64_bitmap_t,
    probes: []const u64,
) !void {
    for (probes) |value| {
        if (rbm.rank(value) != c.roaring64_bitmap_rank(cr, value)) return error.RankMismatch;

        const rawr_index = rbm.getIndex(value);
        var cr_index: u64 = undefined;
        const cr_present = c.roaring64_bitmap_get_index(cr, value, &cr_index);
        if ((rawr_index != null) != cr_present) return error.GetIndexPresenceMismatch;
        if (rawr_index) |idx| {
            if (idx != cr_index) return error.GetIndexMismatch;
        }
    }

    const card = rbm.cardinality();
    if (card == 0) {
        if (rbm.select(0) != null) return error.SelectMismatch;
        var cr_value: u64 = undefined;
        if (c.roaring64_bitmap_select(cr, 0, &cr_value)) return error.SelectPresenceMismatch;
        return;
    }

    const ranks = [_]u64{ 0, card / 2, card - 1, card };
    for (ranks) |rank| {
        const rawr_value = rbm.select(rank);
        var cr_value: u64 = undefined;
        const cr_present = c.roaring64_bitmap_select(cr, rank, &cr_value);
        if ((rawr_value != null) != cr_present) return error.SelectPresenceMismatch;
        if (rawr_value) |value| {
            if (value != cr_value) return error.SelectMismatch;
        }
    }
}

fn validateRangeOps(allocator: std.mem.Allocator) !void {
    var rbm = try Roaring64Bitmap.init(allocator);
    defer rbm.deinit();

    const cr = c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr);

    try applyAddRange(allocator, &rbm, cr, (@as(u64, 3) << 32) | 10, (@as(u64, 3) << 32) | 20);
    try applyAddRange(allocator, &rbm, cr, (@as(u64, 4) << 32) | 0xffff_fffe, (@as(u64, 5) << 32) | 2);
    try applyAddRange(allocator, &rbm, cr, std.math.maxInt(u64), std.math.maxInt(u64));

    try assertRangeAgreement(&rbm, cr, (@as(u64, 3) << 32) | 10, (@as(u64, 3) << 32) | 20);
    try assertRangeAgreement(&rbm, cr, (@as(u64, 4) << 32) | 0xffff_fffe, (@as(u64, 5) << 32) | 2);
    try assertRangeAgreement(&rbm, cr, std.math.maxInt(u64), std.math.maxInt(u64));
    try assertRangeAgreement(&rbm, cr, (@as(u64, 6) << 32), (@as(u64, 6) << 32) | 10);

    try applyRemoveRange(allocator, &rbm, cr, (@as(u64, 3) << 32) | 12, (@as(u64, 3) << 32) | 18);
    try assertRangeAgreement(&rbm, cr, (@as(u64, 3) << 32) | 10, (@as(u64, 3) << 32) | 20);

    try applyRemoveRange(allocator, &rbm, cr, std.math.maxInt(u64), std.math.maxInt(u64));
    try assertRangeAgreement(&rbm, cr, std.math.maxInt(u64), std.math.maxInt(u64));
}

fn applyAddRange(
    allocator: std.mem.Allocator,
    rbm: *Roaring64Bitmap,
    cr: *c.roaring64_bitmap_t,
    lo: u64,
    hi: u64,
) !void {
    try rbm.addRange(lo, hi);
    c.roaring64_bitmap_add_range_closed(cr, lo, hi);
    const probes = [_]u64{ lo, hi };
    try assertAgreement(allocator, "range_add", rbm, cr, &probes);
}

fn applyRemoveRange(
    allocator: std.mem.Allocator,
    rbm: *Roaring64Bitmap,
    cr: *c.roaring64_bitmap_t,
    lo: u64,
    hi: u64,
) !void {
    try rbm.removeRange(lo, hi);
    c.roaring64_bitmap_remove_range_closed(cr, lo, hi);
    const probes = [_]u64{ lo, hi };
    try assertAgreement(allocator, "range_remove", rbm, cr, &probes);
}

fn assertRangeAgreement(rbm: *const Roaring64Bitmap, cr: *const c.roaring64_bitmap_t, lo: u64, hi: u64) !void {
    if (rbm.rangeCardinality(lo, hi) != c.roaring64_bitmap_range_closed_cardinality(cr, lo, hi)) {
        return error.RangeCardinalityMismatch;
    }
    if (rbm.containsRange(lo, hi) != cContainsRangeClosed(cr, lo, hi)) {
        return error.ContainsRangeMismatch;
    }
}

fn cContainsRangeClosed(cr: *const c.roaring64_bitmap_t, lo: u64, hi: u64) bool {
    if (lo > hi) return true;
    if (hi == std.math.maxInt(u64)) {
        return c.roaring64_bitmap_contains_range(cr, lo, hi) and c.roaring64_bitmap_contains(cr, hi);
    }
    return c.roaring64_bitmap_contains_range(cr, lo, hi + 1);
}

fn fillGeneratedCorpus(out: []u64) void {
    for (out, 0..) |*slot, i| {
        const idx: u64 = @intCast(i);
        const hi: u32 = switch (i % 8) {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 17,
            4 => 0x0001_0000,
            5 => 0x7fff_ffff,
            6 => 0x8000_0000,
            else => 0xffff_ffff,
        };
        const lo: u32 = @truncate((idx * 2_654_435_761) ^ (idx << 17) ^ (idx >> 3));
        slot.* = (@as(u64, hi) << 32) | lo;
    }
}
