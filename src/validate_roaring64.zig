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
