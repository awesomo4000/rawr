const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");

const Roaring64Bitmap = rawr.Roaring64Bitmap;

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();

    {
        try runPerValueAgreement(allocator);
    }

    if (gpa.deinit() != .ok) return error.MemoryLeak;
    std.debug.print("difftest64: OK\n", .{});
}

fn runPerValueAgreement(allocator: std.mem.Allocator) !void {
    var rbm = try Roaring64Bitmap.init(allocator);
    defer rbm.deinit();

    const cr = c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr);

    var values: [384]u64 = undefined;
    fillGeneratedCorpus(&values);
    const probes = [_]u64{
        0,
        1,
        std.math.maxInt(u32),
        @as(u64, 1) << 32,
        (@as(u64, 17) << 32) | 42,
        (@as(u64, 0x8000_0000) << 32) | 9,
        std.math.maxInt(u64),
    };

    try assertAgreement(allocator, &rbm, cr, &probes);

    for (values, 0..) |value, i| {
        const was_present = c.roaring64_bitmap_contains(cr, value);
        const added = try rbm.add(value);
        c.roaring64_bitmap_add(cr, value);
        if (added != !was_present) return error.AddAgreementMismatch;

        if (i % 31 == 0) {
            try assertAgreement(allocator, &rbm, cr, &probes);
        }
    }
    try assertAgreement(allocator, &rbm, cr, &probes);

    for (values, 0..) |value, i| {
        if (i % 3 != 0) continue;

        const removed = try rbm.remove(value);
        const cr_removed = c.roaring64_bitmap_remove_checked(cr, value);
        if (removed != cr_removed) return error.RemoveAgreementMismatch;

        if (i % 39 == 0) {
            try assertAgreement(allocator, &rbm, cr, &probes);
        }
    }
    try assertAgreement(allocator, &rbm, cr, &probes);

    for (probes) |probe| {
        if (rbm.contains(probe) != c.roaring64_bitmap_contains(cr, probe)) {
            return error.ProbeContainsMismatch;
        }
    }
}

fn assertAgreement(
    allocator: std.mem.Allocator,
    rbm: *const Roaring64Bitmap,
    cr: *const c.roaring64_bitmap_t,
    probes: []const u64,
) !void {
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

    var iter = rbm.iterator();
    for (rawr_values) |expected| {
        if (iter.next() != expected) return error.IteratorMismatch;
    }
    if (iter.next() != null) return error.IteratorExtraValue;
}

fn fillGeneratedCorpus(out: []u64) void {
    for (out, 0..) |*slot, i| {
        const idx: u64 = @intCast(i);
        const hi: u32 = switch (i % 9) {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 17,
            4 => 0x0001_0000,
            5 => 0x7fff_ffff,
            6 => 0x8000_0000,
            7 => 0xffff_fffe,
            else => 0xffff_ffff,
        };
        const lo: u32 = @truncate((idx * 1_664_525) ^ (idx << 21) ^ (idx * 1_013_904_223));
        slot.* = (@as(u64, hi) << 32) | lo;
    }
}
