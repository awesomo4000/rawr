const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");

const Roaring64Bitmap = rawr.Roaring64Bitmap;

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();

    {
        try runPerValueAgreement(allocator);
        try runSetOperationMatrix(allocator);
        try runPositionalAgreement(allocator);
        try runRangeAgreement(allocator);
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

const MatrixProfile = enum {
    empty,
    sparse,
    mixed,
};

const BinarySetOp = enum {
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    bitwise_difference,
};

fn runSetOperationMatrix(allocator: std.mem.Allocator) !void {
    const profiles = [_]MatrixProfile{ .empty, .sparse, .mixed };

    for (profiles) |profile_a| {
        for (profiles) |profile_b| {
            var a_buf: [192]u64 = undefined;
            var b_buf: [192]u64 = undefined;
            const a_values = fillMatrixProfile(profile_a, &a_buf);
            const b_values = fillMatrixProfile(profile_b, &b_buf);

            var a = try roaring64FromValues(allocator, a_values);
            defer a.deinit();
            var b = try roaring64FromValues(allocator, b_values);
            defer b.deinit();

            const cr_a = try buildCRoaring(a_values);
            defer c.roaring64_bitmap_free(cr_a);
            const cr_b = try buildCRoaring(b_values);
            defer c.roaring64_bitmap_free(cr_b);

            try assertCardinalityOpsAgree(&a, &b, cr_a, cr_b);
            try assertPredicatesAgree(&a, &b, cr_a, cr_b);

            const ops = [_]BinarySetOp{ .bitwise_and, .bitwise_or, .bitwise_xor, .bitwise_difference };
            for (ops) |op| {
                try assertOutOfPlaceSetOpAgree(allocator, op, &a, &b, cr_a, cr_b);
                try assertInPlaceSetOpAgree(allocator, op, &a, &b, cr_a, cr_b);
            }
        }
    }
}

fn runPositionalAgreement(allocator: std.mem.Allocator) !void {
    var values: [384]u64 = undefined;
    fillGeneratedCorpus(&values);

    var rbm = try roaring64FromValues(allocator, &values);
    defer rbm.deinit();

    const cr = try buildCRoaring(&values);
    defer c.roaring64_bitmap_free(cr);

    const probes = [_]u64{
        0,
        1,
        std.math.maxInt(u32),
        @as(u64, 1) << 32,
        (@as(u64, 17) << 32) | 42,
        (@as(u64, 0x8000_0000) << 32) | 9,
        std.math.maxInt(u64),
    };

    try assertPositionalAgreement(&rbm, cr, &probes);

    const card = rbm.cardinality();
    const ranks = [_]u64{ 0, 1, card / 2, card - 1, card };
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

fn runRangeAgreement(allocator: std.mem.Allocator) !void {
    var rbm = try Roaring64Bitmap.init(allocator);
    defer rbm.deinit();

    const cr = c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr);

    try applyAddRange(allocator, &rbm, cr, (@as(u64, 3) << 32) | 10, (@as(u64, 3) << 32) | 20);
    try applyAddRange(allocator, &rbm, cr, (@as(u64, 4) << 32) | 0xffff_fffe, (@as(u64, 5) << 32) | 2);
    try applyAddRange(allocator, &rbm, cr, std.math.maxInt(u64), std.math.maxInt(u64));

    try assertRangeAgreement(&rbm, cr, (@as(u64, 3) << 32) | 10, (@as(u64, 3) << 32) | 20);
    try assertRangeAgreement(&rbm, cr, (@as(u64, 4) << 32) | 0xffff_fffe, (@as(u64, 5) << 32) | 2);
    try assertRangeAgreement(&rbm, cr, (@as(u64, 6) << 32), (@as(u64, 6) << 32) | 10);
    try assertRangeAgreement(&rbm, cr, std.math.maxInt(u64), std.math.maxInt(u64));

    try applyRemoveRange(allocator, &rbm, cr, (@as(u64, 3) << 32) | 12, (@as(u64, 3) << 32) | 18);
    try assertRangeAgreement(&rbm, cr, (@as(u64, 3) << 32) | 10, (@as(u64, 3) << 32) | 20);

    try applyRemoveRange(allocator, &rbm, cr, (@as(u64, 4) << 32) | 0xffff_ffff, (@as(u64, 5) << 32));
    try assertRangeAgreement(&rbm, cr, (@as(u64, 4) << 32) | 0xffff_fffe, (@as(u64, 5) << 32) | 2);
}

fn assertCardinalityOpsAgree(
    a: *const Roaring64Bitmap,
    b: *const Roaring64Bitmap,
    cr_a: *const c.roaring64_bitmap_t,
    cr_b: *const c.roaring64_bitmap_t,
) !void {
    if (a.andCardinality(b) != c.roaring64_bitmap_and_cardinality(cr_a, cr_b)) return error.AndCardinalityMismatch;
    if (a.orCardinality(b) != c.roaring64_bitmap_or_cardinality(cr_a, cr_b)) return error.OrCardinalityMismatch;
    if (a.xorCardinality(b) != c.roaring64_bitmap_xor_cardinality(cr_a, cr_b)) return error.XorCardinalityMismatch;
    if (a.differenceCardinality(b) != c.roaring64_bitmap_andnot_cardinality(cr_a, cr_b)) return error.DifferenceCardinalityMismatch;
}

fn assertPredicatesAgree(
    a: *const Roaring64Bitmap,
    b: *const Roaring64Bitmap,
    cr_a: *const c.roaring64_bitmap_t,
    cr_b: *const c.roaring64_bitmap_t,
) !void {
    if (a.intersects(b) != c.roaring64_bitmap_intersect(cr_a, cr_b)) return error.IntersectsMismatch;
    if (a.isSubsetOf(b) != c.roaring64_bitmap_is_subset(cr_a, cr_b)) return error.SubsetMismatch;
    if (a.isStrictSubsetOf(b) != c.roaring64_bitmap_is_strict_subset(cr_a, cr_b)) return error.StrictSubsetMismatch;
    if (a.equals(b) != c.roaring64_bitmap_equals(cr_a, cr_b)) return error.EqualsMismatch;
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
    try assertAgreement(allocator, rbm, cr, &probes);
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
    try assertAgreement(allocator, rbm, cr, &probes);
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

fn assertOutOfPlaceSetOpAgree(
    allocator: std.mem.Allocator,
    op: BinarySetOp,
    a: *const Roaring64Bitmap,
    b: *const Roaring64Bitmap,
    cr_a: *const c.roaring64_bitmap_t,
    cr_b: *const c.roaring64_bitmap_t,
) !void {
    var rawr_result = switch (op) {
        .bitwise_and => try a.bitwiseAnd(allocator, b),
        .bitwise_or => try a.bitwiseOr(allocator, b),
        .bitwise_xor => try a.bitwiseXor(allocator, b),
        .bitwise_difference => try a.bitwiseDifference(allocator, b),
    };
    defer rawr_result.deinit();

    const cr_result = switch (op) {
        .bitwise_and => c.roaring64_bitmap_and(cr_a, cr_b) orelse return error.CRoaringAllocFailed,
        .bitwise_or => c.roaring64_bitmap_or(cr_a, cr_b) orelse return error.CRoaringAllocFailed,
        .bitwise_xor => c.roaring64_bitmap_xor(cr_a, cr_b) orelse return error.CRoaringAllocFailed,
        .bitwise_difference => c.roaring64_bitmap_andnot(cr_a, cr_b) orelse return error.CRoaringAllocFailed,
    };
    defer c.roaring64_bitmap_free(cr_result);

    const no_probes = [_]u64{};
    try assertAgreement(allocator, &rawr_result, cr_result, &no_probes);
}

fn assertInPlaceSetOpAgree(
    allocator: std.mem.Allocator,
    op: BinarySetOp,
    a: *const Roaring64Bitmap,
    b: *const Roaring64Bitmap,
    cr_a: *const c.roaring64_bitmap_t,
    cr_b: *const c.roaring64_bitmap_t,
) !void {
    var rawr_result = try a.clone(allocator);
    defer rawr_result.deinit();

    switch (op) {
        .bitwise_and => try rawr_result.bitwiseAndInPlace(b),
        .bitwise_or => try rawr_result.bitwiseOrInPlace(b),
        .bitwise_xor => try rawr_result.bitwiseXorInPlace(b),
        .bitwise_difference => try rawr_result.bitwiseDifferenceInPlace(b),
    }

    const cr_result = c.roaring64_bitmap_copy(cr_a) orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr_result);

    switch (op) {
        .bitwise_and => c.roaring64_bitmap_and_inplace(cr_result, cr_b),
        .bitwise_or => c.roaring64_bitmap_or_inplace(cr_result, cr_b),
        .bitwise_xor => c.roaring64_bitmap_xor_inplace(cr_result, cr_b),
        .bitwise_difference => c.roaring64_bitmap_andnot_inplace(cr_result, cr_b),
    }

    const no_probes = [_]u64{};
    try assertAgreement(allocator, &rawr_result, cr_result, &no_probes);
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

fn roaring64FromValues(allocator: std.mem.Allocator, values: []const u64) !Roaring64Bitmap {
    var bm = try Roaring64Bitmap.init(allocator);
    errdefer bm.deinit();
    try bm.addMany(values);
    return bm;
}

fn buildCRoaring(values: []const u64) !*c.roaring64_bitmap_t {
    const cr = c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
    errdefer c.roaring64_bitmap_free(cr);

    for (values) |value| {
        c.roaring64_bitmap_add(cr, value);
    }
    return cr;
}

fn fillMatrixProfile(profile: MatrixProfile, out: []u64) []const u64 {
    return switch (profile) {
        .empty => out[0..0],
        .sparse => fillSparseProfile(out),
        .mixed => fillMixedProfile(out),
    };
}

fn fillSparseProfile(out: []u64) []const u64 {
    const len: usize = 72;
    for (out[0..len], 0..) |*slot, i| {
        const idx: u64 = @intCast(i);
        const hi: u32 = switch (i % 6) {
            0 => 0,
            1 => 1,
            2 => 17,
            3 => 0x0001_0000,
            4 => 0x8000_0000,
            else => 0xffff_ffff,
        };
        const lo: u32 = @truncate((idx * 97_531) ^ (idx << 19) ^ (idx >> 2));
        slot.* = (@as(u64, hi) << 32) | lo;
    }
    return out[0..len];
}

fn fillMixedProfile(out: []u64) []const u64 {
    const len: usize = 144;
    for (out[0..len], 0..) |*slot, i| {
        const idx: u64 = @intCast(i);
        const hi: u32 = switch (i % 8) {
            0 => 1,
            1 => 2,
            2 => 17,
            3 => 18,
            4 => 0x0001_0000,
            5 => 0x7fff_ffff,
            6 => 0x8000_0000,
            else => 0xffff_fffe,
        };
        const lo: u32 = if (i % 5 == 0)
            @truncate(idx / 5)
        else
            @truncate((idx * 1_103_515_245) ^ (idx << 11) ^ 0xa5a5_a5a5);
        slot.* = (@as(u64, hi) << 32) | lo;
    }
    return out[0..len];
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
