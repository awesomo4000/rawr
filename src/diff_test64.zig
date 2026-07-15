// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");

const RoaringBitmap = rawr.RoaringBitmap;
const Roaring64Bitmap = rawr.Roaring64Bitmap;
const gen64 = rawr.roaring64_test_gen;
const test_support = rawr.roaring64_test_support;
const oracle = @import("roaring64_oracle.zig");
const RANDOM_SEED: u64 = 0x64d1_ff64_0001;
const RANDOM_ITERS: usize = 1000;
const RANDOM_MAX_BUCKETS: usize = 6;

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();

    {
        try runPerValueAgreement(allocator);
        try runSetOperationMatrix(allocator);
        try runPositionalAgreement(allocator);
        try runRangeAgreement(allocator);
        try runConstructorAgreement(allocator);
        try runConversionAgreement(allocator);
        try runBulkAgreement(allocator);
        try runCompactionAgreement(allocator);
        try runClearAgreement(allocator);
        try runRandomizedLoop(allocator);
    }

    if (gpa.deinit() != .ok) return error.MemoryLeak;
    std.debug.print("difftest64: OK\n", .{});
}

fn runPerValueAgreement(allocator: std.mem.Allocator) !void {
    var rbm = try Roaring64Bitmap.init(allocator);
    defer rbm.deinit();

    const cr = c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr);

    var generated = try deterministicGeneratedCorpus(allocator);
    defer generated.deinit();
    const values = generated.values;
    const probes = [_]u64{
        0,
        1,
        std.math.maxInt(u32),
        @as(u64, 1) << 32,
        (@as(u64, 17) << 32) | 42,
        (@as(u64, 0x8000_0000) << 32) | 9,
        std.math.maxInt(u64),
    };

    try oracle.assertAgreement(allocator, &rbm, cr, &probes);

    for (values, 0..) |value, i| {
        const was_present = c.roaring64_bitmap_contains(cr, value);
        const added = try rbm.add(value);
        c.roaring64_bitmap_add(cr, value);
        if (added != !was_present) return error.AddAgreementMismatch;

        if (i % 31 == 0) {
            try oracle.assertAgreement(allocator, &rbm, cr, &probes);
        }
    }
    try oracle.assertAgreement(allocator, &rbm, cr, &probes);

    for (values, 0..) |value, i| {
        if (i % 3 != 0) continue;

        const removed = try rbm.remove(value);
        const cr_removed = c.roaring64_bitmap_remove_checked(cr, value);
        if (removed != cr_removed) return error.RemoveAgreementMismatch;

        if (i % 39 == 0) {
            try oracle.assertAgreement(allocator, &rbm, cr, &probes);
        }
    }
    try oracle.assertAgreement(allocator, &rbm, cr, &probes);

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

            var a = try test_support.fromValues(Roaring64Bitmap, allocator, a_values);
            defer a.deinit();
            var b = try test_support.fromValues(Roaring64Bitmap, allocator, b_values);
            defer b.deinit();

            const cr_a = try oracle.buildCRoaring(a_values);
            defer c.roaring64_bitmap_free(cr_a);
            const cr_b = try oracle.buildCRoaring(b_values);
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
    var generated = try deterministicGeneratedCorpus(allocator);
    defer generated.deinit();
    const values = generated.values;

    var rbm = try gen64.toBitmap(Roaring64Bitmap, allocator, &generated);
    defer rbm.deinit();

    const cr = try oracle.buildCRoaring(values);
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

    try oracle.assertPositionalAgreement(&rbm, cr, &probes);

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

    try oracle.applyAddRange(allocator, &rbm, cr, (@as(u64, 3) << 32) | 10, (@as(u64, 3) << 32) | 20);
    try oracle.applyAddRange(allocator, &rbm, cr, (@as(u64, 4) << 32) | 0xffff_fffe, (@as(u64, 5) << 32) | 2);
    try oracle.applyAddRange(allocator, &rbm, cr, std.math.maxInt(u64), std.math.maxInt(u64));

    try oracle.assertRangeAgreement(&rbm, cr, (@as(u64, 3) << 32) | 10, (@as(u64, 3) << 32) | 20);
    try oracle.assertRangeAgreement(&rbm, cr, (@as(u64, 4) << 32) | 0xffff_fffe, (@as(u64, 5) << 32) | 2);
    try oracle.assertRangeAgreement(&rbm, cr, (@as(u64, 6) << 32), (@as(u64, 6) << 32) | 10);
    try oracle.assertRangeAgreement(&rbm, cr, std.math.maxInt(u64), std.math.maxInt(u64));
    try oracle.assertFlipAgreement(allocator, &rbm, cr, (@as(u64, 3) << 32) | 11, (@as(u64, 3) << 32) | 21);
    try oracle.assertFlipAgreement(allocator, &rbm, cr, (@as(u64, 4) << 32) | 0xffff_fffd, (@as(u64, 5) << 32) | 3);
    try oracle.assertFlipAgreement(allocator, &rbm, cr, std.math.maxInt(u64), std.math.maxInt(u64));

    try oracle.applyRemoveRange(allocator, &rbm, cr, (@as(u64, 3) << 32) | 12, (@as(u64, 3) << 32) | 18);
    try oracle.assertRangeAgreement(&rbm, cr, (@as(u64, 3) << 32) | 10, (@as(u64, 3) << 32) | 20);

    try oracle.applyRemoveRange(allocator, &rbm, cr, (@as(u64, 4) << 32) | 0xffff_ffff, (@as(u64, 5) << 32));
    try oracle.assertRangeAgreement(&rbm, cr, (@as(u64, 4) << 32) | 0xffff_fffe, (@as(u64, 5) << 32) | 2);
}

fn runConstructorAgreement(allocator: std.mem.Allocator) !void {
    const RangeCase = struct {
        min: u64,
        max: u64,
        step: u64,
    };
    const range_cases = [_]RangeCase{
        .{ .min = 0, .max = 0, .step = 1 },
        .{ .min = 0, .max = 100, .step = 0 },
        .{ .min = (@as(u64, 1) << 32) - 2, .max = (@as(u64, 1) << 32) + 3, .step = 1 },
        .{ .min = (@as(u64, 3) << 32) | 5, .max = (@as(u64, 3) << 32) | 30, .step = 4 },
        .{ .min = std.math.maxInt(u64) - 5, .max = std.math.maxInt(u64), .step = 3 },
    };

    for (range_cases) |case| {
        var rbm = try Roaring64Bitmap.fromRange(allocator, case.min, case.max, case.step);
        defer rbm.deinit();
        const cr = try oracle.buildCRoaringFromRange(case.min, case.max, case.step);
        defer c.roaring64_bitmap_free(cr);

        const probes = [_]u64{ case.min, case.max, std.math.maxInt(u64) };
        try oracle.assertAgreement(allocator, &rbm, cr, &probes);
    }

    const sorted_values = [_]u64{
        0,
        0,
        (@as(u64, 1) << 32) | 1,
        (@as(u64, 1) << 32) | 1,
        (@as(u64, 1) << 32) | 2,
        (@as(u64, 9) << 32) | 7,
        std.math.maxInt(u64),
    };
    var from_sorted = try Roaring64Bitmap.fromSortedSlice(allocator, &sorted_values);
    defer from_sorted.deinit();
    const cr_sorted = try oracle.buildCRoaringOfPtr(&sorted_values);
    defer c.roaring64_bitmap_free(cr_sorted);
    try oracle.assertAgreement(allocator, &from_sorted, cr_sorted, &sorted_values);
    try oracle.assertStatisticsAgreement(&from_sorted, cr_sorted);
    try oracle.assertFrozenAgreement(allocator, &from_sorted);

    const unsorted_values = [_]u64{
        std.math.maxInt(u64),
        4,
        (@as(u64, 3) << 32) | 9,
        4,
        (@as(u64, 1) << 32) | 2,
        (@as(u64, 3) << 32) | 9,
    };
    var mutable_values = unsorted_values;
    var from_slice = try Roaring64Bitmap.fromSlice(allocator, &mutable_values);
    defer from_slice.deinit();
    const cr_slice = try oracle.buildCRoaringOfPtr(&unsorted_values);
    defer c.roaring64_bitmap_free(cr_slice);
    try oracle.assertAgreement(allocator, &from_slice, cr_slice, &unsorted_values);
}

fn runConversionAgreement(allocator: std.mem.Allocator) !void {
    var values = [_]u32{ 0, 1, 17, 65_536, 123_456, std.math.maxInt(u32) };
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

    _ = try rbm.add(@as(u64, 1) << 32);
    if (try rbm.toRoaring32(allocator) != null) return error.ToRoaring32Mismatch;
}

fn runBulkAgreement(allocator: std.mem.Allocator) !void {
    var rbm = try Roaring64Bitmap.init(allocator);
    defer rbm.deinit();
    const cr = c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr);

    var ctx = Roaring64Bitmap.BulkContext.init();
    var cr_ctx: c.roaring64_bulk_context_t = std.mem.zeroes(c.roaring64_bulk_context_t);

    const values = [_]u64{
        0,
        1,
        2,
        (@as(u64, 10) << 32) | 1,
        (@as(u64, 10) << 32) | 2,
        (@as(u64, 10) << 32) | 3,
        std.math.maxInt(u64),
    };
    for (values) |value| {
        try rbm.addBulk(&ctx, value);
        c.roaring64_bitmap_add_bulk(cr, &cr_ctx, value);
    }

    for (values) |value| {
        if (rbm.containsBulk(&ctx, value) != c.roaring64_bitmap_contains_bulk(cr, &cr_ctx, value)) {
            return error.BulkContainsMismatch;
        }
    }

    const removed = [_]u64{ 1, (@as(u64, 10) << 32) | 2, std.math.maxInt(u64) };
    for (removed) |value| {
        try rbm.removeBulk(&ctx, value);
        c.roaring64_bitmap_remove_bulk(cr, &cr_ctx, value);
    }

    try oracle.assertAgreement(allocator, &rbm, cr, &values);
}

fn runCompactionAgreement(allocator: std.mem.Allocator) !void {
    var rbm = try Roaring64Bitmap.init(allocator);
    defer rbm.deinit();

    const cr = c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr);

    try oracle.applyAddRange(allocator, &rbm, cr, (@as(u64, 7) << 32) | 100, (@as(u64, 7) << 32) | 5000);
    try oracle.applyAddRange(allocator, &rbm, cr, (@as(u64, 8) << 32), (@as(u64, 8) << 32) | 12_000);
    const extra_values = [_]u64{
        (@as(u64, 9) << 32) | 3,
        (@as(u64, 9) << 32) | 99,
        (@as(u64, 9) << 32) | 65_535,
    };
    for (extra_values) |value| {
        _ = try rbm.add(value);
        c.roaring64_bitmap_add(cr, value);
    }

    const rawr_has_runs = try rbm.runOptimize();
    const cr_has_runs = c.roaring64_bitmap_run_optimize(cr);
    if (rawr_has_runs != cr_has_runs) return error.RunOptimizeMismatch;
    if (rawr_has_runs != test_support.hasRunContainers(&rbm)) return error.RunContainerMismatch;
    try oracle.assertAgreement(allocator, &rbm, cr, &extra_values);
    try oracle.assertStatisticsAgreement(&rbm, cr);

    const freed = try rbm.shrinkToFit();
    _ = c.roaring64_bitmap_shrink_to_fit(cr);
    if (freed == 0) return error.ShrinkToFitNoop;
    try oracle.assertAgreement(allocator, &rbm, cr, &extra_values);

    const second_freed = try rbm.shrinkToFit();
    if (second_freed != 0) return error.ShrinkToFitNotIdempotent;
}

fn runClearAgreement(allocator: std.mem.Allocator) !void {
    const values = [_]u64{
        0,
        (@as(u64, 1) << 32) | 3,
        (@as(u64, 17) << 32) | 42,
        std.math.maxInt(u64),
    };

    var rbm = try test_support.fromValues(Roaring64Bitmap, allocator, values[0..]);
    defer rbm.deinit();

    const cr = try oracle.buildCRoaring(values[0..]);
    defer c.roaring64_bitmap_free(cr);

    rbm.clearRetainingCapacity();
    c.roaring64_bitmap_clear(cr);

    const probes = [_]u64{ 0, 1, (@as(u64, 17) << 32) | 42, std.math.maxInt(u64) };
    try oracle.assertAgreement(allocator, &rbm, cr, &probes);

    const reused = (@as(u64, 99) << 32) | 1234;
    if (!(try rbm.add(reused))) return error.ClearReuseMismatch;
    c.roaring64_bitmap_add(cr, reused);
    const reused_probes = [_]u64{ reused, std.math.maxInt(u64) };
    try oracle.assertAgreement(allocator, &rbm, cr, &reused_probes);
}

fn runRandomizedLoop(allocator: std.mem.Allocator) !void {
    std.debug.print("difftest64 random seed=0x{x}, iters={d}, max_buckets={d}\n", .{
        RANDOM_SEED,
        RANDOM_ITERS,
        RANDOM_MAX_BUCKETS,
    });

    var prng = std.Random.DefaultPrng.init(RANDOM_SEED);
    const rng = prng.random();

    for (0..RANDOM_ITERS) |i| {
        runRandomIteration(allocator, rng, i) catch |err| {
            std.debug.print("FAIL: difftest64 random iteration {d}, seed=0x{x}: {s}\n", .{
                i,
                RANDOM_SEED,
                @errorName(err),
            });
            return err;
        };
    }
}

fn runRandomIteration(allocator: std.mem.Allocator, rng: std.Random, iteration: usize) !void {
    var gen_a = try gen64.randomMixed(allocator, rng, RANDOM_MAX_BUCKETS);
    defer gen_a.deinit();
    var gen_b = try gen64.randomMixed(allocator, rng, RANDOM_MAX_BUCKETS);
    defer gen_b.deinit();
    var gen_c = try gen64.randomMixed(allocator, rng, RANDOM_MAX_BUCKETS);
    defer gen_c.deinit();

    var a = try gen64.toBitmap(Roaring64Bitmap, allocator, &gen_a);
    defer a.deinit();
    var b = try gen64.toBitmap(Roaring64Bitmap, allocator, &gen_b);
    defer b.deinit();
    var third = try gen64.toBitmap(Roaring64Bitmap, allocator, &gen_c);
    defer third.deinit();

    const cr_a = try oracle.buildCRoaring(gen_a.values);
    defer c.roaring64_bitmap_free(cr_a);
    const cr_b = try oracle.buildCRoaring(gen_b.values);
    defer c.roaring64_bitmap_free(cr_b);
    const cr_c = try oracle.buildCRoaring(gen_c.values);
    defer c.roaring64_bitmap_free(cr_c);

    const probes = randomProbes(iteration);
    try oracle.assertAgreement(allocator, &a, cr_a, &probes);
    try oracle.assertAgreement(allocator, &b, cr_b, &probes);
    try oracle.assertAgreement(allocator, &third, cr_c, &probes);
    try oracle.assertPositionalAgreement(&a, cr_a, &probes);

    try assertCardinalityOpsAgree(&a, &b, cr_a, cr_b);
    try assertPredicatesAgree(&a, &b, cr_a, cr_b);

    const ops = [_]BinarySetOp{ .bitwise_and, .bitwise_or, .bitwise_xor, .bitwise_difference };
    for (ops) |op| {
        try assertOutOfPlaceSetOpAgree(allocator, op, &a, &b, cr_a, cr_b);
        try assertInPlaceSetOpAgree(allocator, op, &a, &b, cr_a, cr_b);
    }

    try assertTripleCompositionAgree(allocator, &a, &b, &third, cr_a, cr_b, cr_c);
    try assertRandomRangeMutationAgree(allocator, iteration, &a, cr_a);
}

fn assertTripleCompositionAgree(
    allocator: std.mem.Allocator,
    a: *const Roaring64Bitmap,
    b: *const Roaring64Bitmap,
    third: *const Roaring64Bitmap,
    cr_a: *const c.roaring64_bitmap_t,
    cr_b: *const c.roaring64_bitmap_t,
    cr_c: *const c.roaring64_bitmap_t,
) !void {
    var b_or_c = try b.bitwiseOr(allocator, third);
    defer b_or_c.deinit();
    var rawr_result = try a.bitwiseAnd(allocator, &b_or_c);
    defer rawr_result.deinit();

    const cr_b_or_c = c.roaring64_bitmap_or(cr_b, cr_c) orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr_b_or_c);
    const cr_result = c.roaring64_bitmap_and(cr_a, cr_b_or_c) orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr_result);

    const no_probes = [_]u64{};
    try oracle.assertAgreement(allocator, &rawr_result, cr_result, &no_probes);
}

fn assertRandomRangeMutationAgree(
    allocator: std.mem.Allocator,
    iteration: usize,
    source: *const Roaring64Bitmap,
    source_cr: *const c.roaring64_bitmap_t,
) !void {
    var rbm = try source.clone(allocator);
    defer rbm.deinit();

    const cr = c.roaring64_bitmap_copy(source_cr) orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr);

    const range = randomRange(iteration);
    try oracle.assertFlipAgreement(allocator, source, source_cr, range.lo, range.hi);

    try oracle.applyAddRange(allocator, &rbm, cr, range.lo, range.hi);
    try oracle.assertRangeAgreement(&rbm, cr, range.lo, range.hi);

    try oracle.applyRemoveRange(allocator, &rbm, cr, range.lo, range.hi);
    try oracle.assertRangeAgreement(&rbm, cr, range.lo, range.hi);
}

fn randomRange(iteration: usize) gen64.Range {
    return switch (iteration % 6) {
        0 => .{ .lo = (@as(u64, 17) << 32) | 100, .hi = (@as(u64, 17) << 32) | 160 },
        1 => .{ .lo = (@as(u64, 18) << 32) | 0x0000_fffc, .hi = (@as(u64, 18) << 32) | 0x0001_0004 },
        2 => .{ .lo = (@as(u64, 2) << 32) | 0xffff_fffc, .hi = (@as(u64, 3) << 32) | 4 },
        3 => .{ .lo = std.math.maxInt(u64), .hi = std.math.maxInt(u64) },
        4 => .{ .lo = 0, .hi = 1 },
        else => .{ .lo = (@as(u64, 0xffff_fffe) << 32) | 0xffff_fffe, .hi = (@as(u64, 0xffff_ffff) << 32) | 1 },
    };
}

fn randomProbes(iteration: usize) [8]u64 {
    return .{
        0,
        1,
        std.math.maxInt(u32),
        @as(u64, 1) << 32,
        (@as(u64, 17) << 32) | @as(u64, @intCast(iteration & 0xffff)),
        (@as(u64, 0x8000_0000) << 32) | 9,
        (@as(u64, 0xffff_fffe) << 32) | 0xffff_ffff,
        std.math.maxInt(u64),
    };
}

fn deterministicGeneratedCorpus(allocator: std.mem.Allocator) !gen64.Generated {
    var prng = std.Random.DefaultPrng.init(0x6400_d1ff);
    const rng = prng.random();
    const specs = [_]gen64.BucketProfile{
        .{ .hi = 0, .profile = .boundary },
        .{ .hi = 1, .profile = .boundary },
        .{ .hi = 2, .profile = .sparse },
        .{ .hi = 17, .profile = .runs },
        .{ .hi = 0x0001_0000, .profile = .sparse },
        .{ .hi = 0x7fff_ffff, .profile = .boundary },
        .{ .hi = 0x8000_0000, .profile = .sparse },
        .{ .hi = 0xffff_fffe, .profile = .boundary },
        .{ .hi = 0xffff_ffff, .profile = .boundary },
    };

    return gen64.build(allocator, rng, &specs);
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
    try oracle.assertJaccardAgreement(a, b, cr_a, cr_b);
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
    try oracle.assertAgreement(allocator, &rawr_result, cr_result, &no_probes);
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
    try oracle.assertAgreement(allocator, &rawr_result, cr_result, &no_probes);
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
