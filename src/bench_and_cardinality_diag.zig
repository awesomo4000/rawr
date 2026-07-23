// SPDX-License-Identifier: MPL-2.0

//! Focused diagnosis for skewed array-array andCardinality performance.

const std = @import("std");
const c = @import("c");
const bitmap_mod = @import("bitmap.zig");
const container = @import("container.zig");
const kernels = @import("array_kernels.zig");
const array_simd = @import("array_simd.zig");
const bench_time = @import("bench_time.zig");

const RoaringBitmap = bitmap_mod.RoaringBitmap;
const max_cardinality = 4096;
const full_container_count = 200;
const full_matching_containers = 180;
const full_expected_cardinality = full_matching_containers * 32;
const full_warmup_runs = 2;
const full_timed_runs = 9;
const kernel_trial_count = 11;
const min_batch_ns = 1 * std.time.ns_per_ms;
const max_batch_iterations = 1 << 24;
const seed = 0x21_00_2026;

const Case = struct {
    name: []const u8,
    small_len: usize,
    large_len: usize,
};

const cases = [_]Case{
    .{ .name = "32x256", .small_len = 32, .large_len = 256 },
    .{ .name = "32x1024", .small_len = 32, .large_len = 1024 },
    .{ .name = "32x2016", .small_len = 32, .large_len = 2016 },
    .{ .name = "32x2048", .small_len = 32, .large_len = 2048 },
    .{ .name = "32x2080", .small_len = 32, .large_len = 2080 },
    .{ .name = "32x4096", .small_len = 32, .large_len = 4096 },
    .{ .name = "8x1024", .small_len = 8, .large_len = 1024 },
    .{ .name = "16x2048", .small_len = 16, .large_len = 2048 },
    .{ .name = "1x4096", .small_len = 1, .large_len = 4096 },
    .{ .name = "8x4096", .small_len = 8, .large_len = 4096 },
    .{ .name = "64x2496", .small_len = 64, .large_len = 2496 },
    .{ .name = "64x2560", .small_len = 64, .large_len = 2560 },
    .{ .name = "64x2624", .small_len = 64, .large_len = 2624 },
};

const Distribution = enum {
    all_hit,
    disjoint,
    mixed,

    fn name(self: Distribution) []const u8 {
        return switch (self) {
            .all_hit => "all-hit",
            .disjoint => "disjoint",
            .mixed => "mixed",
        };
    }
};

const distributions = [_]Distribution{ .all_hit, .disjoint, .mixed };

const Kernel = struct {
    name: []const u8,
    func: *const fn ([]const u16, []const u16) u64,
};

const direct_kernels = [_]Kernel{
    .{ .name = "rawr-gallop", .func = rawrGallop },
    .{ .name = "croaring-gallop", .func = croaringGallop },
    .{ .name = "rawr-dispatch", .func = rawrDispatch },
    .{ .name = "croaring-dispatch", .func = croaringDispatch },
    .{ .name = "rawr-simd", .func = rawrSimd },
    .{ .name = "rawr-merge", .func = rawrMerge },
};

const FullInputs = struct {
    rawr_small: RoaringBitmap,
    rawr_large: RoaringBitmap,
    croaring_small: *c.roaring_bitmap_t,
    croaring_large: *c.roaring_bitmap_t,

    fn deinit(self: *FullInputs) void {
        self.rawr_small.deinit();
        self.rawr_large.deinit();
        c.roaring_bitmap_free(self.croaring_small);
        c.roaring_bitmap_free(self.croaring_large);
    }
};

const FullStats = struct {
    median_ns: u64,
    min_ns: u64,
    max_ns: u64,
};

var small_storage: [max_cardinality]u16 = undefined;
var large_storage: [max_cardinality]u16 = undefined;
var large_present: [1 << 16]bool = undefined;
var small_present: [1 << 16]bool = undefined;

pub fn main() !void {
    bench_time.printBenchEnvironment();
    bench_time.print("Skewed andCardinality diagnosis\n", .{});
    bench_time.print("================================\n", .{});
    bench_time.print("seed=0x{x}; full=200 containers, 180 matching; kernel trials={d}\n\n", .{
        seed,
        kernel_trial_count,
    });

    var full_inputs = try buildFullInputs(std.heap.smp_allocator);
    defer full_inputs.deinit();
    try validateFullInputs(&full_inputs);
    try runFullApi(&full_inputs);

    bench_time.print("\nDirect array-cardinality kernels (batched)\n", .{});
    bench_time.print("case | distribution | kernel | ns/container\n", .{});
    try runDirectSweep();
}

fn buildFullInputs(allocator: std.mem.Allocator) !FullInputs {
    var rawr_small = try RoaringBitmap.init(allocator);
    errdefer rawr_small.deinit();
    var rawr_large = try RoaringBitmap.init(allocator);
    errdefer rawr_large.deinit();

    try addRawrArrayContainers(&rawr_small, 0, 2048, 32);
    try addRawrArrayContainers(&rawr_large, 20, 0, 4096);

    const croaring_small = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    errdefer c.roaring_bitmap_free(croaring_small);
    const croaring_large = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    errdefer c.roaring_bitmap_free(croaring_large);

    addCRoaringArrayContainers(croaring_small, 0, 2048, 32);
    addCRoaringArrayContainers(croaring_large, 20, 0, 4096);

    return .{
        .rawr_small = rawr_small,
        .rawr_large = rawr_large,
        .croaring_small = croaring_small,
        .croaring_large = croaring_large,
    };
}

fn addRawrArrayContainers(
    bitmap: *RoaringBitmap,
    first_key: usize,
    first_low: usize,
    cardinality: usize,
) !void {
    for (first_key..first_key + full_container_count) |key| {
        const base = @as(u32, @intCast(key)) << 16;
        for (first_low..first_low + cardinality) |low| {
            _ = try bitmap.add(base | @as(u32, @intCast(low)));
        }
    }
}

fn addCRoaringArrayContainers(
    bitmap: *c.roaring_bitmap_t,
    first_key: usize,
    first_low: usize,
    cardinality: usize,
) void {
    for (first_key..first_key + full_container_count) |key| {
        const base = @as(u32, @intCast(key)) << 16;
        for (first_low..first_low + cardinality) |low| {
            c.roaring_bitmap_add(bitmap, base | @as(u32, @intCast(low)));
        }
    }
}

fn validateFullInputs(inputs: *const FullInputs) !void {
    if (!rawrAllArrays(&inputs.rawr_small, full_container_count, 32)) {
        return error.RawrSmallRepresentationMismatch;
    }
    if (!rawrAllArrays(&inputs.rawr_large, full_container_count, 4096)) {
        return error.RawrLargeRepresentationMismatch;
    }
    if (!c.rawr_cr_and_card_all_arrays(inputs.croaring_small, full_container_count, 32)) {
        return error.CRoaringSmallRepresentationMismatch;
    }
    if (!c.rawr_cr_and_card_all_arrays(inputs.croaring_large, full_container_count, 4096)) {
        return error.CRoaringLargeRepresentationMismatch;
    }

    const rawr_cardinality = inputs.rawr_small.andCardinality(&inputs.rawr_large);
    const croaring_cardinality = c.roaring_bitmap_and_cardinality(
        inputs.croaring_small,
        inputs.croaring_large,
    );
    if (rawr_cardinality != full_expected_cardinality or
        croaring_cardinality != full_expected_cardinality or
        rawr_cardinality != croaring_cardinality)
    {
        return error.FullCardinalityMismatch;
    }
}

fn rawrAllArrays(bitmap: *const RoaringBitmap, expected_count: usize, expected_cardinality: u32) bool {
    if (bitmap.size != expected_count) return false;
    for (bitmap.containers[0..bitmap.size]) |tagged| {
        switch (container.Container.fromTagged(tagged)) {
            .array => |array| if (array.getCardinality() != expected_cardinality) return false,
            else => return false,
        }
    }
    return true;
}

fn runFullApi(inputs: *const FullInputs) !void {
    const rawr_stats = try measureFull(rawrFullApi, .{inputs});
    const croaring_stats = try measureFull(croaringFullApi, .{inputs});

    bench_time.print("Full API (original 32x4096 all-hit corpus)\n", .{});
    bench_time.print("variant | median ms | min ms | max ms\n", .{});
    printFullResult("rawr", rawr_stats);
    printFullResult("croaring", croaring_stats);
}

fn rawrFullApi(inputs: *const FullInputs) !void {
    const cardinality = inputs.rawr_small.andCardinality(&inputs.rawr_large);
    std.mem.doNotOptimizeAway(cardinality);
}

fn croaringFullApi(inputs: *const FullInputs) !void {
    const cardinality = c.roaring_bitmap_and_cardinality(
        inputs.croaring_small,
        inputs.croaring_large,
    );
    std.mem.doNotOptimizeAway(cardinality);
}

fn measureFull(comptime operation: anytype, args: anytype) !FullStats {
    for (0..full_warmup_runs) |_| try @call(.auto, operation, args);

    var times: [full_timed_runs]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        try @call(.auto, operation, args);
        elapsed.* = bench_time.monotonicNanos() - start;
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return .{
        .median_ns = times[full_timed_runs / 2],
        .min_ns = times[0],
        .max_ns = times[full_timed_runs - 1],
    };
}

fn printFullResult(name: []const u8, stats: FullStats) void {
    bench_time.print("{s:<10} {d:>10.3} {d:>10.3} {d:>10.3}\n", .{
        name,
        nsToMs(stats.median_ns),
        nsToMs(stats.min_ns),
        nsToMs(stats.max_ns),
    });
    bench_time.print("FULL_RESULT\t{s}\t{d}\n", .{ name, stats.median_ns });
}

fn runDirectSweep() !void {
    for (cases, 0..) |case, case_index| {
        for (distributions, 0..) |distribution, distribution_index| {
            const small = small_storage[0..case.small_len];
            const large = large_storage[0..case.large_len];
            const expected = fillCase(distribution, case_index, distribution_index, small, large);
            try validateDirectCase(small, large, expected);

            for (direct_kernels) |kernel| {
                const iterations = calibrateKernel(kernel.func, small, large);
                const median_ps = medianKernelPicoseconds(kernel.func, small, large, iterations);
                bench_time.print("{s} | {s} | {s} | {d:.3}\n", .{
                    case.name,
                    distribution.name(),
                    kernel.name,
                    @as(f64, @floatFromInt(median_ps)) / 1000.0,
                });
                bench_time.print("KERNEL_RESULT\t{s}\t{s}\t{s}\t{d}\n", .{
                    case.name,
                    distribution.name(),
                    kernel.name,
                    median_ps,
                });
            }
        }
    }
}

fn fillCase(
    distribution: Distribution,
    case_index: usize,
    distribution_index: usize,
    small: []u16,
    large: []u16,
) usize {
    switch (distribution) {
        .all_hit => {
            for (large, 0..) |*value, i| value.* = @intCast(i);
            const start = large.len / 2;
            for (small, 0..) |*value, i| value.* = @intCast(start + i);
            return small.len;
        },
        .disjoint => {
            for (large, 0..) |*value, i| value.* = @intCast(i);
            const start = (1 << 16) - small.len;
            for (small, 0..) |*value, i| value.* = @intCast(start + i);
            return 0;
        },
        .mixed => {
            var prng = std.Random.DefaultPrng.init(
                seed ^ (@as(u64, case_index) << 32) ^ @as(u64, distribution_index),
            );
            fillExactSorted(prng.random(), large);
            @memset(&small_present, false);

            const hit_count = (small.len + 1) / 2;
            for (small[0..hit_count], 0..) |*value, i| {
                const index = (i * large.len) / hit_count;
                value.* = large[index];
                small_present[value.*] = true;
            }

            var count = hit_count;
            while (count < small.len) {
                const value = prng.random().int(u16);
                if (!large_present[value] and !small_present[value]) {
                    small[count] = value;
                    small_present[value] = true;
                    count += 1;
                }
            }
            std.mem.sort(u16, small, {}, std.sort.asc(u16));
            return hit_count;
        },
    }
}

fn fillExactSorted(random: std.Random, out: []u16) void {
    @memset(&large_present, false);
    var count: usize = 0;
    while (count < out.len) {
        const value = random.int(u16);
        if (!large_present[value]) {
            large_present[value] = true;
            count += 1;
        }
    }

    count = 0;
    for (large_present, 0..) |present, value| {
        if (present) {
            out[count] = @intCast(value);
            count += 1;
        }
    }
}

fn validateDirectCase(small: []const u16, large: []const u16, expected: usize) !void {
    if (small.len > max_cardinality or large.len > max_cardinality) {
        return error.DirectCaseNotArraySized;
    }
    if (!isStrictlySorted(small) or !isStrictlySorted(large)) {
        return error.DirectCaseNotSorted;
    }

    const reference = kernels.intersectCardMerge(small, large);
    if (reference != expected) return error.DirectReferenceMismatch;
    for (direct_kernels) |kernel| {
        if (kernel.func(small, large) != reference) return error.DirectKernelMismatch;
    }
}

fn isStrictlySorted(values: []const u16) bool {
    for (values[1..], values[0 .. values.len - 1]) |value, previous| {
        if (value <= previous) return false;
    }
    return true;
}

noinline fn rawrGallop(small: []const u16, large: []const u16) u64 {
    return kernels.intersectCardGallop(small, large);
}

noinline fn croaringGallop(small: []const u16, large: []const u16) u64 {
    return @intCast(c.rawr_cr_and_card_gallop(small.ptr, small.len, large.ptr, large.len));
}

noinline fn rawrDispatch(small: []const u16, large: []const u16) u64 {
    return kernels.intersectCard(small, large);
}

noinline fn croaringDispatch(small: []const u16, large: []const u16) u64 {
    return @intCast(c.rawr_cr_and_card_dispatch(small.ptr, small.len, large.ptr, large.len));
}

noinline fn rawrSimd(small: []const u16, large: []const u16) u64 {
    if (comptime array_simd.has_x86_simd) return array_simd.intersectCardX86(small, large);
    if (comptime array_simd.has_neon) return array_simd.intersectCardNeon(small, large);
    return kernels.intersectCardMerge(small, large);
}

noinline fn rawrMerge(small: []const u16, large: []const u16) u64 {
    return kernels.intersectCardMerge(small, large);
}

fn calibrateKernel(func: *const fn ([]const u16, []const u16) u64, small: []const u16, large: []const u16) usize {
    var iterations: usize = 1;
    while (iterations < max_batch_iterations) : (iterations *= 2) {
        const start = bench_time.monotonicNanos();
        runKernelBatch(func, small, large, iterations);
        if (bench_time.monotonicNanos() - start >= min_batch_ns) break;
    }
    return @min(iterations, max_batch_iterations);
}

fn medianKernelPicoseconds(
    func: *const fn ([]const u16, []const u16) u64,
    small: []const u16,
    large: []const u16,
    iterations: usize,
) u64 {
    var times: [kernel_trial_count]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        runKernelBatch(func, small, large, iterations);
        elapsed.* = bench_time.monotonicNanos() - start;
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return times[kernel_trial_count / 2] * 1000 / iterations;
}

noinline fn runKernelBatch(
    func: *const fn ([]const u16, []const u16) u64,
    small: []const u16,
    large: []const u16,
    iterations: usize,
) void {
    for (0..iterations) |_| {
        const cardinality = func(small, large);
        std.mem.doNotOptimizeAway(cardinality);
    }
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / @as(f64, std.time.ns_per_ms);
}
