// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");

const RoaringBitmap = rawr.RoaringBitmap;
const N_SPARSE_VALUES = 500_000;
const N_ARRAY_CONTAINERS = 200;
const WARMUP_RUNS = 2;
const TIMED_RUNS = 9;

var sparse_values: [N_SPARSE_VALUES]u32 = undefined;

const Inputs = struct {
    sparse_a: RoaringBitmap,
    sparse_b: RoaringBitmap,
    skewed_a: RoaringBitmap,
    skewed_b: RoaringBitmap,
    cr_sparse_a: *c.roaring_bitmap_t,
    cr_sparse_b: *c.roaring_bitmap_t,
    cr_skewed_a: *c.roaring_bitmap_t,
    cr_skewed_b: *c.roaring_bitmap_t,

    fn deinit(self: *Inputs) void {
        self.sparse_a.deinit();
        self.sparse_b.deinit();
        self.skewed_a.deinit();
        self.skewed_b.deinit();
        c.roaring_bitmap_free(self.cr_sparse_a);
        c.roaring_bitmap_free(self.cr_sparse_b);
        c.roaring_bitmap_free(self.cr_skewed_a);
        c.roaring_bitmap_free(self.cr_skewed_b);
    }
};

const Stats = struct {
    median_ns: u64,
    min_ns: u64,
    max_ns: u64,
};

fn initSparseValues() usize {
    var prng = std.Random.DefaultPrng.init(54321);
    for (sparse_values[0..]) |*value| value.* = prng.random().int(u32);
    std.mem.sort(u32, sparse_values[0..], {}, std.sort.asc(u32));

    var len: usize = 1;
    for (sparse_values[1..]) |value| {
        if (value != sparse_values[len - 1]) {
            sparse_values[len] = value;
            len += 1;
        }
    }
    return len;
}

fn addArrayContainersRawr(bm: *RoaringBitmap, first_key: usize, first_low: usize, cardinality: usize) !void {
    for (first_key..first_key + N_ARRAY_CONTAINERS) |key| {
        const base = @as(u32, @intCast(key)) << 16;
        for (first_low..first_low + cardinality) |low| {
            _ = try bm.add(base | @as(u32, @intCast(low)));
        }
    }
}

fn addArrayContainersCRoaring(bm: *c.roaring_bitmap_t, first_key: usize, first_low: usize, cardinality: usize) void {
    for (first_key..first_key + N_ARRAY_CONTAINERS) |key| {
        const base = @as(u32, @intCast(key)) << 16;
        for (first_low..first_low + cardinality) |low| {
            c.roaring_bitmap_add(bm, base | @as(u32, @intCast(low)));
        }
    }
}

fn buildInputs(allocator: std.mem.Allocator, sparse_len: usize) !Inputs {
    var sparse_a = try RoaringBitmap.init(allocator);
    errdefer sparse_a.deinit();
    var sparse_b = try RoaringBitmap.init(allocator);
    errdefer sparse_b.deinit();
    var skewed_a = try RoaringBitmap.init(allocator);
    errdefer skewed_a.deinit();
    var skewed_b = try RoaringBitmap.init(allocator);
    errdefer skewed_b.deinit();

    const half = sparse_len / 2;
    try sparse_a.addMany(sparse_values[0..half]);
    try sparse_b.addMany(sparse_values[half / 2 .. sparse_len]);
    try addArrayContainersRawr(&skewed_a, 0, 2048, 32);
    try addArrayContainersRawr(&skewed_b, 20, 0, 4096);

    const cr_sparse_a = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    errdefer c.roaring_bitmap_free(cr_sparse_a);
    const cr_sparse_b = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    errdefer c.roaring_bitmap_free(cr_sparse_b);
    const cr_skewed_a = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    errdefer c.roaring_bitmap_free(cr_skewed_a);
    const cr_skewed_b = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    errdefer c.roaring_bitmap_free(cr_skewed_b);

    c.roaring_bitmap_add_many(cr_sparse_a, half, sparse_values[0..half].ptr);
    c.roaring_bitmap_add_many(cr_sparse_b, sparse_len - half / 2, sparse_values[half / 2 .. sparse_len].ptr);
    addArrayContainersCRoaring(cr_skewed_a, 0, 2048, 32);
    addArrayContainersCRoaring(cr_skewed_b, 20, 0, 4096);

    return .{
        .sparse_a = sparse_a,
        .sparse_b = sparse_b,
        .skewed_a = skewed_a,
        .skewed_b = skewed_b,
        .cr_sparse_a = cr_sparse_a,
        .cr_sparse_b = cr_sparse_b,
        .cr_skewed_a = cr_skewed_a,
        .cr_skewed_b = cr_skewed_b,
    };
}

fn expectPortableEqual(result: *const RoaringBitmap, oracle: *const c.roaring_bitmap_t) !void {
    const rawr_bytes = try result.serialize(std.heap.smp_allocator);
    defer std.heap.smp_allocator.free(rawr_bytes);
    const oracle_len = c.roaring_bitmap_portable_size_in_bytes(oracle);
    if (rawr_bytes.len != oracle_len) return error.SerializedSizeMismatch;
    const oracle_bytes = try std.heap.smp_allocator.alloc(u8, oracle_len);
    defer std.heap.smp_allocator.free(oracle_bytes);
    if (c.roaring_bitmap_portable_serialize(oracle, @ptrCast(oracle_bytes.ptr)) != oracle_len) {
        return error.SerializedSizeMismatch;
    }
    if (!std.mem.eql(u8, rawr_bytes, oracle_bytes)) return error.CRoaringMismatch;
}

fn validate(inputs: *const Inputs) !void {
    var rawr_and = try inputs.sparse_a.bitwiseAnd(std.heap.smp_allocator, &inputs.sparse_b);
    defer rawr_and.deinit();
    var libc_and = try inputs.sparse_a.bitwiseAnd(bench_time.cAllocator(), &inputs.sparse_b);
    defer libc_and.deinit();
    if (!rawr_and.equals(&libc_and)) return error.RawrAllocatorMismatch;
    const cr_and = c.roaring_bitmap_and(inputs.cr_sparse_a, inputs.cr_sparse_b) orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(cr_and);
    try expectPortableEqual(&rawr_and, cr_and);

    var rawr_or = try inputs.sparse_a.bitwiseOr(std.heap.smp_allocator, &inputs.sparse_b);
    defer rawr_or.deinit();
    var libc_or = try inputs.sparse_a.bitwiseOr(bench_time.cAllocator(), &inputs.sparse_b);
    defer libc_or.deinit();
    if (!rawr_or.equals(&libc_or)) return error.RawrAllocatorMismatch;
    const cr_or = c.roaring_bitmap_or(inputs.cr_sparse_a, inputs.cr_sparse_b) orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(cr_or);
    try expectPortableEqual(&rawr_or, cr_or);

    const rawr_cardinality = inputs.skewed_a.andCardinality(&inputs.skewed_b);
    const cr_cardinality = c.roaring_bitmap_and_cardinality(inputs.cr_skewed_a, inputs.cr_skewed_b);
    if (rawr_cardinality != cr_cardinality) return error.CardinalityMismatch;
}

fn rawrAnd(inputs: *const Inputs, allocator: std.mem.Allocator) !void {
    var result = try inputs.sparse_a.bitwiseAnd(allocator, &inputs.sparse_b);
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn crAnd(inputs: *const Inputs) !void {
    const result = c.roaring_bitmap_and(inputs.cr_sparse_a, inputs.cr_sparse_b) orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
}

fn rawrOr(inputs: *const Inputs, allocator: std.mem.Allocator) !void {
    var result = try inputs.sparse_a.bitwiseOr(allocator, &inputs.sparse_b);
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn crOr(inputs: *const Inputs) !void {
    const result = c.roaring_bitmap_or(inputs.cr_sparse_a, inputs.cr_sparse_b) orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
}

fn rawrAndCardinality(inputs: *const Inputs) !void {
    const cardinality = inputs.skewed_a.andCardinality(&inputs.skewed_b);
    std.mem.doNotOptimizeAway(cardinality);
}

fn crAndCardinality(inputs: *const Inputs) !void {
    const cardinality = c.roaring_bitmap_and_cardinality(inputs.cr_skewed_a, inputs.cr_skewed_b);
    std.mem.doNotOptimizeAway(cardinality);
}

fn measure(comptime operation: anytype, args: anytype) !Stats {
    for (0..WARMUP_RUNS) |_| try @call(.auto, operation, args);

    var times: [TIMED_RUNS]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        try @call(.auto, operation, args);
        elapsed.* = bench_time.monotonicNanos() - start;
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return .{ .median_ns = times[TIMED_RUNS / 2], .min_ns = times[0], .max_ns = times[TIMED_RUNS - 1] };
}

fn printResult(target: []const u8, variant: []const u8, stats: Stats) void {
    bench_time.print("{s:<24} {s:<12} {d:>10.3} {d:>10.3} {d:>10.3}\n", .{
        target,
        variant,
        @as(f64, @floatFromInt(stats.median_ns)) / 1_000_000.0,
        @as(f64, @floatFromInt(stats.min_ns)) / 1_000_000.0,
        @as(f64, @floatFromInt(stats.max_ns)) / 1_000_000.0,
    });
    bench_time.print("RESULT\t{s}\t{s}\t{d}\n", .{ target, variant, stats.median_ns });
}

pub fn main() !void {
    bench_time.print("Isolated CRoaring parity board\n", .{});
    bench_time.print("==============================\n", .{});
    bench_time.printRunTimestamp();
    bench_time.printBenchEnvironment();
    bench_time.print("sparse N={d}, array containers={d}, warmup={d}, timed={d}\n\n", .{
        N_SPARSE_VALUES,
        N_ARRAY_CONTAINERS,
        WARMUP_RUNS,
        TIMED_RUNS,
    });

    const sparse_len = initSparseValues();
    var inputs = try buildInputs(std.heap.smp_allocator, sparse_len);
    defer inputs.deinit();
    try validate(&inputs);
    bench_time.print("VALIDATION\trawr-smp=rawr-libc=croaring-portable\n\n", .{});
    bench_time.print("{s:<24} {s:<12} {s:>10} {s:>10} {s:>10}\n", .{ "target", "variant", "median ms", "min ms", "max ms" });
    bench_time.print("{s:-<24} {s:-<12} {s:->10} {s:->10} {s:->10}\n", .{ "", "", "", "", "" });

    const libc = bench_time.cAllocator();
    printResult("sparse-and", "rawr-smp", try measure(rawrAnd, .{ &inputs, std.heap.smp_allocator }));
    printResult("sparse-and", "rawr-libc", try measure(rawrAnd, .{ &inputs, libc }));
    printResult("sparse-and", "croaring", try measure(crAnd, .{&inputs}));
    printResult("sparse-or", "rawr-smp", try measure(rawrOr, .{ &inputs, std.heap.smp_allocator }));
    printResult("sparse-or", "rawr-libc", try measure(rawrOr, .{ &inputs, libc }));
    printResult("sparse-or", "croaring", try measure(crOr, .{&inputs}));
    printResult("skewed-and-cardinality", "rawr", try measure(rawrAndCardinality, .{&inputs}));
    printResult("skewed-and-cardinality", "croaring", try measure(crAndCardinality, .{&inputs}));
}
