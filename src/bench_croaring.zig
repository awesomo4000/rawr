// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const builtin = @import("builtin");
const rawr = @import("rawr");
const RoaringBitmap = rawr.RoaringBitmap;
const c = @import("c");
const bench_time = @import("bench_time.zig");

const allocator = if (builtin.os.tag == .openbsd) bench_time.openbsd_c_allocator else std.heap.smp_allocator;
const libc_allocator = bench_time.cAllocator();

const WARMUP_RUNS = 3;
const BENCH_RUNS = 21;
const N_VALUES = 1_000_000;
const N_RANK_MANY_PROBES = 200_000;
const N_MANY_BITMAPS = 32;
const N_ARRAY_BENCH_CONTAINERS = 200;

pub const ParityImplementation = enum {
    rawr,
    croaring,
};

pub const ParityAllocator = enum {
    none,
    smp,
    libc,
    arena,
};

pub const ParityTiming = enum {
    external,
    internal,
};

pub const ParityRow = enum {
    add_random,
    add_sequential,
    add_many_random,
    add_many_sequential,
    add_range,
    contains_hit,
    contains_miss,
    sparse_and,
    sparse_and_arena,
    dense_and,
    sparse_or,
    sparse_or_arena,
    dense_or,
    lazy_or_repair,
    lazy_or_construction,
    lazy_or_repair_only,
    or_many,
    or_many_heap,
    xor_many,
    array_balanced_and,
    array_balanced_and_cardinality,
    array_balanced_xor,
    array_skewed_and,
    array_skewed_and_cardinality,
    iterate,
    to_array,
    to_array_alloc,
    serialize,
    deserialize,
    deserialize_arena,
    cardinality,
    rank,
    select,
    rank_many,
    range_cardinality_small,
    range_cardinality_large,
    flip,
    clone,
    remove_range,
};

const BenchResult = struct {
    median_ns: u64,
    p25_ns: u64,
    p75_ns: u64,
};

const LazyPhase = enum {
    construction,
    repair,
};

const LazyContext = enum {
    target_only,
    full_init_first,
    full_init_last,
    allocator_prime,
    cache_prime,
};

const Protocol = struct {
    name: []const u8,
    warmup_runs: usize,
    timed_runs: usize,
};

fn benchmark(comptime func: anytype, comptime args: anytype) BenchResult {
    var times: [BENCH_RUNS]u64 = undefined;

    // Warmup
    for (0..WARMUP_RUNS) |_| {
        _ = @call(.auto, func, args);
    }

    // Timed runs
    for (0..BENCH_RUNS) |i| {
        const start = bench_time.monotonicNanos();
        _ = @call(.auto, func, args);
        times[i] = bench_time.monotonicNanos() - start;
    }

    // Sort for percentiles
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));

    return .{
        .p25_ns = times[BENCH_RUNS / 4],
        .median_ns = times[BENCH_RUNS / 2],
        .p75_ns = times[3 * BENCH_RUNS / 4],
    };
}

fn benchmarkInternallyTimed(comptime func: anytype, comptime args: anytype) BenchResult {
    var times: [BENCH_RUNS]u64 = undefined;

    for (0..WARMUP_RUNS) |_| {
        _ = @call(.auto, func, args);
    }

    for (0..BENCH_RUNS) |i| {
        times[i] = @call(.auto, func, args);
    }

    std.mem.sort(u64, &times, {}, std.sort.asc(u64));

    return .{
        .p25_ns = times[BENCH_RUNS / 4],
        .median_ns = times[BENCH_RUNS / 2],
        .p75_ns = times[3 * BENCH_RUNS / 4],
    };
}

fn benchmarkInternallyTimedProtocol(
    comptime func: anytype,
    comptime args: anytype,
    protocol: Protocol,
) BenchResult {
    std.debug.assert(protocol.timed_runs > 0 and protocol.timed_runs <= BENCH_RUNS);
    var times: [BENCH_RUNS]u64 = undefined;

    for (0..protocol.warmup_runs) |_| _ = @call(.auto, func, args);
    for (times[0..protocol.timed_runs]) |*time| time.* = @call(.auto, func, args);
    std.mem.sort(u64, times[0..protocol.timed_runs], {}, std.sort.asc(u64));

    return .{
        .p25_ns = times[protocol.timed_runs / 4],
        .median_ns = times[protocol.timed_runs / 2],
        .p75_ns = times[(3 * protocol.timed_runs) / 4],
    };
}

fn printHeader() void {
    bench_time.print("\n{s:<40} {s:>12} {s:>12} {s:>8}\n", .{ "Operation", "rawr (ms)", "CRoaring", "ratio" });
    bench_time.print("{s:-<40} {s:->12} {s:->12} {s:->8}\n", .{ "", "", "", "" });
}

fn printResult(name: []const u8, rawr_ns: u64, cr_ns: u64) void {
    const rawr_ms = @as(f64, @floatFromInt(rawr_ns)) / 1_000_000.0;
    const cr_ms = @as(f64, @floatFromInt(cr_ns)) / 1_000_000.0;
    const ratio = if (cr_ns > 0) rawr_ms / cr_ms else 0;
    bench_time.print("{s:<40} {d:>12.2} {d:>12.2} {d:>8.2}x\n", .{ name, rawr_ms, cr_ms, ratio });
}

// ============================================================================
// Test data
// ============================================================================

var random_values: [N_VALUES]u32 = undefined;
var sequential_values: [N_VALUES]u32 = undefined;
var sparse_values: [500000]u32 = undefined;
var sparse_len: usize = 0;
// Iterate these large fixed arrays as slices (`values[0..]`), not as array
// values (`values`), so ReleaseFast does not spill multi-megabyte copies to the
// stack on OpenBSD's default 4 MB stack.
var rank_queries: [N_VALUES]u32 = undefined;
var select_queries: [N_VALUES]u32 = undefined;
var rank_many_probes: [N_RANK_MANY_PROBES]u32 = undefined;
var rank_many_out: [N_RANK_MANY_PROBES]u64 = undefined;
var range_query_lo: [N_VALUES]u32 = undefined;
var range_query_hi: [N_VALUES]u32 = undefined;
var range_large_query_lo: [N_VALUES]u32 = undefined;
var range_large_query_hi: [N_VALUES]u32 = undefined;
var sparse_values_initialized = false;

fn initTestData() void {
    var prng = std.Random.DefaultPrng.init(12345);

    for (0..N_VALUES) |i| {
        random_values[i] = prng.random().int(u32);
        sequential_values[i] = @intCast(i);
        rank_queries[i] = @intCast(prng.random().uintLessThan(u32, 500_000));
        select_queries[i] = @intCast(prng.random().uintLessThan(u32, 500_000));
        const range_start = prng.random().uintLessThan(u32, 50_000);
        range_query_lo[i] = range_start;
        range_query_hi[i] = range_start + prng.random().uintLessThan(u32, 1024);

        const large_start = prng.random().uintLessThan(u32, 20_000);
        const large_len = 30_000 + prng.random().uintLessThan(u32, 20_000);
        range_large_query_lo[i] = large_start;
        range_large_query_hi[i] = @min(59_999, large_start + large_len);
    }

    for (0..N_RANK_MANY_PROBES) |i| {
        rank_many_probes[i] = @intCast((i * 500_000) / N_RANK_MANY_PROBES);
    }

    initSparseValues();
}

fn initSparseValues() void {
    if (sparse_values_initialized) return;

    // Sparse values for set operations (across u32 space)
    var prng2 = std.Random.DefaultPrng.init(54321);
    for (0..500000) |i| {
        sparse_values[i] = prng2.random().int(u32);
    }
    std.mem.sort(u32, &sparse_values, {}, std.sort.asc(u32));
    // Dedupe
    sparse_len = 1;
    for (1..500000) |i| {
        if (sparse_values[i] != sparse_values[sparse_len - 1]) {
            sparse_values[sparse_len] = sparse_values[i];
            sparse_len += 1;
        }
    }
    sparse_values_initialized = true;
}

// ============================================================================
// Rawr benchmarks
// ============================================================================

fn benchRawrAddRandom() void {
    benchRawrAddRandomWithAllocator(allocator);
}

fn benchRawrAddRandomWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.init(result_allocator) catch unreachable;
    defer bm.deinit();
    for (random_values[0..]) |v| {
        _ = bm.add(v) catch unreachable;
    }
    std.mem.doNotOptimizeAway(&bm);
}

fn benchRawrAddSequential() void {
    benchRawrAddSequentialWithAllocator(allocator);
}

fn benchRawrAddSequentialWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.init(result_allocator) catch unreachable;
    defer bm.deinit();
    for (sequential_values[0..]) |v| {
        _ = bm.add(v) catch unreachable;
    }
    std.mem.doNotOptimizeAway(&bm);
}

fn benchRawrAddManyRandom() void {
    benchRawrAddManyRandomWithAllocator(allocator);
}

fn benchRawrAddManyRandomWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.init(result_allocator) catch unreachable;
    defer bm.deinit();
    bm.addMany(random_values[0..]) catch unreachable;
    std.mem.doNotOptimizeAway(&bm);
}

fn benchRawrAddManySequential() void {
    benchRawrAddManySequentialWithAllocator(allocator);
}

fn benchRawrAddManySequentialWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.init(result_allocator) catch unreachable;
    defer bm.deinit();
    bm.addMany(sequential_values[0..]) catch unreachable;
    std.mem.doNotOptimizeAway(&bm);
}

fn benchRawrAddRange() void {
    benchRawrAddRangeWithAllocator(allocator);
}

fn benchRawrAddRangeWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.init(result_allocator) catch unreachable;
    defer bm.deinit();
    _ = bm.addRange(0, N_VALUES - 1) catch unreachable;
    std.mem.doNotOptimizeAway(&bm);
}

var rawr_contains_bm: ?RoaringBitmap = null;
var rawr_cardinality_bm_alt: ?RoaringBitmap = null;

fn initRawrContainsBm() void {
    if (rawr_contains_bm != null) return;
    var bm = RoaringBitmap.init(allocator) catch unreachable;
    for (random_values[0..]) |v| {
        _ = bm.add(v) catch unreachable;
    }
    rawr_contains_bm = bm;
}

fn initRawrCardinalityAlt() void {
    if (rawr_cardinality_bm_alt != null) return;
    rawr_cardinality_bm_alt = rawr_contains_bm.?.clone(allocator) catch unreachable;
}

fn benchRawrContainsHit() void {
    const bm = &rawr_contains_bm.?;
    var hits: u32 = 0;
    for (random_values[0..]) |v| {
        if (bm.contains(v)) hits += 1;
    }
    std.mem.doNotOptimizeAway(hits);
}

fn benchRawrContainsMiss() void {
    const bm = &rawr_contains_bm.?;
    var hits: u32 = 0;
    for (random_values[0..]) |v| {
        if (bm.contains(v | 0x80000000)) hits += 1;
    }
    std.mem.doNotOptimizeAway(hits);
}

var rawr_sparse_a: ?RoaringBitmap = null;
var rawr_sparse_b: ?RoaringBitmap = null;
var rawr_array_balanced_a: ?RoaringBitmap = null;
var rawr_array_balanced_b: ?RoaringBitmap = null;
var rawr_array_skewed_a: ?RoaringBitmap = null;
var rawr_array_skewed_b: ?RoaringBitmap = null;
var rawr_many_bms: [N_MANY_BITMAPS]?RoaringBitmap = [_]?RoaringBitmap{null} ** N_MANY_BITMAPS;
var rawr_many_inputs: [N_MANY_BITMAPS]*const RoaringBitmap = undefined;

fn initRawrSparseBitmaps() void {
    if (rawr_sparse_a != null) return;

    var a = RoaringBitmap.init(allocator) catch unreachable;
    var b = RoaringBitmap.init(allocator) catch unreachable;

    const half = sparse_len / 2;
    for (sparse_values[0..half]) |v| {
        _ = a.add(v) catch unreachable;
    }
    for (sparse_values[half / 2 ..]) |v| {
        _ = b.add(v) catch unreachable;
    }

    rawr_sparse_a = a;
    rawr_sparse_b = b;
}

fn initRawrArrayBitmaps() void {
    if (rawr_array_balanced_a != null) return;

    var balanced_a = RoaringBitmap.init(allocator) catch unreachable;
    var balanced_b = RoaringBitmap.init(allocator) catch unreachable;
    var skewed_a = RoaringBitmap.init(allocator) catch unreachable;
    var skewed_b = RoaringBitmap.init(allocator) catch unreachable;

    addArrayContainersRawr(&balanced_a, 0, 0, 2048);
    addArrayContainersRawr(&balanced_b, 20, 1024, 2048);
    addArrayContainersRawr(&skewed_a, 0, 2048, 32);
    addArrayContainersRawr(&skewed_b, 20, 0, 4096);

    rawr_array_balanced_a = balanced_a;
    rawr_array_balanced_b = balanced_b;
    rawr_array_skewed_a = skewed_a;
    rawr_array_skewed_b = skewed_b;
}

fn addArrayContainersRawr(bm: *RoaringBitmap, first_key: usize, first_low: usize, cardinality: usize) void {
    for (first_key..first_key + N_ARRAY_BENCH_CONTAINERS) |key| {
        const base = @as(u32, @intCast(key)) << 16;
        for (first_low..first_low + cardinality) |low| {
            _ = bm.add(base | @as(u32, @intCast(low))) catch unreachable;
        }
    }
}

fn initRawrManyBitmaps() void {
    if (rawr_many_bms[0] != null) return;

    for (0..N_MANY_BITMAPS) |i| {
        var bm = RoaringBitmap.init(allocator) catch unreachable;
        addManyPatternRawr(&bm, i) catch unreachable;
        if (i % 3 == 0) {
            _ = bm.runOptimize() catch unreachable;
        }
        rawr_many_bms[i] = bm;
        rawr_many_inputs[i] = &rawr_many_bms[i].?;
    }
}

fn addManyPatternRawr(bm: *RoaringBitmap, bitmap_idx: usize) !void {
    for (0..6) |chunk| {
        const base: u32 = @as(u32, @intCast(chunk)) << 16;
        switch ((bitmap_idx + chunk) % 4) {
            0 => {
                for (0..128) |j| {
                    const low: u32 = @intCast((j * 521 + bitmap_idx * 17) & 0xffff);
                    _ = try bm.add(base | low);
                }
            },
            1 => {
                for (0..5000) |j| {
                    const low: u32 = @intCast((j * 13 + bitmap_idx * 29) & 0xffff);
                    _ = try bm.add(base | low);
                }
            },
            2 => {
                const start: u32 = @intCast((bitmap_idx * 97) % 20_000);
                _ = try bm.addRange(base | start, base | (start + 12_000));
            },
            else => {
                _ = try bm.add(base);
                _ = try bm.add(base | 1);
                _ = try bm.add(base | 65_534);
                _ = try bm.add(base | 65_535);
            },
        }
    }
}

fn benchRawrAndSparse() void {
    benchRawrAndSparseWithAllocator(allocator);
}

fn benchRawrAndSparseWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    const a = &rawr_sparse_a.?;
    const b = &rawr_sparse_b.?;
    var result = a.bitwiseAnd(result_allocator, b) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrAndArrayBalanced() void {
    benchRawrAndArrayBalancedWithAllocator(allocator);
}

fn benchRawrAndArrayBalancedWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    var result = rawr_array_balanced_a.?.bitwiseAnd(result_allocator, &rawr_array_balanced_b.?) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrAndCardinalityArrayBalanced() void {
    const cardinality = rawr_array_balanced_a.?.andCardinality(&rawr_array_balanced_b.?);
    std.mem.doNotOptimizeAway(cardinality);
}

fn benchRawrXorArrayBalanced() void {
    benchRawrXorArrayBalancedWithAllocator(allocator);
}

fn benchRawrXorArrayBalancedWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    var result = rawr_array_balanced_a.?.bitwiseXor(result_allocator, &rawr_array_balanced_b.?) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrAndArraySkewed() void {
    benchRawrAndArraySkewedWithAllocator(allocator);
}

fn benchRawrAndArraySkewedWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    var result = rawr_array_skewed_a.?.bitwiseAnd(result_allocator, &rawr_array_skewed_b.?) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrAndCardinalityArraySkewed() void {
    const cardinality = rawr_array_skewed_a.?.andCardinality(&rawr_array_skewed_b.?);
    std.mem.doNotOptimizeAway(cardinality);
}

fn benchRawrOrSparse() void {
    benchRawrOrSparseWithAllocator(allocator);
}

fn benchRawrOrSparseWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    const a = &rawr_sparse_a.?;
    const b = &rawr_sparse_b.?;
    var result = a.bitwiseOr(result_allocator, b) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrLazyOrSparseRepair() void {
    benchRawrLazyOrSparseRepairWithAllocator(allocator);
}

fn benchRawrLazyOrSparseRepairWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    const a = &rawr_sparse_a.?;
    const b = &rawr_sparse_b.?;
    var result = a.lazyOr(result_allocator, b, true) catch unreachable;
    defer result.deinit();
    result.repairAfterLazy() catch unreachable;
    std.mem.doNotOptimizeAway(&result);
}

fn timeRawrLazyOrSparse(comptime phase: LazyPhase) u64 {
    return timeRawrLazyOrSparseWithAllocator(phase, allocator);
}

fn timeRawrLazyOrSparseWithAllocator(comptime phase: LazyPhase, comptime result_allocator: std.mem.Allocator) u64 {
    const a = &rawr_sparse_a.?;
    const b = &rawr_sparse_b.?;

    if (phase == .construction) {
        const start = bench_time.monotonicNanos();
        var result = a.lazyOr(result_allocator, b, true) catch unreachable;
        const elapsed = bench_time.monotonicNanos() - start;
        std.mem.doNotOptimizeAway(&result);
        result.deinit();
        return elapsed;
    }

    var result = a.lazyOr(result_allocator, b, true) catch unreachable;
    const start = bench_time.monotonicNanos();
    result.repairAfterLazy() catch unreachable;
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(&result);
    result.deinit();
    return elapsed;
}

fn benchRawrAndSparseArena() void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = &rawr_sparse_a.?;
    const b = &rawr_sparse_b.?;
    var result = a.bitwiseAnd(arena.allocator(), b) catch unreachable;
    // Don't call result.deinit() — arena.deinit() handles cleanup
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrOrSparseArena() void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = &rawr_sparse_a.?;
    const b = &rawr_sparse_b.?;
    var result = a.bitwiseOr(arena.allocator(), b) catch unreachable;
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrOrMany() void {
    benchRawrOrManyWithAllocator(allocator);
}

fn benchRawrOrManyWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    var result = RoaringBitmap.orMany(result_allocator, &rawr_many_inputs) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrOrManyHeap() void {
    benchRawrOrManyHeapWithAllocator(allocator);
}

fn benchRawrOrManyHeapWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    var result = RoaringBitmap.orManyHeap(result_allocator, &rawr_many_inputs) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrXorMany() void {
    benchRawrXorManyWithAllocator(allocator);
}

fn benchRawrXorManyWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    var result = RoaringBitmap.xorMany(result_allocator, &rawr_many_inputs) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

var rawr_dense_a: ?RoaringBitmap = null;
var rawr_dense_b: ?RoaringBitmap = null;
var rawr_bitset_range_bm: ?RoaringBitmap = null;

fn initRawrDenseBitmaps() void {
    if (rawr_dense_a != null) return;

    var a = RoaringBitmap.init(allocator) catch unreachable;
    var b = RoaringBitmap.init(allocator) catch unreachable;

    _ = a.addRange(0, 499999) catch unreachable;
    _ = b.addRange(250000, 749999) catch unreachable;

    rawr_dense_a = a;
    rawr_dense_b = b;
}

fn initRawrBitsetRangeBm() void {
    if (rawr_bitset_range_bm != null) return;

    var bm = RoaringBitmap.init(allocator) catch unreachable;
    var value: u32 = 0;
    while (value < 60_000) : (value += 3) {
        _ = bm.add(value) catch unreachable;
    }

    rawr_bitset_range_bm = bm;
}

fn benchRawrAndDense() void {
    benchRawrAndDenseWithAllocator(allocator);
}

fn benchRawrAndDenseWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    const a = &rawr_dense_a.?;
    const b = &rawr_dense_b.?;
    var result = a.bitwiseAnd(result_allocator, b) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrOrDense() void {
    benchRawrOrDenseWithAllocator(allocator);
}

fn benchRawrOrDenseWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    const a = &rawr_dense_a.?;
    const b = &rawr_dense_b.?;
    var result = a.bitwiseOr(result_allocator, b) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrIterate() void {
    const bm = &rawr_contains_bm.?;
    var count: u64 = 0;
    var sum: u64 = 0;
    var it = bm.iterator();
    while (it.next()) |v| {
        count +%= 1;
        sum +%= v;
    }
    std.mem.doNotOptimizeAway(count);
    std.mem.doNotOptimizeAway(sum);
}

var rawr_to_array_out: ?[]u32 = null;
var cr_to_array_out: ?[]u32 = null;

fn initToArrayBuffers() void {
    initRawrToArrayBuffer();
    initCRoaringToArrayBuffer();
}

fn initRawrToArrayBuffer() void {
    if (rawr_to_array_out != null) return;
    const rawr_card: usize = @intCast(rawr_contains_bm.?.cardinality());
    rawr_to_array_out = allocator.alloc(u32, rawr_card) catch unreachable;
}

fn initCRoaringToArrayBuffer() void {
    if (cr_to_array_out != null) return;
    const cr_card: usize = @intCast(c.roaring_bitmap_get_cardinality(cr_contains_bm.?));
    cr_to_array_out = allocator.alloc(u32, cr_card) catch unreachable;
}

fn benchRawrToArray() void {
    const bm = &rawr_contains_bm.?;
    const written = bm.toArray(rawr_to_array_out.?);
    std.mem.doNotOptimizeAway(written);
    std.mem.doNotOptimizeAway(rawr_to_array_out.?[rawr_to_array_out.?.len - 1]);
}

fn benchCRoaringToArray() void {
    const bm = cr_contains_bm.?;
    c.roaring_bitmap_to_uint32_array(bm, cr_to_array_out.?.ptr);
    std.mem.doNotOptimizeAway(cr_to_array_out.?[cr_to_array_out.?.len - 1]);
}

fn benchRawrToArrayAlloc() void {
    benchRawrToArrayAllocWithAllocator(allocator);
}

fn benchRawrToArrayAllocWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    const bm = &rawr_contains_bm.?;
    const values = bm.toArrayAlloc(result_allocator) catch unreachable;
    defer result_allocator.free(values);
    std.mem.doNotOptimizeAway(values.ptr);
}

fn benchCRoaringToArrayAlloc() void {
    benchCRoaringToArrayAllocWithAllocator(allocator);
}

fn benchCRoaringToArrayAllocWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    const bm = cr_contains_bm.?;
    const card: usize = @intCast(c.roaring_bitmap_get_cardinality(bm));
    const values = result_allocator.alloc(u32, card) catch unreachable;
    defer result_allocator.free(values);
    c.roaring_bitmap_to_uint32_array(bm, values.ptr);
    std.mem.doNotOptimizeAway(values.ptr);
}

var rawr_serialized: ?[]u8 = null;

fn initRawrSerialized() void {
    if (rawr_serialized != null) return;
    const bm = &rawr_contains_bm.?;
    rawr_serialized = RoaringBitmap.serialize(bm, allocator) catch unreachable;
}

fn benchRawrSerialize() void {
    benchRawrSerializeWithAllocator(allocator);
}

fn benchRawrSerializeWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    const bm = &rawr_contains_bm.?;
    const bytes = RoaringBitmap.serialize(bm, result_allocator) catch unreachable;
    defer result_allocator.free(bytes);
    std.mem.doNotOptimizeAway(bytes.ptr);
}

fn benchRawrDeserialize() void {
    benchRawrDeserializeWithAllocator(allocator);
}

fn benchRawrDeserializeWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.deserialize(result_allocator, rawr_serialized.?) catch unreachable;
    defer bm.deinit();
    std.mem.doNotOptimizeAway(&bm);
}

fn benchRawrDeserializeArena() void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var bm = RoaringBitmap.deserialize(arena.allocator(), rawr_serialized.?) catch unreachable;
    // Don't call bm.deinit() — arena.deinit() handles cleanup
    std.mem.doNotOptimizeAway(&bm);
}

fn benchRawrCardinality() void {
    const bm = &rawr_contains_bm.?;
    const card = bm.cardinality();
    std.mem.doNotOptimizeAway(card);
}

var parity_cardinality_selector: usize = 0;

noinline fn benchRawrCardinalityVarying() void {
    parity_cardinality_selector +%= 1;
    const bm = if (parity_cardinality_selector & 1 == 0)
        &rawr_contains_bm.?
    else
        &rawr_cardinality_bm_alt.?;
    const card = bm.cardinality();
    std.mem.doNotOptimizeAway(card);
}

fn benchRawrRankDense() void {
    const bm = &rawr_dense_a.?;
    var total: u64 = 0;
    for (rank_queries[0..]) |query| {
        total +%= bm.rank(query);
    }
    std.mem.doNotOptimizeAway(total);
}

noinline fn rawrSelectForBenchmark(bm: *const RoaringBitmap, query: u32) ?u32 {
    return @call(.always_inline, RoaringBitmap.select, .{ bm, query });
}

fn benchRawrSelectDense() void {
    const bm = &rawr_dense_a.?;
    var total: u64 = 0;
    for (select_queries[0..]) |query| {
        total +%= rawrSelectForBenchmark(bm, query).?;
    }
    std.mem.doNotOptimizeAway(total);
}

fn benchRawrRankManyDense() void {
    const bm = &rawr_dense_a.?;
    bm.rankMany(rank_many_probes[0..], rank_many_out[0..]);
    std.mem.doNotOptimizeAway(rank_many_out[rank_many_out.len - 1]);
}

fn benchRawrFlipWideDense() void {
    benchRawrFlipWideDenseWithAllocator(allocator);
}

fn benchRawrFlipWideDenseWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    const bm = &rawr_dense_a.?;
    var result = bm.flip(result_allocator, 100_000, 650_000) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrRemoveRangeWideDense() void {
    benchRawrRemoveRangeWideDenseWithAllocator(allocator);
}

fn benchRawrCloneDenseWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    const bm = &rawr_dense_a.?;
    var result = bm.clone(result_allocator) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrRemoveRangeWideDenseWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    const bm = &rawr_dense_a.?;
    var result = bm.clone(result_allocator) catch unreachable;
    defer result.deinit();
    const removed = result.removeRange(100_000, 650_000) catch unreachable;
    std.mem.doNotOptimizeAway(removed);
    std.mem.doNotOptimizeAway(&result);
}

noinline fn rawrRangeCardinality(bm: *const RoaringBitmap, lo: u32, hi: u32) u64 {
    return bm.rangeCardinality(lo, hi);
}

fn benchRawrRangeCardinalityBitset() void {
    const bm = &rawr_bitset_range_bm.?;
    var total: u64 = 0;
    for (range_query_lo[0..], range_query_hi[0..]) |lo, hi| {
        total +%= rawrRangeCardinality(bm, lo, hi);
    }
    std.mem.doNotOptimizeAway(total);
}

fn benchRawrRangeCardinalityBitsetLarge() void {
    const bm = &rawr_bitset_range_bm.?;
    var total: u64 = 0;
    for (range_large_query_lo[0..], range_large_query_hi[0..]) |lo, hi| {
        total +%= rawrRangeCardinality(bm, lo, hi);
    }
    std.mem.doNotOptimizeAway(total);
}

// ============================================================================
// CRoaring benchmarks
// ============================================================================

fn benchCRoaringAddRandom() void {
    const bm = c.roaring_bitmap_create() orelse unreachable;
    defer c.roaring_bitmap_free(bm);
    for (random_values[0..]) |v| {
        c.roaring_bitmap_add(bm, v);
    }
    std.mem.doNotOptimizeAway(bm);
}

fn benchCRoaringAddSequential() void {
    const bm = c.roaring_bitmap_create() orelse unreachable;
    defer c.roaring_bitmap_free(bm);
    for (sequential_values[0..]) |v| {
        c.roaring_bitmap_add(bm, v);
    }
    std.mem.doNotOptimizeAway(bm);
}

fn benchCRoaringAddManyRandom() void {
    const bm = c.roaring_bitmap_create() orelse unreachable;
    defer c.roaring_bitmap_free(bm);
    c.roaring_bitmap_add_many(bm, N_VALUES, random_values[0..].ptr);
    std.mem.doNotOptimizeAway(bm);
}

fn benchCRoaringAddManySequential() void {
    const bm = c.roaring_bitmap_create() orelse unreachable;
    defer c.roaring_bitmap_free(bm);
    c.roaring_bitmap_add_many(bm, N_VALUES, sequential_values[0..].ptr);
    std.mem.doNotOptimizeAway(bm);
}

fn benchCRoaringAddRange() void {
    const bm = c.roaring_bitmap_create() orelse unreachable;
    defer c.roaring_bitmap_free(bm);
    c.roaring_bitmap_add_range(bm, 0, N_VALUES);
    std.mem.doNotOptimizeAway(bm);
}

var cr_contains_bm: ?*c.roaring_bitmap_t = null;
var cr_cardinality_bm_alt: ?*c.roaring_bitmap_t = null;

fn initCRoaringContainsBm() void {
    if (cr_contains_bm != null) return;
    const bm = c.roaring_bitmap_create() orelse unreachable;
    for (random_values[0..]) |v| {
        c.roaring_bitmap_add(bm, v);
    }
    cr_contains_bm = bm;
}

fn initCRoaringCardinalityAlt() void {
    if (cr_cardinality_bm_alt != null) return;
    cr_cardinality_bm_alt = c.roaring_bitmap_copy(cr_contains_bm.?) orelse unreachable;
}

fn benchCRoaringContainsHit() void {
    const bm = cr_contains_bm.?;
    var hits: u32 = 0;
    for (random_values[0..]) |v| {
        if (c.roaring_bitmap_contains(bm, v)) hits += 1;
    }
    std.mem.doNotOptimizeAway(hits);
}

fn benchCRoaringContainsMiss() void {
    const bm = cr_contains_bm.?;
    var hits: u32 = 0;
    for (random_values[0..]) |v| {
        if (c.roaring_bitmap_contains(bm, v | 0x80000000)) hits += 1;
    }
    std.mem.doNotOptimizeAway(hits);
}

var cr_sparse_a: ?*c.roaring_bitmap_t = null;
var cr_sparse_b: ?*c.roaring_bitmap_t = null;
var cr_array_balanced_a: ?*c.roaring_bitmap_t = null;
var cr_array_balanced_b: ?*c.roaring_bitmap_t = null;
var cr_array_skewed_a: ?*c.roaring_bitmap_t = null;
var cr_array_skewed_b: ?*c.roaring_bitmap_t = null;
var cr_many_bms: [N_MANY_BITMAPS]?*c.roaring_bitmap_t = [_]?*c.roaring_bitmap_t{null} ** N_MANY_BITMAPS;
var cr_many_inputs: [N_MANY_BITMAPS]*c.roaring_bitmap_t = undefined;

fn initCRoaringSparseBitmaps() void {
    if (cr_sparse_a != null) return;

    const a = c.roaring_bitmap_create() orelse unreachable;
    const b_bm = c.roaring_bitmap_create() orelse unreachable;

    const half = sparse_len / 2;
    for (sparse_values[0..half]) |v| {
        c.roaring_bitmap_add(a, v);
    }
    for (sparse_values[half / 2 ..]) |v| {
        c.roaring_bitmap_add(b_bm, v);
    }

    cr_sparse_a = a;
    cr_sparse_b = b_bm;
}

fn initCRoaringArrayBitmaps() void {
    if (cr_array_balanced_a != null) return;

    const balanced_a = c.roaring_bitmap_create() orelse unreachable;
    const balanced_b = c.roaring_bitmap_create() orelse unreachable;
    const skewed_a = c.roaring_bitmap_create() orelse unreachable;
    const skewed_b = c.roaring_bitmap_create() orelse unreachable;

    addArrayContainersCRoaring(balanced_a, 0, 0, 2048);
    addArrayContainersCRoaring(balanced_b, 20, 1024, 2048);
    addArrayContainersCRoaring(skewed_a, 0, 2048, 32);
    addArrayContainersCRoaring(skewed_b, 20, 0, 4096);

    cr_array_balanced_a = balanced_a;
    cr_array_balanced_b = balanced_b;
    cr_array_skewed_a = skewed_a;
    cr_array_skewed_b = skewed_b;
}

fn addArrayContainersCRoaring(bm: *c.roaring_bitmap_t, first_key: usize, first_low: usize, cardinality: usize) void {
    for (first_key..first_key + N_ARRAY_BENCH_CONTAINERS) |key| {
        const base = @as(u32, @intCast(key)) << 16;
        for (first_low..first_low + cardinality) |low| {
            c.roaring_bitmap_add(bm, base | @as(u32, @intCast(low)));
        }
    }
}

fn initCRoaringManyBitmaps() void {
    if (cr_many_bms[0] != null) return;

    for (0..N_MANY_BITMAPS) |i| {
        const bm = c.roaring_bitmap_create() orelse unreachable;
        addManyPatternCRoaring(bm, i);
        if (i % 3 == 0) {
            _ = c.roaring_bitmap_run_optimize(bm);
        }
        cr_many_bms[i] = bm;
        cr_many_inputs[i] = bm;
    }
}

fn addManyPatternCRoaring(bm: *c.roaring_bitmap_t, bitmap_idx: usize) void {
    for (0..6) |chunk| {
        const base: u32 = @as(u32, @intCast(chunk)) << 16;
        switch ((bitmap_idx + chunk) % 4) {
            0 => {
                for (0..128) |j| {
                    const low: u32 = @intCast((j * 521 + bitmap_idx * 17) & 0xffff);
                    c.roaring_bitmap_add(bm, base | low);
                }
            },
            1 => {
                for (0..5000) |j| {
                    const low: u32 = @intCast((j * 13 + bitmap_idx * 29) & 0xffff);
                    c.roaring_bitmap_add(bm, base | low);
                }
            },
            2 => {
                const start: u32 = @intCast((bitmap_idx * 97) % 20_000);
                c.roaring_bitmap_add_range(bm, base | start, (base | (start + 12_000)) + 1);
            },
            else => {
                c.roaring_bitmap_add(bm, base);
                c.roaring_bitmap_add(bm, base | 1);
                c.roaring_bitmap_add(bm, base | 65_534);
                c.roaring_bitmap_add(bm, base | 65_535);
            },
        }
    }
}

fn benchCRoaringAndSparse() void {
    const a = cr_sparse_a.?;
    const b_bm = cr_sparse_b.?;
    const result = c.roaring_bitmap_and(a, b_bm) orelse unreachable;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
}

fn benchCRoaringAndArrayBalanced() void {
    const result = c.roaring_bitmap_and(cr_array_balanced_a.?, cr_array_balanced_b.?) orelse unreachable;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
}

fn benchCRoaringAndCardinalityArrayBalanced() void {
    const cardinality = c.roaring_bitmap_and_cardinality(cr_array_balanced_a.?, cr_array_balanced_b.?);
    std.mem.doNotOptimizeAway(cardinality);
}

fn benchCRoaringXorArrayBalanced() void {
    const result = c.roaring_bitmap_xor(cr_array_balanced_a.?, cr_array_balanced_b.?) orelse unreachable;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
}

fn benchCRoaringAndArraySkewed() void {
    const result = c.roaring_bitmap_and(cr_array_skewed_a.?, cr_array_skewed_b.?) orelse unreachable;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
}

fn benchCRoaringAndCardinalityArraySkewed() void {
    const cardinality = c.roaring_bitmap_and_cardinality(cr_array_skewed_a.?, cr_array_skewed_b.?);
    std.mem.doNotOptimizeAway(cardinality);
}

fn benchCRoaringOrSparse() void {
    const a = cr_sparse_a.?;
    const b_bm = cr_sparse_b.?;
    const result = c.roaring_bitmap_or(a, b_bm) orelse unreachable;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
}

fn benchCRoaringLazyOrSparseRepair() void {
    const a = cr_sparse_a.?;
    const b_bm = cr_sparse_b.?;
    const result = c.roaring_bitmap_lazy_or(a, b_bm, true) orelse unreachable;
    defer c.roaring_bitmap_free(result);
    c.roaring_bitmap_repair_after_lazy(result);
    std.mem.doNotOptimizeAway(result);
}

fn timeCRoaringLazyOrSparse(comptime phase: LazyPhase) u64 {
    const a = cr_sparse_a.?;
    const b_bm = cr_sparse_b.?;

    if (phase == .construction) {
        const start = bench_time.monotonicNanos();
        const result = c.roaring_bitmap_lazy_or(a, b_bm, true) orelse unreachable;
        const elapsed = bench_time.monotonicNanos() - start;
        std.mem.doNotOptimizeAway(result);
        c.roaring_bitmap_free(result);
        return elapsed;
    }

    const result = c.roaring_bitmap_lazy_or(a, b_bm, true) orelse unreachable;
    const start = bench_time.monotonicNanos();
    c.roaring_bitmap_repair_after_lazy(result);
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(result);
    c.roaring_bitmap_free(result);
    return elapsed;
}

fn benchCRoaringOrMany() void {
    const result = c.roaring_bitmap_or_many(N_MANY_BITMAPS, @ptrCast(&cr_many_inputs)) orelse unreachable;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
}

fn benchCRoaringOrManyHeap() void {
    const result = c.roaring_bitmap_or_many_heap(N_MANY_BITMAPS, @ptrCast(&cr_many_inputs)) orelse unreachable;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
}

fn benchCRoaringXorMany() void {
    const result = c.roaring_bitmap_xor_many(N_MANY_BITMAPS, @ptrCast(&cr_many_inputs)) orelse unreachable;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
}

var cr_dense_a: ?*c.roaring_bitmap_t = null;
var cr_dense_b: ?*c.roaring_bitmap_t = null;
var cr_bitset_range_bm: ?*c.roaring_bitmap_t = null;

fn initCRoaringDenseBitmaps() void {
    if (cr_dense_a != null) return;

    const a = c.roaring_bitmap_create() orelse unreachable;
    const b_bm = c.roaring_bitmap_create() orelse unreachable;

    c.roaring_bitmap_add_range(a, 0, 500000);
    c.roaring_bitmap_add_range(b_bm, 250000, 750000);

    cr_dense_a = a;
    cr_dense_b = b_bm;
}

fn initCRoaringBitsetRangeBm() void {
    if (cr_bitset_range_bm != null) return;

    const bm = c.roaring_bitmap_create() orelse unreachable;
    var value: u32 = 0;
    while (value < 60_000) : (value += 3) {
        c.roaring_bitmap_add(bm, value);
    }

    cr_bitset_range_bm = bm;
}

fn benchCRoaringAndDense() void {
    const a = cr_dense_a.?;
    const b_bm = cr_dense_b.?;
    const result = c.roaring_bitmap_and(a, b_bm) orelse unreachable;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
}

fn benchCRoaringOrDense() void {
    const a = cr_dense_a.?;
    const b_bm = cr_dense_b.?;
    const result = c.roaring_bitmap_or(a, b_bm) orelse unreachable;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
}

fn benchCRoaringIterate() void {
    const bm = cr_contains_bm.?;
    const result = c.rawr_cr_iterate_pull(bm);
    std.mem.doNotOptimizeAway(result.count);
    std.mem.doNotOptimizeAway(result.sum);
}

var cr_serialized: ?[]u8 = null;

fn initCRoaringSerialized() void {
    if (cr_serialized != null) return;
    const bm = cr_contains_bm.?;
    const size = c.roaring_bitmap_portable_size_in_bytes(bm);
    const buf = allocator.alloc(u8, size) catch unreachable;
    _ = c.roaring_bitmap_portable_serialize(bm, @ptrCast(buf.ptr));
    cr_serialized = buf;
}

fn benchCRoaringSerialize() void {
    benchCRoaringSerializeWithAllocator(allocator);
}

fn benchCRoaringSerializeWithAllocator(comptime result_allocator: std.mem.Allocator) void {
    const bm = cr_contains_bm.?;
    const size = c.roaring_bitmap_portable_size_in_bytes(bm);
    const buf = result_allocator.alloc(u8, size) catch unreachable;
    defer result_allocator.free(buf);
    _ = c.roaring_bitmap_portable_serialize(bm, @ptrCast(buf.ptr));
    std.mem.doNotOptimizeAway(buf.ptr);
}

fn benchCRoaringDeserialize() void {
    const bm = c.roaring_bitmap_portable_deserialize_safe(@ptrCast(cr_serialized.?.ptr), cr_serialized.?.len) orelse unreachable;
    defer c.roaring_bitmap_free(bm);
    std.mem.doNotOptimizeAway(bm);
}

fn benchCRoaringCardinality() void {
    const bm = cr_contains_bm.?;
    const card = c.roaring_bitmap_get_cardinality(bm);
    std.mem.doNotOptimizeAway(card);
}

noinline fn benchCRoaringCardinalityVarying() void {
    parity_cardinality_selector +%= 1;
    const bm = if (parity_cardinality_selector & 1 == 0)
        cr_contains_bm.?
    else
        cr_cardinality_bm_alt.?;
    const card = c.roaring_bitmap_get_cardinality(bm);
    std.mem.doNotOptimizeAway(card);
}

fn benchCRoaringRankDense() void {
    const bm = cr_dense_a.?;
    var total: u64 = 0;
    for (rank_queries[0..]) |query| {
        total +%= c.roaring_bitmap_rank(bm, query);
    }
    std.mem.doNotOptimizeAway(total);
}

fn benchCRoaringSelectDense() void {
    const bm = cr_dense_a.?;
    const result = c.rawr_cr_select_loop(bm, &select_queries, select_queries.len);
    std.mem.doNotOptimizeAway(result.count);
    std.mem.doNotOptimizeAway(result.sum);
}

fn benchCRoaringRankManyDense() void {
    const bm = cr_dense_a.?;
    c.roaring_bitmap_rank_many(
        bm,
        rank_many_probes[0..].ptr,
        rank_many_probes[rank_many_probes.len..].ptr,
        rank_many_out[0..].ptr,
    );
    std.mem.doNotOptimizeAway(rank_many_out[rank_many_out.len - 1]);
}

fn benchCRoaringFlipWideDense() void {
    const bm = cr_dense_a.?;
    const result = c.roaring_bitmap_flip_closed(bm, 100_000, 650_000) orelse unreachable;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
}

fn benchCRoaringRemoveRangeWideDense() void {
    const bm = c.roaring_bitmap_copy(cr_dense_a.?) orelse unreachable;
    defer c.roaring_bitmap_free(bm);
    c.roaring_bitmap_remove_range_closed(bm, 100_000, 650_000);
    std.mem.doNotOptimizeAway(bm);
}

fn benchCRoaringCloneDense() void {
    const result = c.roaring_bitmap_copy(cr_dense_a.?) orelse unreachable;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
}

fn benchCRoaringRangeCardinalityBitset() void {
    const bm = cr_bitset_range_bm.?;
    var total: u64 = 0;
    for (range_query_lo[0..], range_query_hi[0..]) |lo, hi| {
        total +%= c.roaring_bitmap_range_cardinality_closed(bm, lo, hi);
    }
    std.mem.doNotOptimizeAway(total);
}

fn benchCRoaringRangeCardinalityBitsetLarge() void {
    const bm = cr_bitset_range_bm.?;
    var total: u64 = 0;
    for (range_large_query_lo[0..], range_large_query_hi[0..]) |lo, hi| {
        total +%= c.roaring_bitmap_range_cardinality_closed(bm, lo, hi);
    }
    std.mem.doNotOptimizeAway(total);
}

// ============================================================================
// Isolated parity harness adapter
// ============================================================================

pub fn parityTiming(row: ParityRow) ParityTiming {
    return switch (row) {
        .lazy_or_construction, .lazy_or_repair_only => .internal,
        else => .external,
    };
}

pub fn parityRequiresAllocator(row: ParityRow) bool {
    return switch (row) {
        .contains_hit,
        .contains_miss,
        .array_balanced_and_cardinality,
        .array_skewed_and_cardinality,
        .iterate,
        .to_array,
        .cardinality,
        .rank,
        .select,
        .rank_many,
        .range_cardinality_small,
        .range_cardinality_large,
        => false,
        else => true,
    };
}

pub fn parityPrepare(row: ParityRow, implementation: ParityImplementation) void {
    initTestData();
    switch (row) {
        .add_random, .add_sequential, .add_many_random, .add_many_sequential, .add_range => {},
        .contains_hit, .contains_miss, .iterate, .to_array, .to_array_alloc, .serialize, .cardinality => {
            initContainsFor(implementation);
            if (row == .to_array) initToArrayFor(implementation);
            if (row == .cardinality) initCardinalityAltFor(implementation);
        },
        .deserialize, .deserialize_arena => {
            initContainsFor(implementation);
            initSerializedFor(implementation);
        },
        .sparse_and,
        .sparse_and_arena,
        .sparse_or,
        .sparse_or_arena,
        .lazy_or_repair,
        .lazy_or_construction,
        .lazy_or_repair_only,
        => initSparseFor(implementation),
        .dense_and, .dense_or, .rank, .select, .rank_many, .flip, .clone, .remove_range => initDenseFor(implementation),
        .or_many, .or_many_heap, .xor_many => initManyFor(implementation),
        .array_balanced_and,
        .array_balanced_and_cardinality,
        .array_balanced_xor,
        .array_skewed_and,
        .array_skewed_and_cardinality,
        => initArraysFor(implementation),
        .range_cardinality_small, .range_cardinality_large => initBitsetRangeFor(implementation),
    }
}

fn initContainsFor(implementation: ParityImplementation) void {
    switch (implementation) {
        .rawr => initRawrContainsBm(),
        .croaring => initCRoaringContainsBm(),
    }
}

fn initSparseFor(implementation: ParityImplementation) void {
    switch (implementation) {
        .rawr => initRawrSparseBitmaps(),
        .croaring => initCRoaringSparseBitmaps(),
    }
}

fn initArraysFor(implementation: ParityImplementation) void {
    switch (implementation) {
        .rawr => initRawrArrayBitmaps(),
        .croaring => initCRoaringArrayBitmaps(),
    }
}

fn initDenseFor(implementation: ParityImplementation) void {
    switch (implementation) {
        .rawr => initRawrDenseBitmaps(),
        .croaring => initCRoaringDenseBitmaps(),
    }
}

fn initManyFor(implementation: ParityImplementation) void {
    switch (implementation) {
        .rawr => initRawrManyBitmaps(),
        .croaring => initCRoaringManyBitmaps(),
    }
}

fn initBitsetRangeFor(implementation: ParityImplementation) void {
    switch (implementation) {
        .rawr => initRawrBitsetRangeBm(),
        .croaring => initCRoaringBitsetRangeBm(),
    }
}

fn initToArrayFor(implementation: ParityImplementation) void {
    switch (implementation) {
        .rawr => initRawrToArrayBuffer(),
        .croaring => initCRoaringToArrayBuffer(),
    }
}

fn initSerializedFor(implementation: ParityImplementation) void {
    switch (implementation) {
        .rawr => initRawrSerialized(),
        .croaring => initCRoaringSerialized(),
    }
}

fn initCardinalityAltFor(implementation: ParityImplementation) void {
    switch (implementation) {
        .rawr => initRawrCardinalityAlt(),
        .croaring => initCRoaringCardinalityAlt(),
    }
}

pub noinline fn parityRun(row: ParityRow, implementation: ParityImplementation, allocator_kind: ParityAllocator) u64 {
    return switch (implementation) {
        .rawr => parityRunRawr(row, allocator_kind),
        .croaring => parityRunCRoaring(row),
    };
}

noinline fn parityRunRawr(row: ParityRow, allocator_kind: ParityAllocator) u64 {
    switch (row) {
        .add_random => runRawrAllocator(allocator_kind, benchRawrAddRandomWithAllocator),
        .add_sequential => runRawrAllocator(allocator_kind, benchRawrAddSequentialWithAllocator),
        .add_many_random => runRawrAllocator(allocator_kind, benchRawrAddManyRandomWithAllocator),
        .add_many_sequential => runRawrAllocator(allocator_kind, benchRawrAddManySequentialWithAllocator),
        .add_range => runRawrAllocator(allocator_kind, benchRawrAddRangeWithAllocator),
        .contains_hit => benchRawrContainsHit(),
        .contains_miss => benchRawrContainsMiss(),
        .sparse_and => runRawrAllocator(allocator_kind, benchRawrAndSparseWithAllocator),
        .sparse_and_arena => benchRawrAndSparseArena(),
        .dense_and => runRawrAllocator(allocator_kind, benchRawrAndDenseWithAllocator),
        .sparse_or => runRawrAllocator(allocator_kind, benchRawrOrSparseWithAllocator),
        .sparse_or_arena => benchRawrOrSparseArena(),
        .dense_or => runRawrAllocator(allocator_kind, benchRawrOrDenseWithAllocator),
        .lazy_or_repair => runRawrAllocator(allocator_kind, benchRawrLazyOrSparseRepairWithAllocator),
        .lazy_or_construction => return runRawrLazyPhase(allocator_kind, .construction),
        .lazy_or_repair_only => return runRawrLazyPhase(allocator_kind, .repair),
        .or_many => runRawrAllocator(allocator_kind, benchRawrOrManyWithAllocator),
        .or_many_heap => runRawrAllocator(allocator_kind, benchRawrOrManyHeapWithAllocator),
        .xor_many => runRawrAllocator(allocator_kind, benchRawrXorManyWithAllocator),
        .array_balanced_and => runRawrAllocator(allocator_kind, benchRawrAndArrayBalancedWithAllocator),
        .array_balanced_and_cardinality => benchRawrAndCardinalityArrayBalanced(),
        .array_balanced_xor => runRawrAllocator(allocator_kind, benchRawrXorArrayBalancedWithAllocator),
        .array_skewed_and => runRawrAllocator(allocator_kind, benchRawrAndArraySkewedWithAllocator),
        .array_skewed_and_cardinality => benchRawrAndCardinalityArraySkewed(),
        .iterate => benchRawrIterate(),
        .to_array => benchRawrToArray(),
        .to_array_alloc => runRawrAllocator(allocator_kind, benchRawrToArrayAllocWithAllocator),
        .serialize => runRawrAllocator(allocator_kind, benchRawrSerializeWithAllocator),
        .deserialize => runRawrAllocator(allocator_kind, benchRawrDeserializeWithAllocator),
        .deserialize_arena => benchRawrDeserializeArena(),
        .cardinality => benchRawrCardinalityVarying(),
        .rank => benchRawrRankDense(),
        .select => benchRawrSelectDense(),
        .rank_many => benchRawrRankManyDense(),
        .range_cardinality_small => benchRawrRangeCardinalityBitset(),
        .range_cardinality_large => benchRawrRangeCardinalityBitsetLarge(),
        .flip => runRawrAllocator(allocator_kind, benchRawrFlipWideDenseWithAllocator),
        .clone => runRawrAllocator(allocator_kind, benchRawrCloneDenseWithAllocator),
        .remove_range => runRawrAllocator(allocator_kind, benchRawrRemoveRangeWideDenseWithAllocator),
    }
    return 0;
}

fn runRawrAllocator(kind: ParityAllocator, comptime operation: anytype) void {
    switch (kind) {
        .smp => operation(std.heap.smp_allocator),
        .libc => operation(libc_allocator),
        else => unreachable,
    }
}

fn runRawrLazyPhase(kind: ParityAllocator, comptime phase: LazyPhase) u64 {
    return switch (kind) {
        .smp => timeRawrLazyOrSparseWithAllocator(phase, std.heap.smp_allocator),
        .libc => timeRawrLazyOrSparseWithAllocator(phase, libc_allocator),
        else => unreachable,
    };
}

noinline fn parityRunCRoaring(row: ParityRow) u64 {
    switch (row) {
        .add_random => benchCRoaringAddRandom(),
        .add_sequential => benchCRoaringAddSequential(),
        .add_many_random => benchCRoaringAddManyRandom(),
        .add_many_sequential => benchCRoaringAddManySequential(),
        .add_range => benchCRoaringAddRange(),
        .contains_hit => benchCRoaringContainsHit(),
        .contains_miss => benchCRoaringContainsMiss(),
        .sparse_and, .sparse_and_arena => benchCRoaringAndSparse(),
        .dense_and => benchCRoaringAndDense(),
        .sparse_or, .sparse_or_arena => benchCRoaringOrSparse(),
        .dense_or => benchCRoaringOrDense(),
        .lazy_or_repair => benchCRoaringLazyOrSparseRepair(),
        .lazy_or_construction => return timeCRoaringLazyOrSparse(.construction),
        .lazy_or_repair_only => return timeCRoaringLazyOrSparse(.repair),
        .or_many => benchCRoaringOrMany(),
        .or_many_heap => benchCRoaringOrManyHeap(),
        .xor_many => benchCRoaringXorMany(),
        .array_balanced_and => benchCRoaringAndArrayBalanced(),
        .array_balanced_and_cardinality => benchCRoaringAndCardinalityArrayBalanced(),
        .array_balanced_xor => benchCRoaringXorArrayBalanced(),
        .array_skewed_and => benchCRoaringAndArraySkewed(),
        .array_skewed_and_cardinality => benchCRoaringAndCardinalityArraySkewed(),
        .iterate => benchCRoaringIterate(),
        .to_array => benchCRoaringToArray(),
        .to_array_alloc => benchCRoaringToArrayAllocWithAllocator(libc_allocator),
        .serialize => benchCRoaringSerializeWithAllocator(libc_allocator),
        .deserialize, .deserialize_arena => benchCRoaringDeserialize(),
        .cardinality => benchCRoaringCardinalityVarying(),
        .rank => benchCRoaringRankDense(),
        .select => benchCRoaringSelectDense(),
        .rank_many => benchCRoaringRankManyDense(),
        .range_cardinality_small => benchCRoaringRangeCardinalityBitset(),
        .range_cardinality_large => benchCRoaringRangeCardinalityBitsetLarge(),
        .flip => benchCRoaringFlipWideDense(),
        .clone => benchCRoaringCloneDense(),
        .remove_range => benchCRoaringRemoveRangeWideDense(),
    }
    return 0;
}

pub noinline fn parityValidate(row: ParityRow, allocator_kind: ParityAllocator) !void {
    parityPrepare(row, .rawr);
    parityPrepare(row, .croaring);

    switch (row) {
        .contains_hit, .contains_miss => try validateContains(row),
        .array_balanced_and_cardinality, .array_skewed_and_cardinality => try validateAndCardinality(row),
        .cardinality => try validateCardinalityParity(),
        .rank, .select, .range_cardinality_small, .range_cardinality_large => try validateQueries(row),
        .rank_many => try validateRankManyParity(),
        .iterate, .to_array, .to_array_alloc => try validateArrayParity(row, allocator_kind),
        .serialize => try expectPortableEqual(&rawr_contains_bm.?, cr_contains_bm.?),
        else => try validateBitmapResult(row, allocator_kind),
    }
}

fn validateContains(row: ParityRow) !void {
    for (random_values[0..]) |value| {
        const query = if (row == .contains_hit) value else value | 0x80000000;
        if (rawr_contains_bm.?.contains(query) != c.roaring_bitmap_contains(cr_contains_bm.?, query)) {
            return error.ContainsMismatch;
        }
    }
}

fn validateAndCardinality(row: ParityRow) !void {
    const rawr_cardinality, const cr_cardinality = switch (row) {
        .array_balanced_and_cardinality => .{
            rawr_array_balanced_a.?.andCardinality(&rawr_array_balanced_b.?),
            c.roaring_bitmap_and_cardinality(cr_array_balanced_a.?, cr_array_balanced_b.?),
        },
        .array_skewed_and_cardinality => .{
            rawr_array_skewed_a.?.andCardinality(&rawr_array_skewed_b.?),
            c.roaring_bitmap_and_cardinality(cr_array_skewed_a.?, cr_array_skewed_b.?),
        },
        else => unreachable,
    };
    if (rawr_cardinality != cr_cardinality) return error.CardinalityMismatch;
}

fn validateCardinalityParity() !void {
    if (rawr_contains_bm.?.cardinality() != c.roaring_bitmap_get_cardinality(cr_contains_bm.?)) {
        return error.CardinalityMismatch;
    }
    if (rawr_cardinality_bm_alt.?.cardinality() != c.roaring_bitmap_get_cardinality(cr_cardinality_bm_alt.?)) {
        return error.CardinalityMismatch;
    }
}

fn validateQueries(row: ParityRow) !void {
    switch (row) {
        .rank => for (rank_queries[0..]) |query| {
            if (rawr_dense_a.?.rank(query) != c.roaring_bitmap_rank(cr_dense_a.?, query)) return error.QueryMismatch;
        },
        .select => for (select_queries[0..]) |query| {
            var cr_value: u32 = undefined;
            if (!c.roaring_bitmap_select(cr_dense_a.?, query, &cr_value)) return error.QueryMismatch;
            if (rawr_dense_a.?.select(query).? != cr_value) return error.QueryMismatch;
        },
        .range_cardinality_small => for (range_query_lo[0..], range_query_hi[0..]) |lo, hi| {
            if (rawr_bitset_range_bm.?.rangeCardinality(lo, hi) !=
                c.roaring_bitmap_range_cardinality_closed(cr_bitset_range_bm.?, lo, hi)) return error.QueryMismatch;
        },
        .range_cardinality_large => for (range_large_query_lo[0..], range_large_query_hi[0..]) |lo, hi| {
            if (rawr_bitset_range_bm.?.rangeCardinality(lo, hi) !=
                c.roaring_bitmap_range_cardinality_closed(cr_bitset_range_bm.?, lo, hi)) return error.QueryMismatch;
        },
        else => unreachable,
    }
}

noinline fn validateRankManyParity() !void {
    rawr_dense_a.?.rankMany(rank_many_probes[0..], rank_many_out[0..]);
    var cr_out: [N_RANK_MANY_PROBES]u64 = undefined;
    c.roaring_bitmap_rank_many(
        cr_dense_a.?,
        rank_many_probes[0..].ptr,
        rank_many_probes[rank_many_probes.len..].ptr,
        cr_out[0..].ptr,
    );
    if (!std.mem.eql(u64, rank_many_out[0..], cr_out[0..])) return error.QueryMismatch;
}

fn validateArrayParity(row: ParityRow, allocator_kind: ParityAllocator) !void {
    const cardinality: usize = @intCast(rawr_contains_bm.?.cardinality());
    if (cardinality != c.roaring_bitmap_get_cardinality(cr_contains_bm.?)) return error.CardinalityMismatch;
    const validation_allocator = std.heap.page_allocator;
    const rawr_values = switch (row) {
        .to_array_alloc => try rawr_contains_bm.?.toArrayAlloc(switch (allocator_kind) {
            .libc => libc_allocator,
            else => std.heap.smp_allocator,
        }),
        else => try validation_allocator.alloc(u32, cardinality),
    };
    defer switch (row) {
        .to_array_alloc => switch (allocator_kind) {
            .libc => libc_allocator.free(rawr_values),
            else => std.heap.smp_allocator.free(rawr_values),
        },
        else => validation_allocator.free(rawr_values),
    };
    const cr_values = try validation_allocator.alloc(u32, cardinality);
    defer validation_allocator.free(cr_values);

    switch (row) {
        .iterate => {
            var iterator = rawr_contains_bm.?.iterator();
            var index: usize = 0;
            while (iterator.next()) |value| : (index += 1) rawr_values[index] = value;
            if (index != rawr_values.len) return error.ArrayMismatch;
        },
        .to_array => _ = rawr_contains_bm.?.toArray(rawr_values),
        .to_array_alloc => {},
        else => unreachable,
    }
    if (row == .iterate) {
        const written = c.rawr_cr_iterate_pull_values(cr_contains_bm.?, cr_values.ptr, cr_values.len);
        if (written != cr_values.len) return error.ArrayMismatch;
    } else {
        c.roaring_bitmap_to_uint32_array(cr_contains_bm.?, cr_values.ptr);
    }
    if (!std.mem.eql(u32, rawr_values, cr_values)) return error.ArrayMismatch;
}

fn validateBitmapResult(row: ParityRow, allocator_kind: ParityAllocator) !void {
    const cr_result = try makeCRoaringResult(row);
    defer c.roaring_bitmap_free(cr_result);

    if (allocator_kind == .arena) {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        var rawr_result = try makeRawrResult(row, arena.allocator());
        if (row == .clone) try expectRawrPortableIdentical(&rawr_dense_a.?, &rawr_result);
        try expectPortableEqual(&rawr_result, cr_result);
        return;
    }

    const result_allocator = switch (allocator_kind) {
        .libc => libc_allocator,
        else => std.heap.smp_allocator,
    };
    var rawr_result = try makeRawrResult(row, result_allocator);
    defer rawr_result.deinit();
    if (row == .clone) try expectRawrPortableIdentical(&rawr_dense_a.?, &rawr_result);
    try expectPortableEqual(&rawr_result, cr_result);
}

fn makeRawrResult(row: ParityRow, result_allocator: std.mem.Allocator) !RoaringBitmap {
    return switch (row) {
        .add_random => result: {
            var result = try RoaringBitmap.init(result_allocator);
            errdefer result.deinit();
            for (random_values[0..]) |value| _ = try result.add(value);
            break :result result;
        },
        .add_sequential => result: {
            var result = try RoaringBitmap.init(result_allocator);
            errdefer result.deinit();
            for (sequential_values[0..]) |value| _ = try result.add(value);
            break :result result;
        },
        .add_many_random => result: {
            var result = try RoaringBitmap.init(result_allocator);
            errdefer result.deinit();
            try result.addMany(random_values[0..]);
            break :result result;
        },
        .add_many_sequential => result: {
            var result = try RoaringBitmap.init(result_allocator);
            errdefer result.deinit();
            try result.addMany(sequential_values[0..]);
            break :result result;
        },
        .add_range => result: {
            var result = try RoaringBitmap.init(result_allocator);
            errdefer result.deinit();
            _ = try result.addRange(0, N_VALUES - 1);
            break :result result;
        },
        .sparse_and, .sparse_and_arena => try rawr_sparse_a.?.bitwiseAnd(result_allocator, &rawr_sparse_b.?),
        .dense_and => try rawr_dense_a.?.bitwiseAnd(result_allocator, &rawr_dense_b.?),
        .sparse_or, .sparse_or_arena => try rawr_sparse_a.?.bitwiseOr(result_allocator, &rawr_sparse_b.?),
        .dense_or => try rawr_dense_a.?.bitwiseOr(result_allocator, &rawr_dense_b.?),
        .lazy_or_repair, .lazy_or_construction, .lazy_or_repair_only => result: {
            var result = try rawr_sparse_a.?.lazyOr(result_allocator, &rawr_sparse_b.?, true);
            errdefer result.deinit();
            try result.repairAfterLazy();
            break :result result;
        },
        .or_many => try RoaringBitmap.orMany(result_allocator, &rawr_many_inputs),
        .or_many_heap => try RoaringBitmap.orManyHeap(result_allocator, &rawr_many_inputs),
        .xor_many => try RoaringBitmap.xorMany(result_allocator, &rawr_many_inputs),
        .array_balanced_and => try rawr_array_balanced_a.?.bitwiseAnd(result_allocator, &rawr_array_balanced_b.?),
        .array_balanced_xor => try rawr_array_balanced_a.?.bitwiseXor(result_allocator, &rawr_array_balanced_b.?),
        .array_skewed_and => try rawr_array_skewed_a.?.bitwiseAnd(result_allocator, &rawr_array_skewed_b.?),
        .deserialize, .deserialize_arena => try RoaringBitmap.deserialize(result_allocator, rawr_serialized.?),
        .flip => try rawr_dense_a.?.flip(result_allocator, 100_000, 650_000),
        .clone => try rawr_dense_a.?.clone(result_allocator),
        .remove_range => result: {
            var result = try rawr_dense_a.?.clone(result_allocator);
            errdefer result.deinit();
            _ = try result.removeRange(100_000, 650_000);
            break :result result;
        },
        else => unreachable,
    };
}

fn makeCRoaringResult(row: ParityRow) !*c.roaring_bitmap_t {
    return switch (row) {
        .add_random => result: {
            const result = c.roaring_bitmap_create() orelse return error.OutOfMemory;
            for (random_values[0..]) |value| c.roaring_bitmap_add(result, value);
            break :result result;
        },
        .add_sequential => result: {
            const result = c.roaring_bitmap_create() orelse return error.OutOfMemory;
            for (sequential_values[0..]) |value| c.roaring_bitmap_add(result, value);
            break :result result;
        },
        .add_many_random => result: {
            const result = c.roaring_bitmap_create() orelse return error.OutOfMemory;
            c.roaring_bitmap_add_many(result, N_VALUES, random_values[0..].ptr);
            break :result result;
        },
        .add_many_sequential => result: {
            const result = c.roaring_bitmap_create() orelse return error.OutOfMemory;
            c.roaring_bitmap_add_many(result, N_VALUES, sequential_values[0..].ptr);
            break :result result;
        },
        .add_range => result: {
            const result = c.roaring_bitmap_create() orelse return error.OutOfMemory;
            c.roaring_bitmap_add_range(result, 0, N_VALUES);
            break :result result;
        },
        .sparse_and, .sparse_and_arena => c.roaring_bitmap_and(cr_sparse_a.?, cr_sparse_b.?) orelse error.OutOfMemory,
        .dense_and => c.roaring_bitmap_and(cr_dense_a.?, cr_dense_b.?) orelse error.OutOfMemory,
        .sparse_or, .sparse_or_arena => c.roaring_bitmap_or(cr_sparse_a.?, cr_sparse_b.?) orelse error.OutOfMemory,
        .dense_or => c.roaring_bitmap_or(cr_dense_a.?, cr_dense_b.?) orelse error.OutOfMemory,
        .lazy_or_repair, .lazy_or_construction, .lazy_or_repair_only => result: {
            const result = c.roaring_bitmap_lazy_or(cr_sparse_a.?, cr_sparse_b.?, true) orelse return error.OutOfMemory;
            c.roaring_bitmap_repair_after_lazy(result);
            break :result result;
        },
        .or_many => c.roaring_bitmap_or_many(N_MANY_BITMAPS, @ptrCast(&cr_many_inputs)) orelse error.OutOfMemory,
        .or_many_heap => c.roaring_bitmap_or_many_heap(N_MANY_BITMAPS, @ptrCast(&cr_many_inputs)) orelse error.OutOfMemory,
        .xor_many => c.roaring_bitmap_xor_many(N_MANY_BITMAPS, @ptrCast(&cr_many_inputs)) orelse error.OutOfMemory,
        .array_balanced_and => c.roaring_bitmap_and(cr_array_balanced_a.?, cr_array_balanced_b.?) orelse error.OutOfMemory,
        .array_balanced_xor => c.roaring_bitmap_xor(cr_array_balanced_a.?, cr_array_balanced_b.?) orelse error.OutOfMemory,
        .array_skewed_and => c.roaring_bitmap_and(cr_array_skewed_a.?, cr_array_skewed_b.?) orelse error.OutOfMemory,
        .deserialize, .deserialize_arena => c.roaring_bitmap_portable_deserialize_safe(
            @ptrCast(cr_serialized.?.ptr),
            cr_serialized.?.len,
        ) orelse error.OutOfMemory,
        .flip => c.roaring_bitmap_flip_closed(cr_dense_a.?, 100_000, 650_000) orelse error.OutOfMemory,
        .clone => c.roaring_bitmap_copy(cr_dense_a.?) orelse error.OutOfMemory,
        .remove_range => result: {
            const result = c.roaring_bitmap_copy(cr_dense_a.?) orelse return error.OutOfMemory;
            c.roaring_bitmap_remove_range_closed(result, 100_000, 650_000);
            break :result result;
        },
        else => unreachable,
    };
}

fn expectPortableEqual(rawr_result: *const RoaringBitmap, cr_result: *const c.roaring_bitmap_t) !void {
    const validation_allocator = std.heap.page_allocator;
    const rawr_bytes = try rawr_result.serialize(validation_allocator);
    defer validation_allocator.free(rawr_bytes);
    const cr_len = c.roaring_bitmap_portable_size_in_bytes(cr_result);
    if (rawr_bytes.len != cr_len) return error.SerializedSizeMismatch;
    const cr_bytes = try validation_allocator.alloc(u8, cr_len);
    defer validation_allocator.free(cr_bytes);
    if (c.roaring_bitmap_portable_serialize(cr_result, @ptrCast(cr_bytes.ptr)) != cr_len) {
        return error.SerializedSizeMismatch;
    }
    if (!std.mem.eql(u8, rawr_bytes, cr_bytes)) return error.CRoaringMismatch;
}

fn expectRawrPortableIdentical(source: *const RoaringBitmap, clone: *const RoaringBitmap) !void {
    const validation_allocator = std.heap.page_allocator;
    const source_bytes = try source.serialize(validation_allocator);
    defer validation_allocator.free(source_bytes);
    const clone_bytes = try clone.serialize(validation_allocator);
    defer validation_allocator.free(clone_bytes);
    if (!std.mem.eql(u8, source_bytes, clone_bytes)) return error.CloneMismatch;
}

pub fn parityCleanup() void {
    cleanupBenchmarks();
}

// ============================================================================
// Main
// ============================================================================

fn initAllBenchmarkData() void {
    initTestData();
    initRawrContainsBm();
    initCRoaringContainsBm();
    initRawrSparseBitmaps();
    initCRoaringSparseBitmaps();
    initRawrArrayBitmaps();
    initCRoaringArrayBitmaps();
    initRawrDenseBitmaps();
    initCRoaringDenseBitmaps();
    initRawrManyBitmaps();
    initCRoaringManyBitmaps();
    initRawrBitsetRangeBm();
    initCRoaringBitsetRangeBm();
    initToArrayBuffers();
    initRawrSerialized();
    initCRoaringSerialized();
}

fn validateLazyContext() !void {
    var smp_result = try rawr_sparse_a.?.lazyOr(allocator, &rawr_sparse_b.?, true);
    defer smp_result.deinit();
    try smp_result.repairAfterLazy();
    var libc_result = try rawr_sparse_a.?.lazyOr(libc_allocator, &rawr_sparse_b.?, true);
    defer libc_result.deinit();
    try libc_result.repairAfterLazy();
    if (!smp_result.equals(&libc_result)) return error.RawrAllocatorMismatch;

    const cr_result = c.roaring_bitmap_lazy_or(cr_sparse_a.?, cr_sparse_b.?, true) orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(cr_result);
    c.roaring_bitmap_repair_after_lazy(cr_result);

    const rawr_bytes = try smp_result.serialize(std.heap.smp_allocator);
    defer std.heap.smp_allocator.free(rawr_bytes);
    const cr_len = c.roaring_bitmap_portable_size_in_bytes(cr_result);
    if (rawr_bytes.len != cr_len) return error.SerializedSizeMismatch;
    const cr_bytes = try std.heap.smp_allocator.alloc(u8, cr_len);
    defer std.heap.smp_allocator.free(cr_bytes);
    if (c.roaring_bitmap_portable_serialize(cr_result, @ptrCast(cr_bytes.ptr)) != cr_len) {
        return error.SerializedSizeMismatch;
    }
    if (!std.mem.eql(u8, rawr_bytes, cr_bytes)) return error.CRoaringMismatch;
}

noinline fn primeExecutionHistoryBeforeLazy() void {
    _ = benchmark(benchRawrAddRandom, .{});
    _ = benchmark(benchCRoaringAddRandom, .{});
    _ = benchmark(benchRawrAddRandomWithAllocator, .{libc_allocator});
    _ = benchmark(benchRawrAddSequential, .{});
    _ = benchmark(benchCRoaringAddSequential, .{});
    _ = benchmark(benchRawrAddSequentialWithAllocator, .{libc_allocator});
    _ = benchmark(benchRawrAddManyRandom, .{});
    _ = benchmark(benchCRoaringAddManyRandom, .{});
    _ = benchmark(benchRawrAddManyRandomWithAllocator, .{libc_allocator});
    _ = benchmark(benchRawrAddManySequential, .{});
    _ = benchmark(benchCRoaringAddManySequential, .{});
    _ = benchmark(benchRawrAddManySequentialWithAllocator, .{libc_allocator});
    _ = benchmark(benchRawrAddRange, .{});
    _ = benchmark(benchCRoaringAddRange, .{});
    _ = benchmark(benchRawrAddRangeWithAllocator, .{libc_allocator});
    _ = benchmark(benchRawrContainsHit, .{});
    _ = benchmark(benchCRoaringContainsHit, .{});
    _ = benchmark(benchRawrContainsMiss, .{});
    _ = benchmark(benchCRoaringContainsMiss, .{});
    _ = benchmark(benchRawrAndSparse, .{});
    _ = benchmark(benchCRoaringAndSparse, .{});
    _ = benchmark(benchRawrAndSparseWithAllocator, .{libc_allocator});
    _ = benchmark(benchRawrAndSparseArena, .{});
    _ = benchmark(benchRawrAndDense, .{});
    _ = benchmark(benchCRoaringAndDense, .{});
    _ = benchmark(benchRawrAndDenseWithAllocator, .{libc_allocator});
    _ = benchmark(benchRawrOrSparse, .{});
    _ = benchmark(benchCRoaringOrSparse, .{});
    _ = benchmark(benchRawrOrSparseWithAllocator, .{libc_allocator});
    _ = benchmark(benchRawrOrSparseArena, .{});
    _ = benchmark(benchRawrLazyOrSparseRepair, .{});
    _ = benchmark(benchCRoaringLazyOrSparseRepair, .{});
    _ = benchmark(benchRawrLazyOrSparseRepairWithAllocator, .{libc_allocator});
}

fn primeAllocatorOnly(target_allocator: std.mem.Allocator) void {
    const block_count = 16_384;
    var blocks: [block_count][]u8 = undefined;
    for (&blocks) |*block| block.* = target_allocator.alloc(u8, 8192) catch unreachable;
    var index = blocks.len;
    while (index > 0) {
        index -= 1;
        target_allocator.free(blocks[index]);
    }
}

noinline fn primeAllocatorsOnly() void {
    primeAllocatorOnly(allocator);
    primeAllocatorOnly(libc_allocator);
}

noinline fn primeCachesOnly() void {
    var checksum: u64 = 0;
    for (random_values[0..]) |value| checksum +%= value;
    for (sequential_values[0..]) |value| checksum +%= value;
    for (rank_queries[0..]) |value| checksum +%= value;
    for (select_queries[0..]) |value| checksum +%= value;
    for (range_query_lo[0..]) |value| checksum +%= value;
    for (range_query_hi[0..]) |value| checksum +%= value;
    for (range_large_query_lo[0..]) |value| checksum +%= value;
    for (range_large_query_hi[0..]) |value| checksum +%= value;
    for (sparse_values[0..sparse_len]) |value| checksum +%= value;
    std.mem.doNotOptimizeAway(checksum);
}

fn runLazyContext(context: LazyContext, protocol: Protocol) !void {
    bench_time.printBenchEnvironment();
    bench_time.print("Lazy-OR broad-context diagnostic\n", .{});
    bench_time.print("context={s}, warmup={d}, timed={d}\n", .{ @tagName(context), protocol.warmup_runs, protocol.timed_runs });

    switch (context) {
        .target_only => {
            initSparseValues();
            initRawrSparseBitmaps();
            initCRoaringSparseBitmaps();
        },
        .full_init_first, .full_init_last, .allocator_prime, .cache_prime => initAllBenchmarkData(),
    }
    defer cleanupBenchmarks();
    try validateLazyContext();
    bench_time.print("VALIDATION\trawr-smp=rawr-libc=croaring-portable\n", .{});

    switch (context) {
        .full_init_last => primeExecutionHistoryBeforeLazy(),
        .allocator_prime => primeAllocatorsOnly(),
        .cache_prime => primeCachesOnly(),
        else => {},
    }

    const smp = benchmarkInternallyTimedProtocol(
        timeRawrLazyOrSparseWithAllocator,
        .{ LazyPhase.construction, allocator },
        protocol,
    );
    const cr = benchmarkInternallyTimedProtocol(
        timeCRoaringLazyOrSparse,
        .{LazyPhase.construction},
        protocol,
    );
    const rawr_libc = benchmarkInternallyTimedProtocol(
        timeRawrLazyOrSparseWithAllocator,
        .{ LazyPhase.construction, libc_allocator },
        protocol,
    );

    bench_time.print("CONTEXT_RESULT\t{s}\t{s}\trawr-smp\t{d}\n", .{ @tagName(context), protocol.name, smp.median_ns });
    bench_time.print("CONTEXT_RESULT\t{s}\t{s}\trawr-libc\t{d}\n", .{ @tagName(context), protocol.name, rawr_libc.median_ns });
    bench_time.print("CONTEXT_RESULT\t{s}\t{s}\tcroaring\t{d}\n", .{ @tagName(context), protocol.name, cr.median_ns });
}

fn parseLazyContext(name: []const u8) ?LazyContext {
    if (std.mem.eql(u8, name, "target-only")) return .target_only;
    if (std.mem.eql(u8, name, "full-init-first")) return .full_init_first;
    if (std.mem.eql(u8, name, "full-init-last")) return .full_init_last;
    if (std.mem.eql(u8, name, "allocator-prime")) return .allocator_prime;
    if (std.mem.eql(u8, name, "cache-prime")) return .cache_prime;
    return null;
}

fn parseProtocol(name: []const u8) ?Protocol {
    if (std.mem.eql(u8, name, "2x9")) return .{ .name = "2x9", .warmup_runs = 2, .timed_runs = 9 };
    if (std.mem.eql(u8, name, "3x21")) return .{ .name = "3x21", .warmup_runs = 3, .timed_runs = 21 };
    return null;
}

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.smp_allocator);
    defer args.deinit();
    _ = args.skip();

    var lazy_context: ?LazyContext = null;
    var protocol: Protocol = .{ .name = "3x21", .warmup_runs = 3, .timed_runs = 21 };
    while (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "--lazy-context=")) {
            lazy_context = parseLazyContext(arg[15..]) orelse return error.UnknownLazyContext;
        } else if (std.mem.startsWith(u8, arg, "--protocol=")) {
            protocol = parseProtocol(arg[11..]) orelse return error.UnknownBenchmarkProtocol;
        } else {
            return error.UnknownArgument;
        }
    }

    if (lazy_context) |context| return runLazyContext(context, protocol);

    bench_time.printBenchEnvironment();

    bench_time.print("Rawr vs CRoaring Benchmark Comparison\n", .{});
    bench_time.print("======================================\n", .{});
    bench_time.printRunTimestamp();
    bench_time.print("N = {d} values, {d} warmup, {d} timed runs (median)\n", .{ N_VALUES, WARMUP_RUNS, BENCH_RUNS });

    bench_time.print("\nInitializing test data...\n", .{});
    initTestData();

    runAddBenchmarks();
    runContainsBenchmarks();
    runSetBenchmarks();
    runIterationBenchmarks();
    runSerializationBenchmarks();
    runCardinalityBenchmarks();
    runPositionalBenchmarks();
    runRangeBenchmarks();
    cleanupBenchmarks();

    bench_time.print("\nDone.\n", .{});
    bench_time.print("\nNote: ratio < 1.0 = rawr faster, > 1.0 = CRoaring faster\n", .{});
}

// Keep benchmark sections out of main so ReleaseFast does not build a single
// multi-megabyte stack frame on OpenBSD's default 4 MB stack.
noinline fn runAddBenchmarks() void {
    printHeader();
    bench_time.print("ADD OPERATIONS\n", .{});

    var r = benchmark(benchRawrAddRandom, .{});
    var cr = benchmark(benchCRoaringAddRandom, .{});
    printResult("add (random 1M)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrAddSequential, .{});
    cr = benchmark(benchCRoaringAddSequential, .{});
    printResult("add (sequential 1M)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrAddManyRandom, .{});
    cr = benchmark(benchCRoaringAddManyRandom, .{});
    printResult("addMany (random 1M)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrAddManySequential, .{});
    cr = benchmark(benchCRoaringAddManySequential, .{});
    printResult("addMany (sequential 1M)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrAddRange, .{});
    cr = benchmark(benchCRoaringAddRange, .{});
    printResult("addRange (1M)", r.median_ns, cr.median_ns);
}

noinline fn runContainsBenchmarks() void {
    bench_time.print("\nCONTAINS OPERATIONS\n", .{});
    initRawrContainsBm();
    initCRoaringContainsBm();

    var r = benchmark(benchRawrContainsHit, .{});
    var cr = benchmark(benchCRoaringContainsHit, .{});
    printResult("contains (hit)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrContainsMiss, .{});
    cr = benchmark(benchCRoaringContainsMiss, .{});
    printResult("contains (miss)", r.median_ns, cr.median_ns);
}

noinline fn runSetBenchmarks() void {
    bench_time.print("\nSET OPERATIONS (new bitmap)\n", .{});
    initRawrSparseBitmaps();
    initCRoaringSparseBitmaps();
    initRawrArrayBitmaps();
    initCRoaringArrayBitmaps();
    initRawrDenseBitmaps();
    initCRoaringDenseBitmaps();
    initRawrManyBitmaps();
    initCRoaringManyBitmaps();

    var r = benchmark(benchRawrAndSparse, .{});
    var cr = benchmark(benchCRoaringAndSparse, .{});
    printResult("bitwiseAnd (sparse)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrAndSparseArena, .{});
    printResult("bitwiseAnd (sparse, arena)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrAndDense, .{});
    cr = benchmark(benchCRoaringAndDense, .{});
    printResult("bitwiseAnd (dense)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrOrSparse, .{});
    cr = benchmark(benchCRoaringOrSparse, .{});
    printResult("bitwiseOr (sparse)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrOrSparseArena, .{});
    printResult("bitwiseOr (sparse, arena)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrLazyOrSparseRepair, .{});
    cr = benchmark(benchCRoaringLazyOrSparseRepair, .{});
    printResult("lazyOr+repair (sparse)", r.median_ns, cr.median_ns);

    r = benchmarkInternallyTimed(timeRawrLazyOrSparse, .{LazyPhase.construction});
    cr = benchmarkInternallyTimed(timeCRoaringLazyOrSparse, .{LazyPhase.construction});
    printResult("lazyOr construction (sparse)", r.median_ns, cr.median_ns);

    r = benchmarkInternallyTimed(timeRawrLazyOrSparse, .{LazyPhase.repair});
    cr = benchmarkInternallyTimed(timeCRoaringLazyOrSparse, .{LazyPhase.repair});
    printResult("lazyOr repair (sparse)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrOrDense, .{});
    cr = benchmark(benchCRoaringOrDense, .{});
    printResult("bitwiseOr (dense)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrOrMany, .{});
    cr = benchmark(benchCRoaringOrMany, .{});
    printResult("orMany (32 mixed)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrOrManyHeap, .{});
    cr = benchmark(benchCRoaringOrManyHeap, .{});
    printResult("orManyHeap (32 mixed)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrXorMany, .{});
    cr = benchmark(benchCRoaringXorMany, .{});
    printResult("xorMany (32 mixed)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrAndArrayBalanced, .{});
    cr = benchmark(benchCRoaringAndArrayBalanced, .{});
    printResult("bitwiseAnd (array balanced)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrAndCardinalityArrayBalanced, .{});
    cr = benchmark(benchCRoaringAndCardinalityArrayBalanced, .{});
    printResult("andCardinality (array balanced)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrXorArrayBalanced, .{});
    cr = benchmark(benchCRoaringXorArrayBalanced, .{});
    printResult("bitwiseXor (array balanced)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrAndArraySkewed, .{});
    cr = benchmark(benchCRoaringAndArraySkewed, .{});
    printResult("bitwiseAnd (array skewed)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrAndCardinalityArraySkewed, .{});
    cr = benchmark(benchCRoaringAndCardinalityArraySkewed, .{});
    printResult("andCardinality (array skewed)", r.median_ns, cr.median_ns);
}

noinline fn runIterationBenchmarks() void {
    bench_time.print("\nITERATION\n", .{});

    var r = benchmark(benchRawrIterate, .{});
    var cr = benchmark(benchCRoaringIterate, .{});
    printResult("iterate (1M values)", r.median_ns, cr.median_ns);

    initToArrayBuffers();
    r = benchmark(benchRawrToArray, .{});
    cr = benchmark(benchCRoaringToArray, .{});
    printResult("toArray (1M values)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrToArrayAlloc, .{});
    cr = benchmark(benchCRoaringToArrayAlloc, .{});
    printResult("toArrayAlloc (1M values)", r.median_ns, cr.median_ns);
}

noinline fn runSerializationBenchmarks() void {
    bench_time.print("\nSERIALIZATION\n", .{});
    initRawrSerialized();
    initCRoaringSerialized();

    var r = benchmark(benchRawrSerialize, .{});
    var cr = benchmark(benchCRoaringSerialize, .{});
    printResult("serialize", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrDeserialize, .{});
    cr = benchmark(benchCRoaringDeserialize, .{});
    printResult("deserialize", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrDeserializeArena, .{});
    printResult("deserialize (arena)", r.median_ns, cr.median_ns);
}

noinline fn runCardinalityBenchmarks() void {
    bench_time.print("\nCARDINALITY\n", .{});

    const r = benchmark(benchRawrCardinality, .{});
    const cr = benchmark(benchCRoaringCardinality, .{});
    printResult("cardinality", r.median_ns, cr.median_ns);
}

noinline fn runPositionalBenchmarks() void {
    bench_time.print("\nPOSITIONAL QUERIES\n", .{});

    var r = benchmark(benchRawrRankDense, .{});
    var cr = benchmark(benchCRoaringRankDense, .{});
    printResult("rank (dense)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrSelectDense, .{});
    cr = benchmark(benchCRoaringSelectDense, .{});
    printResult("select (dense)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrRankManyDense, .{});
    cr = benchmark(benchCRoaringRankManyDense, .{});
    printResult("rankMany (dense)", r.median_ns, cr.median_ns);
}

noinline fn runRangeBenchmarks() void {
    bench_time.print("\nRANGE OPERATIONS\n", .{});
    initRawrBitsetRangeBm();
    initCRoaringBitsetRangeBm();

    var r = benchmark(benchRawrRangeCardinalityBitset, .{});
    var cr = benchmark(benchCRoaringRangeCardinalityBitset, .{});
    printResult("rangeCardinality small (bitset)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrRangeCardinalityBitsetLarge, .{});
    cr = benchmark(benchCRoaringRangeCardinalityBitsetLarge, .{});
    printResult("rangeCardinality large (bitset)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrFlipWideDense, .{});
    cr = benchmark(benchCRoaringFlipWideDense, .{});
    printResult("flip wide range (dense)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrRemoveRangeWideDense, .{});
    cr = benchmark(benchCRoaringRemoveRangeWideDense, .{});
    printResult("removeRange wide (dense)", r.median_ns, cr.median_ns);
}

noinline fn cleanupBenchmarks() void {
    if (rawr_contains_bm) |*bm| bm.deinit();
    if (rawr_cardinality_bm_alt) |*bm| bm.deinit();
    if (rawr_sparse_a) |*bm| bm.deinit();
    if (rawr_sparse_b) |*bm| bm.deinit();
    if (rawr_array_balanced_a) |*bm| bm.deinit();
    if (rawr_array_balanced_b) |*bm| bm.deinit();
    if (rawr_array_skewed_a) |*bm| bm.deinit();
    if (rawr_array_skewed_b) |*bm| bm.deinit();
    if (rawr_dense_a) |*bm| bm.deinit();
    if (rawr_dense_b) |*bm| bm.deinit();
    if (rawr_bitset_range_bm) |*bm| bm.deinit();
    for (&rawr_many_bms) |*maybe_bm| {
        if (maybe_bm.*) |*bm| bm.deinit();
    }
    if (rawr_to_array_out) |s| allocator.free(s);
    if (cr_to_array_out) |s| allocator.free(s);
    if (rawr_serialized) |s| allocator.free(s);

    if (cr_contains_bm) |bm| c.roaring_bitmap_free(bm);
    if (cr_cardinality_bm_alt) |bm| c.roaring_bitmap_free(bm);
    if (cr_sparse_a) |bm| c.roaring_bitmap_free(bm);
    if (cr_sparse_b) |bm| c.roaring_bitmap_free(bm);
    if (cr_array_balanced_a) |bm| c.roaring_bitmap_free(bm);
    if (cr_array_balanced_b) |bm| c.roaring_bitmap_free(bm);
    if (cr_array_skewed_a) |bm| c.roaring_bitmap_free(bm);
    if (cr_array_skewed_b) |bm| c.roaring_bitmap_free(bm);
    if (cr_dense_a) |bm| c.roaring_bitmap_free(bm);
    if (cr_dense_b) |bm| c.roaring_bitmap_free(bm);
    if (cr_bitset_range_bm) |bm| c.roaring_bitmap_free(bm);
    for (cr_many_bms[0..]) |maybe_bm| {
        if (maybe_bm) |bm| c.roaring_bitmap_free(bm);
    }
    if (cr_serialized) |s| allocator.free(s);
}
