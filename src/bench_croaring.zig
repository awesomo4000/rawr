// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const rawr = @import("rawr");
const RoaringBitmap = rawr.RoaringBitmap;
const c = @import("c");
const bench_time = @import("bench_time.zig");

const smp_allocator = std.heap.smp_allocator;
const libc_allocator = bench_time.cAllocator();

const WARMUP_RUNS = 3;
const BENCH_RUNS = 21;
const N_VALUES = 1_000_000;
const N_RANK_MANY_PROBES = 200_000;
const N_MANY_BITMAPS = 32;
const N_ARRAY_BENCH_CONTAINERS = 200;

const BenchResult = struct {
    median_ns: u64,
    p25_ns: u64,
    p75_ns: u64,
};

const LazyPhase = enum {
    construction,
    repair,
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

fn printHeader() void {
    bench_time.print("\n{s:<40} {s:>12} {s:>12} {s:>12} {s:>9} {s:>9} {s:>9}\n", .{
        "Operation",
        "rawr smp",
        "rawr c",
        "CRoaring",
        "smp/CR",
        "c/CR",
        "c/smp",
    });
    bench_time.print("{s:-<40} {s:->12} {s:->12} {s:->12} {s:->9} {s:->9} {s:->9}\n", .{
        "", "", "", "", "", "", "",
    });
}

fn printResult(name: []const u8, smp_ns: u64, c_ns: ?u64, cr_ns: u64) void {
    const smp_ms = @as(f64, @floatFromInt(smp_ns)) / 1_000_000.0;
    const cr_ms = @as(f64, @floatFromInt(cr_ns)) / 1_000_000.0;
    const smp_cr_ratio = if (cr_ns > 0) smp_ms / cr_ms else 0;

    if (c_ns) |libc_ns| {
        const c_ms = @as(f64, @floatFromInt(libc_ns)) / 1_000_000.0;
        const c_cr_ratio = if (cr_ns > 0) c_ms / cr_ms else 0;
        const c_smp_ratio = if (smp_ns > 0) c_ms / smp_ms else 0;
        bench_time.print("{s:<40} {d:>12.2} {d:>12.2} {d:>12.2} {d:>8.2}x {d:>8.2}x {d:>8.2}x\n", .{
            name,
            smp_ms,
            c_ms,
            cr_ms,
            smp_cr_ratio,
            c_cr_ratio,
            c_smp_ratio,
        });
        bench_time.print("RESULT\t{s}\t{d}\t{d}\t{d}\n", .{ name, smp_ns, libc_ns, cr_ns });
        return;
    }

    bench_time.print("{s:<40} {d:>12.2} {s:>12} {d:>12.2} {d:>8.2}x {s:>9} {s:>9}\n", .{
        name,
        smp_ms,
        "N/A",
        cr_ms,
        smp_cr_ratio,
        "N/A",
        "N/A",
    });
    bench_time.print("RESULT\t{s}\t{d}\tN/A\t{d}\n", .{ name, smp_ns, cr_ns });
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
}

// ============================================================================
// Rawr benchmarks
// ============================================================================

fn benchRawrAddRandom(comptime result_allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.init(result_allocator) catch unreachable;
    defer bm.deinit();
    for (random_values[0..]) |v| {
        _ = bm.add(v) catch unreachable;
    }
    std.mem.doNotOptimizeAway(&bm);
}

fn benchRawrAddSequential(comptime result_allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.init(result_allocator) catch unreachable;
    defer bm.deinit();
    for (sequential_values[0..]) |v| {
        _ = bm.add(v) catch unreachable;
    }
    std.mem.doNotOptimizeAway(&bm);
}

fn benchRawrAddManyRandom(comptime result_allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.init(result_allocator) catch unreachable;
    defer bm.deinit();
    bm.addMany(random_values[0..]) catch unreachable;
    std.mem.doNotOptimizeAway(&bm);
}

fn benchRawrAddManySequential(comptime result_allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.init(result_allocator) catch unreachable;
    defer bm.deinit();
    bm.addMany(sequential_values[0..]) catch unreachable;
    std.mem.doNotOptimizeAway(&bm);
}

fn benchRawrAddRange(comptime result_allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.init(result_allocator) catch unreachable;
    defer bm.deinit();
    _ = bm.addRange(0, N_VALUES - 1) catch unreachable;
    std.mem.doNotOptimizeAway(&bm);
}

var rawr_contains_bm: ?RoaringBitmap = null;

fn initRawrContainsBm() void {
    if (rawr_contains_bm != null) return;
    var bm = RoaringBitmap.init(smp_allocator) catch unreachable;
    for (random_values[0..]) |v| {
        _ = bm.add(v) catch unreachable;
    }
    rawr_contains_bm = bm;
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

    var a = RoaringBitmap.init(smp_allocator) catch unreachable;
    var b = RoaringBitmap.init(smp_allocator) catch unreachable;

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

    var balanced_a = RoaringBitmap.init(smp_allocator) catch unreachable;
    var balanced_b = RoaringBitmap.init(smp_allocator) catch unreachable;
    var skewed_a = RoaringBitmap.init(smp_allocator) catch unreachable;
    var skewed_b = RoaringBitmap.init(smp_allocator) catch unreachable;

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
        var bm = RoaringBitmap.init(smp_allocator) catch unreachable;
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

fn benchRawrAndSparse(comptime result_allocator: std.mem.Allocator) void {
    const a = &rawr_sparse_a.?;
    const b = &rawr_sparse_b.?;
    var result = a.bitwiseAnd(result_allocator, b) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrAndArrayBalanced(comptime result_allocator: std.mem.Allocator) void {
    var result = rawr_array_balanced_a.?.bitwiseAnd(result_allocator, &rawr_array_balanced_b.?) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrAndCardinalityArrayBalanced() void {
    const cardinality = rawr_array_balanced_a.?.andCardinality(&rawr_array_balanced_b.?);
    std.mem.doNotOptimizeAway(cardinality);
}

fn benchRawrXorArrayBalanced(comptime result_allocator: std.mem.Allocator) void {
    var result = rawr_array_balanced_a.?.bitwiseXor(result_allocator, &rawr_array_balanced_b.?) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrAndArraySkewed(comptime result_allocator: std.mem.Allocator) void {
    var result = rawr_array_skewed_a.?.bitwiseAnd(result_allocator, &rawr_array_skewed_b.?) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrAndCardinalityArraySkewed() void {
    const cardinality = rawr_array_skewed_a.?.andCardinality(&rawr_array_skewed_b.?);
    std.mem.doNotOptimizeAway(cardinality);
}

fn benchRawrOrSparse(comptime result_allocator: std.mem.Allocator) void {
    const a = &rawr_sparse_a.?;
    const b = &rawr_sparse_b.?;
    var result = a.bitwiseOr(result_allocator, b) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrLazyOrSparseRepair(comptime result_allocator: std.mem.Allocator) void {
    const a = &rawr_sparse_a.?;
    const b = &rawr_sparse_b.?;
    var result = a.lazyOr(result_allocator, b, true) catch unreachable;
    defer result.deinit();
    result.repairAfterLazy() catch unreachable;
    std.mem.doNotOptimizeAway(&result);
}

fn timeRawrLazyOrSparse(comptime phase: LazyPhase, comptime result_allocator: std.mem.Allocator) u64 {
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

fn benchRawrOrMany(comptime result_allocator: std.mem.Allocator) void {
    var result = RoaringBitmap.orMany(result_allocator, &rawr_many_inputs) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrOrManyHeap(comptime result_allocator: std.mem.Allocator) void {
    var result = RoaringBitmap.orManyHeap(result_allocator, &rawr_many_inputs) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrXorMany(comptime result_allocator: std.mem.Allocator) void {
    var result = RoaringBitmap.xorMany(result_allocator, &rawr_many_inputs) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

var rawr_dense_a: ?RoaringBitmap = null;
var rawr_dense_b: ?RoaringBitmap = null;
var rawr_bitset_range_bm: ?RoaringBitmap = null;

fn initRawrDenseBitmaps() void {
    if (rawr_dense_a != null) return;

    var a = RoaringBitmap.init(smp_allocator) catch unreachable;
    var b = RoaringBitmap.init(smp_allocator) catch unreachable;

    _ = a.addRange(0, 499999) catch unreachable;
    _ = b.addRange(250000, 749999) catch unreachable;

    rawr_dense_a = a;
    rawr_dense_b = b;
}

fn initRawrBitsetRangeBm() void {
    if (rawr_bitset_range_bm != null) return;

    var bm = RoaringBitmap.init(smp_allocator) catch unreachable;
    var value: u32 = 0;
    while (value < 60_000) : (value += 3) {
        _ = bm.add(value) catch unreachable;
    }

    rawr_bitset_range_bm = bm;
}

fn benchRawrAndDense(comptime result_allocator: std.mem.Allocator) void {
    const a = &rawr_dense_a.?;
    const b = &rawr_dense_b.?;
    var result = a.bitwiseAnd(result_allocator, b) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrOrDense(comptime result_allocator: std.mem.Allocator) void {
    const a = &rawr_dense_a.?;
    const b = &rawr_dense_b.?;
    var result = a.bitwiseOr(result_allocator, b) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrIterate() void {
    const bm = &rawr_contains_bm.?;
    var sum: u64 = 0;
    var it = bm.iterator();
    while (it.next()) |v| {
        sum +%= v;
    }
    std.mem.doNotOptimizeAway(sum);
}

var rawr_to_array_out: ?[]u32 = null;
var cr_to_array_out: ?[]u32 = null;

fn initToArrayBuffers() void {
    if (rawr_to_array_out != null) return;

    const rawr_card: usize = @intCast(rawr_contains_bm.?.cardinality());
    const cr_card: usize = @intCast(c.roaring_bitmap_get_cardinality(cr_contains_bm.?));
    rawr_to_array_out = smp_allocator.alloc(u32, rawr_card) catch unreachable;
    cr_to_array_out = smp_allocator.alloc(u32, cr_card) catch unreachable;
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

fn benchRawrToArrayAlloc(comptime result_allocator: std.mem.Allocator) void {
    const bm = &rawr_contains_bm.?;
    const values = bm.toArrayAlloc(result_allocator) catch unreachable;
    defer result_allocator.free(values);
    std.mem.doNotOptimizeAway(values.ptr);
}

fn benchCRoaringToArrayAlloc() void {
    const bm = cr_contains_bm.?;
    const card: usize = @intCast(c.roaring_bitmap_get_cardinality(bm));
    const values = smp_allocator.alloc(u32, card) catch unreachable;
    defer smp_allocator.free(values);
    c.roaring_bitmap_to_uint32_array(bm, values.ptr);
    std.mem.doNotOptimizeAway(values.ptr);
}

var rawr_serialized: ?[]u8 = null;

fn initRawrSerialized() void {
    if (rawr_serialized != null) return;
    const bm = &rawr_contains_bm.?;
    rawr_serialized = RoaringBitmap.serialize(bm, smp_allocator) catch unreachable;
}

fn benchRawrSerialize(comptime result_allocator: std.mem.Allocator) void {
    const bm = &rawr_contains_bm.?;
    const bytes = RoaringBitmap.serialize(bm, result_allocator) catch unreachable;
    defer result_allocator.free(bytes);
    std.mem.doNotOptimizeAway(bytes.ptr);
}

fn benchRawrDeserialize(comptime result_allocator: std.mem.Allocator) void {
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

fn benchRawrRankDense() void {
    const bm = &rawr_dense_a.?;
    var total: u64 = 0;
    for (rank_queries[0..]) |query| {
        total +%= bm.rank(query);
    }
    std.mem.doNotOptimizeAway(total);
}

fn benchRawrSelectDense() void {
    const bm = &rawr_dense_a.?;
    var total: u64 = 0;
    for (select_queries[0..]) |query| {
        total +%= bm.select(query).?;
    }
    std.mem.doNotOptimizeAway(total);
}

fn benchRawrRankManyDense() void {
    const bm = &rawr_dense_a.?;
    bm.rankMany(rank_many_probes[0..], rank_many_out[0..]);
    std.mem.doNotOptimizeAway(rank_many_out[rank_many_out.len - 1]);
}

fn benchRawrFlipWideDense(comptime result_allocator: std.mem.Allocator) void {
    const bm = &rawr_dense_a.?;
    var result = bm.flip(result_allocator, 100_000, 650_000) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrRemoveRangeWideDense(comptime result_allocator: std.mem.Allocator) void {
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

fn initCRoaringContainsBm() void {
    if (cr_contains_bm != null) return;
    const bm = c.roaring_bitmap_create() orelse unreachable;
    for (random_values[0..]) |v| {
        c.roaring_bitmap_add(bm, v);
    }
    cr_contains_bm = bm;
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

var cr_iterate_sum: u64 = 0;

fn crIterateCallback(value: u32, _: ?*anyopaque) callconv(.c) bool {
    cr_iterate_sum +%= value;
    return true;
}

fn benchCRoaringIterate() void {
    const bm = cr_contains_bm.?;
    cr_iterate_sum = 0;
    _ = c.roaring_iterate(bm, crIterateCallback, null);
    std.mem.doNotOptimizeAway(cr_iterate_sum);
}

var cr_serialized: ?[]u8 = null;

fn initCRoaringSerialized() void {
    if (cr_serialized != null) return;
    const bm = cr_contains_bm.?;
    const size = c.roaring_bitmap_portable_size_in_bytes(bm);
    const buf = smp_allocator.alloc(u8, size) catch unreachable;
    _ = c.roaring_bitmap_portable_serialize(bm, @ptrCast(buf.ptr));
    cr_serialized = buf;
}

fn benchCRoaringSerialize() void {
    const bm = cr_contains_bm.?;
    const size = c.roaring_bitmap_portable_size_in_bytes(bm);
    const buf = smp_allocator.alloc(u8, size) catch unreachable;
    defer smp_allocator.free(buf);
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
    var total: u64 = 0;
    for (select_queries[0..]) |query| {
        var value: u32 = undefined;
        _ = c.roaring_bitmap_select(bm, query, &value);
        total +%= value;
    }
    std.mem.doNotOptimizeAway(total);
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
// Main
// ============================================================================

pub fn main() !void {
    bench_time.printBenchEnvironment();
    bench_time.print("# benchmark allocators: rawr smp=std.heap.smp_allocator | rawr c=libc via bench_time.cAllocator() | CRoaring=libc internal\n", .{});
    bench_time.print("# allocator-match caveat: serialize and toArrayAlloc CRoaring output buffers use rawr smp; rawr c/CRoaring is matched only where CRoaring owns allocation\n\n", .{});

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
    bench_time.print("\nNote: smp/CR and c/CR < 1.0 = rawr faster; c/smp < 1.0 = libc allocator faster\n", .{});
}

// Keep benchmark sections out of main so ReleaseFast does not build a single
// multi-megabyte stack frame on OpenBSD's default 4 MB stack.
noinline fn runAddBenchmarks() void {
    printHeader();
    bench_time.print("ADD OPERATIONS\n", .{});
    var r = benchmark(benchRawrAddRandom, .{smp_allocator});
    var cr = benchmark(benchCRoaringAddRandom, .{});
    var rawr_c = benchmark(benchRawrAddRandom, .{libc_allocator});
    printResult("add (random 1M)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrAddSequential, .{smp_allocator});
    cr = benchmark(benchCRoaringAddSequential, .{});
    rawr_c = benchmark(benchRawrAddSequential, .{libc_allocator});
    printResult("add (sequential 1M)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrAddManyRandom, .{smp_allocator});
    cr = benchmark(benchCRoaringAddManyRandom, .{});
    rawr_c = benchmark(benchRawrAddManyRandom, .{libc_allocator});
    printResult("addMany (random 1M)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrAddManySequential, .{smp_allocator});
    cr = benchmark(benchCRoaringAddManySequential, .{});
    rawr_c = benchmark(benchRawrAddManySequential, .{libc_allocator});
    printResult("addMany (sequential 1M)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrAddRange, .{smp_allocator});
    cr = benchmark(benchCRoaringAddRange, .{});
    rawr_c = benchmark(benchRawrAddRange, .{libc_allocator});
    printResult("addRange (1M)", r.median_ns, rawr_c.median_ns, cr.median_ns);
}

noinline fn runContainsBenchmarks() void {
    bench_time.print("\nCONTAINS OPERATIONS\n", .{});
    initRawrContainsBm();
    initCRoaringContainsBm();

    var r = benchmark(benchRawrContainsHit, .{});
    var cr = benchmark(benchCRoaringContainsHit, .{});
    printResult("contains (hit)", r.median_ns, null, cr.median_ns);

    r = benchmark(benchRawrContainsMiss, .{});
    cr = benchmark(benchCRoaringContainsMiss, .{});
    printResult("contains (miss)", r.median_ns, null, cr.median_ns);
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

    var r = benchmark(benchRawrAndSparse, .{smp_allocator});
    var cr = benchmark(benchCRoaringAndSparse, .{});
    var rawr_c = benchmark(benchRawrAndSparse, .{libc_allocator});
    printResult("bitwiseAnd (sparse)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrAndSparseArena, .{});
    printResult("bitwiseAnd (sparse, arena)", r.median_ns, null, cr.median_ns);

    r = benchmark(benchRawrAndDense, .{smp_allocator});
    cr = benchmark(benchCRoaringAndDense, .{});
    rawr_c = benchmark(benchRawrAndDense, .{libc_allocator});
    printResult("bitwiseAnd (dense)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrOrSparse, .{smp_allocator});
    cr = benchmark(benchCRoaringOrSparse, .{});
    rawr_c = benchmark(benchRawrOrSparse, .{libc_allocator});
    printResult("bitwiseOr (sparse)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrOrSparseArena, .{});
    printResult("bitwiseOr (sparse, arena)", r.median_ns, null, cr.median_ns);

    r = benchmark(benchRawrLazyOrSparseRepair, .{smp_allocator});
    cr = benchmark(benchCRoaringLazyOrSparseRepair, .{});
    rawr_c = benchmark(benchRawrLazyOrSparseRepair, .{libc_allocator});
    printResult("lazyOr+repair (sparse)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmarkInternallyTimed(timeRawrLazyOrSparse, .{ LazyPhase.construction, smp_allocator });
    cr = benchmarkInternallyTimed(timeCRoaringLazyOrSparse, .{LazyPhase.construction});
    rawr_c = benchmarkInternallyTimed(timeRawrLazyOrSparse, .{ LazyPhase.construction, libc_allocator });
    printResult("lazyOr construction (sparse)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmarkInternallyTimed(timeRawrLazyOrSparse, .{ LazyPhase.repair, smp_allocator });
    cr = benchmarkInternallyTimed(timeCRoaringLazyOrSparse, .{LazyPhase.repair});
    rawr_c = benchmarkInternallyTimed(timeRawrLazyOrSparse, .{ LazyPhase.repair, libc_allocator });
    printResult("lazyOr repair (sparse)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrOrDense, .{smp_allocator});
    cr = benchmark(benchCRoaringOrDense, .{});
    rawr_c = benchmark(benchRawrOrDense, .{libc_allocator});
    printResult("bitwiseOr (dense)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrOrMany, .{smp_allocator});
    cr = benchmark(benchCRoaringOrMany, .{});
    rawr_c = benchmark(benchRawrOrMany, .{libc_allocator});
    printResult("orMany (32 mixed)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrOrManyHeap, .{smp_allocator});
    cr = benchmark(benchCRoaringOrManyHeap, .{});
    rawr_c = benchmark(benchRawrOrManyHeap, .{libc_allocator});
    printResult("orManyHeap (32 mixed)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrXorMany, .{smp_allocator});
    cr = benchmark(benchCRoaringXorMany, .{});
    rawr_c = benchmark(benchRawrXorMany, .{libc_allocator});
    printResult("xorMany (32 mixed)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrAndArrayBalanced, .{smp_allocator});
    cr = benchmark(benchCRoaringAndArrayBalanced, .{});
    rawr_c = benchmark(benchRawrAndArrayBalanced, .{libc_allocator});
    printResult("bitwiseAnd (array balanced)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrAndCardinalityArrayBalanced, .{});
    cr = benchmark(benchCRoaringAndCardinalityArrayBalanced, .{});
    printResult("andCardinality (array balanced)", r.median_ns, null, cr.median_ns);

    r = benchmark(benchRawrXorArrayBalanced, .{smp_allocator});
    cr = benchmark(benchCRoaringXorArrayBalanced, .{});
    rawr_c = benchmark(benchRawrXorArrayBalanced, .{libc_allocator});
    printResult("bitwiseXor (array balanced)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrAndArraySkewed, .{smp_allocator});
    cr = benchmark(benchCRoaringAndArraySkewed, .{});
    rawr_c = benchmark(benchRawrAndArraySkewed, .{libc_allocator});
    printResult("bitwiseAnd (array skewed)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrAndCardinalityArraySkewed, .{});
    cr = benchmark(benchCRoaringAndCardinalityArraySkewed, .{});
    printResult("andCardinality (array skewed)", r.median_ns, null, cr.median_ns);
}

noinline fn runIterationBenchmarks() void {
    bench_time.print("\nITERATION\n", .{});

    var r = benchmark(benchRawrIterate, .{});
    var cr = benchmark(benchCRoaringIterate, .{});
    printResult("iterate (1M values)", r.median_ns, null, cr.median_ns);

    initToArrayBuffers();
    r = benchmark(benchRawrToArray, .{});
    cr = benchmark(benchCRoaringToArray, .{});
    printResult("toArray (1M values)", r.median_ns, null, cr.median_ns);

    r = benchmark(benchRawrToArrayAlloc, .{smp_allocator});
    cr = benchmark(benchCRoaringToArrayAlloc, .{});
    const rawr_c = benchmark(benchRawrToArrayAlloc, .{libc_allocator});
    printResult("toArrayAlloc (1M values)", r.median_ns, rawr_c.median_ns, cr.median_ns);
}

noinline fn runSerializationBenchmarks() void {
    bench_time.print("\nSERIALIZATION\n", .{});
    initRawrSerialized();
    initCRoaringSerialized();

    var r = benchmark(benchRawrSerialize, .{smp_allocator});
    var cr = benchmark(benchCRoaringSerialize, .{});
    var rawr_c = benchmark(benchRawrSerialize, .{libc_allocator});
    printResult("serialize", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrDeserialize, .{smp_allocator});
    cr = benchmark(benchCRoaringDeserialize, .{});
    rawr_c = benchmark(benchRawrDeserialize, .{libc_allocator});
    printResult("deserialize", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrDeserializeArena, .{});
    printResult("deserialize (arena)", r.median_ns, null, cr.median_ns);
}

noinline fn runCardinalityBenchmarks() void {
    bench_time.print("\nCARDINALITY\n", .{});

    const r = benchmark(benchRawrCardinality, .{});
    const cr = benchmark(benchCRoaringCardinality, .{});
    printResult("cardinality", r.median_ns, null, cr.median_ns);
}

noinline fn runPositionalBenchmarks() void {
    bench_time.print("\nPOSITIONAL QUERIES\n", .{});

    var r = benchmark(benchRawrRankDense, .{});
    var cr = benchmark(benchCRoaringRankDense, .{});
    printResult("rank (dense)", r.median_ns, null, cr.median_ns);

    r = benchmark(benchRawrSelectDense, .{});
    cr = benchmark(benchCRoaringSelectDense, .{});
    printResult("select (dense)", r.median_ns, null, cr.median_ns);

    r = benchmark(benchRawrRankManyDense, .{});
    cr = benchmark(benchCRoaringRankManyDense, .{});
    printResult("rankMany (dense)", r.median_ns, null, cr.median_ns);
}

noinline fn runRangeBenchmarks() void {
    bench_time.print("\nRANGE OPERATIONS\n", .{});
    initRawrBitsetRangeBm();
    initCRoaringBitsetRangeBm();

    var r = benchmark(benchRawrRangeCardinalityBitset, .{});
    var cr = benchmark(benchCRoaringRangeCardinalityBitset, .{});
    printResult("rangeCardinality small (bitset)", r.median_ns, null, cr.median_ns);

    r = benchmark(benchRawrRangeCardinalityBitsetLarge, .{});
    cr = benchmark(benchCRoaringRangeCardinalityBitsetLarge, .{});
    printResult("rangeCardinality large (bitset)", r.median_ns, null, cr.median_ns);

    r = benchmark(benchRawrFlipWideDense, .{smp_allocator});
    cr = benchmark(benchCRoaringFlipWideDense, .{});
    var rawr_c = benchmark(benchRawrFlipWideDense, .{libc_allocator});
    printResult("flip wide range (dense)", r.median_ns, rawr_c.median_ns, cr.median_ns);

    r = benchmark(benchRawrRemoveRangeWideDense, .{smp_allocator});
    cr = benchmark(benchCRoaringRemoveRangeWideDense, .{});
    rawr_c = benchmark(benchRawrRemoveRangeWideDense, .{libc_allocator});
    printResult("removeRange wide (dense)", r.median_ns, rawr_c.median_ns, cr.median_ns);
}

noinline fn cleanupBenchmarks() void {
    if (rawr_contains_bm) |*bm| bm.deinit();
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
    if (rawr_to_array_out) |s| smp_allocator.free(s);
    if (cr_to_array_out) |s| smp_allocator.free(s);
    if (rawr_serialized) |s| smp_allocator.free(s);

    if (cr_contains_bm) |bm| c.roaring_bitmap_free(bm);
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
    if (cr_serialized) |s| smp_allocator.free(s);
}
