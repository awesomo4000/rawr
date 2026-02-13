const std = @import("std");
const rawr = @import("rawr");
const RoaringBitmap = rawr.RoaringBitmap;
const FrozenBitmap = rawr.FrozenBitmap;

const WARMUP_RUNS = 3;
const BENCH_RUNS = 7;
const N_VALUES = 1_000_000;

/// Result of a benchmark run
const BenchResult = struct {
    name: []const u8,
    median_ns: u64,
    min_ns: u64,
    max_ns: u64,
    ops_per_sec: f64,
    ns_per_op: f64,
};

/// Run a benchmark function, return median timing
fn benchmark(name: []const u8, comptime func: anytype, args: anytype, n_ops: u64) BenchResult {
    var times: [BENCH_RUNS]u64 = undefined;

    // Warmup
    for (0..WARMUP_RUNS) |_| {
        _ = @call(.auto, func, args);
    }

    // Timed runs
    for (0..BENCH_RUNS) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        _ = @call(.auto, func, args);
        times[i] = timer.read();
    }

    // Sort for median
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    const median = times[BENCH_RUNS / 2];
    const min = times[0];
    const max = times[BENCH_RUNS - 1];

    const ops_per_sec = if (median > 0) @as(f64, @floatFromInt(n_ops)) * 1_000_000_000.0 / @as(f64, @floatFromInt(median)) else 0;
    const ns_per_op = if (n_ops > 0) @as(f64, @floatFromInt(median)) / @as(f64, @floatFromInt(n_ops)) else 0;

    return .{
        .name = name,
        .median_ns = median,
        .min_ns = min,
        .max_ns = max,
        .ops_per_sec = ops_per_sec,
        .ns_per_op = ns_per_op,
    };
}

fn printResult(r: BenchResult) void {
    const median_ms = @as(f64, @floatFromInt(r.median_ns)) / 1_000_000.0;
    std.debug.print("{s:<45} {d:>10.2} ms  {d:>12.0} ops/s  {d:>8.1} ns/op\n", .{
        r.name,
        median_ms,
        r.ops_per_sec,
        r.ns_per_op,
    });
}

fn printHeader() void {
    std.debug.print("\n{s:<45} {s:>10}     {s:>12}     {s:>8}\n", .{ "Benchmark", "Time", "Throughput", "Latency" });
    std.debug.print("{s:-<45} {s:->10}     {s:->12}     {s:->8}\n", .{ "", "", "", "" });
}

// ============================================================================
// Benchmark implementations
// ============================================================================

var sequential_values: [N_VALUES]u32 = undefined;
var random_values: [N_VALUES]u32 = undefined;
var clustered_values: [N_VALUES]u32 = undefined;
var sorted_values: [N_VALUES]u32 = undefined;

fn initTestData() void {
    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    // Sequential: 0, 1, 2, ...
    for (0..N_VALUES) |i| {
        sequential_values[i] = @intCast(i);
    }

    // Random: uniform distribution across u32 space
    for (0..N_VALUES) |i| {
        random_values[i] = random.int(u32);
    }

    // Clustered: values in clusters (simulates real-world data)
    var idx: usize = 0;
    var cluster_start: u32 = 0;
    while (idx < N_VALUES) {
        const cluster_size = random.intRangeAtMost(u32, 10, 1000);
        for (0..cluster_size) |j| {
            if (idx >= N_VALUES) break;
            clustered_values[idx] = cluster_start + @as(u32, @intCast(j));
            idx += 1;
        }
        cluster_start += random.intRangeAtMost(u32, 100, 10000);
    }

    // Sorted copy of random for fromSorted
    @memcpy(&sorted_values, &random_values);
    std.mem.sort(u32, &sorted_values, {}, std.sort.asc(u32));
}

// --- Add benchmarks ---

fn benchAddSequential(allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.init(allocator) catch unreachable;
    defer bm.deinit();
    for (sequential_values) |v| {
        _ = bm.add(v) catch unreachable;
    }
}

fn benchAddRandom(allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.init(allocator) catch unreachable;
    defer bm.deinit();
    for (random_values) |v| {
        _ = bm.add(v) catch unreachable;
    }
}

fn benchAddClustered(allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.init(allocator) catch unreachable;
    defer bm.deinit();
    for (clustered_values) |v| {
        _ = bm.add(v) catch unreachable;
    }
}

fn benchAddRange(allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.init(allocator) catch unreachable;
    defer bm.deinit();
    _ = bm.addRange(0, N_VALUES - 1) catch unreachable;
}

fn benchFromSorted(allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.fromSorted(allocator, &sorted_values) catch unreachable;
    defer bm.deinit();
}

// --- Contains benchmarks ---

var contains_bitmap: ?RoaringBitmap = null;

fn setupContainsBitmap(allocator: std.mem.Allocator) void {
    if (contains_bitmap != null) return;
    contains_bitmap = RoaringBitmap.fromSorted(allocator, &sorted_values) catch unreachable;
}

fn benchContainsHit(_: std.mem.Allocator) void {
    const bm = &contains_bitmap.?;
    for (sorted_values) |v| {
        _ = bm.contains(v);
    }
}

fn benchContainsMiss(_: std.mem.Allocator) void {
    const bm = &contains_bitmap.?;
    // Probe values that are unlikely to exist
    for (0..N_VALUES) |i| {
        _ = bm.contains(@as(u32, @intCast(i)) | 0x80000000);
    }
}

// --- Set operation benchmarks ---

var sparse_a: ?RoaringBitmap = null;
var sparse_b: ?RoaringBitmap = null;
var dense_a: ?RoaringBitmap = null;
var dense_b: ?RoaringBitmap = null;

fn setupSetOpBitmaps(allocator: std.mem.Allocator) void {
    if (sparse_a != null) return;

    var prng = std.Random.DefaultPrng.init(54321);
    const random = prng.random();

    // Sparse: random values across wide range
    sparse_a = RoaringBitmap.init(allocator) catch unreachable;
    sparse_b = RoaringBitmap.init(allocator) catch unreachable;
    for (0..500_000) |_| {
        _ = sparse_a.?.add(random.int(u32)) catch unreachable;
        _ = sparse_b.?.add(random.int(u32)) catch unreachable;
    }

    // Dense: consecutive ranges with some overlap
    dense_a = RoaringBitmap.init(allocator) catch unreachable;
    dense_b = RoaringBitmap.init(allocator) catch unreachable;
    _ = dense_a.?.addRange(0, 499_999) catch unreachable;
    _ = dense_b.?.addRange(250_000, 749_999) catch unreachable;
}

fn benchAndSparse(allocator: std.mem.Allocator) void {
    var result = sparse_a.?.bitwiseAnd(allocator, &sparse_b.?) catch unreachable;
    defer result.deinit();
}

fn benchAndDense(allocator: std.mem.Allocator) void {
    var result = dense_a.?.bitwiseAnd(allocator, &dense_b.?) catch unreachable;
    defer result.deinit();
}

fn benchOrSparse(allocator: std.mem.Allocator) void {
    var result = sparse_a.?.bitwiseOr(allocator, &sparse_b.?) catch unreachable;
    defer result.deinit();
}

fn benchOrDense(allocator: std.mem.Allocator) void {
    var result = dense_a.?.bitwiseOr(allocator, &dense_b.?) catch unreachable;
    defer result.deinit();
}

fn benchDiffSparse(allocator: std.mem.Allocator) void {
    var result = sparse_a.?.bitwiseDifference(allocator, &sparse_b.?) catch unreachable;
    defer result.deinit();
}

fn benchDiffDense(allocator: std.mem.Allocator) void {
    var result = dense_a.?.bitwiseDifference(allocator, &dense_b.?) catch unreachable;
    defer result.deinit();
}

// --- Clone benchmarks (shows overhead for in-place ops) ---

fn benchCloneSparse(allocator: std.mem.Allocator) void {
    var a = sparse_a.?.clone(allocator) catch unreachable;
    defer a.deinit();
}

fn benchCloneDense(allocator: std.mem.Allocator) void {
    var a = dense_a.?.clone(allocator) catch unreachable;
    defer a.deinit();
}

// --- In-place operation benchmarks (operation only, no clone in timed region) ---

var precloned_sparse: [BENCH_RUNS + WARMUP_RUNS]?RoaringBitmap = .{null} ** (BENCH_RUNS + WARMUP_RUNS);
var precloned_dense: [BENCH_RUNS + WARMUP_RUNS]?RoaringBitmap = .{null} ** (BENCH_RUNS + WARMUP_RUNS);
var preclone_idx: usize = 0;

fn setupPreclonedSparse(allocator: std.mem.Allocator) void {
    for (&precloned_sparse) |*slot| {
        slot.* = sparse_a.?.clone(allocator) catch unreachable;
    }
    preclone_idx = 0;
}

fn setupPreclonedDense(allocator: std.mem.Allocator) void {
    for (&precloned_dense) |*slot| {
        slot.* = dense_a.?.clone(allocator) catch unreachable;
    }
    preclone_idx = 0;
}

fn cleanupPrecloned() void {
    for (&precloned_sparse) |*slot| {
        if (slot.*) |*bm| {
            bm.deinit();
            slot.* = null;
        }
    }
    for (&precloned_dense) |*slot| {
        if (slot.*) |*bm| {
            bm.deinit();
            slot.* = null;
        }
    }
}

fn benchOrInPlaceSparseOpOnly(_: std.mem.Allocator) void {
    precloned_sparse[preclone_idx].?.bitwiseOrInPlace(&sparse_b.?) catch unreachable;
    preclone_idx += 1;
}

fn benchOrInPlaceDenseOpOnly(_: std.mem.Allocator) void {
    precloned_dense[preclone_idx].?.bitwiseOrInPlace(&dense_b.?) catch unreachable;
    preclone_idx += 1;
}

fn benchAndInPlaceSparseOpOnly(_: std.mem.Allocator) void {
    precloned_sparse[preclone_idx].?.bitwiseAndInPlace(&sparse_b.?) catch unreachable;
    preclone_idx += 1;
}

fn benchAndInPlaceDenseOpOnly(_: std.mem.Allocator) void {
    precloned_dense[preclone_idx].?.bitwiseAndInPlace(&dense_b.?) catch unreachable;
    preclone_idx += 1;
}

// --- Iterator benchmark ---

fn benchIterator(_: std.mem.Allocator) void {
    const bm = &contains_bitmap.?;
    var iter = bm.iterator();
    var sum: u64 = 0;
    while (iter.next()) |v| {
        sum +%= v;
    }
    std.mem.doNotOptimizeAway(sum);
}

// --- Serialization benchmarks ---

var serialized_data: ?[]u8 = null;

fn setupSerializedData(allocator: std.mem.Allocator) void {
    if (serialized_data != null) return;
    const bm = &contains_bitmap.?;
    serialized_data = bm.serialize(allocator) catch unreachable;
}

fn benchSerialize(allocator: std.mem.Allocator) void {
    const bm = &contains_bitmap.?;
    const data = bm.serialize(allocator) catch unreachable;
    defer allocator.free(data);
}

fn benchDeserialize(allocator: std.mem.Allocator) void {
    var bm = RoaringBitmap.deserialize(allocator, serialized_data.?) catch unreachable;
    defer bm.deinit();
}

// --- FrozenBitmap benchmarks ---

var frozen_bitmap: ?FrozenBitmap = null;

fn setupFrozenBitmap(_: std.mem.Allocator) void {
    if (frozen_bitmap != null) return;
    frozen_bitmap = FrozenBitmap.init(serialized_data.?) catch unreachable;
}

fn benchFrozenContainsHit(_: std.mem.Allocator) void {
    const fb = &frozen_bitmap.?;
    for (sorted_values) |v| {
        _ = fb.contains(v);
    }
}

fn benchFrozenContainsMiss(_: std.mem.Allocator) void {
    const fb = &frozen_bitmap.?;
    for (0..N_VALUES) |i| {
        _ = fb.contains(@as(u32, @intCast(i)) | 0x80000000);
    }
}

fn benchFrozenIterator(_: std.mem.Allocator) void {
    const fb = &frozen_bitmap.?;
    var iter = fb.iterator();
    var sum: u64 = 0;
    while (iter.next()) |v| {
        sum +%= v;
    }
    std.mem.doNotOptimizeAway(sum);
}

// --- runOptimize benchmark ---

fn benchRunOptimize(allocator: std.mem.Allocator) void {
    // Create bitmap with mixed container types
    var bm = RoaringBitmap.init(allocator) catch unreachable;
    defer bm.deinit();

    // Some sparse arrays
    for (0..1000) |i| {
        _ = bm.add(@as(u32, @intCast(i * 100))) catch unreachable;
    }
    // Some dense ranges (will become bitsets)
    _ = bm.addRange(1_000_000, 1_010_000) catch unreachable;
    // Some clustered data (good for runs)
    _ = bm.addRange(2_000_000, 2_000_100) catch unreachable;
    _ = bm.addRange(2_001_000, 2_001_100) catch unreachable;

    _ = bm.runOptimize() catch unreachable;
}

// ============================================================================
// Main
// ============================================================================

pub fn main() !void {
    // Use c_allocator for benchmarks to measure algorithm performance,
    // not GPA bookkeeping overhead. GPA is better for tests (leak detection).
    const allocator = std.heap.c_allocator;

    std.debug.print("Rawr Roaring Bitmap Benchmarks\n", .{});
    std.debug.print("==============================\n", .{});
    std.debug.print("N = {d} values, {d} warmup runs, {d} timed runs (median reported)\n", .{ N_VALUES, WARMUP_RUNS, BENCH_RUNS });

    // Initialize test data
    std.debug.print("\nInitializing test data...\n", .{});
    initTestData();

    // --- Add benchmarks ---
    printHeader();
    std.debug.print("ADD OPERATIONS\n", .{});

    printResult(benchmark("add (sequential)", benchAddSequential, .{allocator}, N_VALUES));
    printResult(benchmark("add (random)", benchAddRandom, .{allocator}, N_VALUES));
    printResult(benchmark("add (clustered)", benchAddClustered, .{allocator}, N_VALUES));
    printResult(benchmark("addRange (1M values)", benchAddRange, .{allocator}, N_VALUES));
    printResult(benchmark("fromSorted (1M values)", benchFromSorted, .{allocator}, N_VALUES));

    // --- Contains benchmarks ---
    std.debug.print("\nCONTAINS OPERATIONS\n", .{});
    setupContainsBitmap(allocator);

    printResult(benchmark("contains (hit)", benchContainsHit, .{allocator}, N_VALUES));
    printResult(benchmark("contains (miss)", benchContainsMiss, .{allocator}, N_VALUES));

    // --- Set operation benchmarks ---
    std.debug.print("\nSET OPERATIONS (new bitmap)\n", .{});
    setupSetOpBitmaps(allocator);

    printResult(benchmark("bitwiseAnd (sparse 500K x 500K)", benchAndSparse, .{allocator}, 1));
    printResult(benchmark("bitwiseAnd (dense 500K x 500K)", benchAndDense, .{allocator}, 1));
    printResult(benchmark("bitwiseOr (sparse 500K x 500K)", benchOrSparse, .{allocator}, 1));
    printResult(benchmark("bitwiseOr (dense 500K x 500K)", benchOrDense, .{allocator}, 1));
    printResult(benchmark("bitwiseDifference (sparse)", benchDiffSparse, .{allocator}, 1));
    printResult(benchmark("bitwiseDifference (dense)", benchDiffDense, .{allocator}, 1));

    // --- Clone benchmarks ---
    std.debug.print("\nCLONE\n", .{});

    printResult(benchmark("clone (sparse ~65K containers)", benchCloneSparse, .{allocator}, 1));
    printResult(benchmark("clone (dense 8 containers)", benchCloneDense, .{allocator}, 1));

    // --- In-place operation benchmarks (op only, clone done in setup) ---
    std.debug.print("\nSET OPERATIONS (in-place, operation time only)\n", .{});

    setupPreclonedSparse(allocator);
    printResult(benchmark("bitwiseOrInPlace (sparse)", benchOrInPlaceSparseOpOnly, .{allocator}, 1));
    cleanupPrecloned();

    setupPreclonedDense(allocator);
    printResult(benchmark("bitwiseOrInPlace (dense)", benchOrInPlaceDenseOpOnly, .{allocator}, 1));
    cleanupPrecloned();

    setupPreclonedSparse(allocator);
    printResult(benchmark("bitwiseAndInPlace (sparse)", benchAndInPlaceSparseOpOnly, .{allocator}, 1));
    cleanupPrecloned();

    setupPreclonedDense(allocator);
    printResult(benchmark("bitwiseAndInPlace (dense)", benchAndInPlaceDenseOpOnly, .{allocator}, 1));
    cleanupPrecloned();

    // --- Iterator benchmark ---
    std.debug.print("\nITERATION\n", .{});

    printResult(benchmark("iterator (1M values)", benchIterator, .{allocator}, N_VALUES));

    // --- Serialization benchmarks ---
    std.debug.print("\nSERIALIZATION\n", .{});
    setupSerializedData(allocator);

    printResult(benchmark("serialize (1M values)", benchSerialize, .{allocator}, N_VALUES));
    printResult(benchmark("deserialize (1M values)", benchDeserialize, .{allocator}, N_VALUES));

    // --- FrozenBitmap benchmarks ---
    std.debug.print("\nFROZEN BITMAP (zero-copy)\n", .{});
    setupFrozenBitmap(allocator);

    printResult(benchmark("FrozenBitmap.contains (hit)", benchFrozenContainsHit, .{allocator}, N_VALUES));
    printResult(benchmark("FrozenBitmap.contains (miss)", benchFrozenContainsMiss, .{allocator}, N_VALUES));
    printResult(benchmark("FrozenBitmap.iterator (1M values)", benchFrozenIterator, .{allocator}, N_VALUES));

    // --- runOptimize benchmark ---
    std.debug.print("\nOPTIMIZATION\n", .{});

    printResult(benchmark("runOptimize (mixed containers)", benchRunOptimize, .{allocator}, 1));

    // Cleanup
    if (contains_bitmap) |*bm| bm.deinit();
    if (sparse_a) |*bm| bm.deinit();
    if (sparse_b) |*bm| bm.deinit();
    if (dense_a) |*bm| bm.deinit();
    if (dense_b) |*bm| bm.deinit();
    if (serialized_data) |data| allocator.free(data);

    std.debug.print("\nDone.\n", .{});
}
