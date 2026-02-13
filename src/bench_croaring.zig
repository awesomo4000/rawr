const std = @import("std");
const rawr = @import("rawr");
const RoaringBitmap = rawr.RoaringBitmap;
const c = @cImport(@cInclude("croaring_wrapper.h"));

const allocator = std.heap.c_allocator;

const WARMUP_RUNS = 3;
const BENCH_RUNS = 21;
const N_VALUES = 1_000_000;

const BenchResult = struct {
    median_ns: u64,
    p25_ns: u64,
    p75_ns: u64,
};

fn benchmark(comptime func: anytype, args: anytype) BenchResult {
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

    // Sort for percentiles
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));

    return .{
        .p25_ns = times[BENCH_RUNS / 4],
        .median_ns = times[BENCH_RUNS / 2],
        .p75_ns = times[3 * BENCH_RUNS / 4],
    };
}

fn printHeader() void {
    std.debug.print("\n{s:<40} {s:>12} {s:>12} {s:>8}\n", .{ "Operation", "rawr (ms)", "CRoaring", "ratio" });
    std.debug.print("{s:-<40} {s:->12} {s:->12} {s:->8}\n", .{ "", "", "", "" });
}

fn printResult(name: []const u8, rawr_ns: u64, cr_ns: u64) void {
    const rawr_ms = @as(f64, @floatFromInt(rawr_ns)) / 1_000_000.0;
    const cr_ms = @as(f64, @floatFromInt(cr_ns)) / 1_000_000.0;
    const ratio = if (cr_ns > 0) rawr_ms / cr_ms else 0;
    std.debug.print("{s:<40} {d:>12.2} {d:>12.2} {d:>8.2}x\n", .{ name, rawr_ms, cr_ms, ratio });
}

// ============================================================================
// Test data
// ============================================================================

var random_values: [N_VALUES]u32 = undefined;
var sequential_values: [N_VALUES]u32 = undefined;
var sparse_values: [500000]u32 = undefined;
var sparse_len: usize = 0;

fn initTestData() void {
    var prng = std.Random.DefaultPrng.init(12345);

    for (0..N_VALUES) |i| {
        random_values[i] = prng.random().int(u32);
        sequential_values[i] = @intCast(i);
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

fn benchRawrAddRandom() void {
    var bm = RoaringBitmap.init(allocator) catch unreachable;
    defer bm.deinit();
    for (random_values) |v| {
        _ = bm.add(v) catch unreachable;
    }
    std.mem.doNotOptimizeAway(&bm);
}

fn benchRawrAddSequential() void {
    var bm = RoaringBitmap.init(allocator) catch unreachable;
    defer bm.deinit();
    for (sequential_values) |v| {
        _ = bm.add(v) catch unreachable;
    }
    std.mem.doNotOptimizeAway(&bm);
}

fn benchRawrAddRange() void {
    var bm = RoaringBitmap.init(allocator) catch unreachable;
    defer bm.deinit();
    _ = bm.addRange(0, N_VALUES - 1) catch unreachable;
    std.mem.doNotOptimizeAway(&bm);
}

var rawr_contains_bm: ?RoaringBitmap = null;

fn initRawrContainsBm() void {
    if (rawr_contains_bm != null) return;
    var bm = RoaringBitmap.init(allocator) catch unreachable;
    for (random_values) |v| {
        _ = bm.add(v) catch unreachable;
    }
    rawr_contains_bm = bm;
}

fn benchRawrContainsHit() void {
    const bm = &rawr_contains_bm.?;
    var hits: u32 = 0;
    for (random_values) |v| {
        if (bm.contains(v)) hits += 1;
    }
    std.mem.doNotOptimizeAway(hits);
}

fn benchRawrContainsMiss() void {
    const bm = &rawr_contains_bm.?;
    var hits: u32 = 0;
    for (random_values) |v| {
        if (bm.contains(v | 0x80000000)) hits += 1;
    }
    std.mem.doNotOptimizeAway(hits);
}

var rawr_sparse_a: ?RoaringBitmap = null;
var rawr_sparse_b: ?RoaringBitmap = null;

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

fn benchRawrAndSparse() void {
    const a = &rawr_sparse_a.?;
    const b = &rawr_sparse_b.?;
    var result = a.bitwiseAnd(allocator, b) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrOrSparse() void {
    const a = &rawr_sparse_a.?;
    const b = &rawr_sparse_b.?;
    var result = a.bitwiseOr(allocator, b) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
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

var rawr_dense_a: ?RoaringBitmap = null;
var rawr_dense_b: ?RoaringBitmap = null;

fn initRawrDenseBitmaps() void {
    if (rawr_dense_a != null) return;

    var a = RoaringBitmap.init(allocator) catch unreachable;
    var b = RoaringBitmap.init(allocator) catch unreachable;

    _ = a.addRange(0, 499999) catch unreachable;
    _ = b.addRange(250000, 749999) catch unreachable;

    rawr_dense_a = a;
    rawr_dense_b = b;
}

fn benchRawrAndDense() void {
    const a = &rawr_dense_a.?;
    const b = &rawr_dense_b.?;
    var result = a.bitwiseAnd(allocator, b) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrOrDense() void {
    const a = &rawr_dense_a.?;
    const b = &rawr_dense_b.?;
    var result = a.bitwiseOr(allocator, b) catch unreachable;
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

var rawr_serialized: ?[]u8 = null;

fn initRawrSerialized() void {
    if (rawr_serialized != null) return;
    const bm = &rawr_contains_bm.?;
    rawr_serialized = RoaringBitmap.serialize(bm, allocator) catch unreachable;
}

fn benchRawrSerialize() void {
    const bm = &rawr_contains_bm.?;
    const bytes = RoaringBitmap.serialize(bm, allocator) catch unreachable;
    defer allocator.free(bytes);
    std.mem.doNotOptimizeAway(bytes.ptr);
}

fn benchRawrDeserialize() void {
    var bm = RoaringBitmap.deserialize(allocator, rawr_serialized.?) catch unreachable;
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

// ============================================================================
// CRoaring benchmarks
// ============================================================================

fn benchCRoaringAddRandom() void {
    const bm = c.roaring_bitmap_create() orelse unreachable;
    defer c.roaring_bitmap_free(bm);
    for (random_values) |v| {
        c.roaring_bitmap_add(bm, v);
    }
    std.mem.doNotOptimizeAway(bm);
}

fn benchCRoaringAddSequential() void {
    const bm = c.roaring_bitmap_create() orelse unreachable;
    defer c.roaring_bitmap_free(bm);
    for (sequential_values) |v| {
        c.roaring_bitmap_add(bm, v);
    }
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
    for (random_values) |v| {
        c.roaring_bitmap_add(bm, v);
    }
    cr_contains_bm = bm;
}

fn benchCRoaringContainsHit() void {
    const bm = cr_contains_bm.?;
    var hits: u32 = 0;
    for (random_values) |v| {
        if (c.roaring_bitmap_contains(bm, v)) hits += 1;
    }
    std.mem.doNotOptimizeAway(hits);
}

fn benchCRoaringContainsMiss() void {
    const bm = cr_contains_bm.?;
    var hits: u32 = 0;
    for (random_values) |v| {
        if (c.roaring_bitmap_contains(bm, v | 0x80000000)) hits += 1;
    }
    std.mem.doNotOptimizeAway(hits);
}

var cr_sparse_a: ?*c.roaring_bitmap_t = null;
var cr_sparse_b: ?*c.roaring_bitmap_t = null;

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

fn benchCRoaringAndSparse() void {
    const a = cr_sparse_a.?;
    const b_bm = cr_sparse_b.?;
    const result = c.roaring_bitmap_and(a, b_bm) orelse unreachable;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
}

fn benchCRoaringOrSparse() void {
    const a = cr_sparse_a.?;
    const b_bm = cr_sparse_b.?;
    const result = c.roaring_bitmap_or(a, b_bm) orelse unreachable;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
}

var cr_dense_a: ?*c.roaring_bitmap_t = null;
var cr_dense_b: ?*c.roaring_bitmap_t = null;

fn initCRoaringDenseBitmaps() void {
    if (cr_dense_a != null) return;

    const a = c.roaring_bitmap_create() orelse unreachable;
    const b_bm = c.roaring_bitmap_create() orelse unreachable;

    c.roaring_bitmap_add_range(a, 0, 500000);
    c.roaring_bitmap_add_range(b_bm, 250000, 750000);

    cr_dense_a = a;
    cr_dense_b = b_bm;
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
    const buf = allocator.alloc(u8, size) catch unreachable;
    _ = c.roaring_bitmap_portable_serialize(bm, @ptrCast(buf.ptr));
    cr_serialized = buf;
}

fn benchCRoaringSerialize() void {
    const bm = cr_contains_bm.?;
    const size = c.roaring_bitmap_portable_size_in_bytes(bm);
    const buf = allocator.alloc(u8, size) catch unreachable;
    defer allocator.free(buf);
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

// ============================================================================
// Main
// ============================================================================

pub fn main() !void {
    // Print header with timestamp
    const ts = std.time.timestamp();
    const epoch_seconds = std.time.epoch.EpochSeconds{ .secs = @intCast(ts) };
    const day_seconds = epoch_seconds.getDaySeconds();
    const year_day = epoch_seconds.getEpochDay().calculateYearDay();
    const month_day = year_day.calculateMonthDay();

    std.debug.print("Rawr vs CRoaring Benchmark Comparison\n", .{});
    std.debug.print("======================================\n", .{});
    std.debug.print("Run: {d}-{d:0>2}-{d:0>2} {d:0>2}:{d:0>2}:{d:0>2} UTC\n", .{
        year_day.year,
        @intFromEnum(month_day.month),
        month_day.day_index + 1,
        day_seconds.getHoursIntoDay(),
        day_seconds.getMinutesIntoHour(),
        day_seconds.getSecondsIntoMinute(),
    });
    std.debug.print("N = {d} values, {d} warmup, {d} timed runs (median)\n", .{ N_VALUES, WARMUP_RUNS, BENCH_RUNS });

    std.debug.print("\nInitializing test data...\n", .{});
    initTestData();

    // --- Add benchmarks ---
    printHeader();
    std.debug.print("ADD OPERATIONS\n", .{});

    var r = benchmark(benchRawrAddRandom, .{});
    var cr = benchmark(benchCRoaringAddRandom, .{});
    printResult("add (random 1M)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrAddSequential, .{});
    cr = benchmark(benchCRoaringAddSequential, .{});
    printResult("add (sequential 1M)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrAddRange, .{});
    cr = benchmark(benchCRoaringAddRange, .{});
    printResult("addRange (1M)", r.median_ns, cr.median_ns);

    // --- Contains benchmarks ---
    std.debug.print("\nCONTAINS OPERATIONS\n", .{});
    initRawrContainsBm();
    initCRoaringContainsBm();

    r = benchmark(benchRawrContainsHit, .{});
    cr = benchmark(benchCRoaringContainsHit, .{});
    printResult("contains (hit)", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrContainsMiss, .{});
    cr = benchmark(benchCRoaringContainsMiss, .{});
    printResult("contains (miss)", r.median_ns, cr.median_ns);

    // --- Set operations ---
    std.debug.print("\nSET OPERATIONS (new bitmap)\n", .{});
    initRawrSparseBitmaps();
    initCRoaringSparseBitmaps();
    initRawrDenseBitmaps();
    initCRoaringDenseBitmaps();

    r = benchmark(benchRawrAndSparse, .{});
    cr = benchmark(benchCRoaringAndSparse, .{});
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

    r = benchmark(benchRawrOrDense, .{});
    cr = benchmark(benchCRoaringOrDense, .{});
    printResult("bitwiseOr (dense)", r.median_ns, cr.median_ns);

    // --- Iteration ---
    std.debug.print("\nITERATION\n", .{});

    r = benchmark(benchRawrIterate, .{});
    cr = benchmark(benchCRoaringIterate, .{});
    printResult("iterate (1M values)", r.median_ns, cr.median_ns);

    // --- Serialization ---
    std.debug.print("\nSERIALIZATION\n", .{});
    initRawrSerialized();
    initCRoaringSerialized();

    r = benchmark(benchRawrSerialize, .{});
    cr = benchmark(benchCRoaringSerialize, .{});
    printResult("serialize", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrDeserialize, .{});
    cr = benchmark(benchCRoaringDeserialize, .{});
    printResult("deserialize", r.median_ns, cr.median_ns);

    r = benchmark(benchRawrDeserializeArena, .{});
    printResult("deserialize (arena)", r.median_ns, cr.median_ns);

    // --- Cardinality ---
    std.debug.print("\nCARDINALITY\n", .{});

    r = benchmark(benchRawrCardinality, .{});
    cr = benchmark(benchCRoaringCardinality, .{});
    printResult("cardinality", r.median_ns, cr.median_ns);

    // Cleanup
    if (rawr_contains_bm) |*bm| bm.deinit();
    if (rawr_sparse_a) |*bm| bm.deinit();
    if (rawr_sparse_b) |*bm| bm.deinit();
    if (rawr_dense_a) |*bm| bm.deinit();
    if (rawr_dense_b) |*bm| bm.deinit();
    if (rawr_serialized) |s| allocator.free(s);

    if (cr_contains_bm) |bm| c.roaring_bitmap_free(bm);
    if (cr_sparse_a) |bm| c.roaring_bitmap_free(bm);
    if (cr_sparse_b) |bm| c.roaring_bitmap_free(bm);
    if (cr_dense_a) |bm| c.roaring_bitmap_free(bm);
    if (cr_dense_b) |bm| c.roaring_bitmap_free(bm);
    if (cr_serialized) |s| allocator.free(s);

    std.debug.print("\nDone.\n", .{});
    std.debug.print("\nNote: ratio < 1.0 = rawr faster, > 1.0 = CRoaring faster\n", .{});
}
