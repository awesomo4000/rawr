// SPDX-License-Identifier: MPL-2.0

//! Fresh-process select call-boundary and cost-attribution benchmark for spec 24-00.

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");

const RoaringBitmap = rawr.RoaringBitmap;
const Container = rawr.Container;
const ops = rawr.container_ops;

const query_count = 1_000_000;
const bitmap_cardinality = 500_000;
const seed = 12345;
const warmup_runs = 3;
const timed_runs = 21;

const Path = enum {
    rawr_inline,
    rawr_noinline,
    croaring_zig,
    croaring_c,
    checksum_baseline,
    rawr_skip,
    rawr_intra,

    fn name(self: Path) []const u8 {
        return switch (self) {
            .rawr_inline => "rawr-inline",
            .rawr_noinline => "rawr-noinline",
            .croaring_zig => "croaring-zig",
            .croaring_c => "croaring-c",
            .checksum_baseline => "checksum-baseline",
            .rawr_skip => "rawr-skip",
            .rawr_intra => "rawr-intra",
        };
    }

    fn parse(text: []const u8) ?Path {
        inline for (std.meta.fields(Path)) |field| {
            const value: Path = @enumFromInt(field.value);
            if (std.mem.eql(u8, text, value.name())) return value;
        }
        return null;
    }
};

const ScanResult = struct {
    count: u64 = 0,
    sum: u64 = 0,
};

const SelectTarget = struct {
    container: Container,
    key: u16,
    local_rank: u32,
};

const ContainerCounts = struct {
    arrays: u32 = 0,
    bitsets: u32 = 0,
    runs: u32 = 0,
};

const Corpus = struct {
    queries: []u32,
    expected: []u32,
    targets: []SelectTarget,
    target_hits: []u32,
    rank_min: u32,
    rank_max: u32,
    rank_sum: u64,

    fn init(allocator: std.mem.Allocator, bitmap: *const RoaringBitmap) !Corpus {
        const queries = try allocator.alloc(u32, query_count);
        errdefer allocator.free(queries);
        const expected = try allocator.alloc(u32, query_count);
        errdefer allocator.free(expected);
        const targets = try allocator.alloc(SelectTarget, query_count);
        errdefer allocator.free(targets);
        const target_hits = try allocator.alloc(u32, bitmap.size);
        errdefer allocator.free(target_hits);
        @memset(target_hits, 0);

        fillCanonicalSelectQueries(queries);
        var rank_min: u32 = std.math.maxInt(u32);
        var rank_max: u32 = 0;
        var rank_sum: u64 = 0;
        for (queries, expected, targets) |query, *value, *target| {
            rank_min = @min(rank_min, query);
            rank_max = @max(rank_max, query);
            rank_sum += query;

            const found = findSelectTarget(bitmap, query) orelse return error.InvalidCanonicalRank;
            target.* = found.target;
            target_hits[found.index] += 1;
            value.* = selectTarget(found.target) orelse return error.InvalidCanonicalRank;
        }

        return .{
            .queries = queries,
            .expected = expected,
            .targets = targets,
            .target_hits = target_hits,
            .rank_min = rank_min,
            .rank_max = rank_max,
            .rank_sum = rank_sum,
        };
    }

    fn deinit(self: *Corpus, allocator: std.mem.Allocator) void {
        allocator.free(self.queries);
        allocator.free(self.expected);
        allocator.free(self.targets);
        allocator.free(self.target_hits);
    }
};

const Measurement = struct {
    median_ns: u64,
    result: ScanResult,
};

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var requested_path: ?Path = null;
    var header = false;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.startsWith(u8, arg, "--path=")) {
            requested_path = Path.parse(arg[7..]) orelse return error.UnknownPath;
        } else {
            return error.UnknownArgument;
        }
    }

    if (header) {
        if (requested_path != null) return error.ConflictingArguments;
        bench_time.printBenchEnvironment();
        bench_time.print("# select diagnosis protocol: {d}w/{d}t median; seed={d}; queries={d}\n", .{
            warmup_runs,
            timed_runs,
            seed,
            query_count,
        });
        return;
    }

    const path = requested_path orelse return error.MissingPath;
    const allocator = std.heap.smp_allocator;

    var bitmap = try RoaringBitmap.init(allocator);
    defer bitmap.deinit();
    _ = try bitmap.addRange(0, bitmap_cardinality - 1);

    const c_bitmap = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(c_bitmap);
    c.roaring_bitmap_add_range(c_bitmap, 0, bitmap_cardinality);

    var corpus = try Corpus.init(allocator, &bitmap);
    defer corpus.deinit(allocator);
    try validateCorpus(&bitmap, c_bitmap, &corpus);

    const measurement = switch (path) {
        .rawr_inline => measure(scanRawrInline, .{ &bitmap, corpus.queries }),
        .rawr_noinline => measure(scanRawrNoInline, .{ &bitmap, corpus.queries }),
        .croaring_zig => measure(scanCRoaringZig, .{ c_bitmap, corpus.queries }),
        .croaring_c => measure(scanCRoaringC, .{ c_bitmap, corpus.queries }),
        .checksum_baseline => measure(scanChecksumBaseline, .{corpus.expected}),
        .rawr_skip => measure(scanRawrSkip, .{ &bitmap, corpus.queries }),
        .rawr_intra => measure(scanRawrIntra, .{corpus.targets}),
    };
    try validateMeasurement(path, &corpus, measurement.result);
    printResult(path, &bitmap, c_bitmap, &corpus, measurement);
}

fn fillCanonicalSelectQueries(queries: []u32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (queries) |*query| {
        // Keep this draw sequence aligned with bench_croaring.initTestData.
        _ = random.int(u32);
        _ = random.uintLessThan(u32, bitmap_cardinality);
        query.* = random.uintLessThan(u32, bitmap_cardinality);
        _ = random.uintLessThan(u32, 50_000);
        _ = random.uintLessThan(u32, 1024);
        _ = random.uintLessThan(u32, 20_000);
        _ = random.uintLessThan(u32, 20_000);
    }
}

fn measure(comptime scan: anytype, args: anytype) Measurement {
    var last = ScanResult{};
    for (0..warmup_runs) |_| {
        last = @call(.auto, scan, args);
        std.mem.doNotOptimizeAway(last);
    }

    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        last = @call(.auto, scan, args);
        elapsed.* = bench_time.monotonicNanos() - start;
        std.mem.doNotOptimizeAway(last);
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return .{ .median_ns = times[timed_runs / 2], .result = last };
}

noinline fn scanRawrInline(bitmap: *const RoaringBitmap, queries: []const u32) ScanResult {
    var result = ScanResult{};
    for (queries) |query| {
        const value = @call(.always_inline, RoaringBitmap.select, .{ bitmap, query }) orelse continue;
        result.count +%= 1;
        result.sum +%= value;
    }
    return result;
}

noinline fn rawrSelectNoInline(bitmap: *const RoaringBitmap, query: u32) ?u32 {
    return @call(.always_inline, RoaringBitmap.select, .{ bitmap, query });
}

noinline fn scanRawrNoInline(bitmap: *const RoaringBitmap, queries: []const u32) ScanResult {
    var result = ScanResult{};
    for (queries) |query| {
        const value = rawrSelectNoInline(bitmap, query) orelse continue;
        result.count +%= 1;
        result.sum +%= value;
    }
    return result;
}

noinline fn scanCRoaringZig(bitmap: *const c.roaring_bitmap_t, queries: []const u32) ScanResult {
    var result = ScanResult{};
    for (queries) |query| {
        var value: u32 = undefined;
        if (!c.roaring_bitmap_select(bitmap, query, &value)) continue;
        result.count +%= 1;
        result.sum +%= value;
    }
    return result;
}

noinline fn scanCRoaringC(bitmap: *const c.roaring_bitmap_t, queries: []const u32) ScanResult {
    const result = c.rawr_cr_select_loop(bitmap, queries.ptr, queries.len);
    return .{ .count = result.count, .sum = result.sum };
}

noinline fn scanChecksumBaseline(values: []const u32) ScanResult {
    var result = ScanResult{};
    for (values) |value| {
        result.count +%= 1;
        result.sum +%= value;
    }
    return result;
}

noinline fn scanRawrSkip(bitmap: *const RoaringBitmap, queries: []const u32) ScanResult {
    var result = ScanResult{};
    for (queries) |query| {
        const found = findSelectTarget(bitmap, query) orelse continue;
        result.count +%= 1;
        result.sum +%= (@as(u64, found.target.key) << 32) | found.target.local_rank;
    }
    return result;
}

noinline fn scanRawrIntra(targets: []const SelectTarget) ScanResult {
    var result = ScanResult{};
    for (targets) |target| {
        const value = selectTarget(target) orelse continue;
        result.count +%= 1;
        result.sum +%= value;
    }
    return result;
}

const FoundTarget = struct {
    target: SelectTarget,
    index: usize,
};

inline fn findSelectTarget(bitmap: *const RoaringBitmap, rank: u64) ?FoundTarget {
    var prior: u64 = 0;
    for (bitmap.keys[0..bitmap.size], bitmap.containers[0..bitmap.size], 0..) |key, tagged, index| {
        const container = Container.fromTagged(tagged);
        const card = container.getCardinality();
        if (rank < prior + card) {
            return .{
                .target = .{
                    .container = container,
                    .key = key,
                    .local_rank = @intCast(rank - prior),
                },
                .index = index,
            };
        }
        prior += card;
    }
    return null;
}

inline fn selectTarget(target: SelectTarget) ?u32 {
    const low = ops.containerSelect(target.container, target.local_rank) orelse return null;
    return (@as(u32, target.key) << 16) | low;
}

fn validateCorpus(bitmap: *const RoaringBitmap, c_bitmap: *const c.roaring_bitmap_t, corpus: *const Corpus) !void {
    for (corpus.queries, corpus.expected) |query, expected| {
        if (bitmap.select(query) != expected) return error.RawrQueryMismatch;
        var c_value: u32 = undefined;
        if (!c.roaring_bitmap_select(c_bitmap, query, &c_value)) return error.CRoaringQueryMissing;
        if (c_value != expected) return error.CRoaringQueryMismatch;
    }

    const boundary_ranks = [_]u32{ 0, 65_535, 65_536, bitmap_cardinality - 1 };
    for (boundary_ranks) |rank| {
        const expected = bitmap.select(rank) orelse return error.RawrBoundaryMissing;
        var c_value: u32 = undefined;
        if (!c.roaring_bitmap_select(c_bitmap, rank, &c_value)) return error.CRoaringBoundaryMissing;
        if (c_value != expected) return error.BoundaryMismatch;
    }
    if (bitmap.select(bitmap_cardinality) != null) return error.RawrOutOfRangePresent;
    var ignored: u32 = undefined;
    if (c.roaring_bitmap_select(c_bitmap, bitmap_cardinality, &ignored)) return error.CRoaringOutOfRangePresent;

    var empty = try RoaringBitmap.init(std.heap.smp_allocator);
    defer empty.deinit();
    if (empty.select(0) != null) return error.RawrEmptyPresent;
    const c_empty = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(c_empty);
    if (c.roaring_bitmap_select(c_empty, 0, &ignored)) return error.CRoaringEmptyPresent;
}

fn validateMeasurement(path: Path, corpus: *const Corpus, measured: ScanResult) !void {
    const expected = switch (path) {
        .rawr_skip => scanRawrSkipExpected(corpus.targets),
        else => scanChecksumBaseline(corpus.expected),
    };
    if (measured.count != expected.count or measured.sum != expected.sum) {
        return error.TimedChecksumMismatch;
    }
    bench_time.print("VALIDATION\t{s}\t{d}\t{d}\n", .{ path.name(), measured.count, measured.sum });
}

fn scanRawrSkipExpected(targets: []const SelectTarget) ScanResult {
    var result = ScanResult{};
    for (targets) |target| {
        result.count += 1;
        result.sum +%= (@as(u64, target.key) << 32) | target.local_rank;
    }
    return result;
}

fn rawrContainerCounts(bitmap: *const RoaringBitmap) ContainerCounts {
    var counts = ContainerCounts{};
    for (bitmap.containers[0..bitmap.size]) |tagged| {
        switch (tagged.getType()) {
            .array => counts.arrays += 1,
            .bitset => counts.bitsets += 1,
            .run => counts.runs += 1,
            .reserved => unreachable,
        }
    }
    return counts;
}

fn printResult(
    path: Path,
    bitmap: *const RoaringBitmap,
    c_bitmap: *const c.roaring_bitmap_t,
    corpus: *const Corpus,
    measurement: Measurement,
) void {
    const rawr_counts = rawrContainerCounts(bitmap);
    const c_counts = c.rawr_cr_select_counts(c_bitmap);
    bench_time.print("CORPUS\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        path.name(),
        corpus.queries.len,
        rawr_counts.arrays,
        rawr_counts.bitsets,
        rawr_counts.runs,
        c_counts.arrays,
        c_counts.bitsets,
        c_counts.runs,
    });
    bench_time.print("RANKS\t{s}\t{d}\t{d}\t{d}\n", .{
        path.name(),
        corpus.rank_min,
        corpus.rank_max,
        corpus.rank_sum,
    });
    for (corpus.target_hits, 0..) |hits, index| {
        bench_time.print("TARGET\t{s}\t{d}\t{d}\n", .{ path.name(), index, hits });
    }
    bench_time.print("RESULT\t{s}\t{d}\t{d}\t{d}\n", .{
        path.name(),
        measurement.result.count,
        measurement.result.sum,
        measurement.median_ns,
    });
}
