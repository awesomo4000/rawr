// SPDX-License-Identifier: MPL-2.0

//! Repository-only select container-walk matrix for spec 34-00.

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");

const RoaringBitmap = rawr.RoaringBitmap;
const TaggedPtr = rawr.TaggedPtr;
const ContainerType = TaggedPtr.ContainerType;
const ops = rawr.container_ops;

const query_count = 1_000_000;
const canonical_cardinality = 500_000;
const seed = 12345;
const warmup_runs = 3;
const timed_runs = 21;
const prefix_build_calibration_runs = 4096;

const all_corpora = [_]CorpusKind{ .canonical, .array8, .bitset8, .mixed8, .run7 };
const all_paths = [_]Path{ .scalar, .unroll2, .unroll4, .prefix, .croaring };

const CorpusKind = enum {
    canonical,
    array8,
    bitset8,
    mixed8,
    run7,

    fn name(self: CorpusKind) []const u8 {
        return switch (self) {
            .canonical => "canonical",
            .array8 => "array-8",
            .bitset8 => "bitset-8",
            .mixed8 => "mixed-8",
            .run7 => "run-7-tail",
        };
    }

    fn parse(text: []const u8) ?CorpusKind {
        for (all_corpora) |kind| {
            if (std.mem.eql(u8, text, kind.name())) return kind;
        }
        return null;
    }
};

const Path = enum {
    scalar,
    unroll2,
    unroll4,
    prefix,
    croaring,

    fn name(self: Path) []const u8 {
        return switch (self) {
            .scalar => "scalar",
            .unroll2 => "unroll-2",
            .unroll4 => "unroll-4",
            .prefix => "prefix-ceiling",
            .croaring => "croaring",
        };
    }

    fn parse(text: []const u8) ?Path {
        for (all_paths) |path| {
            if (std.mem.eql(u8, text, path.name())) return path;
        }
        return null;
    }
};

const ContainerCounts = struct {
    arrays: u32 = 0,
    bitsets: u32 = 0,
    runs: u32 = 0,
};

const QueryStats = struct {
    min: u32 = std.math.maxInt(u32),
    max: u32 = 0,
    sum: u64 = 0,
    hash: u64 = 0xcbf29ce484222325,

    fn add(self: *QueryStats, query: u32) void {
        self.min = @min(self.min, query);
        self.max = @max(self.max, query);
        self.sum +%= query;
        self.hash = (self.hash ^ query) *% 0x100000001b3;
    }
};

const ExpectedQueryStats = struct {
    min: u32,
    max: u32,
    sum: u64,
    hash: u64,
};

const PrefixTable = struct {
    values: []u64,
    retained_build_ns: u64,
    calibrated_build_ns: u64,

    fn init(allocator: std.mem.Allocator, bitmap: *const RoaringBitmap) !PrefixTable {
        const start = bench_time.monotonicNanos();
        const values = try allocator.alloc(u64, bitmap.size + 1);
        errdefer allocator.free(values);

        fillPrefix(values, bitmap);
        const retained_build_ns = bench_time.monotonicNanos() - start;

        const calibration_start = bench_time.monotonicNanos();
        for (0..prefix_build_calibration_runs) |_| {
            const scratch = try allocator.alloc(u64, bitmap.size + 1);
            fillPrefix(scratch, bitmap);
            std.mem.doNotOptimizeAway(scratch[scratch.len - 1]);
            allocator.free(scratch);
        }
        const calibrated_build_ns =
            (bench_time.monotonicNanos() - calibration_start) / prefix_build_calibration_runs;

        return .{
            .values = values,
            .retained_build_ns = retained_build_ns,
            .calibrated_build_ns = calibrated_build_ns,
        };
    }

    fn deinit(self: *PrefixTable, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
    }

    fn footprintBytes(self: PrefixTable) usize {
        return self.values.len * @sizeOf(u64);
    }
};

const Corpus = struct {
    kind: CorpusKind,
    bitmap: RoaringBitmap,
    c_bitmap: *c.roaring_bitmap_t,
    cardinality: u64,
    queries: []u32,
    query_stats: QueryStats,
    prefix: PrefixTable,

    fn init(allocator: std.mem.Allocator, kind: CorpusKind) !Corpus {
        var bitmap = try RoaringBitmap.init(allocator);
        errdefer bitmap.deinit();
        const c_bitmap = c.roaring_bitmap_create() orelse return error.OutOfMemory;
        errdefer c.roaring_bitmap_free(c_bitmap);

        try buildCorpus(kind, &bitmap, c_bitmap);
        const cardinality = bitmap.cardinality();
        const prefix = try PrefixTable.init(allocator, &bitmap);
        errdefer {
            var owned_prefix = prefix;
            owned_prefix.deinit(allocator);
        }

        const queries = try allocator.alloc(u32, query_count);
        errdefer allocator.free(queries);
        const query_stats = fillQueries(kind, cardinality, queries);

        var result = Corpus{
            .kind = kind,
            .bitmap = bitmap,
            .c_bitmap = c_bitmap,
            .cardinality = cardinality,
            .queries = queries,
            .query_stats = query_stats,
            .prefix = prefix,
        };
        try result.assertInventory();
        return result;
    }

    fn deinit(self: *Corpus, allocator: std.mem.Allocator) void {
        self.prefix.deinit(allocator);
        allocator.free(self.queries);
        c.roaring_bitmap_free(self.c_bitmap);
        self.bitmap.deinit();
    }

    fn assertInventory(self: *const Corpus) !void {
        const expected_size = expectedContainerCount(self.kind);
        const expected_cardinality = expectedBitmapCardinality(self.kind);
        if (self.bitmap.size != expected_size) return error.UnexpectedContainerCount;
        if (self.cardinality != expected_cardinality) return error.UnexpectedCardinality;
        if (self.prefix.values.len != expected_size + 1) return error.UnexpectedPrefixLength;
        if (self.prefix.values[0] != 0) return error.InvalidPrefixOrigin;
        if (self.prefix.values[expected_size] != expected_cardinality) return error.InvalidPrefixTotal;

        for (0..expected_size) |i| {
            if (self.bitmap.keys[i] != @as(u16, @intCast(i))) return error.UnexpectedContainerKey;
            const tagged = self.bitmap.containers[i];
            if (tagged.getType() != expectedContainerType(self.kind, i)) return error.UnexpectedContainerType;
            const card = rawr.Container.fromTagged(tagged).getCardinality();
            if (card != expectedContainerCardinality(self.kind, i)) return error.UnexpectedContainerCardinality;
            if (self.prefix.values[i + 1] != self.prefix.values[i] + card) return error.InvalidPrefixValue;
        }

        for (self.queries) |query| {
            if (query >= self.cardinality) return error.QueryOutOfRange;
        }
        if (!queryStatsEqual(self.query_stats, expectedQueryStats(self.kind))) {
            return error.UnexpectedQueryStream;
        }

        const expected_counts = expectedContainerCounts(self.kind);
        const rawr_counts = rawrContainerCounts(&self.bitmap);
        if (!countsEqual(rawr_counts, expected_counts)) return error.UnexpectedRawrContainerMix;

        const c_counts_raw = c.rawr_cr_select_counts(self.c_bitmap);
        const c_counts = ContainerCounts{
            .arrays = c_counts_raw.arrays,
            .bitsets = c_counts_raw.bitsets,
            .runs = c_counts_raw.runs,
        };
        if (!countsEqual(c_counts, expected_counts)) return error.UnexpectedCRoaringContainerMix;
        if (c.roaring_bitmap_get_cardinality(self.c_bitmap) != expected_cardinality) {
            return error.CRoaringCardinalityMismatch;
        }
    }
};

const ScanResult = struct {
    count: u64 = 0,
    sum: u64 = 0,
};

const Validation = struct {
    valid_ranks: u64,
    boundary_cases: usize,
    expected: ScanResult,
};

const Measurement = struct {
    median_ns: u64,
    min_ns: u64,
    max_ns: u64,
    result: ScanResult,
};

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var requested_corpus: ?CorpusKind = null;
    var requested_path: ?Path = null;
    var header = false;
    var list = false;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.eql(u8, arg, "--list")) {
            list = true;
        } else if (std.mem.startsWith(u8, arg, "--corpus=")) {
            requested_corpus = CorpusKind.parse(arg[9..]) orelse return error.UnknownCorpus;
        } else if (std.mem.startsWith(u8, arg, "--path=")) {
            requested_path = Path.parse(arg[7..]) orelse return error.UnknownPath;
        } else {
            return error.UnknownArgument;
        }
    }

    if (header or list) {
        if (header and list) return error.ConflictingArguments;
        if (requested_corpus != null or requested_path != null) return error.ConflictingArguments;
        if (header) printHeader() else printList();
        return;
    }

    const corpus_kind = requested_corpus orelse return error.MissingCorpus;
    const path = requested_path orelse return error.MissingPath;
    const allocator = std.heap.smp_allocator;

    var corpus = try Corpus.init(allocator, corpus_kind);
    defer corpus.deinit(allocator);
    const validation = try validateCorpus(&corpus);
    const measurement = measurePath(path, &corpus);
    if (!scanResultsEqual(measurement.result, validation.expected)) return error.TimedChecksumMismatch;

    printTuple(&corpus, path, validation, measurement);
}

fn printHeader() void {
    bench_time.printBenchEnvironment();
    bench_time.print("PROTOCOL\tselect-kernel-matrix-v1\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        warmup_runs,
        timed_runs,
        5,
        query_count,
        seed,
    });
    bench_time.print("SCHEMA\tPROTOCOL\tname\twarmup_runs\ttimed_runs\tprocess_runs\tquery_count\tseed\n", .{});
    bench_time.print("SCHEMA\tCORPUS\tcorpus\tcardinality\tcontainers\tarrays\tbitsets\truns\n", .{});
    bench_time.print("SCHEMA\tQUERIES\tcorpus\tcount\tmin\tmax\tsum\tfnv1a64\n", .{});
    bench_time.print("SCHEMA\tPREFIX\tcorpus\tconvention\tentries\tbytes\tretained_build_ns\tcalibration_runs\tcalibrated_build_ns\n", .{});
    bench_time.print("SCHEMA\tVALIDATION\tcorpus\toracle\tvalid_ranks\tboundary_cases\tquery_count\tquery_sum\n", .{});
    bench_time.print("SCHEMA\tRESULT\tcorpus\tpath\tquery_count\tmedian_ns\tmin_ns\tmax_ns\tresult_count\tresult_sum\n", .{});
    bench_time.print("PREFIX_CONVENTION\thalf-open\tprefix[i]=sum(cardinality[0..i])\n", .{});
    bench_time.print("DISASM\trawr-scalar\trawrSelectScalarForBenchmark\n", .{});
    bench_time.print("DISASM\trawr-unroll-2\trawrSelectUnroll2ForBenchmark\n", .{});
    bench_time.print("DISASM\trawr-unroll-4\trawrSelectUnroll4ForBenchmark\n", .{});
    bench_time.print("DISASM\trawr-prefix-ceiling\trawrSelectPrefixForBenchmark\n", .{});
    bench_time.print("DISASM\tcroaring\trawr_cr_select_loop|roaring_bitmap_select\n", .{});
}

fn printList() void {
    for (all_corpora) |kind| {
        const counts = expectedContainerCounts(kind);
        bench_time.print("CORPUS_DEF\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
            kind.name(),
            expectedBitmapCardinality(kind),
            expectedContainerCount(kind),
            counts.arrays,
            counts.bitsets,
            counts.runs,
        });
    }
    for (all_paths) |path| {
        bench_time.print("PATH_DEF\t{s}\n", .{path.name()});
    }
    for (all_corpora) |kind| {
        for (all_paths) |path| {
            bench_time.print("TUPLE\t{s}\t{s}\n", .{ kind.name(), path.name() });
        }
    }
}

fn printTuple(corpus: *const Corpus, path: Path, validation: Validation, measurement: Measurement) void {
    const counts = rawrContainerCounts(&corpus.bitmap);
    bench_time.print("CORPUS\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        corpus.kind.name(),
        corpus.cardinality,
        corpus.bitmap.size,
        counts.arrays,
        counts.bitsets,
        counts.runs,
    });
    bench_time.print("QUERIES\t{s}\t{d}\t{d}\t{d}\t{d}\t{x}\n", .{
        corpus.kind.name(),
        corpus.queries.len,
        corpus.query_stats.min,
        corpus.query_stats.max,
        corpus.query_stats.sum,
        corpus.query_stats.hash,
    });
    bench_time.print("PREFIX\t{s}\thalf-open\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        corpus.kind.name(),
        corpus.prefix.values.len,
        corpus.prefix.footprintBytes(),
        corpus.prefix.retained_build_ns,
        prefix_build_calibration_runs,
        corpus.prefix.calibrated_build_ns,
    });
    bench_time.print("VALIDATION\t{s}\tall-rawr-kernels=croaring\t{d}\t{d}\t{d}\t{d}\n", .{
        corpus.kind.name(),
        validation.valid_ranks,
        validation.boundary_cases,
        validation.expected.count,
        validation.expected.sum,
    });
    bench_time.print("RESULT\t{s}\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        corpus.kind.name(),
        path.name(),
        corpus.queries.len,
        measurement.median_ns,
        measurement.min_ns,
        measurement.max_ns,
        measurement.result.count,
        measurement.result.sum,
    });
}

fn buildCorpus(kind: CorpusKind, bitmap: *RoaringBitmap, c_bitmap: *c.roaring_bitmap_t) !void {
    switch (kind) {
        .canonical => try addRunRange(bitmap, c_bitmap, 0, canonical_cardinality - 1),
        .array8 => {
            for (0..8) |key| try addArrayKey(bitmap, c_bitmap, key);
        },
        .bitset8 => {
            for (0..8) |key| try addBitsetKey(bitmap, c_bitmap, key);
        },
        .mixed8 => {
            for (0..8) |key| {
                switch (expectedContainerType(.mixed8, key)) {
                    .array => try addArrayKey(bitmap, c_bitmap, key),
                    .bitset => try addBitsetKey(bitmap, c_bitmap, key),
                    .run => {
                        const base = @as(u32, @intCast(key)) << 16;
                        try addRunRange(bitmap, c_bitmap, base, base + 12_000);
                    },
                    .reserved => unreachable,
                }
            }
        },
        .run7 => {
            try addRunRange(bitmap, c_bitmap, 0, canonical_cardinality - 1);
            const key7_base = @as(u32, 7) << 16;
            const removed = try bitmap.removeRange(key7_base, canonical_cardinality - 1);
            if (removed != canonical_cardinality - key7_base) return error.UnexpectedRemovedCardinality;
            c.roaring_bitmap_remove_range_closed(c_bitmap, key7_base, canonical_cardinality - 1);
        },
    }
}

fn addArrayKey(bitmap: *RoaringBitmap, c_bitmap: *c.roaring_bitmap_t, key: usize) !void {
    const base = @as(u32, @intCast(key)) << 16;
    for (0..2048) |low| {
        const value = base | @as(u32, @intCast(low));
        _ = try bitmap.add(value);
        c.roaring_bitmap_add(c_bitmap, value);
    }
}

fn addBitsetKey(bitmap: *RoaringBitmap, c_bitmap: *c.roaring_bitmap_t, key: usize) !void {
    const base = @as(u32, @intCast(key)) << 16;
    for (0..6000) |j| {
        const value = base | @as(u32, @intCast(2 * j));
        _ = try bitmap.add(value);
        c.roaring_bitmap_add(c_bitmap, value);
    }
}

fn addRunRange(bitmap: *RoaringBitmap, c_bitmap: *c.roaring_bitmap_t, lo: u32, hi: u32) !void {
    _ = try bitmap.addRange(lo, hi);
    c.roaring_bitmap_add_range(c_bitmap, lo, @as(u64, hi) + 1);
}

fn fillQueries(kind: CorpusKind, cardinality: u64, queries: []u32) QueryStats {
    std.debug.assert(queries.len == query_count);
    var stats = QueryStats{};
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    if (kind == .canonical) {
        for (queries) |*query| {
            // Match the parity harness: select is the third draw in each iteration.
            _ = random.int(u32);
            _ = random.uintLessThan(u32, canonical_cardinality);
            query.* = random.uintLessThan(u32, canonical_cardinality);
            _ = random.uintLessThan(u32, 50_000);
            _ = random.uintLessThan(u32, 1024);
            _ = random.uintLessThan(u32, 20_000);
            _ = random.uintLessThan(u32, 20_000);
            stats.add(query.*);
        }
    } else {
        for (queries) |*query| {
            query.* = @intCast(random.uintLessThan(u64, cardinality));
            stats.add(query.*);
        }
    }

    return stats;
}

fn fillPrefix(values: []u64, bitmap: *const RoaringBitmap) void {
    std.debug.assert(values.len == bitmap.size + 1);
    values[0] = 0;
    for (bitmap.containers[0..bitmap.size], 0..) |tagged, i| {
        values[i + 1] = values[i] + rawr.Container.fromTagged(tagged).getCardinality();
    }
}

fn expectedQueryStats(kind: CorpusKind) ExpectedQueryStats {
    return switch (kind) {
        .canonical => .{ .min = 0, .max = 499_999, .sum = 250_254_666_487, .hash = 0xde58fda1d5e2f820 },
        .array8 => .{ .min = 0, .max = 16_383, .sum = 8_189_559_715, .hash = 0xd6f70a81b43ce004 },
        .bitset8 => .{ .min = 0, .max = 47_999, .sum = 23_993_816_270, .hash = 0x30e35b97f34c0cff },
        .mixed8 => .{ .min = 0, .max = 48_145, .sum = 24_066_799_051, .hash = 0xbc53ad83fa943420 },
        .run7 => .{ .min = 1, .max = 458_751, .sum = 229_321_178_148, .hash = 0x05f3e7a84a961f99 },
    };
}

fn queryStatsEqual(actual: QueryStats, expected: ExpectedQueryStats) bool {
    return actual.min == expected.min and
        actual.max == expected.max and
        actual.sum == expected.sum and
        actual.hash == expected.hash;
}

fn expectedContainerCount(kind: CorpusKind) usize {
    return if (kind == .run7) 7 else 8;
}

fn expectedBitmapCardinality(kind: CorpusKind) u64 {
    return switch (kind) {
        .canonical => canonical_cardinality,
        .array8 => 8 * 2048,
        .bitset8 => 8 * 6000,
        .mixed8 => 3 * 2048 + 3 * 6000 + 2 * 12_001,
        .run7 => 7 * 65_536,
    };
}

fn expectedContainerCounts(kind: CorpusKind) ContainerCounts {
    return switch (kind) {
        .canonical => .{ .runs = 8 },
        .array8 => .{ .arrays = 8 },
        .bitset8 => .{ .bitsets = 8 },
        .mixed8 => .{ .arrays = 3, .bitsets = 3, .runs = 2 },
        .run7 => .{ .runs = 7 },
    };
}

fn expectedContainerType(kind: CorpusKind, index: usize) ContainerType {
    return switch (kind) {
        .canonical, .run7 => .run,
        .array8 => .array,
        .bitset8 => .bitset,
        .mixed8 => switch (index) {
            0, 3, 6 => .array,
            1, 4, 7 => .bitset,
            2, 5 => .run,
            else => unreachable,
        },
    };
}

fn expectedContainerCardinality(kind: CorpusKind, index: usize) u32 {
    return switch (kind) {
        .canonical => if (index < 7) 65_536 else canonical_cardinality - 7 * 65_536,
        .array8 => 2048,
        .bitset8 => 6000,
        .mixed8 => switch (expectedContainerType(kind, index)) {
            .array => 2048,
            .bitset => 6000,
            .run => 12_001,
            .reserved => unreachable,
        },
        .run7 => 65_536,
    };
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

fn countsEqual(a: ContainerCounts, b: ContainerCounts) bool {
    return a.arrays == b.arrays and a.bitsets == b.bitsets and a.runs == b.runs;
}

fn combine(key: u16, low: u16) u32 {
    return (@as(u32, key) << 16) | low;
}

inline fn selectContainer(key: u16, tagged: TaggedPtr, remaining: *u32) ?u32 {
    switch (tagged.getType()) {
        .array => {
            const container = tagged.getArray();
            if (remaining.* < container.cardinality) {
                return combine(key, container.values[@intCast(remaining.*)]);
            }
            remaining.* -= container.cardinality;
        },
        .bitset => {
            const container = tagged.getBitset();
            const card = container.getCardinality();
            if (remaining.* < card) {
                const low = ops.containerSelect(.{ .bitset = container }, remaining.*) orelse return null;
                return combine(key, low);
            }
            remaining.* -= card;
        },
        .run => {
            const container = tagged.getRun();
            const card = container.getCardinality();
            if (remaining.* < card) {
                const low = ops.containerSelect(.{ .run = container }, remaining.*) orelse return null;
                return combine(key, low);
            }
            remaining.* -= card;
        },
        .reserved => unreachable,
    }
    return null;
}

noinline fn rawrSelectScalarForBenchmark(bitmap: *const RoaringBitmap, rank: u64) ?u32 {
    return @call(.always_inline, RoaringBitmap.select, .{ bitmap, rank });
}

noinline fn rawrSelectUnroll2ForBenchmark(bitmap: *const RoaringBitmap, rank: u64) ?u32 {
    if (rank > std.math.maxInt(u32)) return null;
    var remaining: u32 = @intCast(rank);
    var i: usize = 0;

    while (i + 2 <= bitmap.size) : (i += 2) {
        if (selectContainer(bitmap.keys[i], bitmap.containers[i], &remaining)) |value| return value;
        if (selectContainer(bitmap.keys[i + 1], bitmap.containers[i + 1], &remaining)) |value| return value;
    }
    while (i < bitmap.size) : (i += 1) {
        if (selectContainer(bitmap.keys[i], bitmap.containers[i], &remaining)) |value| return value;
    }
    return null;
}

noinline fn rawrSelectUnroll4ForBenchmark(bitmap: *const RoaringBitmap, rank: u64) ?u32 {
    if (rank > std.math.maxInt(u32)) return null;
    var remaining: u32 = @intCast(rank);
    var i: usize = 0;

    while (i + 4 <= bitmap.size) : (i += 4) {
        if (selectContainer(bitmap.keys[i], bitmap.containers[i], &remaining)) |value| return value;
        if (selectContainer(bitmap.keys[i + 1], bitmap.containers[i + 1], &remaining)) |value| return value;
        if (selectContainer(bitmap.keys[i + 2], bitmap.containers[i + 2], &remaining)) |value| return value;
        if (selectContainer(bitmap.keys[i + 3], bitmap.containers[i + 3], &remaining)) |value| return value;
    }
    while (i < bitmap.size) : (i += 1) {
        if (selectContainer(bitmap.keys[i], bitmap.containers[i], &remaining)) |value| return value;
    }
    return null;
}

noinline fn rawrSelectPrefixForBenchmark(
    bitmap: *const RoaringBitmap,
    prefix: []const u64,
    rank: u64,
) ?u32 {
    if (rank > std.math.maxInt(u32)) return null;
    if (prefix.len != bitmap.size + 1 or rank >= prefix[prefix.len - 1]) return null;

    var lo: usize = 0;
    var hi: usize = bitmap.size;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (prefix[mid + 1] <= rank) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    var remaining: u32 = @intCast(rank - prefix[lo]);
    return selectContainer(bitmap.keys[lo], bitmap.containers[lo], &remaining);
}

fn scanRawrScalar(bitmap: *const RoaringBitmap, queries: []const u32) ScanResult {
    var result = ScanResult{};
    for (queries) |query| {
        const value = rawrSelectScalarForBenchmark(bitmap, query) orelse continue;
        result.count +%= 1;
        result.sum +%= value;
    }
    return result;
}

fn scanRawrUnroll2(bitmap: *const RoaringBitmap, queries: []const u32) ScanResult {
    var result = ScanResult{};
    for (queries) |query| {
        const value = rawrSelectUnroll2ForBenchmark(bitmap, query) orelse continue;
        result.count +%= 1;
        result.sum +%= value;
    }
    return result;
}

fn scanRawrUnroll4(bitmap: *const RoaringBitmap, queries: []const u32) ScanResult {
    var result = ScanResult{};
    for (queries) |query| {
        const value = rawrSelectUnroll4ForBenchmark(bitmap, query) orelse continue;
        result.count +%= 1;
        result.sum +%= value;
    }
    return result;
}

fn scanRawrPrefix(bitmap: *const RoaringBitmap, prefix: []const u64, queries: []const u32) ScanResult {
    var result = ScanResult{};
    for (queries) |query| {
        const value = rawrSelectPrefixForBenchmark(bitmap, prefix, query) orelse continue;
        result.count +%= 1;
        result.sum +%= value;
    }
    return result;
}

fn scanCRoaring(bitmap: *const c.roaring_bitmap_t, queries: []const u32) ScanResult {
    const result = c.rawr_cr_select_loop(bitmap, queries.ptr, queries.len);
    return .{ .count = result.count, .sum = result.sum };
}

fn measurePath(path: Path, corpus: *const Corpus) Measurement {
    return switch (path) {
        .scalar => measure(scanRawrScalar, .{ &corpus.bitmap, corpus.queries }),
        .unroll2 => measure(scanRawrUnroll2, .{ &corpus.bitmap, corpus.queries }),
        .unroll4 => measure(scanRawrUnroll4, .{ &corpus.bitmap, corpus.queries }),
        .prefix => measure(scanRawrPrefix, .{ &corpus.bitmap, corpus.prefix.values, corpus.queries }),
        .croaring => measure(scanCRoaring, .{ corpus.c_bitmap, corpus.queries }),
    };
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
    return .{
        .median_ns = times[timed_runs / 2],
        .min_ns = times[0],
        .max_ns = times[timed_runs - 1],
        .result = last,
    };
}

fn validateCorpus(corpus: *const Corpus) !Validation {
    var rank: u64 = 0;
    while (rank < corpus.cardinality) : (rank += 1) {
        const expected = try validateRawrRank(&corpus.bitmap, corpus.prefix.values, rank);
        var c_value: u32 = undefined;
        if (!c.roaring_bitmap_select(corpus.c_bitmap, @intCast(rank), &c_value)) {
            return error.CRoaringValidRankMissing;
        }
        if (c_value != expected) return error.CRoaringValidRankMismatch;
    }

    const boundary_cases = try validateBoundaries(corpus);
    const expected = scanRawrScalar(&corpus.bitmap, corpus.queries);
    const c_result = scanCRoaring(corpus.c_bitmap, corpus.queries);
    if (!scanResultsEqual(c_result, expected)) return error.CRoaringQueryChecksumMismatch;

    return .{
        .valid_ranks = corpus.cardinality,
        .boundary_cases = boundary_cases,
        .expected = expected,
    };
}

fn validateBoundaries(corpus: *const Corpus) !usize {
    var cases: usize = 0;

    for (corpus.prefix.values[0..corpus.bitmap.size]) |rank| {
        const expected = try validateRawrRank(&corpus.bitmap, corpus.prefix.values, rank);
        var c_value: u32 = undefined;
        if (!c.roaring_bitmap_select(corpus.c_bitmap, @intCast(rank), &c_value)) {
            return error.CRoaringBoundaryMissing;
        }
        if (c_value != expected) return error.CRoaringBoundaryMismatch;
        cases += 1;
    }

    _ = try validateRawrRank(&corpus.bitmap, corpus.prefix.values, corpus.cardinality - 1);
    cases += 1;

    if (try validateRawrRank(&corpus.bitmap, corpus.prefix.values, corpus.cardinality) != null) {
        return error.RawrOutOfRangePresent;
    }
    var ignored: u32 = undefined;
    if (c.roaring_bitmap_select(corpus.c_bitmap, @intCast(corpus.cardinality), &ignored)) {
        return error.CRoaringOutOfRangePresent;
    }
    cases += 1;

    const above_u32 = @as(u64, std.math.maxInt(u32)) + 1;
    if (try validateRawrRank(&corpus.bitmap, corpus.prefix.values, above_u32) != null) {
        return error.RawrWideRankPresent;
    }
    cases += 1;
    if (try validateRawrRank(&corpus.bitmap, corpus.prefix.values, std.math.maxInt(u64)) != null) {
        return error.RawrMaxRankPresent;
    }
    cases += 1;

    var empty = try RoaringBitmap.init(std.heap.smp_allocator);
    defer empty.deinit();
    const empty_prefix = [_]u64{0};
    if (try validateRawrRank(&empty, &empty_prefix, 0) != null) return error.RawrEmptyPresent;
    if (try validateRawrRank(&empty, &empty_prefix, std.math.maxInt(u64)) != null) return error.RawrEmptyMaxPresent;
    const c_empty = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(c_empty);
    if (c.roaring_bitmap_select(c_empty, 0, &ignored)) return error.CRoaringEmptyPresent;
    cases += 2;

    return cases;
}

fn validateRawrRank(bitmap: *const RoaringBitmap, prefix: []const u64, rank: u64) !?u32 {
    const expected = bitmap.select(rank);
    if (rawrSelectScalarForBenchmark(bitmap, rank) != expected) return error.ScalarSelectMismatch;
    if (rawrSelectUnroll2ForBenchmark(bitmap, rank) != expected) return error.Unroll2SelectMismatch;
    if (rawrSelectUnroll4ForBenchmark(bitmap, rank) != expected) return error.Unroll4SelectMismatch;
    if (rawrSelectPrefixForBenchmark(bitmap, prefix, rank) != expected) return error.PrefixSelectMismatch;
    return expected;
}

fn scanResultsEqual(a: ScanResult, b: ScanResult) bool {
    return a.count == b.count and a.sum == b.sum;
}
