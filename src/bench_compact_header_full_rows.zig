// SPDX-License-Identifier: MPL-2.0

//! Rawr-only full-row worker for the spec-32 three-way layout experiment.

const std = @import("std");
const rawr = @import("rawr");
const bench_time = @import("bench_time.zig");

const RoaringBitmap = rawr.RoaringBitmap;
const warmup_runs = 3;
const timed_runs = 21;
const dense_batch = 8192;
const value_count = 500_000;
const sparse_seed = 54_321;
const select_seed = 12_345;

const Row = enum {
    clone,
    dense_and,
    select,
    lazy_or_construction,

    fn name(self: Row) []const u8 {
        return switch (self) {
            .clone => "clone",
            .dense_and => "dense-and",
            .select => "select",
            .lazy_or_construction => "lazy-or-construction",
        };
    }

    fn parse(text: []const u8) ?Row {
        inline for (std.meta.fields(Row)) |field| {
            const row: Row = @enumFromInt(field.value);
            if (std.mem.eql(u8, text, row.name())) return row;
        }
        return null;
    }

    fn batch(self: Row) usize {
        return switch (self) {
            .clone, .dense_and => dense_batch,
            .select, .lazy_or_construction => 1,
        };
    }
};

const Measurement = struct {
    median_ns: u64,
    min_ns: u64,
    max_ns: u64,
    checksum: u64,
};

const DenseCorpus = struct {
    a: RoaringBitmap,
    b: RoaringBitmap,
    queries: []u32,

    fn init(allocator: std.mem.Allocator) !DenseCorpus {
        var a = try RoaringBitmap.init(allocator);
        errdefer a.deinit();
        var b = try RoaringBitmap.init(allocator);
        errdefer b.deinit();
        _ = try a.addRange(0, 499_999);
        _ = try b.addRange(250_000, 749_999);

        const queries = try allocator.alloc(u32, 1_000_000);
        errdefer allocator.free(queries);
        fillCanonicalSelectQueries(queries);
        return .{ .a = a, .b = b, .queries = queries };
    }

    fn deinit(self: *DenseCorpus, allocator: std.mem.Allocator) void {
        allocator.free(self.queries);
        self.b.deinit();
        self.a.deinit();
    }
};

const SparseCorpus = struct {
    values: []u32,
    a: RoaringBitmap,
    b: RoaringBitmap,

    fn init(allocator: std.mem.Allocator) !SparseCorpus {
        const values = try allocator.alloc(u32, value_count);
        errdefer allocator.free(values);
        var prng = std.Random.DefaultPrng.init(sparse_seed);
        const random = prng.random();
        for (values) |*value| value.* = random.int(u32);
        std.mem.sort(u32, values, {}, std.sort.asc(u32));
        const sparse_len = dedupeSorted(values);
        const half = sparse_len / 2;

        var a = try RoaringBitmap.init(allocator);
        errdefer a.deinit();
        var b = try RoaringBitmap.init(allocator);
        errdefer b.deinit();
        for (values[0..half]) |value| _ = try a.add(value);
        for (values[half / 2 .. sparse_len]) |value| _ = try b.add(value);
        return .{ .values = values, .a = a, .b = b };
    }

    fn deinit(self: *SparseCorpus, allocator: std.mem.Allocator) void {
        self.b.deinit();
        self.a.deinit();
        allocator.free(self.values);
    }
};

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var row: ?Row = null;
    var header = false;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.startsWith(u8, arg, "--row=")) {
            row = Row.parse(arg[6..]) orelse return error.UnknownRow;
        } else {
            return error.UnknownArgument;
        }
    }
    if (header) {
        if (row != null) return error.ConflictingArguments;
        bench_time.printBenchEnvironment();
        bench_time.print("PROTOCOL\tcompact-header-full-rows-v1\t{d}\t{d}\t5\n", .{
            warmup_runs,
            timed_runs,
        });
        return;
    }

    const selected = row orelse return error.MissingRow;
    const allocator = std.heap.smp_allocator;
    const measurement = switch (selected) {
        .clone, .dense_and, .select => result: {
            var corpus = try DenseCorpus.init(allocator);
            defer corpus.deinit(allocator);
            try validateDenseCorpus(allocator, &corpus);
            break :result measureDense(selected, allocator, &corpus);
        },
        .lazy_or_construction => result: {
            var corpus = try SparseCorpus.init(allocator);
            defer corpus.deinit(allocator);
            try validateSparseCorpus(allocator, &corpus);
            break :result measureLazy(allocator, &corpus);
        },
    };

    bench_time.print("VALIDATION\t{s}\tpass\n", .{selected.name()});
    bench_time.print("RESULT\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        selected.name(),
        selected.batch(),
        measurement.median_ns,
        measurement.min_ns,
        measurement.max_ns,
        measurement.checksum,
    });
}

fn measureDense(
    row: Row,
    allocator: std.mem.Allocator,
    corpus: *const DenseCorpus,
) Measurement {
    var checksum: u64 = 0;
    for (0..warmup_runs) |_| checksum +%= runDense(row, allocator, corpus);
    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        checksum +%= runDense(row, allocator, corpus);
        elapsed.* = bench_time.monotonicNanos() - start;
        std.mem.doNotOptimizeAway(checksum);
    }
    return finishMeasurement(times, checksum);
}

fn runDense(row: Row, allocator: std.mem.Allocator, corpus: *const DenseCorpus) u64 {
    return switch (row) {
        .clone => result: {
            var checksum: u64 = 0;
            for (0..dense_batch) |_| {
                var bitmap = corpus.a.clone(allocator) catch unreachable;
                checksum +%= bitmap.size +% bitmap.cardinality();
                bitmap.deinit();
            }
            break :result checksum;
        },
        .dense_and => result: {
            var checksum: u64 = 0;
            for (0..dense_batch) |_| {
                var bitmap = corpus.a.bitwiseAnd(allocator, &corpus.b) catch unreachable;
                checksum +%= bitmap.size +% bitmap.cardinality();
                bitmap.deinit();
            }
            break :result checksum;
        },
        .select => result: {
            var checksum: u64 = 0;
            for (corpus.queries) |query| checksum +%= selectBoundary(&corpus.a, query).?;
            break :result checksum;
        },
        .lazy_or_construction => unreachable,
    };
}

fn measureLazy(allocator: std.mem.Allocator, corpus: *const SparseCorpus) Measurement {
    var checksum: u64 = 0;
    for (0..warmup_runs) |_| {
        const sample = runLazy(allocator, corpus);
        checksum +%= sample.checksum;
    }
    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| {
        const sample = runLazy(allocator, corpus);
        elapsed.* = sample.elapsed_ns;
        checksum +%= sample.checksum;
        std.mem.doNotOptimizeAway(checksum);
    }
    return finishMeasurement(times, checksum);
}

fn runLazy(allocator: std.mem.Allocator, corpus: *const SparseCorpus) struct {
    elapsed_ns: u64,
    checksum: u64,
} {
    const start = bench_time.monotonicNanos();
    var result = corpus.a.lazyOr(allocator, &corpus.b, true) catch unreachable;
    const elapsed_ns = bench_time.monotonicNanos() - start;
    const checksum = result.size +% @as(u64, @intCast(result.cached_cardinality + 1));
    std.mem.doNotOptimizeAway(&result);
    result.deinit();
    return .{ .elapsed_ns = elapsed_ns, .checksum = checksum };
}

fn finishMeasurement(times_unsorted: [timed_runs]u64, checksum: u64) Measurement {
    var times = times_unsorted;
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return .{
        .median_ns = times[timed_runs / 2],
        .min_ns = times[0],
        .max_ns = times[timed_runs - 1],
        .checksum = checksum,
    };
}

noinline fn selectBoundary(bitmap: *const RoaringBitmap, rank: u64) ?u32 {
    return @call(.always_inline, RoaringBitmap.select, .{ bitmap, rank });
}

fn fillCanonicalSelectQueries(queries: []u32) void {
    var prng = std.Random.DefaultPrng.init(select_seed);
    const random = prng.random();
    for (queries) |*query| {
        _ = random.int(u32);
        _ = random.uintLessThan(u32, 500_000);
        query.* = random.uintLessThan(u32, 500_000);
        _ = random.uintLessThan(u32, 50_000);
        _ = random.uintLessThan(u32, 1024);
        _ = random.uintLessThan(u32, 20_000);
        _ = random.uintLessThan(u32, 20_000);
    }
}

fn dedupeSorted(values: []u32) usize {
    if (values.len == 0) return 0;
    var out: usize = 1;
    for (values[1..]) |value| {
        if (value == values[out - 1]) continue;
        values[out] = value;
        out += 1;
    }
    return out;
}

fn validateDenseCorpus(allocator: std.mem.Allocator, corpus: *const DenseCorpus) !void {
    if (corpus.a.cardinality() != 500_000 or corpus.a.size != 8) return error.InvalidDenseA;
    if (corpus.b.cardinality() != 500_000) return error.InvalidDenseB;
    var intersection = try corpus.a.bitwiseAnd(allocator, &corpus.b);
    defer intersection.deinit();
    if (intersection.cardinality() != 250_000) return error.InvalidDenseIntersection;
    var checksum: u64 = 0;
    for (corpus.queries) |query| checksum +%= selectBoundary(&corpus.a, query).?;
    var expected: u64 = 0;
    for (corpus.queries) |query| expected +%= query;
    if (checksum != expected) return error.InvalidSelectQueries;
}

fn validateSparseCorpus(allocator: std.mem.Allocator, corpus: *const SparseCorpus) !void {
    var expected = try corpus.a.bitwiseOr(allocator, &corpus.b);
    defer expected.deinit();
    var lazy = try corpus.a.lazyOr(allocator, &corpus.b, true);
    defer lazy.deinit();
    try lazy.repairAfterLazy();
    if (!expected.equals(&lazy)) return error.InvalidLazyUnion;
}
