// SPDX-License-Identifier: MPL-2.0

//! Fresh-process four-path iteration diagnosis for spec 23-00.

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");

const RoaringBitmap = rawr.RoaringBitmap;
const Container = rawr.Container;

const value_count = 1_000_000;
const seed = 12345;
const warmup_runs = 3;
const timed_runs = 21;

const Path = enum {
    rawr_pull,
    rawr_push,
    croaring_pull,
    croaring_push,

    fn name(self: Path) []const u8 {
        return switch (self) {
            .rawr_pull => "rawr-pull",
            .rawr_push => "rawr-push",
            .croaring_pull => "croaring-pull",
            .croaring_push => "croaring-push",
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

const ContainerCounts = struct {
    arrays: u32 = 0,
    bitsets: u32 = 0,
    runs: u32 = 0,
};

const Corpus = struct {
    values: []u32,
    oracle_storage: []u32,
    oracle_len: usize,

    fn init(allocator: std.mem.Allocator) !Corpus {
        const values = try allocator.alloc(u32, value_count);
        errdefer allocator.free(values);
        const oracle_storage = try allocator.alloc(u32, value_count);
        errdefer allocator.free(oracle_storage);

        fillCanonicalValues(values);
        @memcpy(oracle_storage, values);
        std.mem.sort(u32, oracle_storage, {}, std.sort.asc(u32));

        var unique: usize = 0;
        for (oracle_storage) |value| {
            if (unique == 0 or value != oracle_storage[unique - 1]) {
                oracle_storage[unique] = value;
                unique += 1;
            }
        }

        return .{
            .values = values,
            .oracle_storage = oracle_storage,
            .oracle_len = unique,
        };
    }

    fn deinit(self: *Corpus, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
        allocator.free(self.oracle_storage);
    }

    fn oracle(self: *const Corpus) []const u32 {
        return self.oracle_storage[0..self.oracle_len];
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
        bench_time.print("# iterate diagnosis protocol: {d}w/{d}t median; seed={d}; attempted={d}\n", .{
            warmup_runs,
            timed_runs,
            seed,
            value_count,
        });
        return;
    }

    const path = requested_path orelse return error.MissingPath;
    const allocator = std.heap.smp_allocator;
    var corpus = try Corpus.init(allocator);
    defer corpus.deinit(allocator);

    switch (path) {
        .rawr_pull, .rawr_push => try runRawr(path, allocator, &corpus),
        .croaring_pull, .croaring_push => try runCRoaring(path, allocator, &corpus),
    }
}

fn fillCanonicalValues(values: []u32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (values) |*value| {
        // Keep this draw sequence aligned with bench_croaring.initTestData.
        value.* = random.int(u32);
        _ = random.uintLessThan(u32, 500_000);
        _ = random.uintLessThan(u32, 500_000);
        _ = random.uintLessThan(u32, 50_000);
        _ = random.uintLessThan(u32, 1024);
        _ = random.uintLessThan(u32, 20_000);
        _ = random.uintLessThan(u32, 20_000);
    }
}

fn runRawr(path: Path, allocator: std.mem.Allocator, corpus: *const Corpus) !void {
    var bitmap = try RoaringBitmap.init(allocator);
    defer bitmap.deinit();
    for (corpus.values) |value| _ = try bitmap.add(value);

    const counts = rawrContainerCounts(&bitmap);
    const measurement = switch (path) {
        .rawr_pull => measure(scanRawrPull, .{&bitmap}),
        .rawr_push => measure(scanRawrPush, .{&bitmap}),
        else => unreachable,
    };
    try validateRawr(path, allocator, &bitmap, corpus.oracle(), measurement.result);
    printResult(path, corpus.oracle(), counts, measurement);
}

fn runCRoaring(path: Path, allocator: std.mem.Allocator, corpus: *const Corpus) !void {
    const bitmap = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(bitmap);
    for (corpus.values) |value| c.roaring_bitmap_add(bitmap, value);

    const c_counts = c.rawr_cr_iterate_container_counts(bitmap);
    const counts = ContainerCounts{
        .arrays = c_counts.arrays,
        .bitsets = c_counts.bitsets,
        .runs = c_counts.runs,
    };
    const measurement = switch (path) {
        .croaring_pull => measure(scanCRoaringPull, .{bitmap}),
        .croaring_push => measure(scanCRoaringPush, .{bitmap}),
        else => unreachable,
    };
    try validateCRoaring(path, allocator, bitmap, corpus.oracle(), measurement.result);
    printResult(path, corpus.oracle(), counts, measurement);
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

noinline fn scanRawrPull(bitmap: *const RoaringBitmap) ScanResult {
    var result = ScanResult{};
    var iterator = bitmap.iterator();
    while (iterator.next()) |value| {
        result.count +%= 1;
        result.sum +%= value;
    }
    return result;
}

noinline fn scanRawrPush(bitmap: *const RoaringBitmap) ScanResult {
    var result = ScanResult{};
    rawrForEachDiagnostic(bitmap, &result, accumulate);
    return result;
}

inline fn accumulate(result: *ScanResult, value: u32) void {
    result.count +%= 1;
    result.sum +%= value;
}

fn rawrForEachDiagnostic(
    bitmap: *const RoaringBitmap,
    context: anytype,
    comptime visit: anytype,
) void {
    for (bitmap.keys[0..bitmap.size], bitmap.containers[0..bitmap.size]) |key, tagged| {
        const high = @as(u32, key) << 16;
        switch (Container.fromTagged(tagged)) {
            .array => |array| {
                for (array.values[0..array.cardinality]) |low| visit(context, high | low);
            },
            .bitset => |bitset| {
                for (bitset.words, 0..) |word, word_index| {
                    var bits = word;
                    while (bits != 0) {
                        const bit = @ctz(bits);
                        bits &= bits - 1;
                        visit(context, high | @as(u32, @intCast(word_index * 64 + bit)));
                    }
                }
            },
            .run => |run| {
                for (run.runs[0..run.n_runs]) |pair| {
                    var offset: u32 = 0;
                    while (offset <= pair.length) : (offset += 1) {
                        visit(context, high | (@as(u32, pair.start) + offset));
                    }
                }
            },
            .reserved => unreachable,
        }
    }
}

noinline fn scanCRoaringPull(bitmap: *const c.roaring_bitmap_t) ScanResult {
    const result = c.rawr_cr_iterate_pull(bitmap);
    return .{ .count = result.count, .sum = result.sum };
}

noinline fn scanCRoaringPush(bitmap: *const c.roaring_bitmap_t) ScanResult {
    const result = c.rawr_cr_iterate_push(bitmap);
    return .{ .count = result.count, .sum = result.sum };
}

fn validateRawr(
    path: Path,
    allocator: std.mem.Allocator,
    bitmap: *const RoaringBitmap,
    oracle: []const u32,
    timed_result: ScanResult,
) !void {
    const output = try allocator.alloc(u32, oracle.len);
    defer allocator.free(output);

    const written = switch (path) {
        .rawr_pull => writeRawrPull(bitmap, output),
        .rawr_push => writeRawrPush(bitmap, output),
        else => unreachable,
    };
    if (written == std.math.maxInt(usize)) return error.ValidationBufferOverflow;
    try validateOutput(path, output[0..written], oracle, timed_result);
}

fn validateCRoaring(
    path: Path,
    allocator: std.mem.Allocator,
    bitmap: *const c.roaring_bitmap_t,
    oracle: []const u32,
    timed_result: ScanResult,
) !void {
    const output = try allocator.alloc(u32, oracle.len);
    defer allocator.free(output);

    const written = switch (path) {
        .croaring_pull => c.rawr_cr_iterate_pull_values(bitmap, output.ptr, output.len),
        .croaring_push => c.rawr_cr_iterate_push_values(bitmap, output.ptr, output.len),
        else => unreachable,
    };
    if (written == std.math.maxInt(usize)) return error.ValidationBufferOverflow;
    try validateOutput(path, output[0..written], oracle, timed_result);
}

fn writeRawrPull(bitmap: *const RoaringBitmap, output: []u32) usize {
    var iterator = bitmap.iterator();
    var index: usize = 0;
    while (iterator.next()) |value| : (index += 1) {
        if (index >= output.len) return std.math.maxInt(usize);
        output[index] = value;
    }
    return index;
}

const WriteContext = struct {
    output: []u32,
    index: usize = 0,
    overflow: bool = false,
};

fn writeValue(context: *WriteContext, value: u32) void {
    if (context.index >= context.output.len) {
        context.overflow = true;
        return;
    }
    context.output[context.index] = value;
    context.index += 1;
}

fn writeRawrPush(bitmap: *const RoaringBitmap, output: []u32) usize {
    var context = WriteContext{ .output = output };
    rawrForEachDiagnostic(bitmap, &context, writeValue);
    return if (context.overflow) std.math.maxInt(usize) else context.index;
}

fn validateOutput(path: Path, output: []const u32, oracle: []const u32, timed_result: ScanResult) !void {
    if (output.len != oracle.len) return error.CardinalityMismatch;
    if (!std.mem.eql(u32, output, oracle)) return error.SequenceMismatch;

    const expected = scanSlice(oracle);
    if (timed_result.count != expected.count or timed_result.sum != expected.sum) {
        return error.TimedChecksumMismatch;
    }
    const hash = rollingHash(output);
    if (hash != rollingHash(oracle)) return error.RollingHashMismatch;
    bench_time.print("VALIDATION\t{s}\t{d}\n", .{ path.name(), hash });
}

fn scanSlice(values: []const u32) ScanResult {
    var result = ScanResult{};
    for (values) |value| {
        result.count +%= 1;
        result.sum +%= value;
    }
    return result;
}

fn rollingHash(values: []const u32) u64 {
    var hash: u64 = 0xcbf29ce484222325;
    for (values) |value| {
        hash ^= value;
        hash *%= 0x100000001b3;
    }
    return hash;
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

fn printResult(path: Path, oracle: []const u32, counts: ContainerCounts, measurement: Measurement) void {
    bench_time.print("CORPUS\t{s}\t{d}\t{d}\t{d}\t{d}\n", .{
        path.name(),
        oracle.len,
        counts.arrays,
        counts.bitsets,
        counts.runs,
    });
    bench_time.print("RESULT\t{s}\t{d}\t{d}\t{d}\n", .{
        path.name(),
        measurement.result.count,
        measurement.result.sum,
        measurement.median_ns,
    });
}
