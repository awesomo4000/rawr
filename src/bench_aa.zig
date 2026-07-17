// SPDX-License-Identifier: MPL-2.0

//! Standalone array-intersection kernel benchmark.
//!
//! Build and run:
//!   zig build bench-aa -Dcpu=native
//!   ./zig-out/bin/bench_aa

const std = @import("std");
const builtin = @import("builtin");
const kernels = @import("array_kernels.zig");
const BitsetContainer = @import("bitset_container.zig").BitsetContainer;
const bench_time = @import("bench_time.zig");

const seed = 0x7261_7772_2d61_6121;
const trial_count = 11;
const min_batch_ns = 1 * std.time.ns_per_ms;
const max_batch_ns = 10 * std.time.ns_per_ms;
const max_batch_iterations = 1 << 24;
const max_cardinality = 4096;

const Case = struct {
    name: []const u8,
    card_a: usize,
    card_b: usize,
};

const cases = [_]Case{
    .{ .name = "32x4096", .card_a = 32, .card_b = 4096 },
    .{ .name = "64x4096", .card_a = 64, .card_b = 4096 },
    .{ .name = "96x4096", .card_a = 96, .card_b = 4096 },
    .{ .name = "128x4096", .card_a = 128, .card_b = 4096 },
    .{ .name = "192x4096", .card_a = 192, .card_b = 4096 },
    .{ .name = "256x4096", .card_a = 256, .card_b = 4096 },
    .{ .name = "384x4096", .card_a = 384, .card_b = 4096 },
    .{ .name = "512x4096", .card_a = 512, .card_b = 4096 },
    .{ .name = "1024x1024", .card_a = 1024, .card_b = 1024 },
    .{ .name = "1024x4096", .card_a = 1024, .card_b = 4096 },
    .{ .name = "2048x2048", .card_a = 2048, .card_b = 2048 },
    .{ .name = "4096x4096", .card_a = 4096, .card_b = 4096 },
};

var input_a: [max_cardinality]u16 = undefined;
var input_b: [max_cardinality]u16 = undefined;
var expected_out: [max_cardinality]u16 = undefined;
var kernel_out: [max_cardinality]u16 = undefined;
var timed_out: [max_cardinality]u16 = undefined;
var seen: [1 << 16]bool = undefined;
var conversion_words: [BitsetContainer.NUM_WORDS]u64 align(64) = undefined;
var lookup_keys: [1024]u16 = undefined;

const BenchError = error{KernelMismatch};

pub fn main() !void {
    bench_time.print("Array intersection kernel benchmark\n", .{});
    bench_time.print("===================================\n", .{});
    bench_time.print("seed=0x{x} target={s}-{s} cpu={s}\n", .{
        seed,
        @tagName(builtin.cpu.arch),
        @tagName(builtin.os.tag),
        builtin.cpu.model.name,
    });
    bench_time.print("case | shape | kernel | ns/op | ratio vs reference\n", .{});

    var prng = std.Random.DefaultPrng.init(seed);
    for (cases) |case| {
        const a = input_a[0..case.card_a];
        const b = input_b[0..case.card_b];
        fillExactSorted(prng.random(), a);
        fillExactSorted(prng.random(), b);

        const expected_len = kernels.intersectWriteMerge(a, b, &expected_out);
        try verifyKernels(a, b, expected_out[0..expected_len]);

        try benchWriteCase(case.name, a, b);
        try benchCardCase(case.name, a, b);
        try benchBoolCase(case.name, a, b);
    }

    fillExactSorted(prng.random(), &input_a);
    benchArrayToBitset(input_a[0..]);
    benchFindKey();
}

fn fillExactSorted(random: std.Random, out: []u16) void {
    @memset(&seen, false);

    var count: usize = 0;
    while (count < out.len) {
        const value = random.int(u16);
        if (!seen[value]) {
            seen[value] = true;
            count += 1;
        }
    }

    count = 0;
    for (seen, 0..) |present, value| {
        if (present) {
            out[count] = @intCast(value);
            count += 1;
        }
    }
    std.debug.assert(count == out.len);
}

fn verifyKernels(a: []const u16, b: []const u16, expected: []const u16) BenchError!void {
    for (kernels.write_bench_kernels) |kernel| {
        const len = kernel.func(a, b, &kernel_out);
        if (!std.mem.eql(u16, expected, kernel_out[0..len])) return error.KernelMismatch;
    }
    for (kernels.card_bench_kernels) |kernel| {
        if (kernel.func(a, b) != expected.len) return error.KernelMismatch;
    }
    for (kernels.bool_bench_kernels) |kernel| {
        if (kernel.func(a, b) != (expected.len != 0)) return error.KernelMismatch;
    }
}

fn benchWriteCase(case_name: []const u8, a: []const u16, b: []const u16) BenchError!void {
    const iterations = calibrateWrite(a, b);
    const merge_ns = medianWrite(kernels.intersectWriteMerge, a, b, iterations);

    for (kernels.write_bench_kernels) |kernel| {
        const ns = medianWrite(kernel.func, a, b, iterations);
        printResult(case_name, "write", kernel.name, ns, merge_ns);
    }
}

fn benchCardCase(case_name: []const u8, a: []const u16, b: []const u16) BenchError!void {
    const iterations = calibrateCard(a, b);
    const merge_ns = medianCard(kernels.intersectCardMerge, a, b, iterations);

    for (kernels.card_bench_kernels) |kernel| {
        const ns = medianCard(kernel.func, a, b, iterations);
        printResult(case_name, "card", kernel.name, ns, merge_ns);
    }
}

fn benchBoolCase(case_name: []const u8, a: []const u16, b: []const u16) BenchError!void {
    const iterations = calibrateBool(a, b);
    const merge_ns = medianBool(kernels.intersectBoolMerge, a, b, iterations);

    for (kernels.bool_bench_kernels) |kernel| {
        const ns = medianBool(kernel.func, a, b, iterations);
        printResult(case_name, "bool", kernel.name, ns, merge_ns);
    }
}

fn calibrateWrite(a: []const u16, b: []const u16) usize {
    var iterations: usize = 1;
    while (true) : (iterations *= 2) {
        var fastest: u64 = std.math.maxInt(u64);
        var slowest: u64 = 0;
        for (kernels.write_bench_kernels) |kernel| {
            const start = bench_time.monotonicNanos();
            runWriteBatch(kernel.func, a, b, iterations);
            const elapsed = bench_time.monotonicNanos() - start;
            fastest = @min(fastest, elapsed);
            slowest = @max(slowest, elapsed);
        }
        if (fastest >= min_batch_ns or slowest >= max_batch_ns or iterations >= max_batch_iterations) break;
    }
    return @min(iterations, max_batch_iterations);
}

fn calibrateCard(a: []const u16, b: []const u16) usize {
    var iterations: usize = 1;
    while (true) : (iterations *= 2) {
        var fastest: u64 = std.math.maxInt(u64);
        var slowest: u64 = 0;
        for (kernels.card_bench_kernels) |kernel| {
            const start = bench_time.monotonicNanos();
            runCardBatch(kernel.func, a, b, iterations);
            const elapsed = bench_time.monotonicNanos() - start;
            fastest = @min(fastest, elapsed);
            slowest = @max(slowest, elapsed);
        }
        if (fastest >= min_batch_ns or slowest >= max_batch_ns or iterations >= max_batch_iterations) break;
    }
    return @min(iterations, max_batch_iterations);
}

fn calibrateBool(a: []const u16, b: []const u16) usize {
    var iterations: usize = 1;
    while (true) : (iterations *= 2) {
        var fastest: u64 = std.math.maxInt(u64);
        var slowest: u64 = 0;
        for (kernels.bool_bench_kernels) |kernel| {
            const start = bench_time.monotonicNanos();
            runBoolBatch(kernel.func, a, b, iterations);
            const elapsed = bench_time.monotonicNanos() - start;
            fastest = @min(fastest, elapsed);
            slowest = @max(slowest, elapsed);
        }
        if (fastest >= min_batch_ns or slowest >= max_batch_ns or iterations >= max_batch_iterations) break;
    }
    return @min(iterations, max_batch_iterations);
}

fn medianWrite(func: kernels.WriteKernel, a: []const u16, b: []const u16, iterations: usize) f64 {
    var times: [trial_count]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        runWriteBatch(func, a, b, iterations);
        elapsed.* = bench_time.monotonicNanos() - start;
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return @as(f64, @floatFromInt(times[trial_count / 2])) / @as(f64, @floatFromInt(iterations));
}

fn medianCard(func: kernels.CardKernel, a: []const u16, b: []const u16, iterations: usize) f64 {
    var times: [trial_count]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        runCardBatch(func, a, b, iterations);
        elapsed.* = bench_time.monotonicNanos() - start;
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return @as(f64, @floatFromInt(times[trial_count / 2])) / @as(f64, @floatFromInt(iterations));
}

fn medianBool(func: kernels.BoolKernel, a: []const u16, b: []const u16, iterations: usize) f64 {
    var times: [trial_count]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        runBoolBatch(func, a, b, iterations);
        elapsed.* = bench_time.monotonicNanos() - start;
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return @as(f64, @floatFromInt(times[trial_count / 2])) / @as(f64, @floatFromInt(iterations));
}

noinline fn runWriteBatch(func: kernels.WriteKernel, a: []const u16, b: []const u16, iterations: usize) void {
    for (0..iterations) |_| {
        const len = func(a, b, &timed_out);
        std.mem.doNotOptimizeAway(len);
        std.mem.doNotOptimizeAway(timed_out[0..len]);
    }
}

noinline fn runCardBatch(func: kernels.CardKernel, a: []const u16, b: []const u16, iterations: usize) void {
    for (0..iterations) |_| {
        const cardinality = func(a, b);
        std.mem.doNotOptimizeAway(cardinality);
    }
}

noinline fn runBoolBatch(func: kernels.BoolKernel, a: []const u16, b: []const u16, iterations: usize) void {
    for (0..iterations) |_| {
        const intersects = func(a, b);
        std.mem.doNotOptimizeAway(intersects);
    }
}

fn printResult(case_name: []const u8, shape: []const u8, kernel_name: []const u8, ns: f64, merge_ns: f64) void {
    bench_time.print("{s} | {s} | {s} | {d:.2} | {d:.2}x\n", .{
        case_name,
        shape,
        kernel_name,
        ns,
        ns / merge_ns,
    });
}

const ConversionKernel = *const fn ([]const u16) u32;

fn benchArrayToBitset(values: []const u16) void {
    const iterations = calibrateConversion(values);
    const add_ns = medianConversion(conversionAddLoop, values, iterations);
    const bulk_ns = medianConversion(conversionBulkSet, values, iterations);

    bench_time.print("4096 array->bitset | conversion | add-loop | {d:.2} | {d:.2}x\n", .{ add_ns, add_ns / add_ns });
    bench_time.print("4096 array->bitset | conversion | bulk-set | {d:.2} | {d:.2}x\n", .{ bulk_ns, bulk_ns / add_ns });
}

fn calibrateConversion(values: []const u16) usize {
    var iterations: usize = 1;
    while (true) : (iterations *= 2) {
        var fastest: u64 = std.math.maxInt(u64);
        var slowest: u64 = 0;
        for ([_]ConversionKernel{ conversionAddLoop, conversionBulkSet }) |kernel| {
            const start = bench_time.monotonicNanos();
            runConversionBatch(kernel, values, iterations);
            const elapsed = bench_time.monotonicNanos() - start;
            fastest = @min(fastest, elapsed);
            slowest = @max(slowest, elapsed);
        }
        if (fastest >= min_batch_ns or slowest >= max_batch_ns or iterations >= max_batch_iterations) break;
    }
    return @min(iterations, max_batch_iterations);
}

fn medianConversion(func: ConversionKernel, values: []const u16, iterations: usize) f64 {
    var times: [trial_count]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        runConversionBatch(func, values, iterations);
        elapsed.* = bench_time.monotonicNanos() - start;
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return @as(f64, @floatFromInt(times[trial_count / 2])) / @as(f64, @floatFromInt(iterations));
}

noinline fn runConversionBatch(func: ConversionKernel, values: []const u16, iterations: usize) void {
    for (0..iterations) |_| {
        const cardinality = func(values);
        std.mem.doNotOptimizeAway(cardinality);
        std.mem.doNotOptimizeAway(&conversion_words);
    }
}

fn conversionAddLoop(values: []const u16) u32 {
    @memset(&conversion_words, 0);
    var bitset = BitsetContainer{ .words = &conversion_words, .cardinality = 0 };
    for (values) |value| _ = bitset.add(value);
    return bitset.getCardinality();
}

fn conversionBulkSet(values: []const u16) u32 {
    @memset(&conversion_words, 0);
    var bitset = BitsetContainer{ .words = &conversion_words, .cardinality = 0 };
    bitset.setList(values);
    return bitset.computeCardinality();
}

const LookupKernel = *const fn ([]const u16, u16) ?usize;
const LookupBenchKernel = struct {
    name: []const u8,
    func: LookupKernel,
};

const lookup_bench_kernels = [_]LookupBenchKernel{
    .{ .name = "old-binary", .func = findKeyOldBinary },
    .{ .name = "lower-bound", .func = findKeyCutoff0 },
    .{ .name = "linear-4", .func = findKeyCutoff4 },
    .{ .name = "linear-8", .func = findKeyCutoff8 },
    .{ .name = "linear-16", .func = findKeyCutoff16 },
};

fn benchFindKey() void {
    for (&lookup_keys, 0..) |*key, idx| key.* = @intCast(2 + idx * 4);

    for ([_]usize{ 4, 16, 32, 64, 256, 1024 }) |count| {
        const keys = lookup_keys[0..count];
        const middle = count / 2;
        const queries = [_]struct { name: []const u8, key: u16 }{
            .{ .name = "hit-first", .key = keys[0] },
            .{ .name = "hit-middle", .key = keys[middle] },
            .{ .name = "hit-last", .key = keys[count - 1] },
            .{ .name = "miss-before", .key = keys[0] - 1 },
            .{ .name = "miss-between", .key = keys[middle] + 1 },
            .{ .name = "miss-after", .key = keys[count - 1] + 1 },
        };

        for (queries) |query| benchLookupCase(keys, query.name, query.key);
    }
}

fn benchLookupCase(keys: []const u16, position: []const u8, key: u16) void {
    const expected = findKeyOldBinary(keys, key);
    for (lookup_bench_kernels) |kernel| {
        if (kernel.func(keys, key) != expected) @panic("findKey benchmark mismatch");
    }

    const iterations = calibrateLookup(keys, key);
    const baseline_ns = medianLookup(findKeyOldBinary, keys, key, iterations);
    for (lookup_bench_kernels) |kernel| {
        const ns = medianLookup(kernel.func, keys, key, iterations);
        bench_time.print("{d} keys {s} | lookup | {s} | {d:.2} | {d:.2}x\n", .{
            keys.len,
            position,
            kernel.name,
            ns,
            ns / baseline_ns,
        });
    }
}

fn calibrateLookup(keys: []const u16, key: u16) usize {
    var iterations: usize = 1;
    while (true) : (iterations *= 2) {
        var fastest: u64 = std.math.maxInt(u64);
        var slowest: u64 = 0;
        for (lookup_bench_kernels) |kernel| {
            const start = bench_time.monotonicNanos();
            runLookupBatch(kernel.func, keys, key, iterations);
            const elapsed = bench_time.monotonicNanos() - start;
            fastest = @min(fastest, elapsed);
            slowest = @max(slowest, elapsed);
        }
        if (fastest >= min_batch_ns or slowest >= max_batch_ns or iterations >= max_batch_iterations) break;
    }
    return @min(iterations, max_batch_iterations);
}

fn medianLookup(func: LookupKernel, keys: []const u16, key: u16, iterations: usize) f64 {
    var times: [trial_count]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        runLookupBatch(func, keys, key, iterations);
        elapsed.* = bench_time.monotonicNanos() - start;
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return @as(f64, @floatFromInt(times[trial_count / 2])) / @as(f64, @floatFromInt(iterations));
}

noinline fn runLookupBatch(func: LookupKernel, keys: []const u16, key: u16, iterations: usize) void {
    for (0..iterations) |_| {
        const result = func(keys, key);
        std.mem.doNotOptimizeAway(result);
    }
}

fn findKeyOldBinary(keys: []const u16, key: u16) ?usize {
    var lo: usize = 0;
    var hi = keys.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (keys[mid] < key) {
            lo = mid + 1;
        } else if (keys[mid] > key) {
            hi = mid;
        } else {
            return mid;
        }
    }
    return null;
}

fn findKeyCutoff0(keys: []const u16, key: u16) ?usize {
    return findKeyCandidate(keys, key, 0);
}

fn findKeyCutoff4(keys: []const u16, key: u16) ?usize {
    return findKeyCandidate(keys, key, 4);
}

fn findKeyCutoff8(keys: []const u16, key: u16) ?usize {
    return findKeyCandidate(keys, key, 8);
}

fn findKeyCutoff16(keys: []const u16, key: u16) ?usize {
    return findKeyCandidate(keys, key, 16);
}

inline fn findKeyCandidate(keys: []const u16, key: u16, comptime cutoff: usize) ?usize {
    if (keys.len <= cutoff) {
        for (keys, 0..) |candidate, idx| {
            if (candidate == key) return idx;
            if (candidate > key) return null;
        }
        return null;
    }

    const idx = kernels.lowerBound(keys, key);
    if (idx < keys.len and keys[idx] == key) return idx;
    return null;
}
