// SPDX-License-Identifier: MPL-2.0

//! Fresh-process attribution worker for real-data array OR and ANDNOT gaps.

const std = @import("std");
const builtin = @import("builtin");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");
const corpus_mod = @import("realdata_corpus.zig");

const RoaringBitmap = rawr.RoaringBitmap;
const ArrayContainer = rawr.ArrayContainer;
const Container = rawr.Container;
const warmup_runs = 1;
const timed_runs = 7;
const default_root = "misc/realdata";
const dataset = corpus_mod.Dataset.wikileaks_noquotes;

const Operation = enum {
    pair_or,
    pair_andnot,

    fn id(self: Operation) []const u8 {
        return switch (self) {
            .pair_or => "pair-or",
            .pair_andnot => "pair-andnot",
        };
    }

    fn parse(value: []const u8) ?Operation {
        inline for (std.meta.fields(Operation)) |field| {
            const operation: Operation = @enumFromInt(field.value);
            if (std.mem.eql(u8, value, operation.id())) return operation;
        }
        return null;
    }

    fn cValue(self: Operation) c.rawr_cr_array_operation_t {
        return switch (self) {
            .pair_or => c.RAWR_CR_ARRAY_UNION,
            .pair_andnot => c.RAWR_CR_ARRAY_DIFFERENCE,
        };
    }
};

const Arm = enum {
    e1_rawr_endtoend,
    e2_croaring_endtoend,
    a1_rawr_scalar,
    a2_croaring_scalar,
    a3_croaring_production,
    b1_rawr_production,
    b2_croaring_production,
    b3_rawr_no_normalize,

    fn id(self: Arm) []const u8 {
        return switch (self) {
            .e1_rawr_endtoend => "e1-rawr-endtoend",
            .e2_croaring_endtoend => "e2-croaring-endtoend",
            .a1_rawr_scalar => "a1-rawr-scalar",
            .a2_croaring_scalar => "a2-croaring-scalar",
            .a3_croaring_production => "a3-croaring-production",
            .b1_rawr_production => "b1-rawr-production",
            .b2_croaring_production => "b2-croaring-production",
            .b3_rawr_no_normalize => "b3-rawr-no-normalize",
        };
    }

    fn parse(value: []const u8) ?Arm {
        inline for (std.meta.fields(Arm)) |field| {
            const arm: Arm = @enumFromInt(field.value);
            if (std.mem.eql(u8, value, arm.id())) return arm;
        }
        return null;
    }

    fn isEndToEnd(self: Arm) bool {
        return self == .e1_rawr_endtoend or self == .e2_croaring_endtoend;
    }
};

const operations = [_]Operation{ .pair_or, .pair_andnot };
const arms = [_]Arm{
    .e1_rawr_endtoend,
    .e2_croaring_endtoend,
    .a1_rawr_scalar,
    .a2_croaring_scalar,
    .a3_croaring_production,
    .b1_rawr_production,
    .b2_croaring_production,
    .b3_rawr_no_normalize,
};

const RequestedCell = struct {
    operation: Operation,
    arm: Arm,
    root: []const u8,
};

const Pair = struct {
    left: *ArrayContainer,
    right: *ArrayContainer,
};

const PairAccounting = struct {
    matched_arrays: u64 = 0,
    bitset_path: u64 = 0,
    matched_other: u64 = 0,
    unmatched_left: u64 = 0,
    unmatched_right: u64 = 0,
    input_elements: u64 = 0,
};

const SizeDistribution = struct {
    min: u32 = 0,
    p50: u32 = 0,
    p90: u32 = 0,
    p99: u32 = 0,
    max: u32 = 0,
};

const PairSet = struct {
    allocator: std.mem.Allocator,
    pairs: []Pair,
    c_pairs: []c.rawr_cr_array_pair_t,
    accounting: PairAccounting,
    sizes: SizeDistribution,

    fn init(allocator: std.mem.Allocator, sources: *const RawrSources, operation: Operation) !PairSet {
        var list: std.ArrayList(Pair) = .empty;
        defer list.deinit(allocator);
        var accounting = PairAccounting{};

        for (sources.bitmaps[0 .. sources.bitmaps.len - 1], sources.bitmaps[1..]) |*left, *right| {
            var left_index: usize = 0;
            var right_index: usize = 0;
            while (left_index < left.size or right_index < right.size) {
                if (left_index == left.size) {
                    accounting.unmatched_right += 1;
                    right_index += 1;
                    continue;
                }
                if (right_index == right.size) {
                    accounting.unmatched_left += 1;
                    left_index += 1;
                    continue;
                }

                const left_key = left.keys[left_index];
                const right_key = right.keys[right_index];
                if (left_key < right_key) {
                    accounting.unmatched_left += 1;
                    left_index += 1;
                    continue;
                }
                if (right_key < left_key) {
                    accounting.unmatched_right += 1;
                    right_index += 1;
                    continue;
                }

                const left_tagged = left.containers[left_index];
                const right_tagged = right.containers[right_index];
                if (left_tagged.getType() == .array and right_tagged.getType() == .array) {
                    const left_array = left_tagged.getArray();
                    const right_array = right_tagged.getArray();
                    const max_card = @as(u32, left_array.cardinality) + right_array.cardinality;
                    if (operation == .pair_or and max_card > ArrayContainer.MAX_CARDINALITY) {
                        accounting.bitset_path += 1;
                    } else {
                        try list.append(allocator, .{ .left = left_array, .right = right_array });
                        accounting.matched_arrays += 1;
                        accounting.input_elements += max_card;
                    }
                } else {
                    accounting.matched_other += 1;
                }
                left_index += 1;
                right_index += 1;
            }
        }

        const pairs = try list.toOwnedSlice(allocator);
        errdefer allocator.free(pairs);
        const c_pairs = try allocator.alloc(c.rawr_cr_array_pair_t, pairs.len);
        errdefer allocator.free(c_pairs);
        const pair_sizes = try allocator.alloc(u32, pairs.len);
        defer allocator.free(pair_sizes);
        for (pairs, c_pairs, pair_sizes) |pair, *c_pair, *size| {
            c_pair.* = .{
                .left = pair.left.values.ptr,
                .left_len = pair.left.cardinality,
                .right = pair.right.values.ptr,
                .right_len = pair.right.cardinality,
            };
            size.* = @as(u32, pair.left.cardinality) + pair.right.cardinality;
        }
        std.mem.sort(u32, pair_sizes, {}, std.sort.asc(u32));

        return .{
            .allocator = allocator,
            .pairs = pairs,
            .c_pairs = c_pairs,
            .accounting = accounting,
            .sizes = sizeDistribution(pair_sizes),
        };
    }

    fn deinit(self: *PairSet) void {
        self.allocator.free(self.pairs);
        self.allocator.free(self.c_pairs);
        self.* = undefined;
    }
};

const RawrSources = struct {
    bitmaps: []RoaringBitmap,

    fn init(corpus: *const corpus_mod.Corpus) !RawrSources {
        const allocator = std.heap.page_allocator;
        const bitmaps = try allocator.alloc(RoaringBitmap, corpus.bitmaps.len);
        var built: usize = 0;
        errdefer {
            for (bitmaps[0..built]) |*bitmap| bitmap.deinit();
            allocator.free(bitmaps);
        }
        for (corpus.bitmaps, 0..) |entry, index| {
            bitmaps[index] = try RoaringBitmap.fromSorted(std.heap.smp_allocator, entry.values);
            built += 1;
        }
        return .{ .bitmaps = bitmaps };
    }

    fn deinit(self: *RawrSources) void {
        for (self.bitmaps) |*bitmap| bitmap.deinit();
        std.heap.page_allocator.free(self.bitmaps);
        self.* = undefined;
    }
};

const CRoaringSources = struct {
    bitmaps: []*c.roaring_bitmap_t,

    fn init(corpus: *const corpus_mod.Corpus) !CRoaringSources {
        const bitmaps = try std.heap.page_allocator.alloc(*c.roaring_bitmap_t, corpus.bitmaps.len);
        var built: usize = 0;
        errdefer {
            for (bitmaps[0..built]) |bitmap| c.roaring_bitmap_free(bitmap);
            std.heap.page_allocator.free(bitmaps);
        }
        for (corpus.bitmaps, 0..) |entry, index| {
            const bitmap = c.roaring_bitmap_create() orelse return error.CRoaringAllocFailed;
            c.roaring_bitmap_add_many(bitmap, entry.values.len, entry.values.ptr);
            bitmaps[index] = bitmap;
            built += 1;
        }
        return .{ .bitmaps = bitmaps };
    }

    fn deinit(self: *CRoaringSources) void {
        for (self.bitmaps) |bitmap| c.roaring_bitmap_free(bitmap);
        std.heap.page_allocator.free(self.bitmaps);
        self.* = undefined;
    }
};

const ArmResult = struct {
    checksum: u64 = 0,
    digest: u64 = 0,
    conversions: u64 = 0,
    allocation_calls: u64 = 0,
    normalization_calls: u64 = 0,
    branch: Branch = .not_applicable,
    outputs_distinct: bool = true,
    output_storage_unchanged: bool = true,
};

const Branch = enum {
    rawr_scalar,
    scalar,
    avx2,
    not_applicable,
};

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var list = false;
    var header = false;
    var operation: ?Operation = null;
    var arm: ?Arm = null;
    var root: []const u8 = default_root;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--list")) {
            list = true;
        } else if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.startsWith(u8, arg, "--operation=")) {
            operation = Operation.parse(arg[12..]) orelse return error.UnknownOperation;
        } else if (std.mem.startsWith(u8, arg, "--arm=")) {
            arm = Arm.parse(arg[6..]) orelse return error.UnknownArm;
        } else if (std.mem.startsWith(u8, arg, "--root=")) {
            root = arg[7..];
        } else {
            return error.UnknownArgument;
        }
    }

    if (list) {
        if (header or operation != null or arm != null or !std.mem.eql(u8, root, default_root)) {
            return error.ConflictingArguments;
        }
        printManifest();
        return;
    }
    if (header) {
        if (operation != null or arm != null or !std.mem.eql(u8, root, default_root)) {
            return error.ConflictingArguments;
        }
        printHeader();
        return;
    }

    const requested = RequestedCell{
        .operation = operation orelse return error.MissingOperation,
        .arm = arm orelse return error.MissingArm,
        .root = root,
    };
    var corpus = try corpus_mod.loadDataset(
        std.heap.page_allocator,
        init.io,
        requested.root,
        dataset,
    );
    defer corpus.deinit();
    try runRequested(requested, &corpus);
}

fn printManifest() void {
    for (operations) |operation| {
        bench_time.print("ROW\t{s}\n", .{operation.id()});
        for (arms) |arm| bench_time.print("TUPLE\t{s}\t{s}\n", .{ operation.id(), arm.id() });
    }
}

fn printHeader() void {
    bench_time.printBenchEnvironment();
    bench_time.print("# requested-cpu: native\n", .{});
    bench_time.print("# dataset: {s}\n", .{dataset.name()});
    bench_time.print("# protocol: {d} warmup cycle, {d} timed cycles, process median\n", .{
        warmup_runs,
        timed_runs,
    });
    bench_time.print("# allocator-pairing: rawr=smp_allocator, CRoaring=default-libc\n", .{});
    bench_time.print("# layer-a-output: preallocated and non-aliased\n", .{});
    bench_time.print("# croaring-avx512: {s}\n", .{
        if (c.CROARING_COMPILER_SUPPORTS_AVX512 != 0) "on" else "off",
    });
}

fn runRequested(requested: RequestedCell, corpus: *const corpus_mod.Corpus) !void {
    if (requested.arm == .e1_rawr_endtoend) {
        var sources = try RawrSources.init(corpus);
        defer sources.deinit();
        const median = try measureRawrEndToEnd(requested.operation, &sources);
        const digest = try digestRawrEndToEnd(requested.operation, &sources, corpus.total_values);
        var pairs = try PairSet.init(std.heap.page_allocator, &sources, requested.operation);
        defer pairs.deinit();
        printResult(requested, median, digest, corpus, &pairs, .{});
        return;
    }
    if (requested.arm == .e2_croaring_endtoend) {
        var sources = try CRoaringSources.init(corpus);
        defer sources.deinit();
        const median = try measureCRoaringEndToEnd(requested.operation, &sources);
        const digest = try digestCRoaringEndToEnd(requested.operation, &sources, corpus.total_values);
        var rawr_sources = try RawrSources.init(corpus);
        defer rawr_sources.deinit();
        var pairs = try PairSet.init(std.heap.page_allocator, &rawr_sources, requested.operation);
        defer pairs.deinit();
        printResult(requested, median, digest, corpus, &pairs, .{});
        return;
    }

    var cr_conditioning: ?CRoaringSources = if (requested.arm == .b2_croaring_production)
        try CRoaringSources.init(corpus)
    else
        null;
    defer if (cr_conditioning) |*sources| sources.deinit();
    var sources = try RawrSources.init(corpus);
    defer sources.deinit();
    var pairs = try PairSet.init(std.heap.page_allocator, &sources, requested.operation);
    defer pairs.deinit();

    const median = try measureMatched(requested.operation, requested.arm, &pairs);
    const expected = try runRawrMatched(requested.operation, .a1_rawr_scalar, &pairs, true);
    const actual = try runMatched(requested.operation, requested.arm, &pairs, true);
    if (actual.digest != expected.digest) return error.MatchedDigestMismatch;
    if (actual.outputs_distinct == false) return error.LayerAOutputAliasesInput;
    if (requested.arm == .a1_rawr_scalar or requested.arm == .a2_croaring_scalar or
        requested.arm == .a3_croaring_production)
    {
        if (actual.allocation_calls != 0) return error.LayerAAllocated;
    }
    if (requested.arm == .b3_rawr_no_normalize and actual.normalization_calls != 0) {
        return error.B3Normalized;
    }
    try validateBranch(requested.arm, actual);
    printResult(requested, median, actual.digest, corpus, &pairs, actual);
}

fn measureRawrEndToEnd(operation: Operation, sources: *const RawrSources) !u64 {
    for (0..warmup_runs) |_| _ = try runRawrEndToEnd(operation, sources);
    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        const checksum = try runRawrEndToEnd(operation, sources);
        elapsed.* = bench_time.monotonicNanos() - start;
        std.mem.doNotOptimizeAway(checksum);
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return times[timed_runs / 2];
}

fn runRawrEndToEnd(operation: Operation, sources: *const RawrSources) !u64 {
    var checksum: u64 = 0;
    for (sources.bitmaps[0 .. sources.bitmaps.len - 1], sources.bitmaps[1..]) |*left, *right| {
        var result = switch (operation) {
            .pair_or => try left.bitwiseOr(std.heap.smp_allocator, right),
            .pair_andnot => try left.bitwiseDifference(std.heap.smp_allocator, right),
        };
        checksum +%= result.cardinality();
        result.deinit();
    }
    return checksum;
}

fn measureCRoaringEndToEnd(operation: Operation, sources: *const CRoaringSources) !u64 {
    for (0..warmup_runs) |_| _ = try runCRoaringEndToEnd(operation, sources);
    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        const checksum = try runCRoaringEndToEnd(operation, sources);
        elapsed.* = bench_time.monotonicNanos() - start;
        std.mem.doNotOptimizeAway(checksum);
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return times[timed_runs / 2];
}

fn runCRoaringEndToEnd(operation: Operation, sources: *const CRoaringSources) !u64 {
    var checksum: u64 = 0;
    for (sources.bitmaps[0 .. sources.bitmaps.len - 1], sources.bitmaps[1..]) |left, right| {
        const result = switch (operation) {
            .pair_or => c.roaring_bitmap_or(left, right),
            .pair_andnot => c.roaring_bitmap_andnot(left, right),
        } orelse return error.CRoaringAllocFailed;
        checksum +%= c.roaring_bitmap_get_cardinality(result);
        c.roaring_bitmap_free(result);
    }
    return checksum;
}

fn measureMatched(operation: Operation, arm: Arm, pairs: *const PairSet) !u64 {
    for (0..warmup_runs) |_| _ = try runMatched(operation, arm, pairs, false);
    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        const result = try runMatched(operation, arm, pairs, false);
        elapsed.* = bench_time.monotonicNanos() - start;
        std.mem.doNotOptimizeAway(result.checksum);
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return times[timed_runs / 2];
}

fn runMatched(operation: Operation, arm: Arm, pairs: *const PairSet, digest_outputs: bool) !ArmResult {
    return switch (arm) {
        .a1_rawr_scalar, .b1_rawr_production, .b3_rawr_no_normalize => @call(.never_inline, runRawrMatched, .{ operation, arm, pairs, digest_outputs }),
        .a2_croaring_scalar, .a3_croaring_production, .b2_croaring_production => @call(.never_inline, runCRoaringMatched, .{ operation, arm, pairs, digest_outputs }),
        else => unreachable,
    };
}

fn runRawrMatched(operation: Operation, arm: Arm, pairs: *const PairSet, digest_outputs: bool) !ArmResult {
    var measured = ArmResult{ .branch = if (arm == .a1_rawr_scalar) .rawr_scalar else .not_applicable };
    var output: [ArrayContainer.MAX_CARDINALITY]u16 = undefined;
    var hasher = PairHasher.init();
    for (pairs.pairs, 0..) |pair, pair_index| {
        if (@intFromPtr(&output) == @intFromPtr(pair.left.values.ptr) or
            @intFromPtr(&output) == @intFromPtr(pair.right.values.ptr))
        {
            measured.outputs_distinct = false;
            return error.LayerAOutputAliasesInput;
        }

        if (arm == .a1_rawr_scalar) {
            const count = mergeInto(operation, pair, &output);
            measured.checksum +%= count;
            if (digest_outputs) hashPair(&hasher, pair_index, output[0..count]);
            continue;
        }

        var result = if (arm == .b1_rawr_production)
            switch (operation) {
                .pair_or => try rawr.container_ops.containerUnion(
                    std.heap.smp_allocator,
                    .{ .array = pair.left },
                    .{ .array = pair.right },
                ),
                .pair_andnot => try rawr.container_ops.containerDifference(
                    std.heap.smp_allocator,
                    .{ .array = pair.left },
                    .{ .array = pair.right },
                ),
            }
        else
            try mergeAllocatingWithoutNormalization(operation, pair);
        measured.checksum +%= result.getCardinality();
        const converted: u64 = switch (result) {
            .run => 1,
            else => 0,
        };
        if (arm == .b1_rawr_production) {
            measured.normalization_calls += @intFromBool(operation == .pair_or);
            measured.conversions += converted;
        }
        measured.allocation_calls += 2 + 2 * converted;
        if (digest_outputs) {
            const count = containerToArray(result, &output);
            hashPair(&hasher, pair_index, output[0..count]);
        }
        result.deinit(std.heap.smp_allocator);
    }
    measured.digest = if (digest_outputs) hasher.finish() else 0;
    return measured;
}

fn runCRoaringMatched(operation: Operation, arm: Arm, pairs: *const PairSet, digest_outputs: bool) !ArmResult {
    var result: c.rawr_cr_array_result_t = undefined;
    const c_arm: c.rawr_cr_array_arm_t = switch (arm) {
        .a2_croaring_scalar => c.RAWR_CR_ARRAY_SCALAR,
        .a3_croaring_production => c.RAWR_CR_ARRAY_PRODUCTION,
        .b2_croaring_production => c.RAWR_CR_ARRAY_ALLOCATING,
        else => unreachable,
    };
    if (!c.rawr_cr_array_attribution_run(
        pairs.c_pairs.ptr,
        pairs.c_pairs.len,
        operation.cValue(),
        c_arm,
        digest_outputs,
        &result,
    )) return error.CRoaringAttributionFailed;
    if (result.pair_count != pairs.accounting.matched_arrays or
        result.input_elements != pairs.accounting.input_elements)
    {
        return error.ArmInputCountMismatch;
    }
    return .{
        .checksum = result.checksum,
        .digest = if (digest_outputs) result.digest else 0,
        .allocation_calls = result.allocation_calls,
        .branch = switch (result.branch) {
            c.RAWR_CR_ARRAY_BRANCH_SCALAR => .scalar,
            c.RAWR_CR_ARRAY_BRANCH_AVX2 => .avx2,
            c.RAWR_CR_ARRAY_BRANCH_NOT_APPLICABLE => .not_applicable,
            else => return error.UnknownCRoaringBranch,
        },
        .outputs_distinct = result.outputs_distinct,
        .output_storage_unchanged = result.output_storage_unchanged,
    };
}

fn mergeAllocatingWithoutNormalization(operation: Operation, pair: Pair) !Container {
    const capacity: u16 = switch (operation) {
        .pair_or => @intCast(@as(u32, pair.left.cardinality) + pair.right.cardinality),
        .pair_andnot => pair.left.cardinality,
    };
    const result = try ArrayContainer.init(std.heap.smp_allocator, capacity);
    errdefer result.deinit(std.heap.smp_allocator);
    result.cardinality = @intCast(mergeInto(operation, pair, result.values[0..capacity]));
    return .{ .array = result };
}

fn mergeInto(operation: Operation, pair: Pair, output: []u16) usize {
    return switch (operation) {
        .pair_or => rawr.container_ops.benchmarkArrayUnionWrite(
            pair.left.values[0..pair.left.cardinality],
            pair.right.values[0..pair.right.cardinality],
            output,
        ),
        .pair_andnot => rawr.container_ops.benchmarkArrayDifferenceWrite(
            pair.left.values[0..pair.left.cardinality],
            pair.right.values[0..pair.right.cardinality],
            output,
        ),
    };
}

fn containerToArray(container: Container, output: []u16) usize {
    return switch (container) {
        .array => |array| blk: {
            @memcpy(output[0..array.cardinality], array.values[0..array.cardinality]);
            break :blk array.cardinality;
        },
        .bitset => |bitset| blk: {
            var count: usize = 0;
            for (bitset.words, 0..) |original_word, word_index| {
                var word = original_word;
                while (word != 0) {
                    const bit: u6 = @intCast(@ctz(word));
                    output[count] = @intCast(word_index * 64 + bit);
                    count += 1;
                    word &= word - 1;
                }
            }
            break :blk count;
        },
        .run => |run| blk: {
            var count: usize = 0;
            for (run.runs[0..run.n_runs]) |entry| {
                var value: u32 = entry.start;
                while (value <= entry.end()) : (value += 1) {
                    output[count] = @intCast(value);
                    count += 1;
                }
            }
            break :blk count;
        },
        .reserved => unreachable,
    };
}

fn validateBranch(arm: Arm, result: ArmResult) !void {
    if (arm != .a3_croaring_production) return;
    const has_avx2 = c.rawr_cr_array_runtime_has_avx2();
    if (builtin.cpu.arch == .aarch64) {
        if (result.branch != .scalar) return error.UnexpectedVectorBranch;
        return;
    }
    if (builtin.cpu.arch == .x86_64 and has_avx2 and result.outputs_distinct) {
        if (result.branch != .avx2) return error.ExpectedAVX2Branch;
    } else if (result.branch != .scalar) {
        return error.ExpectedScalarBranch;
    }
}

fn digestRawrEndToEnd(operation: Operation, sources: *const RawrSources, total_values: u64) !u64 {
    const output = try std.heap.page_allocator.alloc(u32, @intCast(total_values));
    defer std.heap.page_allocator.free(output);
    var hasher = FullHasher.init();
    for (sources.bitmaps[0 .. sources.bitmaps.len - 1], sources.bitmaps[1..], 0..) |*left, *right, index| {
        var result = switch (operation) {
            .pair_or => try left.bitwiseOr(std.heap.smp_allocator, right),
            .pair_andnot => try left.bitwiseDifference(std.heap.smp_allocator, right),
        };
        const count = result.toArray(output);
        hashFull(&hasher, index, output[0..count]);
        result.deinit();
    }
    return hasher.finish();
}

fn digestCRoaringEndToEnd(operation: Operation, sources: *const CRoaringSources, total_values: u64) !u64 {
    const output = try std.heap.page_allocator.alloc(u32, @intCast(total_values));
    defer std.heap.page_allocator.free(output);
    var hasher = FullHasher.init();
    for (sources.bitmaps[0 .. sources.bitmaps.len - 1], sources.bitmaps[1..], 0..) |left, right, index| {
        const result = switch (operation) {
            .pair_or => c.roaring_bitmap_or(left, right),
            .pair_andnot => c.roaring_bitmap_andnot(left, right),
        } orelse return error.CRoaringAllocFailed;
        const count: usize = @intCast(c.roaring_bitmap_get_cardinality(result));
        c.roaring_bitmap_to_uint32_array(result, output.ptr);
        hashFull(&hasher, index, output[0..count]);
        c.roaring_bitmap_free(result);
    }
    return hasher.finish();
}

fn printResult(
    requested: RequestedCell,
    median_ns: u64,
    digest: u64,
    corpus: *const corpus_mod.Corpus,
    pairs: *const PairSet,
    result: ArmResult,
) void {
    bench_time.print(
        "RESULT\t{s}\t{s}\t{d}\t0x{x:0>16}\t0x{x:0>16}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n",
        .{
            requested.operation.id(),
            requested.arm.id(),
            median_ns,
            digest,
            corpus.fingerprint,
            pairs.accounting.matched_arrays,
            pairs.accounting.input_elements,
            pairs.accounting.bitset_path,
            pairs.accounting.matched_other,
            pairs.accounting.unmatched_left,
            pairs.accounting.unmatched_right,
            result.conversions,
            result.allocation_calls,
            result.normalization_calls,
            @tagName(result.branch),
            @intFromBool(result.outputs_distinct),
            @intFromBool(result.output_storage_unchanged),
            pairs.sizes.min,
            pairs.sizes.p50,
            pairs.sizes.p90,
            pairs.sizes.p99,
            pairs.sizes.max,
        },
    );
}

fn sizeDistribution(sorted: []const u32) SizeDistribution {
    if (sorted.len == 0) return .{};
    return .{
        .min = sorted[0],
        .p50 = sorted[(sorted.len - 1) * 50 / 100],
        .p90 = sorted[(sorted.len - 1) * 90 / 100],
        .p99 = sorted[(sorted.len - 1) * 99 / 100],
        .max = sorted[sorted.len - 1],
    };
}

const PairHasher = struct {
    state: u64 = 0xcbf29ce484222325,
    const prime: u64 = 0x100000001b3;

    fn init() PairHasher {
        return .{};
    }

    fn addByte(self: *PairHasher, byte: u8) void {
        self.state = (self.state ^ byte) *% prime;
    }

    fn addU16(self: *PairHasher, value: u16) void {
        inline for (0..2) |shift| self.addByte(@truncate(value >> (shift * 8)));
    }

    fn addU64(self: *PairHasher, value: u64) void {
        inline for (0..8) |shift| self.addByte(@truncate(value >> (shift * 8)));
    }

    fn finish(self: PairHasher) u64 {
        return self.state;
    }
};

fn hashPair(hasher: *PairHasher, index: usize, values: []const u16) void {
    hasher.addU64(index);
    hasher.addU64(values.len);
    for (values) |value| hasher.addU16(value);
}

const FullHasher = struct {
    state: u64 = 0xcbf29ce484222325,
    const prime: u64 = 0x100000001b3;

    fn init() FullHasher {
        return .{};
    }

    fn addByte(self: *FullHasher, byte: u8) void {
        self.state = (self.state ^ byte) *% prime;
    }

    fn addU32(self: *FullHasher, value: u32) void {
        inline for (0..4) |shift| self.addByte(@truncate(value >> (shift * 8)));
    }

    fn addU64(self: *FullHasher, value: u64) void {
        inline for (0..8) |shift| self.addByte(@truncate(value >> (shift * 8)));
    }

    fn finish(self: FullHasher) u64 {
        return self.state;
    }
};

fn hashFull(hasher: *FullHasher, index: usize, values: []const u32) void {
    hasher.addU32(@intCast(index));
    hasher.addU64(values.len);
    for (values) |value| hasher.addU32(value);
}
