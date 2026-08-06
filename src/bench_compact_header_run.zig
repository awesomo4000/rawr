// SPDX-License-Identifier: MPL-2.0

//! Standalone RunContainer header-layout diagnostic for specs 32 and 32-01.
//! Full-bitmap clone, dense-AND, and select remain production-code measurements.

const std = @import("std");
const bench_time = @import("bench_time.zig");
const counting_allocator = @import("counting_allocator.zig");
const CountingAllocator = counting_allocator.CountingAllocator;

const warmup_runs = 3;
const timed_runs = 21;
const external_process_runs = 5;
const select_query_count = 1_000_000;
const select_seed = 12_345;
const build_block_values = 256;
const build_repeats = 256;
const allocation_repeats = 4096;
const min_capacity: u16 = 4;

// Filled from the pinned dense inventory and exact third-draw query stream.
const expected_dense_fingerprint: u64 = 0x5ea8bb16f114ae59;
const expected_select_fingerprint: u64 = 0xbef159fe82f3b120;

const Variant = enum {
    baseline,
    compact,
};

const Cell = enum {
    build_reserved,
    build_growth,
    clone,
    deinit,
    membership,
    iterate,
};

const variants = [_]Variant{ .baseline, .compact };
const cells = [_]Cell{
    .build_reserved,
    .build_growth,
    .clone,
    .deinit,
    .membership,
    .iterate,
};

const RunPair = packed struct {
    start: u16,
    length: u16,

    fn end(self: RunPair) u16 {
        return self.start +| self.length;
    }

    fn contains(self: RunPair, value: u16) bool {
        return value >= self.start and value <= self.end();
    }

    fn cardinality(self: RunPair) u32 {
        return @as(u32, self.length) + 1;
    }
};

const DenseEntry = struct {
    key: u16,
    run: RunPair,
};

const dense_a = [_]DenseEntry{
    .{ .key = 0, .run = .{ .start = 0, .length = 0xffff } },
    .{ .key = 1, .run = .{ .start = 0, .length = 0xffff } },
    .{ .key = 2, .run = .{ .start = 0, .length = 0xffff } },
    .{ .key = 3, .run = .{ .start = 0, .length = 0xffff } },
    .{ .key = 4, .run = .{ .start = 0, .length = 0xffff } },
    .{ .key = 5, .run = .{ .start = 0, .length = 0xffff } },
    .{ .key = 6, .run = .{ .start = 0, .length = 0xffff } },
    .{ .key = 7, .run = .{ .start = 0, .length = 41_247 } },
};

const dense_b = [_]DenseEntry{
    .{ .key = 3, .run = .{ .start = 53_392, .length = 12_143 } },
    .{ .key = 4, .run = .{ .start = 0, .length = 0xffff } },
    .{ .key = 5, .run = .{ .start = 0, .length = 0xffff } },
    .{ .key = 6, .run = .{ .start = 0, .length = 0xffff } },
    .{ .key = 7, .run = .{ .start = 0, .length = 0xffff } },
    .{ .key = 8, .run = .{ .start = 0, .length = 0xffff } },
    .{ .key = 9, .run = .{ .start = 0, .length = 0xffff } },
    .{ .key = 10, .run = .{ .start = 0, .length = 0xffff } },
    .{ .key = 11, .run = .{ .start = 0, .length = 29_103 } },
};

const BaselineRun = RunReplica(false);
const CompactRun = RunReplica(true);

comptime {
    std.debug.assert(@sizeOf(RunPair) == 4);
    std.debug.assert(@sizeOf(BaselineRun) == 24);
    std.debug.assert(@sizeOf(CompactRun) == 16);
    std.debug.assert(@alignOf(BaselineRun) >= 4);
    std.debug.assert(@alignOf(CompactRun) >= 4);
    std.debug.assert(dense_a.len == 8);
    std.debug.assert(dense_b.len == 9);
}

fn RunReplica(comptime compact: bool) type {
    return struct {
        runs: if (compact) [*]RunPair else []RunPair,
        n_runs: u16,
        capacity: u16,
        cardinality_cache: i32,

        const Self = @This();

        fn init(allocator: std.mem.Allocator, requested_capacity: u16) !*Self {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);
            const capacity = @max(min_capacity, requested_capacity);
            const runs = try allocator.alloc(RunPair, capacity);
            self.* = .{
                .runs = if (compact) runs.ptr else runs,
                .n_runs = 0,
                .capacity = capacity,
                .cardinality_cache = 0,
            };
            return self;
        }

        fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.storage());
            allocator.destroy(self);
        }

        fn clone(self: *const Self, allocator: std.mem.Allocator) !*Self {
            const copy = try allocator.create(Self);
            errdefer allocator.destroy(copy);
            const runs = try allocator.alloc(RunPair, self.capacity);
            @memcpy(runs[0..self.n_runs], self.readable());
            copy.* = .{
                .runs = if (compact) runs.ptr else runs,
                .n_runs = self.n_runs,
                .capacity = self.capacity,
                .cardinality_cache = self.cardinality_cache,
            };
            return copy;
        }

        fn contains(self: *const Self, value: u16) bool {
            if (self.n_runs == 0) return false;
            const index = self.search(value);
            return index < self.n_runs and self.readable()[index].contains(value);
        }

        fn addRange(self: *Self, allocator: std.mem.Allocator, start: u16, end: u16) !u64 {
            if (start > end) return 0;
            if (self.n_runs == 0) {
                try self.ensureCapacity(allocator, 1);
                self.storage()[0] = .{ .start = start, .length = end - start };
                self.n_runs = 1;
                self.cardinality_cache = -1;
                return @as(u64, end - start) + 1;
            }

            var low: usize = 0;
            var high: usize = self.n_runs;
            while (low < high) {
                const middle = low + (high - low) / 2;
                if (self.readable()[middle].end() < start -| 1) {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }

            const merge_start = low;
            var merge_end = low;
            var new_start = start;
            var new_end = end;
            while (merge_end < self.n_runs) {
                const run = self.readable()[merge_end];
                if (run.start > new_end +| 1) break;
                new_start = @min(new_start, run.start);
                new_end = @max(new_end, run.end());
                merge_end += 1;
            }

            var before: u64 = 0;
            for (self.readable()[merge_start..merge_end]) |run| before += run.cardinality();
            const new_run = RunPair{ .start = new_start, .length = new_end - new_start };
            const runs_removed = merge_end - merge_start;

            if (runs_removed == 0) {
                try self.ensureCapacity(allocator, self.n_runs + 1);
                const runs = self.storage();
                if (merge_start < self.n_runs) {
                    @memmove(runs[merge_start + 1 .. self.n_runs + 1], runs[merge_start..self.n_runs]);
                }
                runs[merge_start] = new_run;
                self.n_runs += 1;
            } else {
                const runs = self.storage();
                runs[merge_start] = new_run;
                if (runs_removed > 1) {
                    const remaining = self.n_runs - merge_end;
                    @memmove(runs[merge_start + 1 ..][0..remaining], runs[merge_end..self.n_runs]);
                }
                self.n_runs -= @intCast(runs_removed - 1);
            }

            self.cardinality_cache = -1;
            return @as(u64, new_run.length) + 1 - before;
        }

        fn getCardinality(self: *Self) u32 {
            if (self.cardinality_cache >= 0) return @intCast(self.cardinality_cache);
            var cardinality: u32 = 0;
            for (self.readable()) |run| cardinality += run.cardinality();
            self.cardinality_cache = @intCast(cardinality);
            return cardinality;
        }

        fn search(self: *const Self, value: u16) usize {
            var low: usize = 0;
            var high: usize = self.n_runs;
            while (low < high) {
                const middle = low + (high - low) / 2;
                if (self.readable()[middle].end() < value) {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            return low;
        }

        fn ensureCapacity(self: *Self, allocator: std.mem.Allocator, needed: u16) !void {
            if (needed <= self.capacity) return;
            const capacity = @max(self.capacity * 2, needed);
            const runs = try allocator.alloc(RunPair, capacity);
            @memcpy(runs[0..self.n_runs], self.readable());
            allocator.free(self.storage());
            self.runs = if (compact) runs.ptr else runs;
            self.capacity = capacity;
        }

        fn readable(self: *const Self) []const RunPair {
            return self.runs[0..self.n_runs];
        }

        fn storage(self: *Self) []RunPair {
            return self.runs[0..self.capacity];
        }
    };
}

const BuildRange = struct {
    start: u16,
    end: u16,
};

const BuildPlan = struct {
    offset: usize,
    count: usize,
    reserved_capacity: u16,
};

const Corpus = struct {
    allocator: std.mem.Allocator,
    build_ranges: []BuildRange,
    build_plans: [dense_a.len]BuildPlan,
    select_queries: []u32,
    dense_fingerprint: u64,
    select_fingerprint: u64,
    intersection_entries: usize,

    fn init(allocator: std.mem.Allocator) !Corpus {
        var total_ranges: usize = 0;
        for (dense_a) |entry| total_ranges += blockCount(entry.run);
        const build_ranges = try allocator.alloc(BuildRange, total_ranges);
        errdefer allocator.free(build_ranges);

        var build_plans: [dense_a.len]BuildPlan = undefined;
        var output_index: usize = 0;
        for (dense_a, 0..) |entry, entry_index| {
            const blocks = blockCount(entry.run);
            build_plans[entry_index] = .{
                .offset = output_index,
                .count = blocks,
                .reserved_capacity = @intCast((blocks + 1) / 2),
            };
            for (0..2) |parity| {
                var block_index = parity;
                while (block_index < blocks) : (block_index += 2) {
                    build_ranges[output_index] = blockRange(entry.run, block_index);
                    output_index += 1;
                }
            }
        }
        std.debug.assert(output_index == build_ranges.len);

        const select_queries = try allocator.alloc(u32, select_query_count);
        errdefer allocator.free(select_queries);
        var prng = std.Random.DefaultPrng.init(select_seed);
        const random = prng.random();
        for (select_queries) |*query| {
            // select_queries is the third draw in bench_croaring.initTestData;
            // consume the remaining draws to preserve the next iteration.
            _ = random.int(u32);
            _ = random.uintLessThan(u32, 500_000);
            query.* = random.uintLessThan(u32, 500_000);
            _ = random.uintLessThan(u32, 50_000);
            _ = random.uintLessThan(u32, 1024);
            _ = random.uintLessThan(u32, 20_000);
            _ = random.uintLessThan(u32, 20_000);
        }

        const corpus = Corpus{
            .allocator = allocator,
            .build_ranges = build_ranges,
            .build_plans = build_plans,
            .select_queries = select_queries,
            .dense_fingerprint = fingerprintDense(),
            .select_fingerprint = fingerprintQueries(select_queries),
            .intersection_entries = intersectionEntryCount(),
        };
        try corpus.assertPinned();
        return corpus;
    }

    fn deinit(self: *Corpus) void {
        self.allocator.free(self.select_queries);
        self.allocator.free(self.build_ranges);
        self.* = undefined;
    }

    fn assertPinned(self: *const Corpus) !void {
        if (dense_a.len != 8) return error.DenseAInventoryMismatch;
        if (dense_b.len != 9) return error.DenseBInventoryMismatch;
        if (denseCardinality(&dense_a) != 500_000 or denseCardinality(&dense_b) != 500_000) {
            return error.DenseCardinalityMismatch;
        }
        if (self.intersection_entries != 5) return error.DenseIntersectionInventoryMismatch;
        if (expected_dense_fingerprint != 0 and self.dense_fingerprint != expected_dense_fingerprint) {
            return error.DenseFingerprintMismatch;
        }
        if (expected_select_fingerprint != 0 and self.select_fingerprint != expected_select_fingerprint) {
            return error.SelectFingerprintMismatch;
        }
        if (self.select_queries.len != select_query_count) return error.SelectQueryCountMismatch;
        for (self.select_queries) |query| {
            if (query >= 500_000) return error.SelectQueryRangeMismatch;
        }
    }

    fn rangesFor(self: *const Corpus, entry_index: usize) []const BuildRange {
        const plan = self.build_plans[entry_index];
        return self.build_ranges[plan.offset..][0..plan.count];
    }
};

const Sample = struct {
    elapsed_ns: u64,
    teardown_ns: u64,
    stats: CountingAllocator.Stats,
    header_allocations: u64,
    checksum: u64,
};

fn blockCount(run: RunPair) usize {
    return (run.cardinality() + build_block_values - 1) / build_block_values;
}

fn blockRange(run: RunPair, block_index: usize) BuildRange {
    const start: u32 = @as(u32, run.start) + @as(u32, @intCast(block_index)) * build_block_values;
    const end = @min(@as(u32, run.end()), start + build_block_values - 1);
    return .{ .start = @intCast(start), .end = @intCast(end) };
}

fn denseCardinality(entries: []const DenseEntry) u64 {
    var cardinality: u64 = 0;
    for (entries) |entry| cardinality += entry.run.cardinality();
    return cardinality;
}

fn intersectionEntryCount() usize {
    var count: usize = 0;
    for (dense_a) |left| {
        for (dense_b) |right| {
            if (left.key == right.key) count += 1;
        }
    }
    return count;
}

fn mixFingerprint(fingerprint: *u64, value: anytype) void {
    fingerprint.* ^= @as(u64, @intCast(value));
    fingerprint.* *%= 0x100000001b3;
}

fn fingerprintDense() u64 {
    var fingerprint: u64 = 0xcbf29ce484222325;
    for (dense_a) |entry| {
        mixFingerprint(&fingerprint, 0xa0);
        mixFingerprint(&fingerprint, entry.key);
        mixFingerprint(&fingerprint, entry.run.start);
        mixFingerprint(&fingerprint, entry.run.length);
    }
    for (dense_b) |entry| {
        mixFingerprint(&fingerprint, 0xb0);
        mixFingerprint(&fingerprint, entry.key);
        mixFingerprint(&fingerprint, entry.run.start);
        mixFingerprint(&fingerprint, entry.run.length);
    }
    return fingerprint;
}

fn fingerprintQueries(queries: []const u32) u64 {
    var fingerprint: u64 = 0xcbf29ce484222325;
    for (queries) |query| mixFingerprint(&fingerprint, query);
    mixFingerprint(&fingerprint, queries.len);
    return fingerprint;
}

fn buildPlannedPopulation(
    comptime T: type,
    slots: []*T,
    allocator: std.mem.Allocator,
    corpus: *const Corpus,
    reserve: bool,
) !void {
    std.debug.assert(slots.len % dense_a.len == 0);
    var initialized: usize = 0;
    errdefer deinitPopulation(T, slots[0..initialized], allocator);
    for (slots, 0..) |*slot, slot_index| {
        const entry_index = slot_index % dense_a.len;
        const plan = corpus.build_plans[entry_index];
        const container = try T.init(allocator, if (reserve) plan.reserved_capacity else 0);
        errdefer container.deinit(allocator);
        for (corpus.rangesFor(entry_index)) |range| _ = try container.addRange(allocator, range.start, range.end);
        slot.* = container;
        initialized += 1;
    }
}

fn buildCanonicalPopulation(comptime T: type, slots: []*T, allocator: std.mem.Allocator) !void {
    std.debug.assert(slots.len % dense_a.len == 0);
    var initialized: usize = 0;
    errdefer deinitPopulation(T, slots[0..initialized], allocator);
    for (slots, 0..) |*slot, slot_index| {
        const entry = dense_a[slot_index % dense_a.len];
        const container = try T.init(allocator, 1);
        errdefer container.deinit(allocator);
        _ = try container.addRange(allocator, entry.run.start, entry.run.end());
        slot.* = container;
        initialized += 1;
    }
}

fn clonePopulation(comptime T: type, output: []*T, allocator: std.mem.Allocator, input: []const *T) !void {
    var initialized: usize = 0;
    errdefer deinitPopulation(T, output[0..initialized], allocator);
    for (output, input) |*slot, container| {
        slot.* = try container.clone(allocator);
        initialized += 1;
    }
}

fn deinitPopulation(comptime T: type, slots: []const *T, allocator: std.mem.Allocator) void {
    for (slots) |container| container.deinit(allocator);
}

fn populationChecksum(comptime T: type, slots: []const *T) u64 {
    var checksum: u64 = 0xcbf29ce484222325;
    for (slots) |container| {
        mixFingerprint(&checksum, container.n_runs);
        for (container.readable()) |run| {
            mixFingerprint(&checksum, run.start);
            mixFingerprint(&checksum, run.length);
        }
    }
    return checksum;
}

fn runBuild(comptime T: type, corpus: *const Corpus, reserve: bool) !Sample {
    const slot_count = dense_a.len * build_repeats;
    const slots = try std.heap.smp_allocator.alloc(*T, slot_count);
    defer std.heap.smp_allocator.free(slots);
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const allocator = counting.allocator();

    const start = bench_time.monotonicNanos();
    try buildPlannedPopulation(T, slots, allocator, corpus, reserve);
    const elapsed_ns = bench_time.monotonicNanos() - start;
    const stats = counting.snapshot();
    const checksum = populationChecksum(T, slots);
    std.mem.doNotOptimizeAway(checksum);

    const teardown_start = bench_time.monotonicNanos();
    deinitPopulation(T, slots, allocator);
    const teardown_ns = bench_time.monotonicNanos() - teardown_start;
    std.debug.assert(counting.stats.live_bytes == 0);
    std.debug.assert(counting.stats.live_class_bytes == 0);
    return .{
        .elapsed_ns = elapsed_ns,
        .teardown_ns = teardown_ns,
        .stats = stats,
        .header_allocations = slot_count,
        .checksum = checksum,
    };
}

fn runClone(comptime T: type) !Sample {
    const slot_count = dense_a.len * allocation_repeats;
    const source = try std.heap.smp_allocator.alloc(*T, slot_count);
    defer std.heap.smp_allocator.free(source);
    try buildCanonicalPopulation(T, source, std.heap.smp_allocator);
    defer deinitPopulation(T, source, std.heap.smp_allocator);

    const clones = try std.heap.smp_allocator.alloc(*T, slot_count);
    defer std.heap.smp_allocator.free(clones);
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const allocator = counting.allocator();
    const start = bench_time.monotonicNanos();
    try clonePopulation(T, clones, allocator, source);
    const elapsed_ns = bench_time.monotonicNanos() - start;
    const stats = counting.snapshot();
    const checksum = populationChecksum(T, clones);
    std.mem.doNotOptimizeAway(checksum);

    const teardown_start = bench_time.monotonicNanos();
    deinitPopulation(T, clones, allocator);
    const teardown_ns = bench_time.monotonicNanos() - teardown_start;
    std.debug.assert(counting.stats.live_bytes == 0);
    std.debug.assert(counting.stats.live_class_bytes == 0);
    return .{
        .elapsed_ns = elapsed_ns,
        .teardown_ns = teardown_ns,
        .stats = stats,
        .header_allocations = slot_count,
        .checksum = checksum,
    };
}

fn runDeinit(comptime T: type) !Sample {
    const slot_count = dense_a.len * allocation_repeats;
    const slots = try std.heap.smp_allocator.alloc(*T, slot_count);
    defer std.heap.smp_allocator.free(slots);
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const allocator = counting.allocator();
    try buildCanonicalPopulation(T, slots, allocator);
    counting.resetStats();

    const start = bench_time.monotonicNanos();
    deinitPopulation(T, slots, allocator);
    const elapsed_ns = bench_time.monotonicNanos() - start;
    const stats = counting.snapshot();
    std.debug.assert(stats.live_bytes == 0);
    std.debug.assert(stats.live_class_bytes == 0);
    return .{
        .elapsed_ns = elapsed_ns,
        .teardown_ns = elapsed_ns,
        .stats = stats,
        .header_allocations = 0,
        .checksum = slot_count,
    };
}

fn runMembership(comptime T: type, corpus: *const Corpus) !Sample {
    var slots: [dense_a.len]*T = undefined;
    try buildCanonicalPopulation(T, &slots, std.heap.smp_allocator);

    var hits: u64 = 0;
    const start = bench_time.monotonicNanos();
    for (corpus.select_queries) |query| {
        const key: usize = @intCast(query >> 16);
        const low: u16 = @truncate(query);
        hits += @intFromBool(slots[key].contains(low));
    }
    const elapsed_ns = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(hits);
    if (hits != select_query_count) return error.MembershipMismatch;

    const teardown_start = bench_time.monotonicNanos();
    deinitPopulation(T, &slots, std.heap.smp_allocator);
    const teardown_ns = bench_time.monotonicNanos() - teardown_start;
    return .{
        .elapsed_ns = elapsed_ns,
        .teardown_ns = teardown_ns,
        .stats = .{},
        .header_allocations = 0,
        .checksum = hits,
    };
}

fn runIteration(comptime T: type) !Sample {
    var slots: [dense_a.len]*T = undefined;
    try buildCanonicalPopulation(T, &slots, std.heap.smp_allocator);

    var checksum: u64 = 0xcbf29ce484222325;
    const start = bench_time.monotonicNanos();
    for (slots) |container| {
        for (container.readable()) |run| {
            var value: u32 = run.start;
            const end: u32 = run.end();
            while (value <= end) : (value += 1) {
                checksum ^= value;
                checksum *%= 0x100000001b3;
            }
        }
    }
    const elapsed_ns = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(checksum);

    const teardown_start = bench_time.monotonicNanos();
    deinitPopulation(T, &slots, std.heap.smp_allocator);
    const teardown_ns = bench_time.monotonicNanos() - teardown_start;
    return .{
        .elapsed_ns = elapsed_ns,
        .teardown_ns = teardown_ns,
        .stats = .{},
        .header_allocations = 0,
        .checksum = checksum,
    };
}

fn runVariant(comptime T: type, corpus: *const Corpus) ![cells.len]Sample {
    var samples: [cells.len]Sample = undefined;
    samples[@intFromEnum(Cell.build_reserved)] = try runBuild(T, corpus, true);
    samples[@intFromEnum(Cell.build_growth)] = try runBuild(T, corpus, false);
    samples[@intFromEnum(Cell.clone)] = try runClone(T);
    samples[@intFromEnum(Cell.deinit)] = try runDeinit(T);
    samples[@intFromEnum(Cell.membership)] = try runMembership(T, corpus);
    samples[@intFromEnum(Cell.iterate)] = try runIteration(T);
    return samples;
}

fn executeVariant(variant: Variant, corpus: *const Corpus) ![cells.len]Sample {
    return switch (variant) {
        .baseline => @call(.never_inline, runVariant, .{ BaselineRun, corpus }),
        .compact => @call(.never_inline, runVariant, .{ CompactRun, corpus }),
    };
}

fn validateAccountingPair(baseline: Sample, compact: Sample) !void {
    if (baseline.checksum != compact.checksum) return error.ReplicaChecksumMismatch;
    if (baseline.header_allocations != compact.header_allocations) return error.HeaderAllocationMismatch;
    if (baseline.stats.alloc_calls != compact.stats.alloc_calls or
        baseline.stats.free_calls != compact.stats.free_calls or
        baseline.stats.resize_calls != compact.stats.resize_calls or
        baseline.stats.remap_calls != compact.stats.remap_calls)
    {
        return error.AllocationCallMismatch;
    }

    const header_allocations = baseline.header_allocations;
    const expected_requested_delta = header_allocations * (@sizeOf(BaselineRun) - @sizeOf(CompactRun));
    const baseline_class = counting_allocator.smpClassBytes(
        @sizeOf(BaselineRun),
        std.mem.Alignment.fromByteUnits(@alignOf(BaselineRun)),
    );
    const compact_class = counting_allocator.smpClassBytes(
        @sizeOf(CompactRun),
        std.mem.Alignment.fromByteUnits(@alignOf(CompactRun)),
    );
    const expected_class_delta = header_allocations * (baseline_class - compact_class);
    if (baseline.stats.cumulative_bytes < compact.stats.cumulative_bytes or
        baseline.stats.cumulative_bytes - compact.stats.cumulative_bytes != expected_requested_delta)
    {
        return error.RequestedByteFirewallMismatch;
    }
    if (baseline.stats.cumulative_class_bytes < compact.stats.cumulative_class_bytes or
        baseline.stats.cumulative_class_bytes - compact.stats.cumulative_class_bytes != expected_class_delta)
    {
        return error.ClassByteFirewallMismatch;
    }
}

fn validateReplicaValues(corpus: *const Corpus) !void {
    var baseline_slots: [dense_a.len]*BaselineRun = undefined;
    var compact_slots: [dense_a.len]*CompactRun = undefined;
    const payload_alignment = std.mem.Alignment.fromByteUnits(@alignOf(RunPair));

    inline for (.{ true, false }) |reserve| {
        try buildPlannedPopulation(BaselineRun, &baseline_slots, std.heap.smp_allocator, corpus, reserve);
        defer deinitPopulation(BaselineRun, &baseline_slots, std.heap.smp_allocator);
        try buildPlannedPopulation(CompactRun, &compact_slots, std.heap.smp_allocator, corpus, reserve);
        defer deinitPopulation(CompactRun, &compact_slots, std.heap.smp_allocator);

        for (baseline_slots, compact_slots, dense_a) |baseline, compact, entry| {
            if (baseline.n_runs != 1 or compact.n_runs != 1) return error.RunInventoryMismatch;
            if (baseline.capacity != compact.capacity) return error.ReplicaShapeMismatch;
            if (!std.mem.eql(RunPair, baseline.readable(), compact.readable())) return error.ReplicaValueMismatch;
            if (!std.meta.eql(entry.run, baseline.readable()[0])) return error.DenseValueMismatch;
            if (baseline.getCardinality() != entry.run.cardinality() or
                compact.getCardinality() != entry.run.cardinality())
            {
                return error.DenseCardinalityMismatch;
            }

            const baseline_requested = @as(usize, baseline.capacity) * @sizeOf(RunPair);
            const compact_requested = @as(usize, compact.capacity) * @sizeOf(RunPair);
            if (baseline_requested != compact_requested or
                counting_allocator.smpClassBytes(baseline_requested, payload_alignment) !=
                    counting_allocator.smpClassBytes(compact_requested, payload_alignment))
            {
                return error.PayloadFirewallMismatch;
            }
        }
    }
}

fn variantName(variant: Variant) []const u8 {
    return @tagName(variant);
}

fn cellName(cell: Cell) []const u8 {
    return switch (cell) {
        .build_reserved => "build-reserved",
        .build_growth => "build-growth",
        .clone => "clone",
        .deinit => "deinit",
        .membership => "membership",
        .iterate => "iteration",
    };
}

fn sortedField(samples: *const [timed_runs]Sample, comptime field: []const u8) [timed_runs]u64 {
    var values: [timed_runs]u64 = undefined;
    for (samples, 0..) |sample, index| values[index] = @field(sample, field);
    std.mem.sort(u64, &values, {}, std.sort.asc(u64));
    return values;
}

fn stableStats(samples: *const [timed_runs]Sample) !CountingAllocator.Stats {
    const expected = samples[0].stats;
    const expected_header_allocations = samples[0].header_allocations;
    const expected_checksum = samples[0].checksum;
    for (samples[1..]) |sample| {
        if (!std.meta.eql(expected, sample.stats) or
            sample.header_allocations != expected_header_allocations or
            sample.checksum != expected_checksum)
        {
            return error.UnstableAccounting;
        }
    }
    return expected;
}

fn printResults(samples: *const [variants.len][cells.len][timed_runs]Sample) !void {
    bench_time.print("\n{s:<9} {s:<15} {s:>12} {s:>12} {s:>12} {s:>9} {s:>9} {s:>14} {s:>14} {s:>14}\n", .{
        "variant", "cell", "median ns", "min ns", "max ns", "alloc", "free", "requested", "class bytes", "teardown ns",
    });
    bench_time.print("{s:-<9} {s:-<15} {s:->12} {s:->12} {s:->12} {s:->9} {s:->9} {s:->14} {s:->14} {s:->14}\n", .{
        "", "", "", "", "", "", "", "", "", "",
    });

    for (variants, 0..) |variant, variant_index| {
        for (cells, 0..) |cell, cell_index| {
            const cell_samples = &samples[variant_index][cell_index];
            const elapsed = sortedField(cell_samples, "elapsed_ns");
            const teardown = sortedField(cell_samples, "teardown_ns");
            const stats = try stableStats(cell_samples);
            const median_ns = elapsed[timed_runs / 2];
            const teardown_median_ns = teardown[timed_runs / 2];
            bench_time.print("{s:<9} {s:<15} {d:>12} {d:>12} {d:>12} {d:>9} {d:>9} {d:>14} {d:>14} {d:>14}\n", .{
                variantName(variant),
                cellName(cell),
                median_ns,
                elapsed[0],
                elapsed[timed_runs - 1],
                stats.alloc_calls,
                stats.free_calls,
                stats.cumulative_bytes,
                stats.cumulative_class_bytes,
                teardown_median_ns,
            });
            bench_time.print("RESULT\t{s}\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
                variantName(variant),
                cellName(cell),
                median_ns,
                elapsed[0],
                elapsed[timed_runs - 1],
                stats.alloc_calls,
                stats.free_calls,
                stats.cumulative_bytes,
                stats.cumulative_class_bytes,
                teardown_median_ns,
                teardown[0],
                teardown[timed_runs - 1],
                cell_samples[0].header_allocations,
                cell_samples[0].checksum,
            });
        }
    }
}

fn printLayout(comptime T: type, name: []const u8) void {
    const alignment = std.mem.Alignment.fromByteUnits(@alignOf(T));
    bench_time.print("LAYOUT\t{s}\theader-requested\t{d}\theader-align\t{d}\theader-class\t{d}\n", .{
        name,
        @sizeOf(T),
        @alignOf(T),
        counting_allocator.smpClassBytes(@sizeOf(T), alignment),
    });
}

pub fn main() !void {
    bench_time.print("Compact RunContainer header replica diagnostic\n", .{});
    bench_time.print("==============================================\n", .{});
    bench_time.printRunTimestamp();
    bench_time.printBenchEnvironment();
    bench_time.print("PROTOCOL\twarmup\t{d}\ttimed\t{d}\texternal-processes\t{d}\tallocator\tsmp\n", .{
        warmup_runs,
        timed_runs,
        external_process_runs,
    });
    bench_time.print("BOUNDARY\toperation timing excludes checksum and teardown; teardown_ns is separate\n", .{});
    bench_time.print("BATCH\tbuild-repeats\t{d}\tallocation-repeats\t{d}\n", .{ build_repeats, allocation_repeats });
    printLayout(BaselineRun, "baseline");
    printLayout(CompactRun, "compact");

    const baseline_header_class = counting_allocator.smpClassBytes(
        @sizeOf(BaselineRun),
        std.mem.Alignment.fromByteUnits(@alignOf(BaselineRun)),
    );
    const compact_header_class = counting_allocator.smpClassBytes(
        @sizeOf(CompactRun),
        std.mem.Alignment.fromByteUnits(@alignOf(CompactRun)),
    );
    if (baseline_header_class != 32 or compact_header_class != 16) return error.HeaderClassMismatch;

    var corpus = try Corpus.init(std.heap.smp_allocator);
    defer corpus.deinit();
    bench_time.print("CORPUS\tdense-fingerprint\t{x}\tselect-seed\t{d}\tselect-fingerprint\t{x}\n", .{
        corpus.dense_fingerprint,
        select_seed,
        corpus.select_fingerprint,
    });
    bench_time.print("INVENTORY\ta-runs\t{d}\tb-runs\t{d}\tand-runs\t{d}\ta-cardinality\t{d}\tb-cardinality\t{d}\tselect-queries\t{d}\n", .{
        dense_a.len,
        dense_b.len,
        corpus.intersection_entries,
        denseCardinality(&dense_a),
        denseCardinality(&dense_b),
        corpus.select_queries.len,
    });
    try validateReplicaValues(&corpus);
    bench_time.print("VALIDATION\truns=equal\tcardinality=equal\tpayload-requested=equal\tpayload-align={d}\tpayload-class=equal\n", .{
        @alignOf(RunPair),
    });

    var samples: [variants.len][cells.len][timed_runs]Sample = undefined;
    for (0..warmup_runs + timed_runs) |round| {
        bench_time.print("round {d}/{d}\n", .{ round + 1, warmup_runs + timed_runs });
        var round_samples: [variants.len][cells.len]Sample = undefined;
        for (0..variants.len) |slot| {
            const variant_index = (round + slot) % variants.len;
            round_samples[variant_index] = try executeVariant(variants[variant_index], &corpus);
        }
        for (cells, 0..) |_, cell_index| {
            try validateAccountingPair(
                round_samples[@intFromEnum(Variant.baseline)][cell_index],
                round_samples[@intFromEnum(Variant.compact)][cell_index],
            );
        }
        if (round >= warmup_runs) {
            const timed_index = round - warmup_runs;
            for (variants, 0..) |_, variant_index| {
                for (cells, 0..) |_, cell_index| {
                    samples[variant_index][cell_index][timed_index] = round_samples[variant_index][cell_index];
                }
            }
        }
    }
    try printResults(&samples);
}

test "run replicas preserve compact layout and payload behavior" {
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(BaselineRun));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(CompactRun));
    try std.testing.expect(@alignOf(BaselineRun) >= 4);
    try std.testing.expect(@alignOf(CompactRun) >= 4);

    const baseline = try BaselineRun.init(std.testing.allocator, 0);
    defer baseline.deinit(std.testing.allocator);
    const compact = try CompactRun.init(std.testing.allocator, 0);
    defer compact.deinit(std.testing.allocator);
    for (0..32) |index| {
        const start: u16 = @intCast(index * 4);
        _ = try baseline.addRange(std.testing.allocator, start, start + 1);
        _ = try compact.addRange(std.testing.allocator, start, start + 1);
    }
    _ = try baseline.addRange(std.testing.allocator, 0, 127);
    _ = try compact.addRange(std.testing.allocator, 0, 127);
    try std.testing.expectEqual(baseline.capacity, compact.capacity);
    try std.testing.expectEqualSlices(RunPair, baseline.readable(), compact.readable());
    try std.testing.expectEqual(@as(u32, 128), baseline.getCardinality());
    try std.testing.expectEqual(@as(u32, 128), compact.getCardinality());
    try std.testing.expect(baseline.contains(64));
    try std.testing.expect(compact.contains(64));

    const baseline_clone = try baseline.clone(std.testing.allocator);
    defer baseline_clone.deinit(std.testing.allocator);
    const compact_clone = try compact.clone(std.testing.allocator);
    defer compact_clone.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(RunPair, baseline_clone.readable(), compact_clone.readable());
}
