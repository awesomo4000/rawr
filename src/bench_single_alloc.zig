// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const bench_time = @import("bench_time.zig");
const CountingAllocator = @import("counting_allocator.zig").CountingAllocator;

const CONTAINER_COUNT = 10_000;
const CORPUS_SEED = 0x13_00_D2_2026;
const PROBES_PER_CONTAINER = 16;
const WARMUP_RUNS = 1;
const TIMED_RUNS = 9;
const MAX_CARDINALITY: u16 = 4096;
const MIN_CAPACITY: u16 = 4;
const BLOCK_ALIGNMENT = 16;

const Variant = enum {
    baseline_32,
    control_16,
    single_stored,
    single_derived,
};

const Workload = enum {
    build_reserved,
    build_growth,
    clone,
    deinit,
    membership,
    iterate,
    cardinality,
};

const variants = [_]Variant{
    .baseline_32,
    .control_16,
    .single_stored,
    .single_derived,
};

const workloads = [_]Workload{
    .build_reserved,
    .build_growth,
    .clone,
    .deinit,
    .membership,
    .iterate,
    .cardinality,
};

const Sample = struct {
    elapsed_ns: u64,
    stats: CountingAllocator.Stats,
};

const Entry = struct {
    offset: usize,
    cardinality: u16,
};

const Corpus = struct {
    allocator: std.mem.Allocator,
    entries: []Entry,
    values: []u16,
    probes: []u16,

    fn init(allocator: std.mem.Allocator) !Corpus {
        var prng = std.Random.DefaultPrng.init(CORPUS_SEED);
        const random = prng.random();

        const entries = try allocator.alloc(Entry, CONTAINER_COUNT);
        errdefer allocator.free(entries);

        var total_values: usize = 0;
        for (entries, 0..) |*entry, i| {
            const cardinality = if (i < CONTAINER_COUNT * 50 / 100)
                random.intRangeAtMost(u16, 1, 64)
            else if (i < CONTAINER_COUNT * 85 / 100)
                random.intRangeAtMost(u16, 256, 1024)
            else
                random.intRangeAtMost(u16, 3840, MAX_CARDINALITY);

            entry.* = .{
                .offset = total_values,
                .cardinality = cardinality,
            };
            total_values += cardinality;
        }

        const values = try allocator.alloc(u16, total_values);
        errdefer allocator.free(values);
        const probes = try allocator.alloc(u16, CONTAINER_COUNT * PROBES_PER_CONTAINER);
        errdefer allocator.free(probes);
        const marks = try allocator.alloc(u16, 1 << 16);
        defer allocator.free(marks);
        @memset(marks, 0);

        for (entries, 0..) |entry, i| {
            const container_values = values[entry.offset..][0..entry.cardinality];
            const stamp: u16 = @intCast(i + 1);

            var count: usize = 0;
            while (count < container_values.len) {
                const value = random.int(u16);
                if (marks[value] == stamp) continue;
                marks[value] = stamp;
                container_values[count] = value;
                count += 1;
            }
            std.mem.sort(u16, container_values, {}, std.sort.asc(u16));

            const container_probes = probes[i * PROBES_PER_CONTAINER ..][0..PROBES_PER_CONTAINER];
            for (container_probes, 0..) |*probe, probe_i| {
                if (probe_i % 2 == 0) {
                    probe.* = container_values[random.uintLessThan(usize, container_values.len)];
                } else {
                    while (true) {
                        const candidate = random.int(u16);
                        if (!sliceContains(container_values, candidate)) {
                            probe.* = candidate;
                            break;
                        }
                    }
                }
            }
        }

        return .{
            .allocator = allocator,
            .entries = entries,
            .values = values,
            .probes = probes,
        };
    }

    fn deinit(self: *Corpus) void {
        self.allocator.free(self.probes);
        self.allocator.free(self.values);
        self.allocator.free(self.entries);
        self.* = undefined;
    }

    fn entryValues(self: *const Corpus, index: usize) []const u16 {
        const entry = self.entries[index];
        return self.values[entry.offset..][0..entry.cardinality];
    }

    fn entryProbes(self: *const Corpus, index: usize) []const u16 {
        return self.probes[index * PROBES_PER_CONTAINER ..][0..PROBES_PER_CONTAINER];
    }
};

const Baseline32 = TwoAllocContainer(32);
const Control16 = TwoAllocContainer(16);
const SingleStored = SingleAllocContainer(true);
const SingleDerived = SingleAllocContainer(false);

fn TwoAllocContainer(comptime value_alignment: comptime_int) type {
    return struct {
        values: []align(value_alignment) u16,
        cardinality: u16,
        capacity: u16,

        const Self = @This();
        const AddResult = struct {
            container: *Self,
            added: bool,
        };

        fn init(allocator: std.mem.Allocator) !*Self {
            return initCapacity(allocator, MIN_CAPACITY);
        }

        fn initCapacity(allocator: std.mem.Allocator, requested: u16) !*Self {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            const capacity = normalizedCapacity(requested);
            const values = try allocator.alignedAlloc(
                u16,
                std.mem.Alignment.fromByteUnits(value_alignment),
                capacity,
            );
            self.* = .{
                .values = values,
                .cardinality = 0,
                .capacity = capacity,
            };
            return self;
        }

        fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.values[0..self.capacity]);
            allocator.destroy(self);
        }

        fn clone(self: *const Self, allocator: std.mem.Allocator) !*Self {
            const copy = try allocator.create(Self);
            errdefer allocator.destroy(copy);
            const values = try allocator.alignedAlloc(
                u16,
                std.mem.Alignment.fromByteUnits(value_alignment),
                self.capacity,
            );
            @memcpy(values[0..self.cardinality], self.values[0..self.cardinality]);
            copy.* = .{
                .values = values,
                .cardinality = self.cardinality,
                .capacity = self.capacity,
            };
            return copy;
        }

        fn add(self: *Self, allocator: std.mem.Allocator, value: u16) !AddResult {
            if (self.cardinality != 0 and value <= self.values[self.cardinality - 1]) {
                const pos = lowerBound(self.values[0..self.cardinality], value);
                if (pos < self.cardinality and self.values[pos] == value) {
                    return .{ .container = self, .added = false };
                }
                try self.ensureCapacity(allocator, self.cardinality + 1);
                @memmove(
                    self.values[pos + 1 .. self.cardinality + 1],
                    self.values[pos..self.cardinality],
                );
                self.values[pos] = value;
            } else {
                try self.ensureCapacity(allocator, self.cardinality + 1);
                self.values[self.cardinality] = value;
            }
            self.cardinality += 1;
            return .{ .container = self, .added = true };
        }

        fn contains(self: *const Self, value: u16) bool {
            const pos = lowerBound(self.values[0..self.cardinality], value);
            return pos < self.cardinality and self.values[pos] == value;
        }

        fn sumValues(self: *const Self) u64 {
            var sum: u64 = 0;
            for (self.values[0..self.cardinality]) |value| sum +%= value;
            return sum;
        }

        fn getCardinality(self: *const Self) u16 {
            return self.cardinality;
        }

        fn ensureCapacity(self: *Self, allocator: std.mem.Allocator, needed: u16) !void {
            if (needed <= self.capacity) return;
            const capacity = nextCapacity(self.capacity, needed);
            const values = try allocator.alignedAlloc(
                u16,
                std.mem.Alignment.fromByteUnits(value_alignment),
                capacity,
            );
            @memcpy(values[0..self.cardinality], self.values[0..self.cardinality]);
            allocator.free(self.values[0..self.capacity]);
            self.values = values;
            self.capacity = capacity;
        }
    };
}

fn SingleAllocContainer(comptime store_slice: bool) type {
    return struct {
        cardinality: u16,
        capacity: u16,
        values_storage: if (store_slice) []align(BLOCK_ALIGNMENT) u16 else void,

        const Self = @This();
        const AddResult = struct {
            container: *Self,
            added: bool,
        };

        comptime {
            std.debug.assert(dataOffset() % BLOCK_ALIGNMENT == 0);
            std.debug.assert(BLOCK_ALIGNMENT >= 4);
        }

        fn init(allocator: std.mem.Allocator) !*Self {
            return initCapacity(allocator, MIN_CAPACITY);
        }

        fn initCapacity(allocator: std.mem.Allocator, requested: u16) !*Self {
            return allocateBlock(allocator, normalizedCapacity(requested));
        }

        fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.blockSlice());
        }

        fn clone(self: *const Self, allocator: std.mem.Allocator) !*Self {
            const copy = try allocateBlock(allocator, self.capacity);
            copy.cardinality = self.cardinality;
            @memcpy(copy.values()[0..self.cardinality], self.values()[0..self.cardinality]);
            return copy;
        }

        fn add(self: *Self, allocator: std.mem.Allocator, value: u16) !AddResult {
            var current = self;
            var current_values = current.values();
            if (current.cardinality != 0 and value <= current_values[current.cardinality - 1]) {
                const pos = lowerBound(current_values[0..current.cardinality], value);
                if (pos < current.cardinality and current_values[pos] == value) {
                    return .{ .container = current, .added = false };
                }
                current = try current.ensureCapacity(allocator, current.cardinality + 1);
                current_values = current.values();
                @memmove(
                    current_values[pos + 1 .. current.cardinality + 1],
                    current_values[pos..current.cardinality],
                );
                current_values[pos] = value;
            } else {
                current = try current.ensureCapacity(allocator, current.cardinality + 1);
                current_values = current.values();
                current_values[current.cardinality] = value;
            }
            current.cardinality += 1;
            return .{ .container = current, .added = true };
        }

        fn contains(self: *const Self, value: u16) bool {
            const current_values = self.values()[0..self.cardinality];
            const pos = lowerBound(current_values, value);
            return pos < current_values.len and current_values[pos] == value;
        }

        fn sumValues(self: *const Self) u64 {
            var sum: u64 = 0;
            for (self.values()[0..self.cardinality]) |value| sum +%= value;
            return sum;
        }

        fn getCardinality(self: *const Self) u16 {
            return self.cardinality;
        }

        fn ensureCapacity(self: *Self, allocator: std.mem.Allocator, needed: u16) !*Self {
            if (needed <= self.capacity) return self;

            const capacity = nextCapacity(self.capacity, needed);
            const old_block = self.blockSlice();
            const new_size = try blockSize(capacity);
            if (allocator.resize(old_block, new_size)) {
                self.capacity = capacity;
                self.refreshValues();
                return self;
            }

            const moved = try allocateBlock(allocator, capacity);
            moved.cardinality = self.cardinality;
            @memcpy(moved.values()[0..self.cardinality], self.values()[0..self.cardinality]);
            allocator.free(old_block);
            return moved;
        }

        fn allocateBlock(allocator: std.mem.Allocator, capacity: u16) !*Self {
            const bytes = try allocator.alignedAlloc(u8, .@"16", try blockSize(capacity));
            const self: *Self = @ptrCast(bytes.ptr);
            self.* = .{
                .cardinality = 0,
                .capacity = capacity,
                .values_storage = if (store_slice) undefined else {},
            };
            self.refreshValues();
            return self;
        }

        fn refreshValues(self: *Self) void {
            if (store_slice) self.values_storage = self.derivedValues();
        }

        fn values(self: *const Self) []align(BLOCK_ALIGNMENT) u16 {
            if (store_slice) return self.values_storage;
            return self.derivedValues();
        }

        fn derivedValues(self: *const Self) []align(BLOCK_ALIGNMENT) u16 {
            const bytes: [*]u8 = @ptrCast(@constCast(self));
            const value_ptr: [*]align(BLOCK_ALIGNMENT) u16 = @ptrCast(@alignCast(bytes + dataOffset()));
            return value_ptr[0..self.capacity];
        }

        fn blockSlice(self: *Self) []align(BLOCK_ALIGNMENT) u8 {
            const bytes: [*]align(BLOCK_ALIGNMENT) u8 = @ptrCast(@alignCast(self));
            return bytes[0 .. blockSize(self.capacity) catch unreachable];
        }

        fn dataOffset() usize {
            return std.mem.alignForward(usize, @sizeOf(Self), BLOCK_ALIGNMENT);
        }

        fn blockSize(capacity: u16) !usize {
            const payload_size = std.math.mul(usize, capacity, @sizeOf(u16)) catch return error.OutOfMemory;
            return std.math.add(usize, dataOffset(), payload_size) catch return error.OutOfMemory;
        }
    };
}

fn normalizedCapacity(requested: u16) u16 {
    if (requested <= MIN_CAPACITY) return MIN_CAPACITY;
    return std.math.ceilPowerOfTwo(u16, requested) catch MAX_CARDINALITY;
}

fn nextCapacity(current: u16, needed: u16) u16 {
    std.debug.assert(current < MAX_CARDINALITY);
    const rounded = std.math.ceilPowerOfTwo(u16, needed) catch MAX_CARDINALITY;
    return @min(MAX_CARDINALITY, @max(current * 2, rounded));
}

fn lowerBound(values: []const u16, needle: u16) usize {
    var lo: usize = 0;
    var hi = values.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (values[mid] < needle) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

fn sliceContains(values: []const u16, needle: u16) bool {
    const pos = lowerBound(values, needle);
    return pos < values.len and values[pos] == needle;
}

fn runVariant(comptime T: type, corpus: *const Corpus) ![workloads.len]Sample {
    const backing = std.heap.smp_allocator;
    var counting = CountingAllocator.init(backing);
    const allocator = counting.allocator();
    var result: [workloads.len]Sample = undefined;

    const slots = try backing.alloc(*T, CONTAINER_COUNT);
    defer backing.free(slots);
    const clones = try backing.alloc(*T, CONTAINER_COUNT);
    defer backing.free(clones);

    counting.resetStats();
    var start = bench_time.monotonicNanos();
    try buildContainers(T, slots, allocator, corpus, false);
    result[@intFromEnum(Workload.build_growth)] = finishSample(&counting, start);
    std.mem.doNotOptimizeAway(slots.ptr);
    deinitContainers(T, slots, allocator);
    std.debug.assert(counting.stats.live_bytes == 0);

    counting.resetStats();
    start = bench_time.monotonicNanos();
    try buildContainers(T, slots, allocator, corpus, true);
    result[@intFromEnum(Workload.build_reserved)] = finishSample(&counting, start);
    std.mem.doNotOptimizeAway(slots.ptr);

    counting.resetStats();
    start = bench_time.monotonicNanos();
    var membership_sum: usize = 0;
    for (slots, 0..) |container, i| {
        for (corpus.entryProbes(i)) |probe| {
            membership_sum += @intFromBool(container.contains(probe));
        }
    }
    result[@intFromEnum(Workload.membership)] = finishSample(&counting, start);
    std.mem.doNotOptimizeAway(membership_sum);

    counting.resetStats();
    start = bench_time.monotonicNanos();
    var iteration_sum: u64 = 0;
    for (slots) |container| iteration_sum +%= container.sumValues();
    result[@intFromEnum(Workload.iterate)] = finishSample(&counting, start);
    std.mem.doNotOptimizeAway(iteration_sum);

    counting.resetStats();
    start = bench_time.monotonicNanos();
    var cardinality_sum: u64 = 0;
    for (slots) |container| cardinality_sum += container.getCardinality();
    result[@intFromEnum(Workload.cardinality)] = finishSample(&counting, start);
    std.mem.doNotOptimizeAway(cardinality_sum);

    counting.resetStats();
    start = bench_time.monotonicNanos();
    for (slots, clones) |container, *clone_slot| {
        clone_slot.* = try container.clone(allocator);
    }
    result[@intFromEnum(Workload.clone)] = finishSample(&counting, start);
    std.mem.doNotOptimizeAway(clones.ptr);
    deinitContainers(T, clones, allocator);

    counting.resetStats();
    start = bench_time.monotonicNanos();
    deinitContainers(T, slots, allocator);
    result[@intFromEnum(Workload.deinit)] = finishSample(&counting, start);
    std.debug.assert(counting.stats.live_bytes == 0);
    std.debug.assert(counting.stats.remap_calls == 0);

    return result;
}

fn buildContainers(
    comptime T: type,
    slots: []*T,
    allocator: std.mem.Allocator,
    corpus: *const Corpus,
    reserve: bool,
) !void {
    for (slots, 0..) |*slot, i| {
        const values = corpus.entryValues(i);
        var container = if (reserve)
            try T.initCapacity(allocator, @intCast(values.len))
        else
            try T.init(allocator);
        for (values) |value| {
            const add_result = try container.add(allocator, value);
            std.debug.assert(add_result.added);
            container = add_result.container;
        }
        slot.* = container;
    }
}

fn deinitContainers(comptime T: type, slots: []*T, allocator: std.mem.Allocator) void {
    for (slots) |container| container.deinit(allocator);
}

fn finishSample(counting: *CountingAllocator, start: u64) Sample {
    const elapsed_ns = bench_time.monotonicNanos() - start;
    const stats = counting.snapshot();
    std.debug.assert(stats.remap_calls == 0);
    return .{ .elapsed_ns = elapsed_ns, .stats = stats };
}

fn executeVariant(variant: Variant, corpus: *const Corpus) ![workloads.len]Sample {
    return switch (variant) {
        .baseline_32 => @call(.never_inline, runVariant, .{ Baseline32, corpus }),
        .control_16 => @call(.never_inline, runVariant, .{ Control16, corpus }),
        .single_stored => @call(.never_inline, runVariant, .{ SingleStored, corpus }),
        .single_derived => @call(.never_inline, runVariant, .{ SingleDerived, corpus }),
    };
}

fn variantName(variant: Variant) []const u8 {
    return switch (variant) {
        .baseline_32 => "baseline-32",
        .control_16 => "control-16",
        .single_stored => "single-stored",
        .single_derived => "single-derived",
    };
}

fn workloadName(workload: Workload) []const u8 {
    return switch (workload) {
        .build_reserved => "build-reserved",
        .build_growth => "build-growth",
        .clone => "clone",
        .deinit => "deinit",
        .membership => "membership",
        .iterate => "iterate",
        .cardinality => "cardinality",
    };
}

fn medianField(samples: *const [TIMED_RUNS]Sample, comptime field: []const u8) u64 {
    var values: [TIMED_RUNS]u64 = undefined;
    for (samples, 0..) |sample, i| values[i] = @field(sample.stats, field);
    std.mem.sort(u64, &values, {}, std.sort.asc(u64));
    return values[TIMED_RUNS / 2];
}

fn medianTime(samples: *const [TIMED_RUNS]Sample) u64 {
    var values: [TIMED_RUNS]u64 = undefined;
    for (samples, 0..) |sample, i| values[i] = sample.elapsed_ns;
    std.mem.sort(u64, &values, {}, std.sort.asc(u64));
    return values[TIMED_RUNS / 2];
}

fn printResults(samples: *const [variants.len][workloads.len][TIMED_RUNS]Sample) void {
    bench_time.print("\n{s:<15} {s:<15} {s:>9} {s:>7} {s:>7} {s:>11} {s:>11} {s:>12} {s:>12} {s:>12}\n", .{
        "variant", "workload", "ms", "alloc", "free", "resize ok", "resize fail", "requested", "live", "peak",
    });
    bench_time.print("{s:-<15} {s:-<15} {s:->9} {s:->7} {s:->7} {s:->11} {s:->11} {s:->12} {s:->12} {s:->12}\n", .{
        "", "", "", "", "", "", "", "", "", "",
    });

    for (variants, 0..) |variant, variant_i| {
        for (workloads, 0..) |workload, workload_i| {
            const workload_samples = &samples[variant_i][workload_i];
            const elapsed_ns = medianTime(workload_samples);
            const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms;
            bench_time.print("{s:<15} {s:<15} {d:>9.3} {d:>7} {d:>7} {d:>11} {d:>11} {d:>12} {d:>12} {d:>12}\n", .{
                variantName(variant),
                workloadName(workload),
                elapsed_ms,
                medianField(workload_samples, "alloc_calls"),
                medianField(workload_samples, "free_calls"),
                medianField(workload_samples, "resize_successes"),
                medianField(workload_samples, "resize_failures"),
                medianField(workload_samples, "cumulative_bytes"),
                medianField(workload_samples, "live_bytes"),
                medianField(workload_samples, "peak_live_bytes"),
            });
            bench_time.print("RESULT\t{s}\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
                variantName(variant),
                workloadName(workload),
                elapsed_ns,
                medianField(workload_samples, "alloc_calls"),
                medianField(workload_samples, "free_calls"),
                medianField(workload_samples, "resize_successes"),
                medianField(workload_samples, "resize_failures"),
                medianField(workload_samples, "cumulative_bytes"),
                medianField(workload_samples, "live_bytes"),
                medianField(workload_samples, "peak_live_bytes"),
            });
        }
    }
}

pub fn main() !void {
    bench_time.print("Single-allocation ArrayContainer prototype\n", .{});
    bench_time.print("==========================================\n", .{});
    bench_time.print("N={d}, seed=0x{x}, probes={d}, warmup={d}, timed={d}\n", .{
        CONTAINER_COUNT,
        CORPUS_SEED,
        PROBES_PER_CONTAINER,
        WARMUP_RUNS,
        TIMED_RUNS,
    });
    bench_time.print("shapes: 50% [1,64], 35% [256,1024], 15% [3840,4096]\n", .{});
    bench_time.printBenchEnvironment();

    var corpus = try Corpus.init(std.heap.smp_allocator);
    defer corpus.deinit();
    bench_time.print("corpus values: {d}\n", .{corpus.values.len});

    var samples: [variants.len][workloads.len][TIMED_RUNS]Sample = undefined;
    for (0..WARMUP_RUNS + TIMED_RUNS) |round| {
        bench_time.print("round {d}/{d}\n", .{ round + 1, WARMUP_RUNS + TIMED_RUNS });
        for (0..variants.len) |slot| {
            const variant_i = (round + slot) % variants.len;
            const run_samples = try executeVariant(variants[variant_i], &corpus);
            if (round >= WARMUP_RUNS) {
                const timed_i = round - WARMUP_RUNS;
                for (0..workloads.len) |workload_i| {
                    samples[variant_i][workload_i][timed_i] = run_samples[workload_i];
                }
            }
        }
    }

    printResults(&samples);
}

test "prototype layouts preserve values across growth and clone" {
    inline for (.{ Baseline32, Control16, SingleStored, SingleDerived }) |T| {
        var counting = CountingAllocator.init(std.testing.allocator);
        const allocator = counting.allocator();
        var container = try T.init(allocator);
        for (0..MAX_CARDINALITY) |value| {
            const result = try container.add(allocator, @intCast(value));
            try std.testing.expect(result.added);
            container = result.container;
        }
        try std.testing.expect(container.contains(0));
        try std.testing.expect(container.contains(MAX_CARDINALITY - 1));
        try std.testing.expect(!container.contains(MAX_CARDINALITY));

        const clone = try container.clone(allocator);
        try std.testing.expectEqual(container.sumValues(), clone.sumValues());
        clone.deinit(allocator);
        container.deinit(allocator);
        try std.testing.expectEqual(@as(u64, 0), counting.stats.live_bytes);
        try std.testing.expectEqual(@as(u64, 0), counting.stats.remap_calls);
    }
}
