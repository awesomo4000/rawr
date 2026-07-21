// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");
const counting_mod = @import("counting_allocator.zig");

const RoaringBitmap = rawr.RoaringBitmap;
const Container = rawr.Container;
const CountingAllocator = counting_mod.CountingAllocator;

const SPARSE_SEED = 54_321;
const SPARSE_VALUE_COUNT = 500_000;
const NWAY_SEED = 0x17_00_2026;
const NWAY_OPERANDS = 16;
const SPARSE_NWAY_KEYS = 1_024;
const SPARSE_VALUES_PER_KEY = 64;
const DENSE_NWAY_KEYS = 256;
const DENSE_VALUES_PER_KEY = 320;
const DENSE_KEY_START = 2_048;
const ARRAY_LIMIT = 4_096;

const Experiment = enum {
    sparse_2way,
    sparse_nway,
    dense_nway,
};

const Phase = enum {
    construction,
    repair,
    combined,
};

const experiments = [_]Experiment{ .sparse_2way, .sparse_nway, .dense_nway };
const phases = [_]Phase{ .construction, .repair, .combined };

const Classification = struct {
    shared_groups: usize,
    eligible_groups: usize,
    unknown_groups: usize,
};

const Measurement = struct {
    elapsed_ns: u64,
    stats: CountingAllocator.Stats = .{},
    arena_capacity: usize = 0,
};

const SparsePair = struct {
    allocator: std.mem.Allocator,
    a: RoaringBitmap,
    b: RoaringBitmap,
    c_a: *c.roaring_bitmap_t,
    c_b: *c.roaring_bitmap_t,
    fingerprint: u64,

    fn init(allocator: std.mem.Allocator) !SparsePair {
        const values = try allocator.alloc(u32, SPARSE_VALUE_COUNT);
        defer allocator.free(values);

        var prng = std.Random.DefaultPrng.init(SPARSE_SEED);
        for (values) |*value| value.* = prng.random().int(u32);
        std.mem.sort(u32, values, {}, std.sort.asc(u32));

        var unique_len: usize = 1;
        for (values[1..]) |value| {
            if (value == values[unique_len - 1]) continue;
            values[unique_len] = value;
            unique_len += 1;
        }

        const half = unique_len / 2;
        const a_values = values[0..half];
        const b_values = values[half / 2 .. unique_len];

        var a = try RoaringBitmap.init(allocator);
        errdefer a.deinit();
        try a.addMany(a_values);

        var b = try RoaringBitmap.init(allocator);
        errdefer b.deinit();
        try b.addMany(b_values);

        const c_a = c.roaring_bitmap_create() orelse return error.OutOfMemory;
        errdefer c.roaring_bitmap_free(c_a);
        c.roaring_bitmap_add_many(c_a, a_values.len, a_values.ptr);

        const c_b = c.roaring_bitmap_create() orelse return error.OutOfMemory;
        errdefer c.roaring_bitmap_free(c_b);
        c.roaring_bitmap_add_many(c_b, b_values.len, b_values.ptr);

        var hasher = std.hash.Wyhash.init(SPARSE_SEED);
        hasher.update(std.mem.sliceAsBytes(a_values));
        hasher.update(std.mem.sliceAsBytes(b_values));

        return .{
            .allocator = allocator,
            .a = a,
            .b = b,
            .c_a = c_a,
            .c_b = c_b,
            .fingerprint = hasher.final(),
        };
    }

    fn deinit(self: *SparsePair) void {
        c.roaring_bitmap_free(self.c_b);
        c.roaring_bitmap_free(self.c_a);
        self.b.deinit();
        self.a.deinit();
        self.* = undefined;
    }
};

const NwayCorpus = struct {
    allocator: std.mem.Allocator,
    rawr_bitmaps: [NWAY_OPERANDS]?RoaringBitmap = [_]?RoaringBitmap{null} ** NWAY_OPERANDS,
    c_bitmaps: [NWAY_OPERANDS]?*c.roaring_bitmap_t = [_]?*c.roaring_bitmap_t{null} ** NWAY_OPERANDS,
    key_start: u16,
    key_count: usize,
    values_per_key: usize,
    fingerprint: u64,

    fn init(
        allocator: std.mem.Allocator,
        key_start: u16,
        key_count: usize,
        values_per_key: usize,
    ) !NwayCorpus {
        var result = NwayCorpus{
            .allocator = allocator,
            .key_start = key_start,
            .key_count = key_count,
            .values_per_key = values_per_key,
            .fingerprint = 0,
        };
        errdefer result.deinit();

        const value_count = key_count * values_per_key;
        const values = try allocator.alloc(u32, value_count);
        defer allocator.free(values);

        var hasher = std.hash.Wyhash.init(
            NWAY_SEED ^ @as(u64, key_start) ^ @as(u64, @intCast(values_per_key)),
        );
        for (0..NWAY_OPERANDS) |operand| {
            var pos: usize = 0;
            for (0..key_count) |key_offset| {
                const key: u32 = @as(u32, key_start) + @as(u32, @intCast(key_offset));
                const low_start = operand * values_per_key;
                for (0..values_per_key) |value_offset| {
                    const low: u32 = @intCast(low_start + value_offset);
                    values[pos] = (key << 16) | low;
                    pos += 1;
                }
            }
            std.debug.assert(pos == values.len);
            hasher.update(std.mem.sliceAsBytes(values));

            var bitmap = try RoaringBitmap.initCapacity(allocator, @intCast(key_count));
            errdefer bitmap.deinit();
            try bitmap.addMany(values);

            const c_bitmap = c.roaring_bitmap_create() orelse return error.OutOfMemory;
            c.roaring_bitmap_add_many(c_bitmap, values.len, values.ptr);

            result.rawr_bitmaps[operand] = bitmap;
            result.c_bitmaps[operand] = c_bitmap;
        }

        result.fingerprint = hasher.final();
        return result;
    }

    fn deinit(self: *NwayCorpus) void {
        for (&self.c_bitmaps) |*maybe_bitmap| {
            if (maybe_bitmap.*) |bitmap| c.roaring_bitmap_free(bitmap);
            maybe_bitmap.* = null;
        }
        for (&self.rawr_bitmaps) |*maybe_bitmap| {
            if (maybe_bitmap.*) |*bitmap| bitmap.deinit();
            maybe_bitmap.* = null;
        }
    }

    fn rawrInputs(self: *const NwayCorpus) [NWAY_OPERANDS]*const RoaringBitmap {
        var inputs: [NWAY_OPERANDS]*const RoaringBitmap = undefined;
        for (0..NWAY_OPERANDS) |i| inputs[i] = &self.rawr_bitmaps[i].?;
        return inputs;
    }

    fn cInputs(self: *const NwayCorpus) [NWAY_OPERANDS]*const c.roaring_bitmap_t {
        var inputs: [NWAY_OPERANDS]*const c.roaring_bitmap_t = undefined;
        for (0..NWAY_OPERANDS) |i| inputs[i] = self.c_bitmaps[i].?;
        return inputs;
    }
};

const Corpora = struct {
    sparse_pair: SparsePair,
    sparse_nway: NwayCorpus,
    dense_nway: NwayCorpus,

    fn init(allocator: std.mem.Allocator) !Corpora {
        var sparse_pair = try SparsePair.init(allocator);
        errdefer sparse_pair.deinit();
        var sparse_nway = try NwayCorpus.init(
            allocator,
            0,
            SPARSE_NWAY_KEYS,
            SPARSE_VALUES_PER_KEY,
        );
        errdefer sparse_nway.deinit();
        const dense_nway = try NwayCorpus.init(
            allocator,
            DENSE_KEY_START,
            DENSE_NWAY_KEYS,
            DENSE_VALUES_PER_KEY,
        );

        return .{
            .sparse_pair = sparse_pair,
            .sparse_nway = sparse_nway,
            .dense_nway = dense_nway,
        };
    }

    fn deinit(self: *Corpora) void {
        self.dense_nway.deinit();
        self.sparse_nway.deinit();
        self.sparse_pair.deinit();
        self.* = undefined;
    }
};

fn storedCardinality(container: Container) ?u32 {
    return switch (container) {
        .array => |array| array.cardinality,
        .bitset => |bitset| if (bitset.cardinality < 0) null else @intCast(bitset.cardinality),
        .run => |run| run.getCardinality(),
        .reserved => unreachable,
    };
}

fn classifyPair(a: *const RoaringBitmap, b: *const RoaringBitmap) Classification {
    var result = Classification{ .shared_groups = 0, .eligible_groups = 0, .unknown_groups = 0 };
    var i: usize = 0;
    var j: usize = 0;
    while (i < a.size and j < b.size) {
        if (a.keys[i] < b.keys[j]) {
            i += 1;
            continue;
        }
        if (a.keys[i] > b.keys[j]) {
            j += 1;
            continue;
        }

        result.shared_groups += 1;
        const a_card = storedCardinality(Container.fromTagged(a.containers[i]));
        const b_card = storedCardinality(Container.fromTagged(b.containers[j]));
        if (a_card == null or b_card == null) {
            result.unknown_groups += 1;
        } else if (@as(u64, a_card.?) + b_card.? <= ARRAY_LIMIT) {
            result.eligible_groups += 1;
        }
        i += 1;
        j += 1;
    }
    return result;
}

fn classifyNway(inputs: [NWAY_OPERANDS]*const RoaringBitmap) Classification {
    var cursors: [NWAY_OPERANDS]usize = @splat(0);
    var result = Classification{ .shared_groups = 0, .eligible_groups = 0, .unknown_groups = 0 };

    while (true) {
        var min_key: ?u16 = null;
        for (inputs, cursors) |bitmap, cursor| {
            if (cursor >= bitmap.size) continue;
            const key = bitmap.keys[cursor];
            if (min_key == null or key < min_key.?) min_key = key;
        }
        const key = min_key orelse break;

        var contributors: usize = 0;
        var sum: u64 = 0;
        var known = true;
        for (inputs, &cursors) |bitmap, *cursor| {
            if (cursor.* >= bitmap.size or bitmap.keys[cursor.*] != key) continue;
            contributors += 1;
            if (storedCardinality(Container.fromTagged(bitmap.containers[cursor.*]))) |cardinality| {
                sum +|= cardinality;
            } else {
                known = false;
            }
            cursor.* += 1;
        }

        if (contributors < 2) continue;
        result.shared_groups += 1;
        if (!known) {
            result.unknown_groups += 1;
        } else if (sum <= ARRAY_LIMIT) {
            result.eligible_groups += 1;
        }
    }
    return result;
}

fn warmupPlaceholder(phase: Phase) !void {
    var result = try RoaringBitmap.init(std.heap.smp_allocator);
    defer result.deinit();
    if (phase != .construction) try result.repairAfterLazy();
    std.mem.doNotOptimizeAway(&result);
}

fn measurePlaceholder(phase: Phase) !Measurement {
    try warmupPlaceholder(phase);

    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const allocator = counting.allocator();

    switch (phase) {
        .construction => {
            counting.resetStats();
            const start = bench_time.monotonicNanos();
            var result = try RoaringBitmap.init(allocator);
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(&result);
            const stats = counting.snapshot();
            result.deinit();
            std.debug.assert(counting.stats.live_bytes == 0);
            return .{ .elapsed_ns = elapsed, .stats = stats };
        },
        .repair => {
            var result = try RoaringBitmap.init(allocator);
            defer result.deinit();
            counting.resetStats();
            const start = bench_time.monotonicNanos();
            try result.repairAfterLazy();
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(&result);
            return .{ .elapsed_ns = elapsed, .stats = counting.snapshot() };
        },
        .combined => {
            counting.resetStats();
            const start = bench_time.monotonicNanos();
            var result = try RoaringBitmap.init(allocator);
            try result.repairAfterLazy();
            std.mem.doNotOptimizeAway(&result);
            result.deinit();
            const elapsed = bench_time.monotonicNanos() - start;
            std.debug.assert(counting.stats.live_bytes == 0);
            return .{ .elapsed_ns = elapsed, .stats = counting.snapshot() };
        },
    }
}

fn measureCRoaringPlaceholder(phase: Phase) Measurement {
    const start = switch (phase) {
        .construction, .combined => bench_time.monotonicNanos(),
        .repair => 0,
    };
    const bitmap = c.roaring_bitmap_create() orelse unreachable;

    if (phase == .construction) {
        const elapsed = bench_time.monotonicNanos() - start;
        std.mem.doNotOptimizeAway(bitmap);
        c.roaring_bitmap_free(bitmap);
        return .{ .elapsed_ns = elapsed };
    }

    const repair_start = if (phase == .repair) bench_time.monotonicNanos() else start;
    c.roaring_bitmap_repair_after_lazy(bitmap);
    std.mem.doNotOptimizeAway(bitmap);
    c.roaring_bitmap_free(bitmap);
    return .{ .elapsed_ns = bench_time.monotonicNanos() - repair_start };
}

fn experimentName(experiment: Experiment) []const u8 {
    return switch (experiment) {
        .sparse_2way => "sparse-2way",
        .sparse_nway => "sparse-nway",
        .dense_nway => "dense-nway",
    };
}

fn phaseName(phase: Phase) []const u8 {
    return @tagName(phase);
}

fn printMeasurement(
    experiment: Experiment,
    variant: []const u8,
    phase: Phase,
    measurement: Measurement,
    classification: Classification,
) void {
    const stats = measurement.stats;
    bench_time.print(
        "{s:<13} {s:<11} {s:<12} {d:>10} ns  alloc={d} free={d} requested={d} class={d} peak-class={d}\n",
        .{
            experimentName(experiment),
            variant,
            phaseName(phase),
            measurement.elapsed_ns,
            stats.alloc_calls,
            stats.free_calls,
            stats.cumulative_bytes,
            stats.cumulative_class_bytes,
            stats.peak_live_class_bytes,
        },
    );
    bench_time.print(
        "RESULT\t{s}\t{s}\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n",
        .{
            experimentName(experiment),
            variant,
            phaseName(phase),
            measurement.elapsed_ns,
            stats.alloc_calls,
            stats.free_calls,
            stats.resize_calls,
            stats.remap_calls,
            stats.cumulative_bytes,
            stats.cumulative_class_bytes,
            stats.live_class_bytes,
            stats.peak_live_class_bytes,
            measurement.arena_capacity,
            classification.eligible_groups,
            classification.shared_groups,
            classification.unknown_groups,
        },
    );
}

fn validateCountingAllocator() !void {
    var storage: [256]u8 align(64) = undefined;
    var fixed = std.heap.FixedBufferAllocator.init(&storage);
    var counting = CountingAllocator.init(fixed.allocator());
    const allocator = counting.allocator();

    const one = try allocator.alloc(u8, 1);
    const aligned = try allocator.alignedAlloc(u8, .@"16", 17);
    const stats = counting.snapshot();
    if (stats.alloc_calls != 2 or
        stats.cumulative_bytes != 18 or
        stats.cumulative_class_bytes != 40 or
        stats.peak_live_class_bytes != 40)
    {
        return error.CountingAllocatorMismatch;
    }
    allocator.free(aligned);
    allocator.free(one);
    if (counting.stats.live_bytes != 0 or counting.stats.live_class_bytes != 0) {
        return error.CountingAllocatorMismatch;
    }
}

pub fn expectByteIdentical(
    allocator: std.mem.Allocator,
    actual: *const RoaringBitmap,
    expected: *const RoaringBitmap,
) !void {
    const actual_bytes = try actual.serialize(allocator);
    defer allocator.free(actual_bytes);
    const expected_bytes = try expected.serialize(allocator);
    defer allocator.free(expected_bytes);
    if (!std.mem.eql(u8, actual_bytes, expected_bytes)) return error.RawrByteMismatch;
}

pub fn expectCRoaringLogicalEqual(
    allocator: std.mem.Allocator,
    actual: *const RoaringBitmap,
    expected: *const c.roaring_bitmap_t,
) !void {
    const cardinality = actual.cardinality();
    if (cardinality != c.roaring_bitmap_get_cardinality(expected)) return error.CardinalityMismatch;

    const actual_values = try actual.toArrayAlloc(allocator);
    defer allocator.free(actual_values);
    const expected_values = try allocator.alloc(u32, @intCast(cardinality));
    defer allocator.free(expected_values);
    c.roaring_bitmap_to_uint32_array(expected, expected_values.ptr);
    if (!std.mem.eql(u32, actual_values, expected_values)) return error.ValueMismatch;
}

pub fn main() !void {
    try validateCountingAllocator();

    bench_time.print("Transient-bitset arena Phase A harness\n", .{});
    bench_time.print("========================================\n", .{});
    bench_time.printBenchEnvironment();
    bench_time.print(
        "seeds: sparse={d}, nway=0x{x}; operands={d}\n",
        .{ SPARSE_SEED, NWAY_SEED, NWAY_OPERANDS },
    );
    bench_time.print(
        "sparse-nway: keys={d}, per-input/key={d}, summed-bound={d}\n",
        .{ SPARSE_NWAY_KEYS, SPARSE_VALUES_PER_KEY, NWAY_OPERANDS * SPARSE_VALUES_PER_KEY },
    );
    bench_time.print(
        "dense-nway: keys={d}, per-input/key={d}, summed-bound={d}\n",
        .{ DENSE_NWAY_KEYS, DENSE_VALUES_PER_KEY, NWAY_OPERANDS * DENSE_VALUES_PER_KEY },
    );
    bench_time.print("Initializing deterministic corpora...\n", .{});

    var corpora = try Corpora.init(std.heap.smp_allocator);
    defer corpora.deinit();

    const sparse_nway_inputs = corpora.sparse_nway.rawrInputs();
    const dense_nway_inputs = corpora.dense_nway.rawrInputs();
    const classifications = [_]Classification{
        classifyPair(&corpora.sparse_pair.a, &corpora.sparse_pair.b),
        classifyNway(sparse_nway_inputs),
        classifyNway(dense_nway_inputs),
    };

    bench_time.print("fingerprint sparse-2way=0x{x}\n", .{corpora.sparse_pair.fingerprint});
    bench_time.print("fingerprint sparse-nway=0x{x}\n", .{corpora.sparse_nway.fingerprint});
    bench_time.print("fingerprint dense-nway=0x{x}\n", .{corpora.dense_nway.fingerprint});
    for (experiments, classifications) |experiment, classification| {
        bench_time.print(
            "eligibility {s}: eligible={d} shared={d} unknown={d}\n",
            .{
                experimentName(experiment),
                classification.eligible_groups,
                classification.shared_groups,
                classification.unknown_groups,
            },
        );
    }

    bench_time.print("\nPlaceholder pipeline (17-01/17-02 register real variants)\n", .{});
    for (experiments, classifications) |experiment, classification| {
        for (phases) |phase| {
            printMeasurement(
                experiment,
                "placeholder",
                phase,
                try measurePlaceholder(phase),
                classification,
            );
            printMeasurement(
                experiment,
                "croaring",
                phase,
                measureCRoaringPlaceholder(phase),
                classification,
            );
        }
    }
}
