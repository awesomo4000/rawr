// SPDX-License-Identifier: MPL-2.0

//! Spec 38-00 diagnosis worker. Address sorting is implemented only in this
//! executable; no production bitmap path or public API is changed.

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");
const CountingAllocator = @import("counting_allocator.zig").CountingAllocator;

const RoaringBitmap = rawr.RoaringBitmap;
const BitsetContainer = rawr.BitsetContainer;
const RunContainer = rawr.RunContainer;
const Container = rawr.Container;

const max_containers = 16_364;
const values_per_input_container = 4;
const warmup_runs = 3;
const timed_runs = 21;
const noise_count = 16_384;

const Phase = enum { repair, teardown, teardown_control };
const AllocatorKind = enum { smp, libc, croaring };
const Strategy = enum { unsorted, header_asc, payload_asc, payload_desc };
const ScratchMode = enum { none, cold, reused };
const NoiseMode = enum { none, shared };
const Lifecycle = enum { steady, first_cycle };

const Config = struct {
    phase: Phase,
    allocator: AllocatorKind,
    strategy: Strategy,
    scratch: ScratchMode,
    noise: NoiseMode,
    lifecycle: Lifecycle,
    count: usize,
};

const RepairTiming = struct {
    total_ns: u64,
    cardinality_ns: u64,
};

const TeardownTiming = struct {
    teardown_ns: u64,
    refill_ns: u64,
    traversal_ns: u64,
    combined_ns: u64,
    teardown_and_combined_ns: u64,
    noise_ns: u64,
};

const RepairStats = struct {
    total: MetricStats,
    cardinality: MetricStats,
};

const TeardownStats = struct {
    teardown: MetricStats,
    refill: MetricStats,
    traversal: MetricStats,
    combined: MetricStats,
    teardown_and_combined: MetricStats,
    noise: MetricStats,
};

const MetricStats = struct {
    median: u64,
    minimum: u64,
    maximum: u64,
};

const NoiseKind = enum { primary, secondary_64, secondary_512, secondary_4096 };
const NoiseDescriptor = struct { kind: NoiseKind };
const NoiseAllocation = struct {
    ptr: [*]u8,
    len: usize,
    kind: NoiseKind,
    live: bool,
};

var left_values: [max_containers * values_per_input_container]u32 = undefined;
var right_values: [max_containers * values_per_input_container]u32 = undefined;
var mass_bitsets: [max_containers]*BitsetContainer = undefined;
var run_containers: [8]*RunContainer = undefined;
var noise_descriptors: [noise_count]NoiseDescriptor = undefined;
var noise_allocations: [noise_count]NoiseAllocation = undefined;

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var header = false;
    var phase: ?Phase = null;
    var allocator: ?AllocatorKind = null;
    var strategy: ?Strategy = null;
    var scratch: ?ScratchMode = null;
    var noise: ?NoiseMode = null;
    var lifecycle: ?Lifecycle = null;
    var count: ?usize = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.startsWith(u8, arg, "--phase=")) {
            phase = std.meta.stringToEnum(Phase, arg[8..]) orelse return error.UnknownPhase;
        } else if (std.mem.startsWith(u8, arg, "--allocator=")) {
            allocator = std.meta.stringToEnum(AllocatorKind, arg[12..]) orelse return error.UnknownAllocator;
        } else if (std.mem.startsWith(u8, arg, "--strategy=")) {
            strategy = std.meta.stringToEnum(Strategy, arg[11..]) orelse return error.UnknownStrategy;
        } else if (std.mem.startsWith(u8, arg, "--scratch=")) {
            scratch = std.meta.stringToEnum(ScratchMode, arg[10..]) orelse return error.UnknownScratchMode;
        } else if (std.mem.startsWith(u8, arg, "--noise=")) {
            noise = std.meta.stringToEnum(NoiseMode, arg[8..]) orelse return error.UnknownNoiseMode;
        } else if (std.mem.startsWith(u8, arg, "--lifecycle=")) {
            lifecycle = std.meta.stringToEnum(Lifecycle, arg[12..]) orelse return error.UnknownLifecycle;
        } else if (std.mem.startsWith(u8, arg, "--count=")) {
            count = try std.fmt.parseInt(usize, arg[8..], 10);
        } else {
            return error.UnknownArgument;
        }
    }

    if (header) {
        if (phase != null or allocator != null or strategy != null or scratch != null or noise != null or lifecycle != null or count != null) {
            return error.ConflictingArguments;
        }
        printHeader();
        return;
    }

    const config = Config{
        .phase = phase orelse return error.MissingPhase,
        .allocator = allocator orelse return error.MissingAllocator,
        .strategy = strategy orelse .unsorted,
        .scratch = scratch orelse .none,
        .noise = noise orelse .none,
        .lifecycle = lifecycle orelse .steady,
        .count = count orelse max_containers,
    };
    try validateConfig(config);
    prepareNoiseDescriptors();

    switch (config.phase) {
        .repair => try runRepair(config),
        .teardown => try runTeardown(config),
        .teardown_control => try runTeardownControl(config),
    }
}

fn printHeader() void {
    bench_time.printBenchEnvironment();
    bench_time.print("# address-sorted diagnosis: production code unchanged\n", .{});
    bench_time.print("# steady protocol: {d}w/{d}t median; first-cycle: one sample per fresh process\n", .{ warmup_runs, timed_runs });
    bench_time.print("# requested-cpu: native\n", .{});
    bench_time.print("# croaring-avx512: {s}\n", .{if (c.CROARING_COMPILER_SUPPORTS_AVX512 != 0) "on" else "off"});
}

fn validateConfig(config: Config) !void {
    if (config.count == 0 or config.count > max_containers) return error.InvalidContainerCount;
    switch (config.phase) {
        .repair => {
            if (config.noise != .none or config.lifecycle != .steady) return error.UnsupportedRepairConfiguration;
            if (config.allocator == .croaring) {
                if (config.strategy != .unsorted or config.scratch != .none) return error.UnsupportedCRoaringConfiguration;
            } else if (config.strategy == .unsorted) {
                if (config.scratch != .none) return error.UnsupportedScratchMode;
            } else if (config.scratch == .none) {
                return error.MissingScratchMode;
            }
        },
        .teardown => {
            if (config.scratch != .none) return error.UnsupportedScratchMode;
            if (config.allocator == .croaring and config.strategy != .unsorted) return error.UnsupportedCRoaringConfiguration;
        },
        .teardown_control => {
            if (config.count != 8 or config.allocator == .croaring or config.noise != .none or
                config.lifecycle != .steady or config.scratch != .none)
            {
                return error.UnsupportedControlConfiguration;
            }
        },
    }
}

fn allocatorFor(kind: AllocatorKind) std.mem.Allocator {
    return switch (kind) {
        .smp => std.heap.smp_allocator,
        .libc, .croaring => bench_time.cAllocator(),
    };
}

fn initRepairValues(count: usize) void {
    for (0..count) |key| {
        const base = @as(u32, @intCast(key)) << 16;
        const offset = key * values_per_input_container;
        left_values[offset + 0] = base + 0;
        left_values[offset + 1] = base + 2;
        left_values[offset + 2] = base + 4;
        left_values[offset + 3] = base + 6;
        right_values[offset + 0] = base + 1;
        right_values[offset + 1] = base + 3;
        right_values[offset + 2] = base + 5;
        right_values[offset + 3] = base + 7;
    }
}

fn buildRawrRepairInputs(count: usize) !struct { RoaringBitmap, RoaringBitmap } {
    initRepairValues(count);
    var left = try RoaringBitmap.init(std.heap.smp_allocator);
    errdefer left.deinit();
    var right = try RoaringBitmap.init(std.heap.smp_allocator);
    errdefer right.deinit();
    try left.addMany(left_values[0 .. count * values_per_input_container]);
    try right.addMany(right_values[0 .. count * values_per_input_container]);
    return .{ left, right };
}

fn buildCRRepairInputs(count: usize) !struct { *c.roaring_bitmap_t, *c.roaring_bitmap_t } {
    initRepairValues(count);
    const left = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    errdefer c.roaring_bitmap_free(left);
    const right = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    errdefer c.roaring_bitmap_free(right);
    const len = count * values_per_input_container;
    c.roaring_bitmap_add_many(left, len, &left_values);
    c.roaring_bitmap_add_many(right, len, &right_values);
    return .{ left, right };
}

fn runRepair(config: Config) !void {
    const stats = if (config.allocator == .croaring) blk: {
        const inputs = try buildCRRepairInputs(config.count);
        defer c.roaring_bitmap_free(inputs[0]);
        defer c.roaring_bitmap_free(inputs[1]);
        const measured = try measureCRRepair(inputs[0], inputs[1]);
        try validateRepair(config.count);
        break :blk measured;
    } else blk: {
        var inputs = try buildRawrRepairInputs(config.count);
        defer inputs[0].deinit();
        defer inputs[1].deinit();
        const measured = try measureRawrRepair(config, &inputs[0], &inputs[1]);
        try validateRepair(config.count);
        break :blk measured;
    };

    bench_time.print("RESULT\trepair\t{s}\t{s}\t{s}\tnone\tsteady\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        @tagName(config.allocator),
        @tagName(config.strategy),
        @tagName(config.scratch),
        config.count,
        stats.total.median,
        stats.total.minimum,
        stats.total.maximum,
        stats.cardinality.median,
        stats.cardinality.minimum,
        stats.cardinality.maximum,
        c.rawr_cr_sorted_peak_rss_bytes(),
    });
    bench_time.print("VALIDATION\trepair\tportable-bytes-identical\n", .{});
}

fn measureRawrRepair(config: Config, left: *const RoaringBitmap, right: *const RoaringBitmap) !RepairStats {
    const allocator = allocatorFor(config.allocator);
    var reusable: ?[]*BitsetContainer = null;
    if (config.scratch == .reused) reusable = try allocator.alloc(*BitsetContainer, config.count);
    defer if (reusable) |storage| allocator.free(storage);

    for (0..warmup_runs) |_| _ = try timeRawrRepair(left, right, allocator, config.strategy, config.scratch, reusable);
    var total: [timed_runs]u64 = undefined;
    var cardinality: [timed_runs]u64 = undefined;
    for (&total, &cardinality) |*total_sample, *cardinality_sample| {
        const sample = try timeRawrRepair(left, right, allocator, config.strategy, config.scratch, reusable);
        total_sample.* = sample.total_ns;
        cardinality_sample.* = sample.cardinality_ns;
    }
    return .{ .total = metricStats(&total), .cardinality = metricStats(&cardinality) };
}

fn timeRawrRepair(
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
    allocator: std.mem.Allocator,
    strategy: Strategy,
    scratch_mode: ScratchMode,
    reusable: ?[]*BitsetContainer,
) !RepairTiming {
    var result = try left.lazyOr(allocator, right, true);
    errdefer result.deinit();

    if (strategy == .unsorted) {
        const start = bench_time.monotonicNanos();
        try result.repairAfterLazy();
        const elapsed = bench_time.monotonicNanos() - start;
        std.mem.doNotOptimizeAway(&result);
        result.deinit();
        return .{ .total_ns = elapsed, .cardinality_ns = elapsed };
    }

    const total_start = bench_time.monotonicNanos();
    var owned: ?[]*BitsetContainer = null;
    const scratch = switch (scratch_mode) {
        .cold => blk: {
            owned = try allocator.alloc(*BitsetContainer, result.size);
            break :blk owned.?;
        },
        .reused => reusable.?[0..result.size],
        .none => unreachable,
    };
    defer if (owned) |storage| allocator.free(storage);

    var bitset_count: usize = 0;
    for (result.containers[0..result.size]) |tagged| {
        if (tagged.getType() != .bitset) continue;
        scratch[bitset_count] = tagged.getBitset();
        bitset_count += 1;
    }
    std.mem.sort(*BitsetContainer, scratch[0..bitset_count], strategy, lessThanBitset);
    for (scratch[0..bitset_count]) |bitset| _ = bitset.computeCardinality();
    const cardinality_elapsed = bench_time.monotonicNanos() - total_start;

    // Cardinality is now cached. Production repair performs the key-ordered
    // conversion/free pass, preserving its existing free order exactly.
    try result.repairAfterLazy();
    if (owned) |storage| {
        allocator.free(storage);
        owned = null;
    }
    const total_elapsed = bench_time.monotonicNanos() - total_start;
    std.mem.doNotOptimizeAway(&result);
    result.deinit();
    return .{ .total_ns = total_elapsed, .cardinality_ns = cardinality_elapsed };
}

fn lessThanBitset(strategy: Strategy, left: *BitsetContainer, right: *BitsetContainer) bool {
    return switch (strategy) {
        .header_asc => @intFromPtr(left) < @intFromPtr(right),
        .payload_asc => @intFromPtr(left.words) < @intFromPtr(right.words),
        .payload_desc => @intFromPtr(left.words) > @intFromPtr(right.words),
        .unsorted => unreachable,
    };
}

fn measureCRRepair(left: *const c.roaring_bitmap_t, right: *const c.roaring_bitmap_t) !RepairStats {
    for (0..warmup_runs) |_| _ = try timeCRRepair(left, right);
    var total: [timed_runs]u64 = undefined;
    for (&total) |*sample| sample.* = try timeCRRepair(left, right);
    const stats = metricStats(&total);
    return .{ .total = stats, .cardinality = stats };
}

fn timeCRRepair(left: *const c.roaring_bitmap_t, right: *const c.roaring_bitmap_t) !u64 {
    const result = c.roaring_bitmap_lazy_or(left, right, true) orelse return error.OutOfMemory;
    const start = bench_time.monotonicNanos();
    c.roaring_bitmap_repair_after_lazy(result);
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(result);
    c.roaring_bitmap_free(result);
    return elapsed;
}

fn validateRepair(count: usize) !void {
    var rawr_inputs = try buildRawrRepairInputs(count);
    defer rawr_inputs[0].deinit();
    defer rawr_inputs[1].deinit();
    const cr_inputs = try buildCRRepairInputs(count);
    defer c.roaring_bitmap_free(cr_inputs[0]);
    defer c.roaring_bitmap_free(cr_inputs[1]);

    var baseline = try rawr_inputs[0].lazyOr(std.heap.smp_allocator, &rawr_inputs[1], true);
    defer baseline.deinit();
    try baseline.repairAfterLazy();

    var payload = try rawr_inputs[0].lazyOr(std.heap.smp_allocator, &rawr_inputs[1], true);
    defer payload.deinit();
    const scratch = try std.heap.smp_allocator.alloc(*BitsetContainer, count);
    defer std.heap.smp_allocator.free(scratch);
    _ = try repairForValidation(&payload, .payload_asc, scratch);
    if (!baseline.equals(&payload)) return error.SortedRepairMismatch;

    var header = try rawr_inputs[0].lazyOr(std.heap.smp_allocator, &rawr_inputs[1], true);
    defer header.deinit();
    _ = try repairForValidation(&header, .header_asc, scratch);
    if (!baseline.equals(&header)) return error.HeaderRepairMismatch;

    const cr_result = c.roaring_bitmap_lazy_or(cr_inputs[0], cr_inputs[1], true) orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(cr_result);
    c.roaring_bitmap_repair_after_lazy(cr_result);

    const rawr_bytes = try baseline.serialize(std.heap.page_allocator);
    defer std.heap.page_allocator.free(rawr_bytes);
    const cr_size = c.roaring_bitmap_portable_size_in_bytes(cr_result);
    if (rawr_bytes.len != cr_size) return error.SerializedSizeMismatch;
    const cr_bytes = try std.heap.page_allocator.alloc(u8, cr_size);
    defer std.heap.page_allocator.free(cr_bytes);
    if (c.roaring_bitmap_portable_serialize(cr_result, @ptrCast(cr_bytes.ptr)) != cr_size or
        !std.mem.eql(u8, rawr_bytes, cr_bytes))
    {
        return error.CRoaringMismatch;
    }
}

fn repairForValidation(bitmap: *RoaringBitmap, strategy: Strategy, scratch: []*BitsetContainer) !RepairTiming {
    var count: usize = 0;
    for (bitmap.containers[0..bitmap.size]) |tagged| {
        if (tagged.getType() == .bitset) {
            scratch[count] = tagged.getBitset();
            count += 1;
        }
    }
    std.mem.sort(*BitsetContainer, scratch[0..count], strategy, lessThanBitset);
    for (scratch[0..count]) |bitset| _ = bitset.computeCardinality();
    try bitmap.repairAfterLazy();
    return .{ .total_ns = 0, .cardinality_ns = 0 };
}

fn runTeardown(config: Config) !void {
    const stats = if (config.allocator == .croaring)
        try measureCRTeardown(config)
    else
        try measureRawrTeardown(config);
    try validateRawrTeardown();

    bench_time.print("RESULT\tteardown\t{s}\t{s}\tnone\t{s}\t{s}\t{d}", .{
        @tagName(config.allocator),
        @tagName(config.strategy),
        @tagName(config.noise),
        @tagName(config.lifecycle),
        config.count,
    });
    bench_time.print("\t{d}\t{d}\t{d}", .{
        stats.teardown.median,
        stats.teardown.minimum,
        stats.teardown.maximum,
    });
    bench_time.print("\t{d}\t{d}\t{d}", .{
        stats.refill.median,
        stats.refill.minimum,
        stats.refill.maximum,
    });
    bench_time.print("\t{d}\t{d}\t{d}", .{
        stats.traversal.median,
        stats.traversal.minimum,
        stats.traversal.maximum,
    });
    bench_time.print("\t{d}\t{d}\t{d}", .{
        stats.combined.median,
        stats.combined.minimum,
        stats.combined.maximum,
    });
    bench_time.print("\t{d}\t{d}\t{d}", .{
        stats.teardown_and_combined.median,
        stats.teardown_and_combined.minimum,
        stats.teardown_and_combined.maximum,
    });
    bench_time.print("\t{d}\t{d}\t{d}\t{d}\n", .{
        stats.noise.median,
        stats.noise.minimum,
        stats.noise.maximum,
        c.rawr_cr_sorted_peak_rss_bytes(),
    });
    bench_time.print("VALIDATION\tteardown\tall-containers-freed\n", .{});
}

fn measureRawrTeardown(config: Config) !TeardownStats {
    const allocator = allocatorFor(config.allocator);
    if (config.lifecycle == .first_cycle) {
        const sample = try timeRawrTeardown(allocator, config.strategy, config.noise, config.count);
        return teardownSingleStats(sample);
    }
    for (0..warmup_runs) |_| _ = try timeRawrTeardown(allocator, config.strategy, config.noise, config.count);
    var samples: [timed_runs]TeardownTiming = undefined;
    for (&samples) |*sample| sample.* = try timeRawrTeardown(allocator, config.strategy, config.noise, config.count);
    return teardownStats(&samples);
}

fn timeRawrTeardown(allocator: std.mem.Allocator, strategy: Strategy, noise: NoiseMode, count: usize) !TeardownTiming {
    try allocateRawrMass(allocator, count);
    // Teardown starts from constructed bitsets, whose payload pages are already
    // resident. This setup remains outside the timed free path.
    zeroRawrMass(count);
    const teardown_start = bench_time.monotonicNanos();
    freeRawrMass(allocator, strategy, count) catch |err| {
        freeRawrMassUnsorted(allocator, count);
        return err;
    };
    const teardown_ns = bench_time.monotonicNanos() - teardown_start;

    const noise_ns = if (noise == .shared) try applyNoise(allocator) else 0;
    const combined_start = bench_time.monotonicNanos();
    const refill_start = combined_start;
    try allocateRawrMass(allocator, count);
    const refill_ns = bench_time.monotonicNanos() - refill_start;
    const traversal_start = bench_time.monotonicNanos();
    zeroRawrMass(count);
    const traversal_ns = bench_time.monotonicNanos() - traversal_start;
    const combined_ns = bench_time.monotonicNanos() - combined_start;
    freeRawrMassUnsorted(allocator, count);
    if (noise == .shared) cleanupNoise(allocator);
    return .{
        .teardown_ns = teardown_ns,
        .refill_ns = refill_ns,
        .traversal_ns = traversal_ns,
        .combined_ns = combined_ns,
        .teardown_and_combined_ns = teardown_ns + combined_ns,
        .noise_ns = noise_ns,
    };
}

fn allocateRawrMass(allocator: std.mem.Allocator, count: usize) !void {
    var created: usize = 0;
    errdefer freeRawrMassUnsorted(allocator, created);
    while (created < count) : (created += 1) {
        const header = try allocator.create(BitsetContainer);
        errdefer allocator.destroy(header);
        const words = try allocator.alignedAlloc(u64, .@"64", BitsetContainer.NUM_WORDS);
        header.* = .{ .words = words[0..BitsetContainer.NUM_WORDS], .cardinality = 0 };
        mass_bitsets[created] = header;
    }
}

fn zeroRawrMass(count: usize) void {
    for (mass_bitsets[0..count]) |bitset| @memset(bitset.words, 0);
    std.mem.doNotOptimizeAway(mass_bitsets[0..count]);
}

fn freeRawrMass(allocator: std.mem.Allocator, strategy: Strategy, count: usize) !void {
    if (strategy == .unsorted) {
        freeRawrMassUnsorted(allocator, count);
        return;
    }
    const scratch = try allocator.alloc(*BitsetContainer, count);
    @memcpy(scratch, mass_bitsets[0..count]);
    std.mem.sort(*BitsetContainer, scratch, strategy, lessThanBitset);
    for (scratch) |bitset| bitset.deinit(allocator);
    allocator.free(scratch);
}

fn freeRawrMassUnsorted(allocator: std.mem.Allocator, count: usize) void {
    for (mass_bitsets[0..count]) |bitset| bitset.deinit(allocator);
}

fn measureCRTeardown(config: Config) !TeardownStats {
    if (config.lifecycle == .first_cycle) return teardownSingleStats(try timeCRTeardown(config.noise, config.count));
    for (0..warmup_runs) |_| _ = try timeCRTeardown(config.noise, config.count);
    var samples: [timed_runs]TeardownTiming = undefined;
    for (&samples) |*sample| sample.* = try timeCRTeardown(config.noise, config.count);
    return teardownStats(&samples);
}

fn timeCRTeardown(noise: NoiseMode, count: usize) !TeardownTiming {
    if (!c.rawr_cr_sorted_mass_alloc(count)) return error.OutOfMemory;
    c.rawr_cr_sorted_mass_zero(count);
    const teardown_start = bench_time.monotonicNanos();
    c.rawr_cr_sorted_mass_free(count);
    const teardown_ns = bench_time.monotonicNanos() - teardown_start;

    const allocator = bench_time.cAllocator();
    const noise_ns = if (noise == .shared) try applyNoise(allocator) else 0;
    const combined_start = bench_time.monotonicNanos();
    const refill_start = combined_start;
    if (!c.rawr_cr_sorted_mass_alloc(count)) return error.OutOfMemory;
    const refill_ns = bench_time.monotonicNanos() - refill_start;
    const traversal_start = bench_time.monotonicNanos();
    c.rawr_cr_sorted_mass_zero(count);
    const traversal_ns = bench_time.monotonicNanos() - traversal_start;
    const combined_ns = bench_time.monotonicNanos() - combined_start;
    c.rawr_cr_sorted_mass_free(count);
    if (noise == .shared) cleanupNoise(allocator);
    return .{
        .teardown_ns = teardown_ns,
        .refill_ns = refill_ns,
        .traversal_ns = traversal_ns,
        .combined_ns = combined_ns,
        .teardown_and_combined_ns = teardown_ns + combined_ns,
        .noise_ns = noise_ns,
    };
}

fn prepareNoiseDescriptors() void {
    for (0..noise_count / 2) |index| {
        noise_descriptors[index] = .{ .kind = .primary };
        noise_descriptors[noise_count / 2 + index] = .{ .kind = switch (index % 3) {
            0 => .secondary_64,
            1 => .secondary_512,
            else => .secondary_4096,
        } };
    }
    var prng = std.Random.DefaultPrng.init(0xA110C);
    prng.random().shuffle(NoiseDescriptor, &noise_descriptors);
}

fn applyNoise(allocator: std.mem.Allocator) !u64 {
    const start = bench_time.monotonicNanos();
    var allocated: usize = 0;
    errdefer {
        for (noise_allocations[0..allocated]) |entry| if (entry.live) freeNoiseAllocation(allocator, entry);
    }
    for (&noise_descriptors, 0..) |descriptor, index| {
        const len: usize = switch (descriptor.kind) {
            .primary => 8192,
            .secondary_64 => 64,
            .secondary_512 => 512,
            .secondary_4096 => 4096,
        };
        const bytes = if (descriptor.kind == .primary)
            try allocator.alignedAlloc(u8, .@"64", len)
        else
            try allocator.alloc(u8, len);
        @memset(bytes[0..64], 0xA5);
        noise_allocations[index] = .{ .ptr = bytes.ptr, .len = len, .kind = descriptor.kind, .live = true };
        allocated += 1;
    }
    for (&noise_allocations, 0..) |*entry, index| {
        if (index % 2 != 0) continue;
        freeNoiseAllocation(allocator, entry.*);
        entry.live = false;
    }
    return bench_time.monotonicNanos() - start;
}

fn cleanupNoise(allocator: std.mem.Allocator) void {
    for (&noise_allocations) |*entry| {
        if (!entry.live) continue;
        freeNoiseAllocation(allocator, entry.*);
        entry.live = false;
    }
}

fn freeNoiseAllocation(allocator: std.mem.Allocator, entry: NoiseAllocation) void {
    if (entry.kind == .primary) {
        const aligned: [*]align(64) u8 = @alignCast(entry.ptr);
        allocator.free(aligned[0..entry.len]);
    } else {
        allocator.free(entry.ptr[0..entry.len]);
    }
}

fn runTeardownControl(config: Config) !void {
    const allocator = allocatorFor(config.allocator);
    for (0..warmup_runs) |_| _ = try timeRunControl(allocator, config.strategy);
    var samples: [timed_runs]u64 = undefined;
    for (&samples) |*sample| sample.* = try timeRunControl(allocator, config.strategy);
    const stats = metricStats(&samples);
    try validateRunControl();
    bench_time.print("RESULT\tteardown-control\t{s}\t{s}\tnone\tnone\tsteady\t8\t{d}\t{d}\t{d}\t{d}\n", .{
        @tagName(config.allocator),
        @tagName(config.strategy),
        stats.median,
        stats.minimum,
        stats.maximum,
        c.rawr_cr_sorted_peak_rss_bytes(),
    });
    bench_time.print("VALIDATION\tteardown-control\teight-run-containers-freed\n", .{});
}

fn validateRawrTeardown() !void {
    inline for (.{ Strategy.unsorted, Strategy.header_asc, Strategy.payload_asc, Strategy.payload_desc }) |strategy| {
        var counting = CountingAllocator.init(std.heap.page_allocator);
        const allocator = counting.allocator();
        try allocateRawrMass(allocator, 64);
        try freeRawrMass(allocator, strategy, 64);
        const stats = counting.snapshot();
        if (stats.live_bytes != 0 or stats.alloc_calls != stats.free_calls) return error.TeardownLeak;
    }
}

fn validateRunControl() !void {
    inline for (.{ Strategy.unsorted, Strategy.header_asc, Strategy.payload_asc, Strategy.payload_desc }) |strategy| {
        var counting = CountingAllocator.init(std.heap.page_allocator);
        const allocator = counting.allocator();
        _ = try timeRunControl(allocator, strategy);
        const stats = counting.snapshot();
        if (stats.live_bytes != 0 or stats.alloc_calls != stats.free_calls) return error.RunControlLeak;
    }
}

fn timeRunControl(allocator: std.mem.Allocator, strategy: Strategy) !u64 {
    for (&run_containers) |*slot| {
        slot.* = try RunContainer.init(allocator, 1);
        _ = try slot.*.addRange(allocator, 0, std.math.maxInt(u16));
    }
    const start = bench_time.monotonicNanos();
    if (strategy == .unsorted) {
        for (run_containers) |run| run.deinit(allocator);
    } else {
        var scratch: [run_containers.len]*RunContainer = run_containers;
        std.mem.sort(*RunContainer, &scratch, strategy, lessThanRun);
        for (scratch) |run| run.deinit(allocator);
    }
    return bench_time.monotonicNanos() - start;
}

fn lessThanRun(strategy: Strategy, left: *RunContainer, right: *RunContainer) bool {
    return switch (strategy) {
        .header_asc => @intFromPtr(left) < @intFromPtr(right),
        .payload_asc => @intFromPtr(left.runs) < @intFromPtr(right.runs),
        .payload_desc => @intFromPtr(left.runs) > @intFromPtr(right.runs),
        .unsorted => unreachable,
    };
}

fn metricStats(values: *[timed_runs]u64) MetricStats {
    std.mem.sort(u64, values, {}, std.sort.asc(u64));
    return .{ .median = values[timed_runs / 2], .minimum = values[0], .maximum = values[timed_runs - 1] };
}

fn teardownSingleStats(sample: TeardownTiming) TeardownStats {
    return .{
        .teardown = singleMetric(sample.teardown_ns),
        .refill = singleMetric(sample.refill_ns),
        .traversal = singleMetric(sample.traversal_ns),
        .combined = singleMetric(sample.combined_ns),
        .teardown_and_combined = singleMetric(sample.teardown_and_combined_ns),
        .noise = singleMetric(sample.noise_ns),
    };
}

fn singleMetric(value: u64) MetricStats {
    return .{ .median = value, .minimum = value, .maximum = value };
}

fn teardownStats(samples: *const [timed_runs]TeardownTiming) TeardownStats {
    var teardown: [timed_runs]u64 = undefined;
    var refill: [timed_runs]u64 = undefined;
    var traversal: [timed_runs]u64 = undefined;
    var combined: [timed_runs]u64 = undefined;
    var teardown_and_combined: [timed_runs]u64 = undefined;
    var noise: [timed_runs]u64 = undefined;
    for (samples, &teardown, &refill, &traversal, &combined, &teardown_and_combined, &noise) |sample, *a, *b, *d, *e, *f, *g| {
        a.* = sample.teardown_ns;
        b.* = sample.refill_ns;
        d.* = sample.traversal_ns;
        e.* = sample.combined_ns;
        f.* = sample.teardown_and_combined_ns;
        g.* = sample.noise_ns;
    }
    return .{
        .teardown = metricStats(&teardown),
        .refill = metricStats(&refill),
        .traversal = metricStats(&traversal),
        .combined = metricStats(&combined),
        .teardown_and_combined = metricStats(&teardown_and_combined),
        .noise = metricStats(&noise),
    };
}
