// SPDX-License-Identifier: MPL-2.0

//! Spec 39-00 diagnosis worker. Deferred free ordering is implemented only in
//! this executable; production repair and the public API remain unchanged.

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");
const dashboard = @import("bench_croaring.zig");

const RoaringBitmap = rawr.RoaringBitmap;
const ArrayContainer = rawr.ArrayContainer;
const BitsetContainer = rawr.BitsetContainer;
const Container = rawr.Container;
const TaggedPtr = rawr.TaggedPtr;
const ops = rawr.container_ops;

const warmup_runs = 3;
const timed_runs = 21;
const max_demotions = 16_364;
const noise_count = 16_384;
const bucket_count = 4096;
const radix_bits = 8;
const radix_bins = 1 << radix_bits;

const Corpus = enum { canonical, sweep };
const AllocatorKind = enum { smp, libc, croaring };
const Strategy = enum { interleaved, key, reverse, bucket, radix, pdq, block };
const NoiseMode = enum { none, shared };

const Config = struct {
    corpus: Corpus,
    allocator: AllocatorKind,
    strategy: Strategy,
    noise: NoiseMode,
    demotions: usize,
};

const CycleSample = struct {
    construction_ns: u64 = 0,
    repair_ns: u64 = 0,
    scratch_ns: u64 = 0,
    reorder_ns: u64 = 0,
    demote_free_ns: u64 = 0,
    teardown_ns: u64 = 0,
    full_ns: u64 = 0,
    demoted: usize = 0,
    arrays: usize = 0,
    bitsets: usize = 0,
    runs: usize = 0,
    travel_ppm: u64 = 0,
    page_local_ppm: u64 = 0,
    descending_ppm: u64 = 0,
    reorder_fallback: bool = false,
};

const CycleStats = struct {
    construction_ns: u64,
    repair_ns: u64,
    scratch_ns: u64,
    reorder_ns: u64,
    demote_free_ns: u64,
    teardown_ns: u64,
    full_ns: u64,
    travel_ppm: u64,
    page_local_ppm: u64,
    descending_ppm: u64,
    last: CycleSample,
};

const RepairObservation = struct {
    scratch_ns: u64 = 0,
    reorder_ns: u64 = 0,
    demote_free_ns: u64 = 0,
    demoted: usize = 0,
    travel_ppm: u64 = 0,
    page_local_ppm: u64 = 0,
    descending_ppm: u64 = 0,
    reorder_fallback: bool = false,
};

const OrderQuality = struct {
    travel_ppm: u64,
    page_local_ppm: u64,
    descending_ppm: u64,
};

const KindCounts = struct {
    arrays: usize = 0,
    bitsets: usize = 0,
    runs: usize = 0,
};

const FailOnceAllocator = struct {
    backing: std.mem.Allocator,
    failed: bool = false,

    const Self = @This();

    fn allocator(self: *Self) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable: std.mem.Allocator.VTable = .{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    };

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        if (!self.failed) {
            self.failed = true;
            return null;
        }
        return self.backing.rawAlloc(len, alignment, ret_addr);
    }

    fn resize(
        ctx: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        ret_addr: usize,
    ) bool {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.backing.rawResize(memory, alignment, new_len, ret_addr);
    }

    fn remap(
        ctx: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        ret_addr: usize,
    ) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.backing.rawRemap(memory, alignment, new_len, ret_addr);
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.backing.rawFree(memory, alignment, ret_addr);
    }
};

const NoiseKind = enum { primary, secondary_64, secondary_512, secondary_4096 };
const NoiseDescriptor = struct { kind: NoiseKind };
const NoiseAllocation = struct {
    ptr: [*]u8,
    len: usize,
    kind: NoiseKind,
    live: bool,
};

var noise_descriptors: [noise_count]NoiseDescriptor = undefined;
var noise_allocations: [noise_count]NoiseAllocation = undefined;

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var header = false;
    var corpus: ?Corpus = null;
    var allocator: ?AllocatorKind = null;
    var strategy: ?Strategy = null;
    var noise: ?NoiseMode = null;
    var demotions: ?usize = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.startsWith(u8, arg, "--corpus=")) {
            corpus = std.meta.stringToEnum(Corpus, arg[9..]) orelse return error.UnknownCorpus;
        } else if (std.mem.startsWith(u8, arg, "--allocator=")) {
            allocator = std.meta.stringToEnum(AllocatorKind, arg[12..]) orelse return error.UnknownAllocator;
        } else if (std.mem.startsWith(u8, arg, "--strategy=")) {
            strategy = std.meta.stringToEnum(Strategy, arg[11..]) orelse return error.UnknownStrategy;
        } else if (std.mem.startsWith(u8, arg, "--noise=")) {
            noise = std.meta.stringToEnum(NoiseMode, arg[8..]) orelse return error.UnknownNoise;
        } else if (std.mem.startsWith(u8, arg, "--demotions=")) {
            demotions = try std.fmt.parseInt(usize, arg[12..], 10);
        } else {
            return error.UnknownArgument;
        }
    }

    if (header) {
        if (corpus != null or allocator != null or strategy != null or noise != null or demotions != null) {
            return error.ConflictingArguments;
        }
        printHeader();
        return;
    }

    const config = Config{
        .corpus = corpus orelse return error.MissingCorpus,
        .allocator = allocator orelse return error.MissingAllocator,
        .strategy = strategy orelse return error.MissingStrategy,
        .noise = noise orelse .none,
        .demotions = demotions orelse max_demotions,
    };
    try validateConfig(config);
    prepareNoiseDescriptors();

    if (config.allocator == .croaring) {
        dashboard.parityPrepare(.lazy_or_repair, .croaring);
        defer dashboard.parityCleanup();
        const stats = measureCRoaring(config.noise);
        try dashboard.parityValidate(.lazy_or_repair, .libc);
        printResult(config, stats);
        bench_time.print("VALIDATION\tcanonical-croaring\tportable-bytes-identical\n", .{});
        return;
    }

    if (config.corpus == .canonical) {
        dashboard.parityPrepare(.lazy_or_repair, .rawr);
        defer dashboard.parityCleanup();
        const inputs = dashboard.parityRawrSparseInputs();
        try runRawr(config, inputs.left, inputs.right);
        return;
    }

    var inputs = try buildSweepInputs(config.demotions);
    defer inputs[0].deinit();
    defer inputs[1].deinit();
    try runRawr(config, &inputs[0], &inputs[1]);
}

fn printHeader() void {
    bench_time.printBenchEnvironment();
    bench_time.print("# spec 39-00 free-order diagnosis: production code unchanged\n", .{});
    bench_time.print("# protocol: {d}w/{d}t median; fresh-process aggregation is controller-side\n", .{ warmup_runs, timed_runs });
    bench_time.print("# requested-cpu: native\n", .{});
    bench_time.print("# croaring-avx512: {s}\n", .{if (c.CROARING_COMPILER_SUPPORTS_AVX512 != 0) "on" else "off"});
}

fn validateConfig(config: Config) !void {
    if (config.demotions == 0 or config.demotions > max_demotions) return error.InvalidDemotionCount;
    if (config.corpus == .canonical and config.demotions != max_demotions) return error.InvalidCanonicalCount;
    if (config.allocator == .croaring and
        (config.corpus != .canonical or config.strategy != .interleaved))
    {
        return error.UnsupportedCRoaringConfiguration;
    }
}

fn allocatorFor(kind: AllocatorKind) std.mem.Allocator {
    return switch (kind) {
        .smp => std.heap.smp_allocator,
        .libc, .croaring => bench_time.cAllocator(),
    };
}

fn buildSweepInputs(demotions: usize) !struct { RoaringBitmap, RoaringBitmap } {
    var left = try RoaringBitmap.init(std.heap.smp_allocator);
    errdefer left.deinit();
    var right = try RoaringBitmap.init(std.heap.smp_allocator);
    errdefer right.deinit();

    const keys = demotions * 4;
    for (0..keys) |key| {
        const base = @as(u32, @intCast(key)) << 16;
        inline for (.{ 1, 17, 33, 49 }) |low| {
            const value = base | low;
            if (key < demotions * 2) _ = try left.add(value);
            if (key >= demotions) _ = try right.add(value);
        }
    }
    return .{ left, right };
}

fn runRawr(config: Config, left: *const RoaringBitmap, right: *const RoaringBitmap) !void {
    const allocator = allocatorFor(config.allocator);
    const stats = try measureRawr(config, left, right, allocator);
    try validateRawr(config, left, right, allocator, stats.last.demoted);
    printResult(config, stats);
    bench_time.print(
        "VALIDATION\t{s}\tbytes-permutation-ownership-scratch-fallback\n",
        .{@tagName(config.corpus)},
    );
}

fn measureRawr(
    config: Config,
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
    allocator: std.mem.Allocator,
) !CycleStats {
    var noise_active = false;
    defer if (noise_active) cleanupNoise(allocator);

    for (0..warmup_runs) |_| {
        _ = try timeRawrCycle(left, right, allocator, config.strategy, config.demotions);
        if (noise_active) cleanupNoise(allocator);
        noise_active = false;
        if (config.noise == .shared) {
            try applyNoise(allocator);
            noise_active = true;
        }
    }

    var samples: [timed_runs]CycleSample = undefined;
    for (&samples) |*sample| {
        sample.* = try timeRawrCycle(left, right, allocator, config.strategy, config.demotions);
        if (noise_active) cleanupNoise(allocator);
        noise_active = false;
        if (config.noise == .shared) {
            try applyNoise(allocator);
            noise_active = true;
        }
    }
    return cycleStats(&samples);
}

fn timeRawrCycle(
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
    allocator: std.mem.Allocator,
    strategy: Strategy,
    expected_demotions: usize,
) !CycleSample {
    const full_start = bench_time.monotonicNanos();
    const construction_start = full_start;
    var result = try left.lazyOr(allocator, right, true);
    const construction_ns = bench_time.monotonicNanos() - construction_start;

    const repair_start = bench_time.monotonicNanos();
    const repair = if (strategy == .interleaved) blk: {
        try result.repairAfterLazy();
        break :blk RepairObservation{};
    } else try repairDeferred(&result, strategy, false);
    const repair_ns = bench_time.monotonicNanos() - repair_start;

    const kinds = countKinds(&result);
    const teardown_start = bench_time.monotonicNanos();
    result.deinit();
    const teardown_ns = bench_time.monotonicNanos() - teardown_start;
    const full_ns = bench_time.monotonicNanos() - full_start;

    return .{
        .construction_ns = construction_ns,
        .repair_ns = repair_ns,
        .scratch_ns = repair.scratch_ns,
        .reorder_ns = repair.reorder_ns,
        .demote_free_ns = repair.demote_free_ns,
        .teardown_ns = teardown_ns,
        .full_ns = full_ns,
        .demoted = if (strategy == .interleaved) expected_demotions else repair.demoted,
        .arrays = kinds.arrays,
        .bitsets = kinds.bitsets,
        .runs = kinds.runs,
        .travel_ppm = repair.travel_ppm,
        .page_local_ppm = repair.page_local_ppm,
        .descending_ppm = repair.descending_ppm,
        .reorder_fallback = repair.reorder_fallback,
    };
}

fn repairDeferred(bitmap: *RoaringBitmap, strategy: Strategy, audit: bool) !RepairObservation {
    const allocator = bitmap.allocator;
    const scratch_start = bench_time.monotonicNanos();
    const scratch = allocator.alloc(*BitsetContainer, bitmap.size) catch {
        try bitmap.repairAfterLazy();
        return .{ .reorder_fallback = true };
    };
    var scratch_live = true;
    var scratch_ns = bench_time.monotonicNanos() - scratch_start;
    defer if (scratch_live) allocator.free(scratch);

    const original_size: usize = bitmap.size;
    var write_idx: usize = 0;
    var collected: usize = 0;
    var total: u64 = 0;

    for (0..original_size) |read_idx| {
        const key = bitmap.keys[read_idx];
        const tagged = bitmap.containers[read_idx];
        switch (Container.fromTagged(tagged)) {
            .array => |array| {
                if (array.cardinality == 0) {
                    array.deinit(allocator);
                    continue;
                }
                bitmap.keys[write_idx] = key;
                bitmap.containers[write_idx] = tagged;
                total += array.cardinality;
                write_idx += 1;
            },
            .bitset => |bitset| {
                const cardinality = bitset.computeCardinality();
                if (cardinality == 0) {
                    bitset.deinit(allocator);
                    continue;
                }
                if (cardinality <= ArrayContainer.MAX_CARDINALITY) {
                    const array = ops.bitsetToArray(allocator, bitset) catch |err| {
                        for (scratch[0..collected]) |owned| owned.deinit(allocator);
                        compactUntouchedTail(bitmap, write_idx, read_idx, original_size);
                        return err;
                    };
                    bitmap.keys[write_idx] = key;
                    bitmap.containers[write_idx] = TaggedPtr.initArray(array);
                    scratch[collected] = bitset;
                    collected += 1;
                } else {
                    bitmap.keys[write_idx] = key;
                    bitmap.containers[write_idx] = tagged;
                }
                total += cardinality;
                write_idx += 1;
            },
            .run => |run| {
                const cardinality = run.getCardinality();
                if (cardinality == 0) {
                    run.deinit(allocator);
                    continue;
                }
                bitmap.keys[write_idx] = key;
                bitmap.containers[write_idx] = tagged;
                total += cardinality;
                write_idx += 1;
            },
            .reserved => unreachable,
        }
    }

    bitmap.size = @intCast(write_idx);
    bitmap.cached_cardinality = @intCast(total);

    var collected_owned = true;
    errdefer if (collected_owned) freeBitsets(allocator, scratch[0..collected], false);
    const original_order = if (audit)
        try std.heap.page_allocator.dupe(*BitsetContainer, scratch[0..collected])
    else
        null;
    defer if (original_order) |items| std.heap.page_allocator.free(items);

    const reorder_start = bench_time.monotonicNanos();
    const reordered = try reorderPointers(allocator, scratch[0..collected], strategy);
    const reorder_ns = bench_time.monotonicNanos() - reorder_start;
    const actual_strategy = if (reordered) strategy else Strategy.key;
    if (original_order) |items| try verifyPermutation(items, scratch[0..collected]);
    const quality = orderQuality(scratch[0..collected], actual_strategy == .reverse);

    const free_start = bench_time.monotonicNanos();
    freeBitsets(allocator, scratch[0..collected], actual_strategy == .reverse);
    collected_owned = false;
    const demote_free_ns = bench_time.monotonicNanos() - free_start;
    const scratch_free_start = bench_time.monotonicNanos();
    allocator.free(scratch);
    scratch_live = false;
    scratch_ns += bench_time.monotonicNanos() - scratch_free_start;

    return .{
        .scratch_ns = scratch_ns,
        .reorder_ns = reorder_ns,
        .demote_free_ns = demote_free_ns,
        .demoted = collected,
        .travel_ppm = quality.travel_ppm,
        .page_local_ppm = quality.page_local_ppm,
        .descending_ppm = quality.descending_ppm,
        .reorder_fallback = !reordered,
    };
}

fn verifyPermutation(original: []*BitsetContainer, reordered: []*BitsetContainer) !void {
    if (original.len != reordered.len) return error.ReorderLengthMismatch;
    const sorted_reordered = try std.heap.page_allocator.dupe(*BitsetContainer, reordered);
    defer std.heap.page_allocator.free(sorted_reordered);
    std.mem.sortUnstable(*BitsetContainer, original, {}, payloadAscending);
    std.mem.sortUnstable(*BitsetContainer, sorted_reordered, {}, payloadAscending);
    for (original, sorted_reordered, 0..) |expected, actual, index| {
        if (expected != actual) return error.ReorderNotPermutation;
        if (index != 0 and expected == original[index - 1]) return error.DuplicateDeferredOwner;
    }
}

fn compactUntouchedTail(bitmap: *RoaringBitmap, write_idx: usize, read_idx: usize, original_size: usize) void {
    const tail_len = original_size - read_idx;
    std.mem.copyForwards(u16, bitmap.keys[write_idx .. write_idx + tail_len], bitmap.keys[read_idx..original_size]);
    std.mem.copyForwards(TaggedPtr, bitmap.containers[write_idx .. write_idx + tail_len], bitmap.containers[read_idx..original_size]);
    bitmap.size = @intCast(write_idx + tail_len);
    bitmap.cached_cardinality = -1;
}

fn reorderPointers(allocator: std.mem.Allocator, items: []*BitsetContainer, strategy: Strategy) !bool {
    if (items.len <= 1) return true;
    return switch (strategy) {
        .interleaved, .key, .reverse => true,
        .bucket => bucketOrder(allocator, items),
        .radix => radixOrder(allocator, items),
        .pdq => blk: {
            std.mem.sortUnstable(*BitsetContainer, items, {}, payloadDescending);
            break :blk true;
        },
        .block => blk: {
            std.mem.sort(*BitsetContainer, items, {}, payloadDescending);
            break :blk true;
        },
    };
}

fn bucketOrder(allocator: std.mem.Allocator, items: []*BitsetContainer) !bool {
    const auxiliary = allocator.alloc(*BitsetContainer, items.len) catch return false;
    defer allocator.free(auxiliary);

    const bounds = addressBounds(items);
    const span = bounds.maximum - bounds.minimum;
    if (span == 0) return true;
    const significant_bits: usize = @bitSizeOf(usize) - @clz(span);
    const bucket_bits = std.math.log2_int(usize, bucket_count);
    const shift: std.math.Log2Int(usize) = if (significant_bits > bucket_bits)
        @intCast(significant_bits - bucket_bits)
    else
        0;

    var counts: [bucket_count]u32 = @splat(0);
    for (items) |item| counts[bucketIndex(item, bounds.minimum, shift)] += 1;

    var offsets: [bucket_count]u32 = undefined;
    var next: u32 = 0;
    var bucket: usize = bucket_count;
    while (bucket > 0) {
        bucket -= 1;
        offsets[bucket] = next;
        next += counts[bucket];
    }
    for (items) |item| {
        const index = bucketIndex(item, bounds.minimum, shift);
        auxiliary[offsets[index]] = item;
        offsets[index] += 1;
    }
    @memcpy(items, auxiliary);
    return true;
}

fn bucketIndex(item: *BitsetContainer, minimum: usize, shift: std.math.Log2Int(usize)) usize {
    const normalized = @intFromPtr(item.words.ptr) - minimum;
    return @min(normalized >> shift, bucket_count - 1);
}

fn radixOrder(allocator: std.mem.Allocator, items: []*BitsetContainer) !bool {
    const auxiliary = allocator.alloc(*BitsetContainer, items.len) catch return false;
    defer allocator.free(auxiliary);

    const bounds = addressBounds(items);
    const span = bounds.maximum - bounds.minimum;
    if (span == 0) return true;
    const significant_bits: usize = @bitSizeOf(usize) - @clz(span);
    const passes = (significant_bits + radix_bits - 1) / radix_bits;
    var source = items;
    var destination = auxiliary;

    for (0..passes) |pass| {
        const shift: std.math.Log2Int(usize) = @intCast(pass * radix_bits);
        var counts: [radix_bins]u32 = @splat(0);
        for (source) |item| {
            const key = (@intFromPtr(item.words.ptr) - bounds.minimum) >> shift;
            counts[@intCast(key & (radix_bins - 1))] += 1;
        }
        var offsets: [radix_bins]u32 = undefined;
        var next: u32 = 0;
        for (counts, 0..) |count, index| {
            offsets[index] = next;
            next += count;
        }
        for (source) |item| {
            const key = (@intFromPtr(item.words.ptr) - bounds.minimum) >> shift;
            const index: usize = @intCast(key & (radix_bins - 1));
            destination[offsets[index]] = item;
            offsets[index] += 1;
        }
        const old_source = source;
        source = destination;
        destination = old_source;
    }
    if (source.ptr != items.ptr) @memcpy(items, source);
    reversePointers(items);
    return true;
}

fn addressBounds(items: []*BitsetContainer) struct { minimum: usize, maximum: usize } {
    var minimum = @intFromPtr(items[0].words.ptr);
    var maximum = minimum;
    for (items[1..]) |item| {
        const address = @intFromPtr(item.words.ptr);
        minimum = @min(minimum, address);
        maximum = @max(maximum, address);
    }
    return .{ .minimum = minimum, .maximum = maximum };
}

fn payloadDescending(_: void, left: *BitsetContainer, right: *BitsetContainer) bool {
    return @intFromPtr(left.words.ptr) > @intFromPtr(right.words.ptr);
}

fn payloadAscending(_: void, left: *BitsetContainer, right: *BitsetContainer) bool {
    return @intFromPtr(left.words.ptr) < @intFromPtr(right.words.ptr);
}

fn reversePointers(items: []*BitsetContainer) void {
    var left: usize = 0;
    var right = items.len;
    while (left < right) : (left += 1) {
        right -= 1;
        if (left >= right) break;
        std.mem.swap(*BitsetContainer, &items[left], &items[right]);
    }
}

fn freeBitsets(allocator: std.mem.Allocator, items: []*BitsetContainer, reverse: bool) void {
    if (!reverse) {
        for (items) |item| item.deinit(allocator);
        return;
    }
    var index = items.len;
    while (index > 0) {
        index -= 1;
        items[index].deinit(allocator);
    }
}

fn orderQuality(items: []*BitsetContainer, reverse: bool) OrderQuality {
    if (items.len <= 1) return .{ .travel_ppm = 0, .page_local_ppm = 1_000_000, .descending_ppm = 1_000_000 };
    const bounds = addressBounds(items);
    const footprint = @max(@as(u128, 1), @as(u128, items.len) * BitsetContainer.NUM_WORDS * @sizeOf(u64));
    var travel: u128 = 0;
    var page_local: u64 = 0;
    var descending: u64 = 0;
    var previous = orderedAddress(items, 0, reverse);
    for (1..items.len) |index| {
        const address = orderedAddress(items, index, reverse);
        travel += if (address >= previous) address - previous else previous - address;
        page_local += @intFromBool((address >> 12) == (previous >> 12));
        descending += @intFromBool(address < previous);
        previous = address;
    }
    _ = bounds;
    const pairs: u64 = @intCast(items.len - 1);
    return .{
        .travel_ppm = @intCast((travel * 1_000_000) / footprint),
        .page_local_ppm = page_local * 1_000_000 / pairs,
        .descending_ppm = descending * 1_000_000 / pairs,
    };
}

fn orderedAddress(items: []*BitsetContainer, index: usize, reverse: bool) usize {
    const item = if (reverse) items[items.len - 1 - index] else items[index];
    return @intFromPtr(item.words.ptr);
}

fn countKinds(bitmap: *const RoaringBitmap) KindCounts {
    var result: KindCounts = .{};
    for (bitmap.containers[0..bitmap.size]) |tagged| switch (tagged.getType()) {
        .array => result.arrays += 1,
        .bitset => result.bitsets += 1,
        .run => result.runs += 1,
        .reserved => unreachable,
    };
    return result;
}

fn validateRawr(
    config: Config,
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
    allocator: std.mem.Allocator,
    expected_demoted: usize,
) !void {
    var baseline = try left.lazyOr(allocator, right, true);
    defer baseline.deinit();
    try baseline.repairAfterLazy();
    try baseline.validate();

    var candidate = try left.lazyOr(allocator, right, true);
    defer candidate.deinit();
    const observation = if (config.strategy == .interleaved) blk: {
        try candidate.repairAfterLazy();
        break :blk RepairObservation{};
    } else try repairDeferred(&candidate, config.strategy, true);
    try candidate.validate();
    if (!baseline.equals(&candidate)) return error.CandidateMismatch;

    const baseline_bytes = try baseline.serialize(std.heap.page_allocator);
    defer std.heap.page_allocator.free(baseline_bytes);
    const candidate_bytes = try candidate.serialize(std.heap.page_allocator);
    defer std.heap.page_allocator.free(candidate_bytes);
    if (!std.mem.eql(u8, baseline_bytes, candidate_bytes)) return error.SerializedMismatch;
    if (config.strategy != .interleaved and observation.demoted != expected_demoted) return error.UnstableDemotionCount;
    try verifyScratchFallback(left, right, allocator, &baseline);
}

fn verifyScratchFallback(
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
    allocator: std.mem.Allocator,
    baseline: *const RoaringBitmap,
) !void {
    var candidate = try left.lazyOr(allocator, right, true);
    var fail_once = FailOnceAllocator{ .backing = allocator };
    candidate.allocator = fail_once.allocator();
    defer candidate.deinit();

    const observation = try repairDeferred(&candidate, .reverse, false);
    if (!fail_once.failed or !observation.reorder_fallback) return error.ScratchFallbackNotExercised;
    try candidate.validate();
    if (!baseline.equals(&candidate)) return error.ScratchFallbackMismatch;
}

fn measureCRoaring(noise: NoiseMode) CycleStats {
    const allocator = bench_time.cAllocator();
    var noise_active = false;
    defer if (noise_active) cleanupNoise(allocator);
    for (0..warmup_runs) |_| {
        _ = dashboard.parityRun(.lazy_or_repair, .croaring, .libc);
        if (noise_active) cleanupNoise(allocator);
        noise_active = false;
        if (noise == .shared) {
            applyNoise(allocator) catch unreachable;
            noise_active = true;
        }
    }
    var samples: [timed_runs]CycleSample = undefined;
    for (&samples) |*sample| {
        const start = bench_time.monotonicNanos();
        _ = dashboard.parityRun(.lazy_or_repair, .croaring, .libc);
        sample.* = .{ .full_ns = bench_time.monotonicNanos() - start };
        if (noise_active) cleanupNoise(allocator);
        noise_active = false;
        if (noise == .shared) {
            applyNoise(allocator) catch unreachable;
            noise_active = true;
        }
    }
    return cycleStats(&samples);
}

fn cycleStats(samples: *const [timed_runs]CycleSample) CycleStats {
    var construction: [timed_runs]u64 = undefined;
    var repair: [timed_runs]u64 = undefined;
    var scratch: [timed_runs]u64 = undefined;
    var reorder: [timed_runs]u64 = undefined;
    var demote_free: [timed_runs]u64 = undefined;
    var teardown: [timed_runs]u64 = undefined;
    var full: [timed_runs]u64 = undefined;
    var travel: [timed_runs]u64 = undefined;
    var page_local: [timed_runs]u64 = undefined;
    var descending: [timed_runs]u64 = undefined;
    for (samples, &construction, &repair, &scratch, &reorder, &demote_free, &teardown, &full, &travel, &page_local, &descending) |
        sample,
        *a,
        *b,
        *d,
        *e,
        *f,
        *g,
        *h,
        *i,
        *j,
        *k,
    | {
        a.* = sample.construction_ns;
        b.* = sample.repair_ns;
        d.* = sample.scratch_ns;
        e.* = sample.reorder_ns;
        f.* = sample.demote_free_ns;
        g.* = sample.teardown_ns;
        h.* = sample.full_ns;
        i.* = sample.travel_ppm;
        j.* = sample.page_local_ppm;
        k.* = sample.descending_ppm;
    }
    return .{
        .construction_ns = median(&construction),
        .repair_ns = median(&repair),
        .scratch_ns = median(&scratch),
        .reorder_ns = median(&reorder),
        .demote_free_ns = median(&demote_free),
        .teardown_ns = median(&teardown),
        .full_ns = median(&full),
        .travel_ppm = median(&travel),
        .page_local_ppm = median(&page_local),
        .descending_ppm = median(&descending),
        .last = samples[timed_runs - 1],
    };
}

fn median(values: *[timed_runs]u64) u64 {
    std.mem.sort(u64, values, {}, std.sort.asc(u64));
    return values[timed_runs / 2];
}

fn printResult(config: Config, stats: CycleStats) void {
    bench_time.print(
        "RESULT\t{s}\t{s}\t{s}\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n",
        .{
            @tagName(config.corpus),
            @tagName(config.allocator),
            @tagName(config.strategy),
            @tagName(config.noise),
            config.demotions,
            stats.construction_ns,
            stats.repair_ns,
            stats.scratch_ns,
            stats.reorder_ns,
            stats.demote_free_ns,
            stats.teardown_ns,
            stats.full_ns,
            c.rawr_bench_peak_rss_bytes(),
            stats.last.demoted,
            stats.last.arrays,
            stats.last.bitsets,
            stats.last.runs,
            stats.travel_ppm,
            stats.page_local_ppm,
            stats.descending_ppm,
            @intFromBool(stats.last.reorder_fallback),
        },
    );
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

fn applyNoise(allocator: std.mem.Allocator) !void {
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
