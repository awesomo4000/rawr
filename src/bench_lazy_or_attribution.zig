// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");
const CountingAllocator = @import("counting_allocator.zig").CountingAllocator;

const RoaringBitmap = rawr.RoaringBitmap;
const ArrayContainer = rawr.ArrayContainer;
const BitsetContainer = rawr.BitsetContainer;
const Container = rawr.Container;
const TaggedPtr = rawr.TaggedPtr;
const container_ops = rawr.container_ops;

const N_VALUES = 500_000;
const DENSE_SHARED_KEYS = 1024;
const WARMUP_RUNS = 3;
const TIMED_RUNS = 21;

const EXPECTED_SPARSE_LEN = 499_964;
const EXPECTED_LEFT_KEYS = 32_691;
const EXPECTED_RIGHT_KEYS = 49_169;
const EXPECTED_SHARED_KEYS = 16_364;
const EXPECTED_LEFT_ONLY_KEYS = 16_327;
const EXPECTED_RIGHT_ONLY_KEYS = 32_805;
const EXPECTED_SPARSE_MIN_CARDINALITY = 1;
const EXPECTED_SPARSE_MAX_CARDINALITY = 21;
const EXPECTED_SPARSE_TOTAL_CARDINALITY = 125_006;
const EXPECTED_SPARSE_CARDINALITY_FINGERPRINT = 0xc2a9f91d23d63807;

var sparse_values: [N_VALUES]u32 = undefined;

const SharedPair = struct {
    first: *const ArrayContainer,
    second: *const ArrayContainer,
};

const CapturedCorpus = struct {
    shared: []SharedPair,
    unique: []TaggedPtr,
    left_only: usize,
    right_only: usize,

    fn deinit(self: *CapturedCorpus, allocator: std.mem.Allocator) void {
        allocator.free(self.shared);
        allocator.free(self.unique);
    }
};

const Stats = struct {
    median_ns: u64,
    min_ns: u64,
    max_ns: u64,
};

const PrototypeVariant = enum {
    headered,
    headerless,
};

const PrototypePhase = enum {
    construction,
    repair,
    combined,
};

const CorpusKind = enum {
    sparse,
    dense_survivor,
};

const LifecycleCounts = struct {
    matched_keys: usize = 0,
    transient_bitsets: usize = 0,
    demoted: usize = 0,
    surviving: usize = 0,
    empty: usize = 0,
    min_cardinality: u32 = std.math.maxInt(u32),
    max_cardinality: u32 = 0,
    total_cardinality: u64 = 0,
    cardinality_fingerprint: u64 = 0xcbf29ce484222325,

    fn record(self: *LifecycleCounts, cardinality: u32) void {
        self.transient_bitsets += 1;
        self.min_cardinality = @min(self.min_cardinality, cardinality);
        self.max_cardinality = @max(self.max_cardinality, cardinality);
        self.total_cardinality += cardinality;
        self.cardinality_fingerprint ^= cardinality;
        self.cardinality_fingerprint *%= 0x100000001b3;
        if (cardinality == 0) {
            self.empty += 1;
        } else if (cardinality <= ArrayContainer.MAX_CARDINALITY) {
            self.demoted += 1;
        } else {
            self.surviving += 1;
        }
    }

    fn finish(self: *LifecycleCounts) void {
        if (self.transient_bitsets == 0) self.min_cardinality = 0;
    }
};

const Accounting = struct {
    construction: CountingAllocator.Stats,
    repair: CountingAllocator.Stats,
    combined: CountingAllocator.Stats,
    teardown: CountingAllocator.Stats,
    teardown_ns: u64,
};

fn initValues() usize {
    var prng = std.Random.DefaultPrng.init(54321);
    for (sparse_values[0..]) |*value| value.* = prng.random().int(u32);
    std.mem.sort(u32, sparse_values[0..], {}, std.sort.asc(u32));

    var len: usize = 1;
    for (sparse_values[1..]) |value| {
        if (value != sparse_values[len - 1]) {
            sparse_values[len] = value;
            len += 1;
        }
    }
    return len;
}

fn buildRawrInputs(allocator: std.mem.Allocator, values: []const u32) !struct { RoaringBitmap, RoaringBitmap } {
    var left = try RoaringBitmap.init(allocator);
    errdefer left.deinit();
    var right = try RoaringBitmap.init(allocator);
    errdefer right.deinit();

    const half = values.len / 2;
    try left.addMany(values[0..half]);
    try right.addMany(values[half / 2 ..]);
    return .{ left, right };
}

fn buildCrInputs(values: []const u32) !struct { *c.roaring_bitmap_t, *c.roaring_bitmap_t } {
    const left = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    errdefer c.roaring_bitmap_free(left);
    const right = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    errdefer c.roaring_bitmap_free(right);

    const half = values.len / 2;
    c.roaring_bitmap_add_many(left, half, values.ptr);
    c.roaring_bitmap_add_many(right, values.len - half / 2, values[half / 2 ..].ptr);
    return .{ left, right };
}

fn appendTagged(bitmap: *RoaringBitmap, key: u16, tagged: TaggedPtr) void {
    std.debug.assert(bitmap.size < bitmap.capacity);
    bitmap.keys[bitmap.size] = key;
    bitmap.containers[bitmap.size] = tagged;
    bitmap.size += 1;
}

fn transientTagged(words: *align(64) [BitsetContainer.NUM_WORDS]u64) TaggedPtr {
    const raw = @intFromPtr(words);
    std.debug.assert(raw & 0x3 == 0);
    return .{
        .tag = .reserved,
        .addr = @truncate(raw >> 2),
    };
}

fn transientWords(tagged: TaggedPtr) *align(64) [BitsetContainer.NUM_WORDS]u64 {
    std.debug.assert(tagged.tag == .reserved);
    return @ptrFromInt(tagged.rawAddr());
}

fn freeTransientWords(allocator: std.mem.Allocator, words: *align(64) [BitsetContainer.NUM_WORDS]u64) void {
    const slice: []align(64) u64 = words;
    allocator.free(slice);
}

fn deinitPrototype(bitmap: *RoaringBitmap) void {
    for (bitmap.containers[0..bitmap.size]) |tagged| {
        if (tagged.tag == .reserved) {
            freeTransientWords(bitmap.allocator, transientWords(tagged));
        } else {
            Container.fromTagged(tagged).deinit(bitmap.allocator);
        }
    }
    bitmap.allocator.free(bitmap.keys[0..bitmap.capacity]);
    bitmap.allocator.free(bitmap.containers[0..bitmap.capacity]);
    bitmap.* = undefined;
}

fn lazyAccumulate(accumulator: *BitsetContainer, container: Container) void {
    switch (container) {
        .array => |array| accumulator.setList(array.values[0..array.cardinality]),
        .bitset => |bitset| accumulator.lazyUnionWith(bitset),
        .run => |run| {
            for (run.runs[0..run.n_runs]) |pair| accumulator.setRange(pair.start, pair.end());
            accumulator.cardinality = -1;
        },
        .reserved => unreachable,
    }
}

fn buildPrototype(
    comptime variant: PrototypeVariant,
    allocator: std.mem.Allocator,
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
) !RoaringBitmap {
    const max_result_size = @min(left.size + right.size, @as(u32, 1) << 16);
    var result = try RoaringBitmap.initCapacity(allocator, max_result_size);
    errdefer deinitPrototype(&result);

    var i: usize = 0;
    var j: usize = 0;
    while (i < left.size and j < right.size) {
        const left_key = left.keys[i];
        const right_key = right.keys[j];
        if (left_key < right_key) {
            appendTagged(&result, left_key, (try Container.fromTagged(left.containers[i]).clone(allocator)).toTagged());
            i += 1;
        } else if (left_key > right_key) {
            appendTagged(&result, right_key, (try Container.fromTagged(right.containers[j]).clone(allocator)).toTagged());
            j += 1;
        } else {
            switch (variant) {
                .headered => {
                    const accumulator = try BitsetContainer.init(allocator);
                    errdefer accumulator.deinit(allocator);
                    lazyAccumulate(accumulator, Container.fromTagged(left.containers[i]));
                    lazyAccumulate(accumulator, Container.fromTagged(right.containers[j]));
                    appendTagged(&result, left_key, TaggedPtr.initBitset(accumulator));
                },
                .headerless => {
                    const allocated = try allocator.alignedAlloc(u64, .@"64", BitsetContainer.NUM_WORDS);
                    const words: *align(64) [BitsetContainer.NUM_WORDS]u64 = allocated[0..BitsetContainer.NUM_WORDS];
                    errdefer freeTransientWords(allocator, words);
                    @memset(words, 0);
                    var view = BitsetContainer{ .words = words, .cardinality = 0 };
                    lazyAccumulate(&view, Container.fromTagged(left.containers[i]));
                    lazyAccumulate(&view, Container.fromTagged(right.containers[j]));
                    appendTagged(&result, left_key, transientTagged(words));
                },
            }
            i += 1;
            j += 1;
        }
    }
    while (i < left.size) : (i += 1) {
        appendTagged(&result, left.keys[i], (try Container.fromTagged(left.containers[i]).clone(allocator)).toTagged());
    }
    while (j < right.size) : (j += 1) {
        appendTagged(&result, right.keys[j], (try Container.fromTagged(right.containers[j]).clone(allocator)).toTagged());
    }
    result.cached_cardinality = -1;
    return result;
}

fn repairPrototype(bitmap: *RoaringBitmap) !void {
    var write_index: usize = 0;
    var total: u64 = 0;

    for (bitmap.keys[0..bitmap.size], bitmap.containers[0..bitmap.size]) |key, tagged| {
        if (tagged.tag == .reserved) {
            const words = transientWords(tagged);
            var view = BitsetContainer{ .words = words, .cardinality = -1 };
            const cardinality = view.computeCardinality();
            if (cardinality == 0) {
                freeTransientWords(bitmap.allocator, words);
                continue;
            }
            bitmap.keys[write_index] = key;
            if (cardinality <= ArrayContainer.MAX_CARDINALITY) {
                const array = try container_ops.bitsetToArray(bitmap.allocator, &view);
                freeTransientWords(bitmap.allocator, words);
                bitmap.containers[write_index] = TaggedPtr.initArray(array);
            } else {
                const header = try bitmap.allocator.create(BitsetContainer);
                header.* = .{ .words = words, .cardinality = @intCast(cardinality) };
                bitmap.containers[write_index] = TaggedPtr.initBitset(header);
            }
            total += cardinality;
            write_index += 1;
            continue;
        }

        switch (Container.fromTagged(tagged)) {
            .array => |array| {
                if (array.cardinality == 0) {
                    array.deinit(bitmap.allocator);
                    continue;
                }
                bitmap.keys[write_index] = key;
                bitmap.containers[write_index] = tagged;
                total += array.cardinality;
                write_index += 1;
            },
            .bitset => |bitset| {
                const cardinality = bitset.computeCardinality();
                if (cardinality == 0) {
                    bitset.deinit(bitmap.allocator);
                    continue;
                }
                bitmap.keys[write_index] = key;
                if (cardinality <= ArrayContainer.MAX_CARDINALITY) {
                    const array = try container_ops.bitsetToArray(bitmap.allocator, bitset);
                    bitset.deinit(bitmap.allocator);
                    bitmap.containers[write_index] = TaggedPtr.initArray(array);
                } else {
                    bitmap.containers[write_index] = tagged;
                }
                total += cardinality;
                write_index += 1;
            },
            .run => |run| {
                const cardinality = run.getCardinality();
                if (cardinality == 0) {
                    run.deinit(bitmap.allocator);
                    continue;
                }
                bitmap.keys[write_index] = key;
                bitmap.containers[write_index] = tagged;
                total += cardinality;
                write_index += 1;
            },
            .reserved => unreachable,
        }
    }
    bitmap.size = @intCast(write_index);
    bitmap.cached_cardinality = @intCast(total);
}

fn buildDenseSurvivorInputs(allocator: std.mem.Allocator) !struct { RoaringBitmap, RoaringBitmap } {
    var left = try RoaringBitmap.initCapacity(allocator, DENSE_SHARED_KEYS);
    errdefer left.deinit();
    var right = try RoaringBitmap.initCapacity(allocator, DENSE_SHARED_KEYS);
    errdefer right.deinit();

    for (0..DENSE_SHARED_KEYS) |index| {
        const left_bitset = try BitsetContainer.init(allocator);
        left_bitset.setRange(0, 4096);
        _ = left_bitset.computeCardinality();
        appendTagged(&left, @intCast(index), TaggedPtr.initBitset(left_bitset));

        const right_bitset = try BitsetContainer.init(allocator);
        right_bitset.setRange(4096, 8192);
        _ = right_bitset.computeCardinality();
        appendTagged(&right, @intCast(index), TaggedPtr.initBitset(right_bitset));
    }
    left.cached_cardinality = -1;
    right.cached_cardinality = -1;
    return .{ left, right };
}

fn captureCorpus(allocator: std.mem.Allocator, left: *const RoaringBitmap, right: *const RoaringBitmap) !CapturedCorpus {
    var shared_count: usize = 0;
    var left_only: usize = 0;
    var right_only: usize = 0;
    var i: usize = 0;
    var j: usize = 0;
    while (i < left.size and j < right.size) {
        if (left.keys[i] < right.keys[j]) {
            left_only += 1;
            i += 1;
        } else if (left.keys[i] > right.keys[j]) {
            right_only += 1;
            j += 1;
        } else {
            shared_count += 1;
            i += 1;
            j += 1;
        }
    }
    left_only += left.size - i;
    right_only += right.size - j;

    const shared = try allocator.alloc(SharedPair, shared_count);
    errdefer allocator.free(shared);
    const unique = try allocator.alloc(TaggedPtr, left_only + right_only);
    errdefer allocator.free(unique);

    i = 0;
    j = 0;
    var shared_index: usize = 0;
    var unique_index: usize = 0;
    while (i < left.size and j < right.size) {
        if (left.keys[i] < right.keys[j]) {
            unique[unique_index] = left.containers[i];
            unique_index += 1;
            i += 1;
        } else if (left.keys[i] > right.keys[j]) {
            unique[unique_index] = right.containers[j];
            unique_index += 1;
            j += 1;
        } else {
            const first = Container.fromTagged(left.containers[i]);
            const second = Container.fromTagged(right.containers[j]);
            shared[shared_index] = .{
                .first = switch (first) {
                    .array => |array| array,
                    else => return error.ExpectedArrayContainer,
                },
                .second = switch (second) {
                    .array => |array| array,
                    else => return error.ExpectedArrayContainer,
                },
            };
            shared_index += 1;
            i += 1;
            j += 1;
        }
    }
    while (i < left.size) : (i += 1) {
        unique[unique_index] = left.containers[i];
        unique_index += 1;
    }
    while (j < right.size) : (j += 1) {
        unique[unique_index] = right.containers[j];
        unique_index += 1;
    }

    return .{
        .shared = shared,
        .unique = unique,
        .left_only = left_only,
        .right_only = right_only,
    };
}

fn inspectLifecycle(bitmap: *const RoaringBitmap, matched_keys: usize) !LifecycleCounts {
    var counts = LifecycleCounts{ .matched_keys = matched_keys };
    for (bitmap.containers[0..bitmap.size]) |tagged| {
        if (tagged.tag != .reserved) continue;
        const words = transientWords(tagged);
        var view = BitsetContainer{ .words = words, .cardinality = -1 };
        counts.record(view.computeCardinality());
    }
    counts.finish();
    if (counts.transient_bitsets != matched_keys) return error.TransientMaterializationMismatch;
    return counts;
}

fn validateSparseCounts(sparse_len: usize, left: *const RoaringBitmap, right: *const RoaringBitmap, corpus: *const CapturedCorpus) !void {
    if (sparse_len != EXPECTED_SPARSE_LEN or
        left.size != EXPECTED_LEFT_KEYS or
        right.size != EXPECTED_RIGHT_KEYS or
        corpus.shared.len != EXPECTED_SHARED_KEYS or
        corpus.left_only != EXPECTED_LEFT_ONLY_KEYS or
        corpus.right_only != EXPECTED_RIGHT_ONLY_KEYS)
    {
        return error.SparseCorpusCountMismatch;
    }
}

fn validatePrototypePair(
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
    expected: *const RoaringBitmap,
) !LifecycleCounts {
    var headered = try buildPrototype(.headered, std.heap.smp_allocator, left, right);
    defer deinitPrototype(&headered);
    var headerless = try buildPrototype(.headerless, std.heap.smp_allocator, left, right);
    defer deinitPrototype(&headerless);

    const lifecycle = try inspectLifecycle(&headerless, countMatchedKeys(left, right));
    try repairPrototype(&headered);
    try repairPrototype(&headerless);
    try headered.validate();
    try headerless.validate();
    if (!headered.equals(&headerless) or !headered.equals(expected)) return error.PrototypeLogicalMismatch;

    const headered_bytes = try headered.serialize(std.heap.smp_allocator);
    defer std.heap.smp_allocator.free(headered_bytes);
    const headerless_bytes = try headerless.serialize(std.heap.smp_allocator);
    defer std.heap.smp_allocator.free(headerless_bytes);
    const expected_bytes = try expected.serialize(std.heap.smp_allocator);
    defer std.heap.smp_allocator.free(expected_bytes);
    if (!std.mem.eql(u8, headered_bytes, headerless_bytes) or
        !std.mem.eql(u8, headered_bytes, expected_bytes))
    {
        return error.PrototypeSerializedMismatch;
    }
    return lifecycle;
}

fn countMatchedKeys(left: *const RoaringBitmap, right: *const RoaringBitmap) usize {
    var i: usize = 0;
    var j: usize = 0;
    var count: usize = 0;
    while (i < left.size and j < right.size) {
        if (left.keys[i] < right.keys[j]) {
            i += 1;
        } else if (left.keys[i] > right.keys[j]) {
            j += 1;
        } else {
            count += 1;
            i += 1;
            j += 1;
        }
    }
    return count;
}

fn prototypeTimed(
    comptime variant: PrototypeVariant,
    phase: PrototypePhase,
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
) !u64 {
    switch (phase) {
        .construction => {
            const start = bench_time.monotonicNanos();
            var result = try buildPrototype(variant, std.heap.smp_allocator, left, right);
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(&result);
            deinitPrototype(&result);
            return elapsed;
        },
        .repair => {
            var result = try buildPrototype(variant, std.heap.smp_allocator, left, right);
            errdefer deinitPrototype(&result);
            const start = bench_time.monotonicNanos();
            try repairPrototype(&result);
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(&result);
            deinitPrototype(&result);
            return elapsed;
        },
        .combined => {
            const start = bench_time.monotonicNanos();
            var result = try buildPrototype(variant, std.heap.smp_allocator, left, right);
            errdefer deinitPrototype(&result);
            try repairPrototype(&result);
            std.mem.doNotOptimizeAway(&result);
            deinitPrototype(&result);
            return bench_time.monotonicNanos() - start;
        },
    }
}

fn productionCombined(left: *const RoaringBitmap, right: *const RoaringBitmap, allocator: std.mem.Allocator) !u64 {
    const start = bench_time.monotonicNanos();
    var result = try left.lazyOr(allocator, right, true);
    errdefer result.deinit();
    try result.repairAfterLazy();
    std.mem.doNotOptimizeAway(&result);
    result.deinit();
    return bench_time.monotonicNanos() - start;
}

fn productionRepair(left: *const RoaringBitmap, right: *const RoaringBitmap, allocator: std.mem.Allocator) !u64 {
    var result = try left.lazyOr(allocator, right, true);
    errdefer result.deinit();
    const start = bench_time.monotonicNanos();
    try result.repairAfterLazy();
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(&result);
    result.deinit();
    return elapsed;
}

fn crCombined(left: *const c.roaring_bitmap_t, right: *const c.roaring_bitmap_t) !u64 {
    const start = bench_time.monotonicNanos();
    const result = c.roaring_bitmap_lazy_or(left, right, true) orelse return error.OutOfMemory;
    c.roaring_bitmap_repair_after_lazy(result);
    std.mem.doNotOptimizeAway(result);
    c.roaring_bitmap_free(result);
    return bench_time.monotonicNanos() - start;
}

fn crRepair(left: *const c.roaring_bitmap_t, right: *const c.roaring_bitmap_t) !u64 {
    const result = c.roaring_bitmap_lazy_or(left, right, true) orelse return error.OutOfMemory;
    const start = bench_time.monotonicNanos();
    c.roaring_bitmap_repair_after_lazy(result);
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(result);
    c.roaring_bitmap_free(result);
    return elapsed;
}

fn addActivity(first: CountingAllocator.Stats, second: CountingAllocator.Stats) CountingAllocator.Stats {
    var result = first;
    result.alloc_calls += second.alloc_calls;
    result.free_calls += second.free_calls;
    result.resize_calls += second.resize_calls;
    result.resize_successes += second.resize_successes;
    result.resize_failures += second.resize_failures;
    result.remap_calls += second.remap_calls;
    result.remap_in_place += second.remap_in_place;
    result.remap_moved += second.remap_moved;
    result.remap_failures += second.remap_failures;
    result.cumulative_bytes += second.cumulative_bytes;
    result.cumulative_class_bytes += second.cumulative_class_bytes;
    result.live_bytes = second.live_bytes;
    result.live_class_bytes = second.live_class_bytes;
    result.peak_live_bytes = @max(first.peak_live_bytes, second.peak_live_bytes);
    result.peak_live_class_bytes = @max(first.peak_live_class_bytes, second.peak_live_class_bytes);
    return result;
}

fn accountPrototype(
    comptime variant: PrototypeVariant,
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
) !Accounting {
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const allocator = counting.allocator();
    var result = try buildPrototype(variant, allocator, left, right);
    errdefer deinitPrototype(&result);
    const construction = counting.snapshot();

    counting.resetStats();
    try repairPrototype(&result);
    const repair = counting.snapshot();
    const combined = addActivity(construction, repair);

    counting.resetStats();
    const teardown_start = bench_time.monotonicNanos();
    deinitPrototype(&result);
    const teardown_ns = bench_time.monotonicNanos() - teardown_start;
    const teardown = counting.snapshot();
    if (teardown.live_bytes != 0 or teardown.live_class_bytes != 0) return error.AccountingLeak;
    return .{
        .construction = construction,
        .repair = repair,
        .combined = combined,
        .teardown = teardown,
        .teardown_ns = teardown_ns,
    };
}

fn printAccount(corpus: CorpusKind, variant: PrototypeVariant, phase: PrototypePhase, stats: CountingAllocator.Stats, teardown_ns: u64) void {
    bench_time.print("ACCOUNT\t{s}\t{s}\t{s}\talloc={d}\tfree={d}\trequested={d}\tclass={d}\tlive={d}\tpeak={d}\tteardown_ns={d}\n", .{
        @tagName(corpus),
        @tagName(variant),
        @tagName(phase),
        stats.alloc_calls,
        stats.free_calls,
        stats.cumulative_bytes,
        stats.cumulative_class_bytes,
        stats.live_bytes,
        stats.peak_live_bytes,
        teardown_ns,
    });
}

fn printAccounting(corpus: CorpusKind, variant: PrototypeVariant, accounting: Accounting) void {
    printAccount(corpus, variant, .construction, accounting.construction, 0);
    printAccount(corpus, variant, .repair, accounting.repair, 0);
    printAccount(corpus, variant, .combined, accounting.combined, accounting.teardown_ns);
    bench_time.print("TEARDOWN\t{s}\t{s}\tfree={d}\tlive={d}\tclass_live={d}\tns={d}\n", .{
        @tagName(corpus),
        @tagName(variant),
        accounting.teardown.free_calls,
        accounting.teardown.live_bytes,
        accounting.teardown.live_class_bytes,
        accounting.teardown_ns,
    });
}

fn printLifecycle(corpus: CorpusKind, counts: LifecycleCounts) void {
    bench_time.print("LIFECYCLE\t{s}\tmatched={d}\tcreated={d}\tdemoted={d}\tsurviving={d}\tempty={d}\tmin_card={d}\tmax_card={d}\ttotal_card={d}\tfingerprint=0x{x}\n", .{
        @tagName(corpus),
        counts.matched_keys,
        counts.transient_bitsets,
        counts.demoted,
        counts.surviving,
        counts.empty,
        counts.min_cardinality,
        counts.max_cardinality,
        counts.total_cardinality,
        counts.cardinality_fingerprint,
    });
}

fn measure(comptime operation: anytype, args: anytype) !Stats {
    for (0..WARMUP_RUNS) |_| _ = try @call(.auto, operation, args);

    var times: [TIMED_RUNS]u64 = undefined;
    for (&times) |*time| time.* = try @call(.auto, operation, args);
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return .{
        .median_ns = times[TIMED_RUNS / 2],
        .min_ns = times[0],
        .max_ns = times[TIMED_RUNS - 1],
    };
}

fn printResult(component: []const u8, variant: []const u8, stats: Stats) void {
    bench_time.print("{s:<20} {s:<12} {d:>10.3} {d:>10.3} {d:>10.3}\n", .{
        component,
        variant,
        @as(f64, @floatFromInt(stats.median_ns)) / 1_000_000.0,
        @as(f64, @floatFromInt(stats.min_ns)) / 1_000_000.0,
        @as(f64, @floatFromInt(stats.max_ns)) / 1_000_000.0,
    });
    bench_time.print("RESULT\t{s}\t{s}\t{d}\n", .{ component, variant, stats.median_ns });
}

fn rawrFull(left: *const RoaringBitmap, right: *const RoaringBitmap, allocator: std.mem.Allocator) !u64 {
    const start = bench_time.monotonicNanos();
    var result = try left.lazyOr(allocator, right, true);
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(&result);
    result.deinit();
    return elapsed;
}

fn crFull(left: *const c.roaring_bitmap_t, right: *const c.roaring_bitmap_t) !u64 {
    const start = bench_time.monotonicNanos();
    const result = c.roaring_bitmap_lazy_or(left, right, true) orelse return error.OutOfMemory;
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(result);
    c.roaring_bitmap_free(result);
    return elapsed;
}

fn rawrFullAfterRepair(left: *const RoaringBitmap, right: *const RoaringBitmap, allocator: std.mem.Allocator) !u64 {
    var priming_result = try left.lazyOr(allocator, right, true);
    try priming_result.repairAfterLazy();
    priming_result.deinit();
    return rawrFull(left, right, allocator);
}

fn crFullAfterRepair(left: *const c.roaring_bitmap_t, right: *const c.roaring_bitmap_t) !u64 {
    const priming_result = c.roaring_bitmap_lazy_or(left, right, true) orelse return error.OutOfMemory;
    c.roaring_bitmap_repair_after_lazy(priming_result);
    c.roaring_bitmap_free(priming_result);
    return crFull(left, right);
}

fn rawrTopLevelInit(allocator: std.mem.Allocator, capacity: u32) !u64 {
    const batch_size = 64;
    var results: [batch_size]RoaringBitmap = undefined;
    var initialized: usize = 0;
    errdefer for (results[0..initialized]) |*result| result.deinit();
    const start = bench_time.monotonicNanos();
    for (&results) |*result| {
        result.* = try RoaringBitmap.initCapacity(allocator, capacity);
        initialized += 1;
    }
    const elapsed = (bench_time.monotonicNanos() - start) / batch_size;
    std.mem.doNotOptimizeAway(&results);
    for (&results) |*result| result.deinit();
    return elapsed;
}

fn rawrHeaderAlloc(allocator: std.mem.Allocator, count: usize) !u64 {
    const headers = try std.heap.smp_allocator.alloc(*BitsetContainer, count);
    defer std.heap.smp_allocator.free(headers);
    var created: usize = 0;
    errdefer for (headers[0..created]) |header| allocator.destroy(header);

    const start = bench_time.monotonicNanos();
    for (headers) |*header| {
        header.* = try allocator.create(BitsetContainer);
        created += 1;
    }
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(headers.ptr);
    for (headers) |header| allocator.destroy(header);
    return elapsed;
}

fn rawrWordsAlloc(allocator: std.mem.Allocator, count: usize) !u64 {
    const words = try std.heap.smp_allocator.alloc([]align(64) u64, count);
    defer std.heap.smp_allocator.free(words);
    var created: usize = 0;
    errdefer for (words[0..created]) |slice| allocator.free(slice);

    const start = bench_time.monotonicNanos();
    for (words) |*slice| {
        slice.* = try allocator.alignedAlloc(u64, .@"64", BitsetContainer.NUM_WORDS);
        created += 1;
    }
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(words.ptr);
    for (words) |slice| allocator.free(slice);
    return elapsed;
}

fn rawrBitsetCreate(allocator: std.mem.Allocator, count: usize) !u64 {
    const bitsets = try std.heap.smp_allocator.alloc(*BitsetContainer, count);
    defer std.heap.smp_allocator.free(bitsets);
    var created: usize = 0;
    errdefer for (bitsets[0..created]) |bitset| bitset.deinit(allocator);

    const start = bench_time.monotonicNanos();
    for (bitsets) |*bitset| {
        bitset.* = try BitsetContainer.init(allocator);
        created += 1;
    }
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(bitsets.ptr);
    for (bitsets) |bitset| bitset.deinit(allocator);
    return elapsed;
}

fn allocateBitsets(allocator: std.mem.Allocator, count: usize) ![]*BitsetContainer {
    const bitsets = try std.heap.smp_allocator.alloc(*BitsetContainer, count);
    errdefer std.heap.smp_allocator.free(bitsets);
    var created: usize = 0;
    errdefer for (bitsets[0..created]) |bitset| bitset.deinit(allocator);
    for (bitsets) |*bitset| {
        bitset.* = try BitsetContainer.init(allocator);
        created += 1;
    }
    return bitsets;
}

fn freeBitsets(allocator: std.mem.Allocator, bitsets: []*BitsetContainer) void {
    for (bitsets) |bitset| bitset.deinit(allocator);
    std.heap.smp_allocator.free(bitsets);
}

fn rawrZero(allocator: std.mem.Allocator, count: usize) !u64 {
    const bitsets = try allocateBitsets(allocator, count);
    defer freeBitsets(allocator, bitsets);
    for (bitsets) |bitset| @memset(bitset.words, 0xA5A5A5A5A5A5A5A5);

    const start = bench_time.monotonicNanos();
    for (bitsets) |bitset| @memset(bitset.words, 0);
    const elapsed = bench_time.monotonicNanos() - start;
    var checksum: u64 = 0;
    for (bitsets) |bitset| checksum +%= bitset.words[0];
    std.mem.doNotOptimizeAway(checksum);
    return elapsed;
}

fn rawrZeroFresh(allocator: std.mem.Allocator, count: usize) !u64 {
    const words = try std.heap.smp_allocator.alloc([]align(64) u64, count);
    defer std.heap.smp_allocator.free(words);
    var created: usize = 0;
    errdefer for (words[0..created]) |slice| allocator.free(slice);
    for (words) |*slice| {
        slice.* = try allocator.alignedAlloc(u64, .@"64", BitsetContainer.NUM_WORDS);
        created += 1;
    }

    const start = bench_time.monotonicNanos();
    for (words) |slice| @memset(slice, 0);
    const elapsed = bench_time.monotonicNanos() - start;
    var checksum: u64 = 0;
    for (words) |slice| checksum +%= slice[0];
    std.mem.doNotOptimizeAway(checksum);
    for (words) |slice| allocator.free(slice);
    return elapsed;
}

fn rawrAccumulateFirst(allocator: std.mem.Allocator, shared: []const SharedPair) !u64 {
    const bitsets = try allocateBitsets(allocator, shared.len);
    defer freeBitsets(allocator, bitsets);

    const start = bench_time.monotonicNanos();
    for (bitsets, shared) |bitset, pair| bitset.setList(pair.first.values[0..pair.first.cardinality]);
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(bitsets[bitsets.len - 1].words[0]);
    return elapsed;
}

fn rawrAccumulateSecond(allocator: std.mem.Allocator, shared: []const SharedPair) !u64 {
    const bitsets = try allocateBitsets(allocator, shared.len);
    defer freeBitsets(allocator, bitsets);
    for (bitsets, shared) |bitset, pair| bitset.setList(pair.first.values[0..pair.first.cardinality]);

    const start = bench_time.monotonicNanos();
    for (bitsets, shared) |bitset, pair| bitset.setList(pair.second.values[0..pair.second.cardinality]);
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(bitsets[bitsets.len - 1].words[0]);
    return elapsed;
}

fn rawrSharedPipeline(allocator: std.mem.Allocator, shared: []const SharedPair) !u64 {
    const bitsets = try std.heap.smp_allocator.alloc(*BitsetContainer, shared.len);
    defer std.heap.smp_allocator.free(bitsets);
    var created: usize = 0;
    errdefer for (bitsets[0..created]) |bitset| bitset.deinit(allocator);

    const start = bench_time.monotonicNanos();
    for (bitsets, shared) |*bitset, pair| {
        bitset.* = try BitsetContainer.init(allocator);
        created += 1;
        bitset.*.setList(pair.first.values[0..pair.first.cardinality]);
        bitset.*.setList(pair.second.values[0..pair.second.cardinality]);
    }
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(bitsets[bitsets.len - 1].words[0]);
    for (bitsets) |bitset| bitset.deinit(allocator);
    return elapsed;
}

fn rawrCloneUnique(allocator: std.mem.Allocator, unique: []const TaggedPtr) !u64 {
    const clones = try std.heap.smp_allocator.alloc(TaggedPtr, unique.len);
    defer std.heap.smp_allocator.free(clones);
    var cloned: usize = 0;
    errdefer for (clones[0..cloned]) |clone| Container.fromTagged(clone).deinit(allocator);

    const start = bench_time.monotonicNanos();
    for (clones, unique) |*clone, source| {
        clone.* = (try Container.fromTagged(source).clone(allocator)).toTagged();
        cloned += 1;
    }
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(clones.ptr);
    for (clones) |clone| Container.fromTagged(clone).deinit(allocator);
    return elapsed;
}

fn rawrMergeAppend(left: *const RoaringBitmap, right: *const RoaringBitmap) !u64 {
    const max_count: usize = @as(usize, left.size) + right.size;
    const keys = try std.heap.smp_allocator.alloc(u16, max_count);
    defer std.heap.smp_allocator.free(keys);
    const containers = try std.heap.smp_allocator.alloc(TaggedPtr, max_count);
    defer std.heap.smp_allocator.free(containers);

    var i: usize = 0;
    var j: usize = 0;
    var out: usize = 0;
    const start = bench_time.monotonicNanos();
    while (i < left.size and j < right.size) {
        if (left.keys[i] < right.keys[j]) {
            keys[out] = left.keys[i];
            containers[out] = left.containers[i];
            i += 1;
        } else if (left.keys[i] > right.keys[j]) {
            keys[out] = right.keys[j];
            containers[out] = right.containers[j];
            j += 1;
        } else {
            keys[out] = left.keys[i];
            containers[out] = left.containers[i];
            i += 1;
            j += 1;
        }
        out += 1;
    }
    while (i < left.size) : (i += 1) {
        keys[out] = left.keys[i];
        containers[out] = left.containers[i];
        out += 1;
    }
    while (j < right.size) : (j += 1) {
        keys[out] = right.keys[j];
        containers[out] = right.containers[j];
        out += 1;
    }
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(keys[out - 1]);
    std.mem.doNotOptimizeAway(containers[out - 1]);
    return elapsed;
}

fn crHeaders(context: *c.rawr_cr_attr_context) !u64 {
    const start = bench_time.monotonicNanos();
    if (!c.rawr_cr_attr_alloc_headers(context)) return error.OutOfMemory;
    const elapsed = bench_time.monotonicNanos() - start;
    c.rawr_cr_attr_free_headers(context);
    return elapsed;
}

fn crWords(context: *c.rawr_cr_attr_context) !u64 {
    if (!c.rawr_cr_attr_alloc_headers(context)) return error.OutOfMemory;
    defer c.rawr_cr_attr_free_headers(context);
    const start = bench_time.monotonicNanos();
    if (!c.rawr_cr_attr_alloc_words(context)) return error.OutOfMemory;
    const elapsed = bench_time.monotonicNanos() - start;
    c.rawr_cr_attr_free_words(context);
    return elapsed;
}

fn crBitsetCreate(context: *c.rawr_cr_attr_context) !u64 {
    const start = bench_time.monotonicNanos();
    if (!c.rawr_cr_attr_create_bitsets(context)) return error.OutOfMemory;
    const elapsed = bench_time.monotonicNanos() - start;
    c.rawr_cr_attr_free_bitsets(context);
    return elapsed;
}

fn crZero(context: *c.rawr_cr_attr_context) !u64 {
    if (!c.rawr_cr_attr_create_bitsets(context)) return error.OutOfMemory;
    defer c.rawr_cr_attr_free_bitsets(context);
    c.rawr_cr_attr_dirty_words(context);
    const start = bench_time.monotonicNanos();
    c.rawr_cr_attr_zero_words(context);
    return bench_time.monotonicNanos() - start;
}

fn crZeroFresh(context: *c.rawr_cr_attr_context) !u64 {
    if (!c.rawr_cr_attr_alloc_headers(context)) return error.OutOfMemory;
    defer c.rawr_cr_attr_free_headers(context);
    if (!c.rawr_cr_attr_alloc_words(context)) return error.OutOfMemory;
    defer c.rawr_cr_attr_free_words(context);
    const start = bench_time.monotonicNanos();
    c.rawr_cr_attr_zero_words(context);
    return bench_time.monotonicNanos() - start;
}

fn crAccumulateFirst(context: *c.rawr_cr_attr_context) !u64 {
    if (!c.rawr_cr_attr_create_bitsets(context)) return error.OutOfMemory;
    defer c.rawr_cr_attr_free_bitsets(context);
    const start = bench_time.monotonicNanos();
    c.rawr_cr_attr_accumulate_first(context);
    return bench_time.monotonicNanos() - start;
}

fn crAccumulateSecond(context: *c.rawr_cr_attr_context) !u64 {
    if (!c.rawr_cr_attr_create_bitsets(context)) return error.OutOfMemory;
    defer c.rawr_cr_attr_free_bitsets(context);
    c.rawr_cr_attr_accumulate_first(context);
    const start = bench_time.monotonicNanos();
    c.rawr_cr_attr_accumulate_second(context);
    return bench_time.monotonicNanos() - start;
}

fn crSharedPipeline(context: *c.rawr_cr_attr_context) !u64 {
    const start = bench_time.monotonicNanos();
    if (!c.rawr_cr_attr_shared_pipeline(context)) return error.OutOfMemory;
    const elapsed = bench_time.monotonicNanos() - start;
    c.rawr_cr_attr_free_bitsets(context);
    return elapsed;
}

fn crCloneUnique(context: *c.rawr_cr_attr_context) !u64 {
    const start = bench_time.monotonicNanos();
    if (!c.rawr_cr_attr_clone_unique(context)) return error.OutOfMemory;
    const elapsed = bench_time.monotonicNanos() - start;
    c.rawr_cr_attr_free_clones(context);
    return elapsed;
}

fn crMergeAppend(context: *c.rawr_cr_attr_context) !u64 {
    const start = bench_time.monotonicNanos();
    const checksum = c.rawr_cr_attr_merge_append(context);
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(checksum);
    return elapsed;
}

fn validateReplica(
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
    cr_left: *const c.roaring_bitmap_t,
    cr_right: *const c.roaring_bitmap_t,
) !void {
    var smp_result = try left.lazyOr(std.heap.smp_allocator, right, true);
    defer smp_result.deinit();
    try smp_result.repairAfterLazy();

    var libc_result = try left.lazyOr(bench_time.cAllocator(), right, true);
    defer libc_result.deinit();
    try libc_result.repairAfterLazy();
    if (!smp_result.equals(&libc_result)) return error.RawrAllocatorMismatch;

    const cr_result = c.roaring_bitmap_lazy_or(cr_left, cr_right, true) orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(cr_result);
    c.roaring_bitmap_repair_after_lazy(cr_result);

    const rawr_bytes = try smp_result.serialize(std.heap.smp_allocator);
    defer std.heap.smp_allocator.free(rawr_bytes);
    const cr_size = c.roaring_bitmap_portable_size_in_bytes(cr_result);
    if (rawr_bytes.len != cr_size) return error.SerializedSizeMismatch;
    const cr_bytes = try std.heap.smp_allocator.alloc(u8, cr_size);
    defer std.heap.smp_allocator.free(cr_bytes);
    if (c.roaring_bitmap_portable_serialize(cr_result, @ptrCast(cr_bytes.ptr)) != cr_size) {
        return error.SerializedSizeMismatch;
    }
    if (!std.mem.eql(u8, rawr_bytes, cr_bytes)) return error.CRoaringMismatch;
}

fn runComponent(comptime operation: anytype, args: anytype, component: []const u8, variant: []const u8) !void {
    printResult(component, variant, try measure(operation, args));
}

fn measurePrototype(
    comptime variant: PrototypeVariant,
    phase: PrototypePhase,
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
) !Stats {
    for (0..WARMUP_RUNS) |_| _ = try prototypeTimed(variant, phase, left, right);
    var times: [TIMED_RUNS]u64 = undefined;
    for (&times) |*time| time.* = try prototypeTimed(variant, phase, left, right);
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return .{
        .median_ns = times[TIMED_RUNS / 2],
        .min_ns = times[0],
        .max_ns = times[TIMED_RUNS - 1],
    };
}

fn runPrototypeCorpus(
    corpus: CorpusKind,
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
) !void {
    inline for (std.meta.tags(PrototypePhase)) |phase| {
        const component = switch (corpus) {
            .sparse => switch (phase) {
                .construction => "prototype-sparse-construction",
                .repair => "prototype-sparse-repair",
                .combined => "prototype-sparse-combined",
            },
            .dense_survivor => switch (phase) {
                .construction => "prototype-dense-construction",
                .repair => "prototype-dense-repair",
                .combined => "prototype-dense-combined",
            },
        };
        printResult(component, "headered", try measurePrototype(.headered, phase, left, right));
        printResult(component, "headerless", try measurePrototype(.headerless, phase, left, right));
    }
}

export fn rawr_lazy_attr_zero_probe(words: *align(64) [BitsetContainer.NUM_WORDS]u64) callconv(.c) void {
    @memset(words, 0);
}

export fn rawr_lazy_attr_accumulate_probe(
    words: *align(64) [BitsetContainer.NUM_WORDS]u64,
    values: [*]const u16,
    count: usize,
) callconv(.c) void {
    for (values[0..count]) |value| words[value >> 6] |= @as(u64, 1) << @as(u6, @truncate(value));
}

fn preserveCodegenProbes() void {
    var words: [BitsetContainer.NUM_WORDS]u64 align(64) = undefined;
    c.rawr_cr_attr_call_zig_probes(&words, @ptrCast(&words), 0);
    std.mem.doNotOptimizeAway(&words);
}

pub fn main() !void {
    bench_time.print("Lazy-OR construction attribution\n", .{});
    bench_time.print("================================\n", .{});
    bench_time.printRunTimestamp();
    bench_time.printBenchEnvironment();
    bench_time.print("N={d}, warmup={d}, timed={d}\n\n", .{ N_VALUES, WARMUP_RUNS, TIMED_RUNS });

    const sparse_len = initValues();
    var rawr_inputs = try buildRawrInputs(std.heap.smp_allocator, sparse_values[0..sparse_len]);
    defer rawr_inputs[0].deinit();
    defer rawr_inputs[1].deinit();
    const cr_inputs = try buildCrInputs(sparse_values[0..sparse_len]);
    defer c.roaring_bitmap_free(cr_inputs[0]);
    defer c.roaring_bitmap_free(cr_inputs[1]);

    var corpus = try captureCorpus(std.heap.smp_allocator, &rawr_inputs[0], &rawr_inputs[1]);
    defer corpus.deinit(std.heap.smp_allocator);
    bench_time.print("COUNT\tsparse_values\t{d}\n", .{sparse_len});
    try validateSparseCounts(sparse_len, &rawr_inputs[0], &rawr_inputs[1], &corpus);
    const cr_context = c.rawr_cr_attr_context_create(cr_inputs[0], cr_inputs[1]) orelse return error.OutOfMemory;
    defer c.rawr_cr_attr_context_free(cr_context);
    const cr_counts = c.rawr_cr_attr_get_counts(cr_context);
    var cr_materialization: c.rawr_cr_attr_materialization_counts = undefined;
    if (!c.rawr_cr_attr_get_materialization_counts(cr_context, &cr_materialization)) return error.OutOfMemory;

    if (cr_counts.shared_keys != corpus.shared.len or
        cr_counts.left_only_keys != corpus.left_only or
        cr_counts.right_only_keys != corpus.right_only or
        cr_counts.non_array_shared_keys != 0)
    {
        return error.AttributionCorpusMismatch;
    }
    if (cr_counts.bitsets_created != corpus.shared.len or
        cr_materialization.before_bitset != corpus.shared.len or
        cr_materialization.before_array != corpus.left_only + corpus.right_only or
        cr_materialization.before_run != 0 or
        cr_materialization.after_array != corpus.shared.len + corpus.left_only + corpus.right_only or
        cr_materialization.after_bitset != 0 or
        cr_materialization.after_run != 0)
    {
        return error.CRoaringMaterializationMismatch;
    }
    try validateReplica(&rawr_inputs[0], &rawr_inputs[1], cr_inputs[0], cr_inputs[1]);

    var sparse_expected = try rawr_inputs[0].lazyOr(std.heap.smp_allocator, &rawr_inputs[1], true);
    defer sparse_expected.deinit();
    try sparse_expected.repairAfterLazy();
    const sparse_lifecycle = try validatePrototypePair(&rawr_inputs[0], &rawr_inputs[1], &sparse_expected);
    if (sparse_lifecycle.demoted != EXPECTED_SHARED_KEYS or
        sparse_lifecycle.surviving != 0 or
        sparse_lifecycle.empty != 0 or
        sparse_lifecycle.min_cardinality != EXPECTED_SPARSE_MIN_CARDINALITY or
        sparse_lifecycle.max_cardinality != EXPECTED_SPARSE_MAX_CARDINALITY or
        sparse_lifecycle.total_cardinality != EXPECTED_SPARSE_TOTAL_CARDINALITY or
        sparse_lifecycle.cardinality_fingerprint != EXPECTED_SPARSE_CARDINALITY_FINGERPRINT)
    {
        return error.SparseLifecycleMismatch;
    }

    var dense_inputs = try buildDenseSurvivorInputs(std.heap.smp_allocator);
    defer dense_inputs[0].deinit();
    defer dense_inputs[1].deinit();
    var dense_expected = try dense_inputs[0].lazyOr(std.heap.smp_allocator, &dense_inputs[1], true);
    defer dense_expected.deinit();
    try dense_expected.repairAfterLazy();
    const dense_lifecycle = try validatePrototypePair(&dense_inputs[0], &dense_inputs[1], &dense_expected);
    if (dense_lifecycle.demoted != 0 or
        dense_lifecycle.surviving != DENSE_SHARED_KEYS or
        dense_lifecycle.empty != 0 or
        dense_lifecycle.min_cardinality != 8193 or
        dense_lifecycle.max_cardinality != 8193 or
        dense_lifecycle.total_cardinality != @as(u64, DENSE_SHARED_KEYS) * 8193)
    {
        return error.DenseLifecycleMismatch;
    }

    const sparse_headered_accounting = try accountPrototype(.headered, &rawr_inputs[0], &rawr_inputs[1]);
    const sparse_headerless_accounting = try accountPrototype(.headerless, &rawr_inputs[0], &rawr_inputs[1]);
    const dense_headered_accounting = try accountPrototype(.headered, &dense_inputs[0], &dense_inputs[1]);
    const dense_headerless_accounting = try accountPrototype(.headerless, &dense_inputs[0], &dense_inputs[1]);
    if (sparse_headered_accounting.construction.alloc_calls != 130_994 or
        sparse_headerless_accounting.construction.alloc_calls != 114_630 or
        sparse_headered_accounting.construction.alloc_calls - sparse_headerless_accounting.construction.alloc_calls != EXPECTED_SHARED_KEYS or
        sparse_headered_accounting.combined.alloc_calls - sparse_headerless_accounting.combined.alloc_calls != EXPECTED_SHARED_KEYS or
        dense_headered_accounting.construction.alloc_calls - dense_headerless_accounting.construction.alloc_calls != DENSE_SHARED_KEYS or
        dense_headered_accounting.combined.alloc_calls != dense_headerless_accounting.combined.alloc_calls)
    {
        return error.PrototypeAccountingMismatch;
    }
    preserveCodegenProbes();

    bench_time.print("COUNT\tleft_keys\t{d}\n", .{rawr_inputs[0].size});
    bench_time.print("COUNT\tright_keys\t{d}\n", .{rawr_inputs[1].size});
    bench_time.print("COUNT\tshared_keys\t{d}\n", .{corpus.shared.len});
    bench_time.print("COUNT\tleft_only_keys\t{d}\n", .{corpus.left_only});
    bench_time.print("COUNT\tright_only_keys\t{d}\n", .{corpus.right_only});
    bench_time.print("COUNT\tbitsets_created\t{d}\n", .{cr_counts.bitsets_created});
    bench_time.print("COUNT\ttransient_header_calls\t{d}\n", .{corpus.shared.len});
    bench_time.print("COUNT\ttransient_words_calls\t{d}\n", .{corpus.shared.len});
    bench_time.print("COUNT\ttransient_total_calls\t{d}\n", .{corpus.shared.len * 2});
    bench_time.print("COUNT\tbytes_cleared\t{d}\n", .{cr_counts.bytes_cleared});
    bench_time.print("MATERIALIZATION\tcroaring-before\tarray={d}\tbitset={d}\trun={d}\n", .{
        cr_materialization.before_array,
        cr_materialization.before_bitset,
        cr_materialization.before_run,
    });
    bench_time.print("MATERIALIZATION\tcroaring-after\tarray={d}\tbitset={d}\trun={d}\n", .{
        cr_materialization.after_array,
        cr_materialization.after_bitset,
        cr_materialization.after_run,
    });
    printLifecycle(.sparse, sparse_lifecycle);
    printLifecycle(.dense_survivor, dense_lifecycle);
    printAccounting(.sparse, .headered, sparse_headered_accounting);
    printAccounting(.sparse, .headerless, sparse_headerless_accounting);
    printAccounting(.dense_survivor, .headered, dense_headered_accounting);
    printAccounting(.dense_survivor, .headerless, dense_headerless_accounting);
    bench_time.print("ACCOUNT_DELTA\tsparse\tconstruction_allocs={d}\tcombined_allocs={d}\theaders_eliminated={d}\theaders_deferred=0\n", .{
        sparse_headered_accounting.construction.alloc_calls - sparse_headerless_accounting.construction.alloc_calls,
        sparse_headered_accounting.combined.alloc_calls - sparse_headerless_accounting.combined.alloc_calls,
        sparse_lifecycle.demoted,
    });
    bench_time.print("ACCOUNT_DELTA\tdense_survivor\tconstruction_allocs={d}\tcombined_allocs={d}\theaders_eliminated=0\theaders_deferred={d}\n", .{
        dense_headered_accounting.construction.alloc_calls - dense_headerless_accounting.construction.alloc_calls,
        dense_headered_accounting.combined.alloc_calls - dense_headerless_accounting.combined.alloc_calls,
        dense_lifecycle.surviving,
    });
    bench_time.print("VALIDATION\theadered=headerless=production=rawr-libc=croaring-portable\n\n", .{});

    bench_time.print("{s:<20} {s:<12} {s:>10} {s:>10} {s:>10}\n", .{ "component", "variant", "median ms", "min ms", "max ms" });
    bench_time.print("{s:-<20} {s:-<12} {s:->10} {s:->10} {s:->10}\n", .{ "", "", "", "", "" });

    const libc = bench_time.cAllocator();
    try runComponent(rawrFull, .{ &rawr_inputs[0], &rawr_inputs[1], std.heap.smp_allocator }, "full", "rawr-smp");
    try runComponent(rawrFull, .{ &rawr_inputs[0], &rawr_inputs[1], libc }, "full", "rawr-libc");
    try runComponent(crFull, .{ cr_inputs[0], cr_inputs[1] }, "full", "croaring");

    try runComponent(productionRepair, .{ &rawr_inputs[0], &rawr_inputs[1], std.heap.smp_allocator }, "canonical-repair", "rawr-smp");
    try runComponent(crRepair, .{ cr_inputs[0], cr_inputs[1] }, "canonical-repair", "croaring");
    try runComponent(productionCombined, .{ &rawr_inputs[0], &rawr_inputs[1], std.heap.smp_allocator }, "canonical-combined", "rawr-smp");
    try runComponent(crCombined, .{ cr_inputs[0], cr_inputs[1] }, "canonical-combined", "croaring");

    try runPrototypeCorpus(.sparse, &rawr_inputs[0], &rawr_inputs[1]);
    try runPrototypeCorpus(.dense_survivor, &dense_inputs[0], &dense_inputs[1]);

    try runComponent(rawrFullAfterRepair, .{ &rawr_inputs[0], &rawr_inputs[1], std.heap.smp_allocator }, "full-after-repair", "rawr-smp");
    try runComponent(rawrFullAfterRepair, .{ &rawr_inputs[0], &rawr_inputs[1], libc }, "full-after-repair", "rawr-libc");
    try runComponent(crFullAfterRepair, .{ cr_inputs[0], cr_inputs[1] }, "full-after-repair", "croaring");

    try runComponent(rawrSharedPipeline, .{ std.heap.smp_allocator, corpus.shared }, "shared-pipeline", "rawr-smp");
    try runComponent(rawrSharedPipeline, .{ libc, corpus.shared }, "shared-pipeline", "rawr-libc");
    try runComponent(crSharedPipeline, .{cr_context}, "shared-pipeline", "croaring");

    const max_result_size = @min(rawr_inputs[0].size + rawr_inputs[1].size, @as(u32, 1) << 16);
    try runComponent(rawrTopLevelInit, .{ std.heap.smp_allocator, max_result_size }, "top-level-init", "rawr-smp");
    try runComponent(rawrTopLevelInit, .{ libc, max_result_size }, "top-level-init", "rawr-libc");

    try runComponent(rawrHeaderAlloc, .{ std.heap.smp_allocator, corpus.shared.len }, "header-alloc", "rawr-smp");
    try runComponent(rawrHeaderAlloc, .{ libc, corpus.shared.len }, "header-alloc", "rawr-libc");
    try runComponent(crHeaders, .{cr_context}, "header-alloc", "croaring");

    try runComponent(rawrWordsAlloc, .{ std.heap.smp_allocator, corpus.shared.len }, "words-alloc", "rawr-smp");
    try runComponent(rawrWordsAlloc, .{ libc, corpus.shared.len }, "words-alloc", "rawr-libc");
    try runComponent(crWords, .{cr_context}, "words-alloc", "croaring");

    try runComponent(rawrBitsetCreate, .{ std.heap.smp_allocator, corpus.shared.len }, "bitset-create", "rawr-smp");
    try runComponent(rawrBitsetCreate, .{ libc, corpus.shared.len }, "bitset-create", "rawr-libc");
    try runComponent(crBitsetCreate, .{cr_context}, "bitset-create", "croaring");

    try runComponent(rawrZeroFresh, .{ std.heap.smp_allocator, corpus.shared.len }, "zero-fresh", "rawr-smp");
    try runComponent(rawrZeroFresh, .{ libc, corpus.shared.len }, "zero-fresh", "rawr-libc");
    try runComponent(crZeroFresh, .{cr_context}, "zero-fresh", "croaring");

    try runComponent(rawrZero, .{ std.heap.smp_allocator, corpus.shared.len }, "zero-dirty", "rawr-smp");
    try runComponent(rawrZero, .{ libc, corpus.shared.len }, "zero-dirty", "rawr-libc");
    try runComponent(crZero, .{cr_context}, "zero-dirty", "croaring");

    try runComponent(rawrAccumulateFirst, .{ std.heap.smp_allocator, corpus.shared }, "accumulate-first", "rawr-smp");
    try runComponent(rawrAccumulateFirst, .{ libc, corpus.shared }, "accumulate-first", "rawr-libc");
    try runComponent(crAccumulateFirst, .{cr_context}, "accumulate-first", "croaring");

    try runComponent(rawrAccumulateSecond, .{ std.heap.smp_allocator, corpus.shared }, "accumulate-second", "rawr-smp");
    try runComponent(rawrAccumulateSecond, .{ libc, corpus.shared }, "accumulate-second", "rawr-libc");
    try runComponent(crAccumulateSecond, .{cr_context}, "accumulate-second", "croaring");

    try runComponent(rawrCloneUnique, .{ std.heap.smp_allocator, corpus.unique }, "clone-unique", "rawr-smp");
    try runComponent(rawrCloneUnique, .{ libc, corpus.unique }, "clone-unique", "rawr-libc");
    try runComponent(crCloneUnique, .{cr_context}, "clone-unique", "croaring");

    try runComponent(rawrMergeAppend, .{ &rawr_inputs[0], &rawr_inputs[1] }, "merge-append", "rawr-smp");
    try runComponent(rawrMergeAppend, .{ &rawr_inputs[0], &rawr_inputs[1] }, "merge-append", "rawr-libc");
    try runComponent(crMergeAppend, .{cr_context}, "merge-append", "croaring");
}
