// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const rawr = @import("rawr");
const bench_time = @import("bench_time.zig");
const CountingAllocator = @import("counting_allocator.zig").CountingAllocator;

const RoaringBitmap = rawr.RoaringBitmap;
const ArrayContainer = rawr.ArrayContainer;
const BitsetContainer = rawr.BitsetContainer;
const RunContainer = rawr.RunContainer;
const Container = rawr.Container;
const TaggedPtr = rawr.TaggedPtr;
const container_ops = rawr.container_ops;

const UNMATCHED_PCTS = [_]u8{ 0, 25, 50, 75, 100 };
const SWEEP_KEYS: usize = 192;
const FIXPOINT_INITIAL_KEYS: usize = 192;
const FIXPOINT_DELTA_KEYS: usize = 64;
const FIXPOINT_ROUNDS: usize = 16;
const SELF_ONLY_KEY_BASE: u16 = 20_000;
const FIXPOINT_NEW_KEY_BASE: u16 = 30_000;
const WARMUP_RUNS = 2;
const TIMED_RUNS = 9;

const Variant = enum {
    baseline,
    consuming,
};

const Attribution = struct {
    total_allocs: u64 = 0,
    index_allocs: u64 = 0,
    matched_allocs: u64 = 0,
    clone_allocs: u64 = 0,
    moved_containers: u64 = 0,

    fn add(self: *Attribution, other: Attribution) void {
        self.total_allocs += other.total_allocs;
        self.index_allocs += other.index_allocs;
        self.matched_allocs += other.matched_allocs;
        self.clone_allocs += other.clone_allocs;
        self.moved_containers += other.moved_containers;
    }

    fn validate(self: Attribution, variant: Variant) !void {
        if (self.total_allocs != self.index_allocs + self.matched_allocs + self.clone_allocs) {
            return error.AllocationAttributionMismatch;
        }
        switch (variant) {
            .baseline => {
                if (self.clone_allocs != 2 * self.moved_containers) {
                    return error.BaselineCloneCountMismatch;
                }
            },
            .consuming => {
                if (self.clone_allocs != 0) return error.ConsumingCloneCountMismatch;
            },
        }
    }
};

const Timing = struct {
    union_ns: u64,
    lifecycle_ns: u64,
};

const BitmapPair = struct {
    left: RoaringBitmap,
    right: RoaringBitmap,

    fn deinit(self: *BitmapPair) void {
        self.right.deinit();
        self.left.deinit();
    }
};

fn allocDelta(counting: *const CountingAllocator, before: u64) u64 {
    return counting.stats.alloc_calls - before;
}

fn taggedEqual(a: TaggedPtr, b: TaggedPtr) bool {
    return a.eql(b);
}

fn unmatchedRightCount(left: *const RoaringBitmap, right: *const RoaringBitmap) u32 {
    var i: usize = 0;
    var j: usize = 0;
    var unmatched: u32 = 0;
    while (i < left.size and j < right.size) {
        if (left.keys[i] < right.keys[j]) {
            i += 1;
        } else if (left.keys[i] > right.keys[j]) {
            unmatched += 1;
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    unmatched += right.size - @as(u32, @intCast(j));
    return unmatched;
}

fn consumingOrMeasured(
    self: *RoaringBitmap,
    other: *RoaringBitmap,
    counting: *CountingAllocator,
    attribution: *Attribution,
) !void {
    return consumingOrImpl(true, self, other, counting, attribution);
}

fn consumingOrImpl(
    comptime measured: bool,
    self: *RoaringBitmap,
    other: *RoaringBitmap,
    counting: if (measured) *CountingAllocator else void,
    attribution: if (measured) *Attribution else void,
) !void {
    if (self.allocator.ptr != other.allocator.ptr or self.allocator.vtable != other.allocator.vtable) {
        return error.AllocatorMismatch;
    }
    if (self == other) return error.AliasedOperands;

    const old_self_size = self.size;
    const unmatched = unmatchedRightCount(self, other);
    const output_size = old_self_size + unmatched;

    const reserve_before = if (measured) counting.stats.alloc_calls else 0;
    try self.ensureTotalCapacity(output_size);
    if (measured) attribution.index_allocs += allocDelta(counting, reserve_before);

    // This is unconditional: even an unmatched-only commit changes self.
    self.cached_cardinality = -1;

    var i: usize = 0;
    var j: usize = 0;
    while (i < old_self_size and j < other.size) {
        const key_a = self.keys[i];
        const key_b = other.keys[j];
        if (key_a < key_b) {
            i += 1;
        } else if (key_a > key_b) {
            j += 1;
        } else {
            const matched_before = if (measured) counting.stats.alloc_calls else 0;
            const old_container = Container.fromTagged(self.containers[i]);
            const result = try container_ops.containerUnionInPlace(
                self.allocator,
                old_container,
                Container.fromTagged(other.containers[j]),
            );
            if (measured) attribution.matched_allocs += allocDelta(counting, matched_before);

            const result_tp = result.toTagged();
            if (!taggedEqual(result_tp, self.containers[i])) {
                old_container.deinit(self.allocator);
                self.containers[i] = result_tp;
            }
            i += 1;
            j += 1;
        }
    }

    // No fallible work occurs after this point.
    i = 0;
    j = 0;
    while (i < old_self_size and j < other.size) {
        if (self.keys[i] < other.keys[j]) {
            i += 1;
        } else if (self.keys[i] > other.keys[j]) {
            j += 1;
        } else {
            Container.fromTagged(other.containers[j]).deinit(other.allocator);
            i += 1;
            j += 1;
        }
    }

    var left_tail: usize = old_self_size;
    var right_tail: usize = other.size;
    var out_tail: usize = output_size;
    while (left_tail > 0 and right_tail > 0) {
        const key_a = self.keys[left_tail - 1];
        const key_b = other.keys[right_tail - 1];
        out_tail -= 1;
        if (key_a > key_b) {
            left_tail -= 1;
            self.keys[out_tail] = key_a;
            self.containers[out_tail] = self.containers[left_tail];
        } else if (key_a < key_b) {
            right_tail -= 1;
            self.keys[out_tail] = key_b;
            self.containers[out_tail] = other.containers[right_tail];
            if (measured) attribution.moved_containers += 1;
        } else {
            left_tail -= 1;
            right_tail -= 1;
            self.keys[out_tail] = key_a;
            self.containers[out_tail] = self.containers[left_tail];
        }
    }
    while (left_tail > 0) {
        left_tail -= 1;
        out_tail -= 1;
        self.keys[out_tail] = self.keys[left_tail];
        self.containers[out_tail] = self.containers[left_tail];
    }
    while (right_tail > 0) {
        right_tail -= 1;
        out_tail -= 1;
        self.keys[out_tail] = other.keys[right_tail];
        self.containers[out_tail] = other.containers[right_tail];
        if (measured) attribution.moved_containers += 1;
    }
    std.debug.assert(out_tail == 0);

    self.size = output_size;
    other.size = 0;
    other.cached_cardinality = 0;

    if (measured) {
        attribution.clone_allocs = 0;
        attribution.total_allocs = counting.stats.alloc_calls;
    }
}

// Allocation-attributed replica of the current eager implementation. Timing always calls
// RoaringBitmap.bitwiseOrInPlace directly; this copy exists only because the allocator
// cannot otherwise identify which production call site requested an allocation.
fn baselineOrMeasured(
    self: *RoaringBitmap,
    other: *const RoaringBitmap,
    counting: *CountingAllocator,
    attribution: *Attribution,
) !void {
    if (other.size == 0) return;
    self.cached_cardinality = -1;

    const max_size = self.size + other.size;
    var before = counting.stats.alloc_calls;
    const new_keys = try self.allocator.alloc(u16, max_size);
    errdefer self.allocator.free(new_keys);
    const new_containers = try self.allocator.alloc(TaggedPtr, max_size);
    errdefer self.allocator.free(new_containers);
    const owned = try self.allocator.alloc(bool, max_size);
    defer self.allocator.free(owned);
    attribution.index_allocs += allocDelta(counting, before);

    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;
    errdefer {
        for (new_containers[0..k], owned[0..k]) |tp, is_owned| {
            if (is_owned) Container.fromTagged(tp).deinit(self.allocator);
        }
    }

    while (i < self.size and j < other.size) {
        const key_a = self.keys[i];
        const key_b = other.keys[j];
        if (key_a < key_b) {
            new_keys[k] = key_a;
            new_containers[k] = self.containers[i];
            owned[k] = false;
            i += 1;
        } else if (key_a > key_b) {
            before = counting.stats.alloc_calls;
            const cloned = try Container.fromTagged(other.containers[j]).clone(self.allocator);
            attribution.clone_allocs += allocDelta(counting, before);
            attribution.moved_containers += 1;
            new_keys[k] = key_b;
            new_containers[k] = cloned.toTagged();
            owned[k] = true;
            j += 1;
        } else {
            const old_container = Container.fromTagged(self.containers[i]);
            before = counting.stats.alloc_calls;
            const result = try container_ops.containerUnionInPlace(
                self.allocator,
                old_container,
                Container.fromTagged(other.containers[j]),
            );
            attribution.matched_allocs += allocDelta(counting, before);
            const result_tp = result.toTagged();
            const is_same = taggedEqual(result_tp, self.containers[i]);
            if (!is_same) old_container.deinit(self.allocator);
            new_keys[k] = key_a;
            new_containers[k] = result_tp;
            owned[k] = !is_same;
            i += 1;
            j += 1;
        }
        k += 1;
    }
    while (i < self.size) : (i += 1) {
        new_keys[k] = self.keys[i];
        new_containers[k] = self.containers[i];
        owned[k] = false;
        k += 1;
    }
    while (j < other.size) : (j += 1) {
        before = counting.stats.alloc_calls;
        const cloned = try Container.fromTagged(other.containers[j]).clone(self.allocator);
        attribution.clone_allocs += allocDelta(counting, before);
        attribution.moved_containers += 1;
        new_keys[k] = other.keys[j];
        new_containers[k] = cloned.toTagged();
        owned[k] = true;
        k += 1;
    }

    self.allocator.free(self.keys[0..self.capacity]);
    self.allocator.free(self.containers[0..self.capacity]);
    before = counting.stats.alloc_calls;
    if (k < max_size) {
        self.keys = self.allocator.realloc(new_keys, k) catch new_keys;
        self.containers = self.allocator.realloc(new_containers, k) catch new_containers;
        self.capacity = @intCast(k);
    } else {
        self.keys = new_keys;
        self.containers = new_containers;
        self.capacity = @intCast(max_size);
    }
    attribution.index_allocs += allocDelta(counting, before);
    self.size = @intCast(k);
    attribution.total_allocs = counting.stats.alloc_calls;
}

fn makeContainer(allocator: std.mem.Allocator, key: u16, salt: u16) !TaggedPtr {
    return switch (key % 3) {
        0 => blk: {
            const array = try ArrayContainer.init(allocator, 64);
            errdefer array.deinit(allocator);
            const base: u16 = (salt % 64) * 128;
            for (0..64) |idx| array.values[idx] = base + @as(u16, @intCast(idx * 2));
            array.cardinality = 64;
            break :blk TaggedPtr.initArray(array);
        },
        1 => blk: {
            const bitset = try BitsetContainer.init(allocator);
            const shift: u6 = @truncate(salt *% 7);
            for (bitset.words, 0..) |*word, idx| {
                const pattern = if ((idx + salt) % 2 == 0)
                    @as(u64, 0x5555_5555_5555_5555)
                else
                    @as(u64, 0x3333_3333_3333_3333);
                word.* = std.math.rotl(u64, pattern, shift);
            }
            bitset.cardinality = -1;
            _ = bitset.computeCardinality();
            break :blk TaggedPtr.initBitset(bitset);
        },
        else => blk: {
            const run = try RunContainer.init(allocator, 4);
            errdefer run.deinit(allocator);
            const base: u16 = (salt % 64) * 128;
            for (0..4) |idx| {
                const start = base + @as(u16, @intCast(idx * 256));
                _ = try run.addRange(allocator, start, start + 95);
            }
            _ = run.getCardinality();
            break :blk TaggedPtr.initRun(run);
        },
    };
}

fn makeBitmap(allocator: std.mem.Allocator, keys: []const u16, salt: u16) !RoaringBitmap {
    var bitmap = try RoaringBitmap.initCapacity(allocator, @intCast(keys.len));
    errdefer bitmap.deinit();
    for (keys, 0..) |key, idx| {
        bitmap.keys[idx] = key;
        bitmap.containers[idx] = try makeContainer(allocator, key, salt);
        bitmap.size += 1;
    }
    bitmap.cached_cardinality = -1;
    try bitmap.validate();
    return bitmap;
}

fn makeSweepPair(allocator: std.mem.Allocator, unmatched_percent: u8) !BitmapPair {
    const unmatched = SWEEP_KEYS * unmatched_percent / 100;
    const shared = SWEEP_KEYS - unmatched;
    var left_keys: [SWEEP_KEYS]u16 = undefined;
    var right_keys: [SWEEP_KEYS]u16 = undefined;
    for (0..shared) |idx| left_keys[idx] = @intCast(idx);
    for (0..unmatched) |idx| left_keys[shared + idx] = SELF_ONLY_KEY_BASE + @as(u16, @intCast(idx));
    for (0..SWEEP_KEYS) |idx| right_keys[idx] = @intCast(idx);

    var left = try makeBitmap(allocator, &left_keys, 1);
    errdefer left.deinit();
    const right = try makeBitmap(allocator, &right_keys, 2);
    return .{ .left = left, .right = right };
}

fn initialFixpoint(allocator: std.mem.Allocator) !RoaringBitmap {
    var keys: [FIXPOINT_INITIAL_KEYS]u16 = undefined;
    for (&keys, 0..) |*key, idx| key.* = @intCast(idx);
    return makeBitmap(allocator, &keys, 0);
}

fn makeFixpointDelta(
    allocator: std.mem.Allocator,
    accumulator: *const RoaringBitmap,
    unmatched_percent: u8,
    round: usize,
    next_key: *u16,
) !RoaringBitmap {
    const unmatched = FIXPOINT_DELTA_KEYS * unmatched_percent / 100;
    const matched = FIXPOINT_DELTA_KEYS - unmatched;
    var keys: [FIXPOINT_DELTA_KEYS]u16 = undefined;
    if (matched > 0) {
        const max_start = @as(usize, accumulator.size) - matched;
        const start = (round * 17) % (max_start + 1);
        @memcpy(keys[0..matched], accumulator.keys[start .. start + matched]);
    }
    for (0..unmatched) |idx| keys[matched + idx] = next_key.* + @as(u16, @intCast(idx));
    next_key.* += @intCast(unmatched);
    std.mem.sort(u16, &keys, {}, std.sort.asc(u16));
    return makeBitmap(allocator, &keys, @intCast(round + 1));
}

fn applyVariant(variant: Variant, left: *RoaringBitmap, right: *RoaringBitmap) !void {
    switch (variant) {
        .baseline => try left.bitwiseOrInPlace(right),
        .consuming => try left.bitwiseOrInPlaceConsume(right),
    }
}

fn median(values: *[TIMED_RUNS]u64) u64 {
    std.mem.sort(u64, values, {}, std.sort.asc(u64));
    return values[TIMED_RUNS / 2];
}

fn measureSweepTiming(variant: Variant, unmatched_percent: u8) !Timing {
    var union_samples: [TIMED_RUNS]u64 = undefined;
    var lifecycle_samples: [TIMED_RUNS]u64 = undefined;
    for (0..WARMUP_RUNS + TIMED_RUNS) |sample| {
        const lifecycle_start = bench_time.monotonicNanos();
        var pair = try makeSweepPair(std.heap.smp_allocator, unmatched_percent);
        const union_start = bench_time.monotonicNanos();
        try applyVariant(variant, &pair.left, &pair.right);
        const union_elapsed = bench_time.monotonicNanos() - union_start;
        pair.deinit();
        const lifecycle_elapsed = bench_time.monotonicNanos() - lifecycle_start;
        if (sample >= WARMUP_RUNS) {
            union_samples[sample - WARMUP_RUNS] = union_elapsed;
            lifecycle_samples[sample - WARMUP_RUNS] = lifecycle_elapsed;
        }
    }
    return .{ .union_ns = median(&union_samples), .lifecycle_ns = median(&lifecycle_samples) };
}

fn measureFixpointTiming(variant: Variant, unmatched_percent: u8) !Timing {
    var union_samples: [TIMED_RUNS]u64 = undefined;
    var lifecycle_samples: [TIMED_RUNS]u64 = undefined;
    for (0..WARMUP_RUNS + TIMED_RUNS) |sample| {
        var accumulator = try initialFixpoint(std.heap.smp_allocator);
        var next_key: u16 = FIXPOINT_NEW_KEY_BASE;
        var union_total: u64 = 0;
        var lifecycle_total: u64 = 0;
        for (0..FIXPOINT_ROUNDS) |round| {
            const lifecycle_start = bench_time.monotonicNanos();
            var delta = try makeFixpointDelta(
                std.heap.smp_allocator,
                &accumulator,
                unmatched_percent,
                round,
                &next_key,
            );
            const union_start = bench_time.monotonicNanos();
            try applyVariant(variant, &accumulator, &delta);
            union_total += bench_time.monotonicNanos() - union_start;
            delta.deinit();
            lifecycle_total += bench_time.monotonicNanos() - lifecycle_start;
        }
        accumulator.deinit();
        if (sample >= WARMUP_RUNS) {
            union_samples[sample - WARMUP_RUNS] = union_total;
            lifecycle_samples[sample - WARMUP_RUNS] = lifecycle_total;
        }
    }
    return .{ .union_ns = median(&union_samples), .lifecycle_ns = median(&lifecycle_samples) };
}

fn measureSweepAlloc(variant: Variant, unmatched_percent: u8) !Attribution {
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const allocator = counting.allocator();
    var pair = try makeSweepPair(allocator, unmatched_percent);
    counting.resetStats();
    var attribution = Attribution{};
    switch (variant) {
        .baseline => try baselineOrMeasured(&pair.left, &pair.right, &counting, &attribution),
        .consuming => try consumingOrMeasured(&pair.left, &pair.right, &counting, &attribution),
    }
    attribution.total_allocs = counting.stats.alloc_calls;
    try attribution.validate(variant);
    pair.deinit();
    if (counting.stats.live_bytes != 0) return error.CountingAllocatorLeak;
    return attribution;
}

fn measureFixpointAlloc(variant: Variant, unmatched_percent: u8) !Attribution {
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const allocator = counting.allocator();
    var accumulator = try initialFixpoint(allocator);
    var next_key: u16 = FIXPOINT_NEW_KEY_BASE;
    var total = Attribution{};
    for (0..FIXPOINT_ROUNDS) |round| {
        var delta = try makeFixpointDelta(
            allocator,
            &accumulator,
            unmatched_percent,
            round,
            &next_key,
        );
        counting.resetStats();
        var operation = Attribution{};
        switch (variant) {
            .baseline => try baselineOrMeasured(&accumulator, &delta, &counting, &operation),
            .consuming => try consumingOrMeasured(&accumulator, &delta, &counting, &operation),
        }
        operation.total_allocs = counting.stats.alloc_calls;
        try operation.validate(variant);
        total.add(operation);
        delta.deinit();
    }
    try total.validate(variant);
    accumulator.deinit();
    if (counting.stats.live_bytes != 0) return error.CountingAllocatorLeak;
    return total;
}

fn measureProductionSweepAlloc(variant: Variant, unmatched_percent: u8) !u64 {
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const allocator = counting.allocator();
    var pair = try makeSweepPair(allocator, unmatched_percent);
    counting.resetStats();
    try applyVariant(variant, &pair.left, &pair.right);
    const allocs = counting.stats.alloc_calls;
    pair.deinit();
    if (counting.stats.live_bytes != 0) return error.CountingAllocatorLeak;
    return allocs;
}

fn measureProductionFixpointAlloc(variant: Variant, unmatched_percent: u8) !u64 {
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const allocator = counting.allocator();
    var accumulator = try initialFixpoint(allocator);
    var next_key: u16 = FIXPOINT_NEW_KEY_BASE;
    var total_allocs: u64 = 0;
    for (0..FIXPOINT_ROUNDS) |round| {
        var delta = try makeFixpointDelta(
            allocator,
            &accumulator,
            unmatched_percent,
            round,
            &next_key,
        );
        counting.resetStats();
        try applyVariant(variant, &accumulator, &delta);
        total_allocs += counting.stats.alloc_calls;
        delta.deinit();
    }
    accumulator.deinit();
    if (counting.stats.live_bytes != 0) return error.CountingAllocatorLeak;
    return total_allocs;
}

fn validatePrototype() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer if (gpa.deinit() != .ok) @panic("consuming OR validation leaked");
    const allocator = gpa.allocator();

    for (UNMATCHED_PCTS) |unmatched_percent| {
        var baseline = try makeSweepPair(allocator, unmatched_percent);
        defer baseline.deinit();
        var consuming = try makeSweepPair(allocator, unmatched_percent);
        defer consuming.deinit();
        if (!baseline.left.equals(&consuming.left) or !baseline.right.equals(&consuming.right)) {
            return error.InputCorpusMismatch;
        }
        try baseline.left.bitwiseOrInPlace(&baseline.right);
        try consuming.left.bitwiseOrInPlaceConsume(&consuming.right);
        try baseline.left.validate();
        try consuming.left.validate();
        try consuming.right.validate();
        if (!baseline.left.equals(&consuming.left)) return error.PrototypeResultMismatch;
        if (consuming.right.cardinality() != 0 or consuming.right.cached_cardinality != 0) {
            return error.ConsumedBitmapNotEmpty;
        }
        _ = try consuming.right.add(0xFFFF_FFFE);
        try consuming.right.validate();
        if (!consuming.right.contains(0xFFFF_FFFE)) return error.ConsumedBitmapReuseFailed;
    }

    // Check every fixpoint round, not only the final result.
    for (UNMATCHED_PCTS) |unmatched_percent| {
        var baseline = try initialFixpoint(allocator);
        defer baseline.deinit();
        var consuming = try initialFixpoint(allocator);
        defer consuming.deinit();
        var baseline_next: u16 = FIXPOINT_NEW_KEY_BASE;
        var consuming_next: u16 = FIXPOINT_NEW_KEY_BASE;
        for (0..FIXPOINT_ROUNDS) |round| {
            var baseline_delta = try makeFixpointDelta(
                allocator,
                &baseline,
                unmatched_percent,
                round,
                &baseline_next,
            );
            defer baseline_delta.deinit();
            var consuming_delta = try makeFixpointDelta(
                allocator,
                &consuming,
                unmatched_percent,
                round,
                &consuming_next,
            );
            defer consuming_delta.deinit();
            try baseline.bitwiseOrInPlace(&baseline_delta);
            try consuming.bitwiseOrInPlaceConsume(&consuming_delta);
            try baseline.validate();
            try consuming.validate();
            try consuming_delta.validate();
            if (!baseline.equals(&consuming)) return error.FixpointResultMismatch;
            if (consuming_delta.cardinality() != 0) return error.ConsumedBitmapNotEmpty;
        }
    }
}

fn printTiming(workload: []const u8, unmatched_percent: u8, variant: Variant, timing: Timing) void {
    bench_time.print(
        "{s:<9} unmatched={d:>3}% {s:<9} union={d:>10} ns lifecycle={d:>10} ns\n",
        .{ workload, unmatched_percent, @tagName(variant), timing.union_ns, timing.lifecycle_ns },
    );
    bench_time.print(
        "RESULT\t{s}\t{d}\t{s}\t{d}\t{d}\n",
        .{ workload, unmatched_percent, @tagName(variant), timing.union_ns, timing.lifecycle_ns },
    );
}

fn printAlloc(workload: []const u8, unmatched_percent: u8, variant: Variant, result: Attribution) void {
    bench_time.print(
        "alloc {s:<9} unmatched={d:>3}% {s:<9} total={d} index={d} matched={d} clones={d} moved={d}\n",
        .{
            workload,
            unmatched_percent,
            @tagName(variant),
            result.total_allocs,
            result.index_allocs,
            result.matched_allocs,
            result.clone_allocs,
            result.moved_containers,
        },
    );
    bench_time.print(
        "ALLOC\t{s}\t{d}\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\n",
        .{
            workload,
            unmatched_percent,
            @tagName(variant),
            result.total_allocs,
            result.index_allocs,
            result.matched_allocs,
            result.clone_allocs,
            result.moved_containers,
        },
    );
}

pub fn main() !void {
    bench_time.print("Consuming in-place OR benchmark\n", .{});
    bench_time.print("================================\n", .{});
    bench_time.printBenchEnvironment();
    bench_time.print(
        "sweep keys={d}; mix=array/bitset/run by key mod 3; unmatched=0/25/50/75/100%\n",
        .{SWEEP_KEYS},
    );
    bench_time.print(
        "fixpoint initial={d}, delta={d}, rounds={d}; warmup={d}, timed={d}\n",
        .{ FIXPOINT_INITIAL_KEYS, FIXPOINT_DELTA_KEYS, FIXPOINT_ROUNDS, WARMUP_RUNS, TIMED_RUNS },
    );
    bench_time.print("Validating ownership and result parity...\n", .{});
    try validatePrototype();
    bench_time.print("validation: passed\n\n", .{});

    for (UNMATCHED_PCTS) |unmatched_percent| {
        inline for (.{ Variant.baseline, Variant.consuming }) |variant| {
            const sweep_alloc = try measureSweepAlloc(variant, unmatched_percent);
            const fixpoint_alloc = try measureFixpointAlloc(variant, unmatched_percent);
            if (sweep_alloc.total_allocs != try measureProductionSweepAlloc(variant, unmatched_percent)) {
                return error.AllocationReplicaMismatch;
            }
            if (fixpoint_alloc.total_allocs != try measureProductionFixpointAlloc(variant, unmatched_percent)) {
                return error.AllocationReplicaMismatch;
            }
            printAlloc("sweep", unmatched_percent, variant, sweep_alloc);
            printTiming(
                "sweep",
                unmatched_percent,
                variant,
                try measureSweepTiming(variant, unmatched_percent),
            );
            printAlloc("fixpoint", unmatched_percent, variant, fixpoint_alloc);
            printTiming(
                "fixpoint",
                unmatched_percent,
                variant,
                try measureFixpointTiming(variant, unmatched_percent),
            );
        }
    }
}
