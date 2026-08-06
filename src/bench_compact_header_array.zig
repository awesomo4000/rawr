// SPDX-License-Identifier: MPL-2.0

//! Standalone ArrayContainer header-layout diagnostic for specs 32 and 32-00.
//! This file deliberately does not import or duplicate bitmap operations.

const std = @import("std");
const bench_time = @import("bench_time.zig");
const counting_allocator = @import("counting_allocator.zig");
const CountingAllocator = counting_allocator.CountingAllocator;

const value_count = 500_000;
const corpus_seed = 54_321;
const warmup_runs = 3;
const timed_runs = 21;
const external_process_runs = 5;
const min_capacity: u16 = 4;
const max_cardinality: u16 = 4096;
const value_alignment = std.mem.Alignment.@"32";

// Filled from the pinned DefaultPrng corpus. These constants intentionally make
// an RNG, sorting, dedupe, or operand-boundary change fail before timing.
const expected_sparse_len: usize = 499_964;
const expected_corpus_fingerprint: u64 = 0xfdf580cd424a30cd;
const expected_left_entries: usize = 32_691;
const expected_right_entries: usize = 49_169;
const expected_attribution_entries: usize = 49_132;

const Variant = enum {
    baseline,
    compact,
};

const Cell = enum {
    build_reserved,
    build_growth,
    clone,
    clone_attribution,
    deinit,
    membership,
    iterate,
};

const variants = [_]Variant{ .baseline, .compact };
const cells = [_]Cell{
    .build_reserved,
    .build_growth,
    .clone,
    .clone_attribution,
    .deinit,
    .membership,
    .iterate,
};

const BaselineArray = ArrayReplica(false);
const CompactArray = ArrayReplica(true);

comptime {
    std.debug.assert(@sizeOf(BaselineArray) == 24);
    std.debug.assert(@sizeOf(CompactArray) == 16);
    std.debug.assert(@alignOf(BaselineArray) >= 4);
    std.debug.assert(@alignOf(CompactArray) >= 4);
}

fn ArrayReplica(comptime compact: bool) type {
    return struct {
        values: if (compact) [*]align(32) u16 else []align(32) u16,
        cardinality: u16,
        capacity: u16,

        const Self = @This();

        fn init(allocator: std.mem.Allocator, requested_capacity: u16) !*Self {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            const capacity = normalizedCapacity(requested_capacity);
            const values = try allocator.alignedAlloc(u16, value_alignment, capacity);
            self.* = .{
                .values = if (compact) values.ptr else values,
                .cardinality = 0,
                .capacity = capacity,
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
            const values = try allocator.alignedAlloc(u16, value_alignment, self.capacity);
            @memcpy(values[0..self.cardinality], self.readable());
            copy.* = .{
                .values = if (compact) values.ptr else values,
                .cardinality = self.cardinality,
                .capacity = self.capacity,
            };
            return copy;
        }

        fn add(self: *Self, allocator: std.mem.Allocator, value: u16) !bool {
            if (self.cardinality == 0 or value > self.readable()[self.cardinality - 1]) {
                try self.ensureCapacity(allocator, self.cardinality + 1);
                self.storage()[self.cardinality] = value;
                self.cardinality += 1;
                return true;
            }

            const position = lowerBound(self.readable(), value);
            if (position < self.cardinality and self.readable()[position] == value) return false;

            try self.ensureCapacity(allocator, self.cardinality + 1);
            const values = self.storage();
            @memmove(values[position + 1 .. self.cardinality + 1], values[position..self.cardinality]);
            values[position] = value;
            self.cardinality += 1;
            return true;
        }

        fn contains(self: *const Self, value: u16) bool {
            const values = self.readable();
            const position = lowerBound(values, value);
            return position < values.len and values[position] == value;
        }

        fn ensureCapacity(self: *Self, allocator: std.mem.Allocator, needed: u16) !void {
            if (needed <= self.capacity) return;
            const capacity = nextCapacity(self.capacity, needed);
            const values = try allocator.alignedAlloc(u16, value_alignment, capacity);
            @memcpy(values[0..self.cardinality], self.readable());
            allocator.free(self.storage());
            self.values = if (compact) values.ptr else values;
            self.capacity = capacity;
        }

        fn readable(self: *const Self) []const u16 {
            return self.values[0..self.cardinality];
        }

        fn storage(self: *Self) []align(32) u16 {
            return self.values[0..self.capacity];
        }
    };
}

const Entry = struct {
    key: u16,
    offset: usize,
    cardinality: u16,
};

const Probe = struct {
    entry_index: usize,
    value: u16,
    expected: bool,
};

const Corpus = struct {
    allocator: std.mem.Allocator,
    values: []u32,
    sparse_len: usize,
    entries: []Entry,
    attribution_entries: []Entry,
    probes: []Probe,
    left_entries: usize,
    right_entries: usize,
    shared_entries: usize,
    expected_probe_hits: u64,
    fingerprint: u64,

    fn init(allocator: std.mem.Allocator) !Corpus {
        const values = try allocator.alloc(u32, value_count);
        errdefer allocator.free(values);

        var prng = std.Random.DefaultPrng.init(corpus_seed);
        const random = prng.random();
        for (values) |*value| value.* = random.int(u32);
        std.mem.sort(u32, values, {}, std.sort.asc(u32));

        var sparse_len: usize = 1;
        for (values[1..]) |value| {
            if (value != values[sparse_len - 1]) {
                values[sparse_len] = value;
                sparse_len += 1;
            }
        }

        const half = sparse_len / 2;
        const left_entries = countEntries(values, 0, half);
        const right_entries = countEntries(values, half / 2, sparse_len);
        const entries = try allocator.alloc(Entry, left_entries + right_entries);
        errdefer allocator.free(entries);
        fillEntries(entries[0..left_entries], values, 0, half);
        fillEntries(entries[left_entries..], values, half / 2, sparse_len);

        const left = entries[0..left_entries];
        const right = entries[left_entries..];
        const inventory = compareInventories(left, right);
        const attribution_entries = try allocator.alloc(Entry, inventory.left_only + inventory.right_only);
        errdefer allocator.free(attribution_entries);
        fillAttributionEntries(attribution_entries, left, right);

        const probes = try allocator.alloc(Probe, entries.len * 2);
        errdefer allocator.free(probes);
        var expected_probe_hits: u64 = 0;
        for (entries, 0..) |entry, entry_index| {
            const first = lowValue(values[entry.offset]);
            probes[entry_index * 2] = .{
                .entry_index = entry_index,
                .value = first,
                .expected = true,
            };
            expected_probe_hits += 1;

            const candidate = first +% 0x8000;
            const expected = entryContains(values, entry, candidate);
            probes[entry_index * 2 + 1] = .{
                .entry_index = entry_index,
                .value = candidate,
                .expected = expected,
            };
            expected_probe_hits += @intFromBool(expected);
        }

        const fingerprint = fingerprintValues(values[0..sparse_len]);
        const corpus = Corpus{
            .allocator = allocator,
            .values = values,
            .sparse_len = sparse_len,
            .entries = entries,
            .attribution_entries = attribution_entries,
            .probes = probes,
            .left_entries = left_entries,
            .right_entries = right_entries,
            .shared_entries = inventory.shared,
            .expected_probe_hits = expected_probe_hits,
            .fingerprint = fingerprint,
        };
        try corpus.assertPinned();
        return corpus;
    }

    fn deinit(self: *Corpus) void {
        self.allocator.free(self.probes);
        self.allocator.free(self.attribution_entries);
        self.allocator.free(self.entries);
        self.allocator.free(self.values);
        self.* = undefined;
    }

    fn assertPinned(self: *const Corpus) !void {
        if (expected_sparse_len != 0 and self.sparse_len != expected_sparse_len) return error.SparseLengthMismatch;
        if (expected_corpus_fingerprint != 0 and self.fingerprint != expected_corpus_fingerprint) return error.CorpusFingerprintMismatch;
        if (expected_left_entries != 0 and self.left_entries != expected_left_entries) return error.LeftInventoryMismatch;
        if (expected_right_entries != 0 and self.right_entries != expected_right_entries) return error.RightInventoryMismatch;
        if (expected_attribution_entries != 0 and self.attribution_entries.len != expected_attribution_entries) return error.AttributionInventoryMismatch;
        if (self.shared_entries * 2 + self.attribution_entries.len != self.left_entries + self.right_entries) {
            return error.InventoryAccountingMismatch;
        }
        for (self.entries) |entry| {
            if (entry.cardinality == 0 or entry.cardinality > max_cardinality) return error.ExpectedArrayPopulation;
        }
    }

    fn entryValues(self: *const Corpus, entry: Entry) []const u32 {
        return self.values[entry.offset..][0..entry.cardinality];
    }
};

const Inventory = struct {
    shared: usize,
    left_only: usize,
    right_only: usize,
};

const Sample = struct {
    elapsed_ns: u64,
    teardown_ns: u64,
    stats: CountingAllocator.Stats,
    header_allocations: u64,
    checksum: u64,
};

fn normalizedCapacity(requested: u16) u16 {
    if (requested <= min_capacity) return min_capacity;
    return std.math.ceilPowerOfTwo(u16, requested) catch max_cardinality;
}

fn nextCapacity(current: u16, needed: u16) u16 {
    const rounded = std.math.ceilPowerOfTwo(u16, needed) catch max_cardinality;
    return @min(max_cardinality, @max(current * 2, rounded));
}

fn lowerBound(values: []const u16, needle: u16) usize {
    var low: usize = 0;
    var high = values.len;
    while (low < high) {
        const middle = low + (high - low) / 2;
        if (values[middle] < needle) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

fn lowValue(value: u32) u16 {
    return @truncate(value);
}

fn countEntries(values: []const u32, start: usize, end: usize) usize {
    if (start == end) return 0;
    var count: usize = 1;
    var previous_key: u16 = @truncate(values[start] >> 16);
    for (values[start + 1 .. end]) |value| {
        const key: u16 = @truncate(value >> 16);
        if (key != previous_key) {
            count += 1;
            previous_key = key;
        }
    }
    return count;
}

fn fillEntries(entries: []Entry, values: []const u32, start: usize, end: usize) void {
    if (start == end) return;
    var entry_index: usize = 0;
    var group_start = start;
    var key: u16 = @truncate(values[start] >> 16);
    for (values[start + 1 .. end], start + 1..) |value, index| {
        const next_key: u16 = @truncate(value >> 16);
        if (next_key != key) {
            entries[entry_index] = .{
                .key = key,
                .offset = group_start,
                .cardinality = @intCast(index - group_start),
            };
            entry_index += 1;
            group_start = index;
            key = next_key;
        }
    }
    entries[entry_index] = .{
        .key = key,
        .offset = group_start,
        .cardinality = @intCast(end - group_start),
    };
    std.debug.assert(entry_index + 1 == entries.len);
}

fn compareInventories(left: []const Entry, right: []const Entry) Inventory {
    var inventory = Inventory{ .shared = 0, .left_only = 0, .right_only = 0 };
    var left_index: usize = 0;
    var right_index: usize = 0;
    while (left_index < left.len and right_index < right.len) {
        if (left[left_index].key < right[right_index].key) {
            inventory.left_only += 1;
            left_index += 1;
        } else if (left[left_index].key > right[right_index].key) {
            inventory.right_only += 1;
            right_index += 1;
        } else {
            inventory.shared += 1;
            left_index += 1;
            right_index += 1;
        }
    }
    inventory.left_only += left.len - left_index;
    inventory.right_only += right.len - right_index;
    return inventory;
}

fn fillAttributionEntries(output: []Entry, left: []const Entry, right: []const Entry) void {
    var output_index: usize = 0;
    var left_index: usize = 0;
    var right_index: usize = 0;
    while (left_index < left.len and right_index < right.len) {
        if (left[left_index].key < right[right_index].key) {
            output[output_index] = left[left_index];
            output_index += 1;
            left_index += 1;
        } else if (left[left_index].key > right[right_index].key) {
            output[output_index] = right[right_index];
            output_index += 1;
            right_index += 1;
        } else {
            left_index += 1;
            right_index += 1;
        }
    }
    while (left_index < left.len) : (left_index += 1) {
        output[output_index] = left[left_index];
        output_index += 1;
    }
    while (right_index < right.len) : (right_index += 1) {
        output[output_index] = right[right_index];
        output_index += 1;
    }
    std.debug.assert(output_index == output.len);
}

fn entryContains(values: []const u32, entry: Entry, needle: u16) bool {
    var low: usize = 0;
    var high: usize = entry.cardinality;
    while (low < high) {
        const middle = low + (high - low) / 2;
        const current = lowValue(values[entry.offset + middle]);
        if (current < needle) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low < entry.cardinality and lowValue(values[entry.offset + low]) == needle;
}

fn fingerprintValues(values: []const u32) u64 {
    var fingerprint: u64 = 0xcbf29ce484222325;
    for (values) |value| {
        fingerprint ^= value;
        fingerprint *%= 0x100000001b3;
    }
    fingerprint ^= values.len;
    fingerprint *%= 0x100000001b3;
    return fingerprint;
}

fn buildPopulation(
    comptime T: type,
    slots: []*T,
    allocator: std.mem.Allocator,
    corpus: *const Corpus,
    entries: []const Entry,
    reserve: bool,
) !void {
    var initialized: usize = 0;
    errdefer deinitPopulation(T, slots[0..initialized], allocator);
    for (slots, entries) |*slot, entry| {
        const container = try T.init(allocator, if (reserve) entry.cardinality else 0);
        errdefer container.deinit(allocator);
        for (corpus.entryValues(entry)) |value| {
            if (!try container.add(allocator, lowValue(value))) return error.DuplicateArrayValue;
        }
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
        checksum ^= container.cardinality;
        checksum *%= 0x100000001b3;
        for (container.readable()) |value| {
            checksum ^= value;
            checksum *%= 0x100000001b3;
        }
    }
    return checksum;
}

fn runBuild(comptime T: type, corpus: *const Corpus, reserve: bool) !Sample {
    const slots = try std.heap.smp_allocator.alloc(*T, corpus.entries.len);
    defer std.heap.smp_allocator.free(slots);
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const allocator = counting.allocator();

    const start = bench_time.monotonicNanos();
    try buildPopulation(T, slots, allocator, corpus, corpus.entries, reserve);
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
        .header_allocations = corpus.entries.len,
        .checksum = checksum,
    };
}

fn runClone(comptime T: type, corpus: *const Corpus, entries: []const Entry) !Sample {
    const source = try std.heap.smp_allocator.alloc(*T, entries.len);
    defer std.heap.smp_allocator.free(source);
    try buildPopulation(T, source, std.heap.smp_allocator, corpus, entries, true);
    defer deinitPopulation(T, source, std.heap.smp_allocator);

    const clones = try std.heap.smp_allocator.alloc(*T, entries.len);
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
        .header_allocations = entries.len,
        .checksum = checksum,
    };
}

fn runDeinit(comptime T: type, corpus: *const Corpus) !Sample {
    const slots = try std.heap.smp_allocator.alloc(*T, corpus.entries.len);
    defer std.heap.smp_allocator.free(slots);
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const allocator = counting.allocator();
    try buildPopulation(T, slots, allocator, corpus, corpus.entries, true);
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
        .checksum = corpus.entries.len,
    };
}

fn runMembership(comptime T: type, corpus: *const Corpus) !Sample {
    const slots = try std.heap.smp_allocator.alloc(*T, corpus.entries.len);
    defer std.heap.smp_allocator.free(slots);
    try buildPopulation(T, slots, std.heap.smp_allocator, corpus, corpus.entries, true);

    var hits: u64 = 0;
    const start = bench_time.monotonicNanos();
    for (corpus.probes) |probe| hits += @intFromBool(slots[probe.entry_index].contains(probe.value));
    const elapsed_ns = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(hits);
    if (hits != corpus.expected_probe_hits) return error.MembershipMismatch;

    const teardown_start = bench_time.monotonicNanos();
    deinitPopulation(T, slots, std.heap.smp_allocator);
    const teardown_ns = bench_time.monotonicNanos() - teardown_start;
    return .{
        .elapsed_ns = elapsed_ns,
        .teardown_ns = teardown_ns,
        .stats = .{},
        .header_allocations = 0,
        .checksum = hits,
    };
}

fn runIteration(comptime T: type, corpus: *const Corpus) !Sample {
    const slots = try std.heap.smp_allocator.alloc(*T, corpus.entries.len);
    defer std.heap.smp_allocator.free(slots);
    try buildPopulation(T, slots, std.heap.smp_allocator, corpus, corpus.entries, true);

    var checksum: u64 = 0;
    const start = bench_time.monotonicNanos();
    for (slots) |container| {
        for (container.readable()) |value| checksum +%= value;
    }
    const elapsed_ns = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(checksum);

    const teardown_start = bench_time.monotonicNanos();
    deinitPopulation(T, slots, std.heap.smp_allocator);
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
    samples[@intFromEnum(Cell.clone)] = try runClone(T, corpus, corpus.entries);
    samples[@intFromEnum(Cell.clone_attribution)] = try runClone(T, corpus, corpus.attribution_entries);
    samples[@intFromEnum(Cell.deinit)] = try runDeinit(T, corpus);
    samples[@intFromEnum(Cell.membership)] = try runMembership(T, corpus);
    samples[@intFromEnum(Cell.iterate)] = try runIteration(T, corpus);
    return samples;
}

fn executeVariant(variant: Variant, corpus: *const Corpus) ![cells.len]Sample {
    return switch (variant) {
        .baseline => @call(.never_inline, runVariant, .{ BaselineArray, corpus }),
        .compact => @call(.never_inline, runVariant, .{ CompactArray, corpus }),
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
    const expected_requested_delta = header_allocations * (@sizeOf(BaselineArray) - @sizeOf(CompactArray));
    const baseline_class = counting_allocator.smpClassBytes(
        @sizeOf(BaselineArray),
        std.mem.Alignment.fromByteUnits(@alignOf(BaselineArray)),
    );
    const compact_class = counting_allocator.smpClassBytes(
        @sizeOf(CompactArray),
        std.mem.Alignment.fromByteUnits(@alignOf(CompactArray)),
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
    const baseline_slots = try std.heap.smp_allocator.alloc(*BaselineArray, corpus.entries.len);
    defer std.heap.smp_allocator.free(baseline_slots);
    const compact_slots = try std.heap.smp_allocator.alloc(*CompactArray, corpus.entries.len);
    defer std.heap.smp_allocator.free(compact_slots);

    inline for (.{ true, false }) |reserve| {
        try buildPopulation(BaselineArray, baseline_slots, std.heap.smp_allocator, corpus, corpus.entries, reserve);
        defer deinitPopulation(BaselineArray, baseline_slots, std.heap.smp_allocator);
        try buildPopulation(CompactArray, compact_slots, std.heap.smp_allocator, corpus, corpus.entries, reserve);
        defer deinitPopulation(CompactArray, compact_slots, std.heap.smp_allocator);

        for (baseline_slots, compact_slots, corpus.entries) |baseline, compact, entry| {
            if (baseline.cardinality != compact.cardinality or baseline.capacity != compact.capacity) {
                return error.ReplicaShapeMismatch;
            }
            if (!std.mem.eql(u16, baseline.readable(), compact.readable())) return error.ReplicaValueMismatch;
            const payload_requested = @as(usize, baseline.capacity) * @sizeOf(u16);
            const baseline_payload_class = counting_allocator.smpClassBytes(payload_requested, value_alignment);
            const compact_payload_class = counting_allocator.smpClassBytes(
                @as(usize, compact.capacity) * @sizeOf(u16),
                value_alignment,
            );
            if (baseline_payload_class != compact_payload_class or baseline.cardinality != entry.cardinality) {
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
        .clone_attribution => "clone-attribution",
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
    bench_time.print("\n{s:<9} {s:<19} {s:>12} {s:>12} {s:>12} {s:>9} {s:>9} {s:>14} {s:>14} {s:>14}\n", .{
        "variant", "cell", "median ns", "min ns", "max ns", "alloc", "free", "requested", "class bytes", "teardown ns",
    });
    bench_time.print("{s:-<9} {s:-<19} {s:->12} {s:->12} {s:->12} {s:->9} {s:->9} {s:->14} {s:->14} {s:->14}\n", .{
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
            bench_time.print("{s:<9} {s:<19} {d:>12} {d:>12} {d:>12} {d:>9} {d:>9} {d:>14} {d:>14} {d:>14}\n", .{
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
    bench_time.print("Compact ArrayContainer header replica diagnostic\n", .{});
    bench_time.print("================================================\n", .{});
    bench_time.printRunTimestamp();
    bench_time.printBenchEnvironment();
    bench_time.print("PROTOCOL\twarmup\t{d}\ttimed\t{d}\texternal-processes\t{d}\tallocator\tsmp\n", .{
        warmup_runs,
        timed_runs,
        external_process_runs,
    });
    bench_time.print("BOUNDARY\toperation timing excludes checksum and teardown; teardown_ns is separate\n", .{});
    printLayout(BaselineArray, "baseline");
    printLayout(CompactArray, "compact");

    const baseline_header_class = counting_allocator.smpClassBytes(
        @sizeOf(BaselineArray),
        std.mem.Alignment.fromByteUnits(@alignOf(BaselineArray)),
    );
    const compact_header_class = counting_allocator.smpClassBytes(
        @sizeOf(CompactArray),
        std.mem.Alignment.fromByteUnits(@alignOf(CompactArray)),
    );
    if (baseline_header_class != 32 or compact_header_class != 16) return error.HeaderClassMismatch;

    var corpus = try Corpus.init(std.heap.smp_allocator);
    defer corpus.deinit();
    bench_time.print("CORPUS\tseed\t{d}\tdraws\t{d}\tsparse-len\t{d}\tfingerprint\t{x}\n", .{
        corpus_seed,
        value_count,
        corpus.sparse_len,
        corpus.fingerprint,
    });
    bench_time.print("INVENTORY\tleft\t{d}\tright\t{d}\tshared\t{d}\tunmatched-clones\t{d}\tprobes\t{d}\thits\t{d}\n", .{
        corpus.left_entries,
        corpus.right_entries,
        corpus.shared_entries,
        corpus.attribution_entries.len,
        corpus.probes.len,
        corpus.expected_probe_hits,
    });
    try validateReplicaValues(&corpus);
    bench_time.print("VALIDATION\tvalues=equal\tpayload-requested=equal\tpayload-align=32\tpayload-class=equal\n", .{});

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

test "array replicas preserve compact layout and payload behavior" {
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(BaselineArray));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(CompactArray));
    try std.testing.expect(@alignOf(BaselineArray) >= 4);
    try std.testing.expect(@alignOf(CompactArray) >= 4);

    const baseline = try BaselineArray.init(std.testing.allocator, 0);
    defer baseline.deinit(std.testing.allocator);
    const compact = try CompactArray.init(std.testing.allocator, 0);
    defer compact.deinit(std.testing.allocator);
    for (0..257) |index| {
        const value: u16 = @intCast(index * 2);
        try std.testing.expect(try baseline.add(std.testing.allocator, value));
        try std.testing.expect(try compact.add(std.testing.allocator, value));
    }
    try std.testing.expectEqual(baseline.capacity, compact.capacity);
    try std.testing.expectEqualSlices(u16, baseline.readable(), compact.readable());
    try std.testing.expect(baseline.contains(256));
    try std.testing.expect(compact.contains(256));

    const baseline_clone = try baseline.clone(std.testing.allocator);
    defer baseline_clone.deinit(std.testing.allocator);
    const compact_clone = try compact.clone(std.testing.allocator);
    defer compact_clone.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u16, baseline_clone.readable(), compact_clone.readable());
}
