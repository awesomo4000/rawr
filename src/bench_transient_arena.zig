// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");
const counting_mod = @import("counting_allocator.zig");

const RoaringBitmap = rawr.RoaringBitmap;
const ArrayContainer = rawr.ArrayContainer;
const BitsetContainer = rawr.BitsetContainer;
const Container = rawr.Container;
const TaggedPtr = rawr.TaggedPtr;
const container_ops = rawr.container_ops;
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

const RawrVariant = enum {
    baseline,
    arena,
    fba,
};

var empty_fba_buffer: [0]u8 align(64) = .{};

const FbaBacking = struct {
    persistent_allocator: std.mem.Allocator,
    slab: ?[]align(64) u8,
    allocator_state: std.heap.FixedBufferAllocator,

    fn init(persistent_allocator: std.mem.Allocator, container_count: usize) !FbaBacking {
        const bytes = try transientFootprint(container_count);
        if (bytes == 0) {
            return .{
                .persistent_allocator = persistent_allocator,
                .slab = null,
                .allocator_state = std.heap.FixedBufferAllocator.init(empty_fba_buffer[0..]),
            };
        }

        const slab = try persistent_allocator.alignedAlloc(u8, .@"64", bytes);
        return .{
            .persistent_allocator = persistent_allocator,
            .slab = slab,
            .allocator_state = std.heap.FixedBufferAllocator.init(slab),
        };
    }

    fn allocator(self: *FbaBacking) std.mem.Allocator {
        return self.allocator_state.allocator();
    }

    fn capacity(self: *const FbaBacking) usize {
        return if (self.slab) |slab| slab.len else 0;
    }

    fn assertFullyConsumed(self: *const FbaBacking) void {
        std.debug.assert(self.allocator_state.end_index == self.capacity());
    }

    fn deinit(self: *FbaBacking) void {
        if (self.slab) |slab| self.persistent_allocator.free(slab);
        self.* = undefined;
    }
};

const TransientBacking = union(RawrVariant) {
    baseline: void,
    arena: std.heap.ArenaAllocator,
    fba: FbaBacking,

    fn init(
        variant: RawrVariant,
        persistent_allocator: std.mem.Allocator,
        eligible_count: usize,
    ) !TransientBacking {
        return switch (variant) {
            .baseline => .{ .baseline = {} },
            .arena => .{ .arena = std.heap.ArenaAllocator.init(persistent_allocator) },
            .fba => .{ .fba = try FbaBacking.init(persistent_allocator, eligible_count) },
        };
    }

    fn allocator(self: *TransientBacking, persistent_allocator: std.mem.Allocator) std.mem.Allocator {
        return switch (self.*) {
            .baseline => persistent_allocator,
            .arena => |*arena| arena.allocator(),
            .fba => |*fba| fba.allocator(),
        };
    }

    fn capacity(self: *const TransientBacking) usize {
        return switch (self.*) {
            .baseline => 0,
            .arena => |*arena| arena.queryCapacity(),
            .fba => |*fba| fba.capacity(),
        };
    }

    fn assertFullyConsumed(self: *const TransientBacking) void {
        switch (self.*) {
            .fba => |*fba| fba.assertFullyConsumed(),
            .baseline, .arena => {},
        }
    }

    fn deinit(self: *TransientBacking) void {
        switch (self.*) {
            .baseline => {},
            .arena => |*arena| arena.deinit(),
            .fba => |*fba| fba.deinit(),
        }
        self.* = undefined;
    }
};

fn transientFootprint(container_count: usize) !usize {
    var offset: usize = 0;
    for (0..container_count) |_| {
        offset = std.mem.alignForward(usize, offset, @alignOf(BitsetContainer));
        offset = try std.math.add(usize, offset, @sizeOf(BitsetContainer));
        offset = std.mem.alignForward(usize, offset, 64);
        offset = try std.math.add(usize, offset, BitsetContainer.SIZE_BYTES);
    }
    return offset;
}

const MixedBitmap = struct {
    bitmap: RoaringBitmap,
    persistent_allocator: std.mem.Allocator,
    transient_flags: ?[]bool = null,

    fn init(persistent_allocator: std.mem.Allocator) !MixedBitmap {
        return .{
            .bitmap = try RoaringBitmap.init(persistent_allocator),
            .persistent_allocator = persistent_allocator,
        };
    }

    fn initCapacity(persistent_allocator: std.mem.Allocator, capacity: u32) !MixedBitmap {
        return .{
            .bitmap = try RoaringBitmap.initCapacity(persistent_allocator, capacity),
            .persistent_allocator = persistent_allocator,
        };
    }

    fn ensureCapacity(self: *MixedBitmap, needed: u32) !void {
        const old_capacity = self.bitmap.capacity;
        try self.bitmap.ensureTotalCapacity(needed);
        if (self.bitmap.capacity == old_capacity) return;

        if (self.transient_flags) |old_flags| {
            const new_flags = try self.persistent_allocator.alloc(bool, self.bitmap.capacity);
            @memcpy(new_flags[0..self.bitmap.size], old_flags[0..self.bitmap.size]);
            @memset(new_flags[self.bitmap.size..], false);
            self.persistent_allocator.free(old_flags);
            self.transient_flags = new_flags;
        }
    }

    fn ensureFlags(self: *MixedBitmap) ![]bool {
        if (self.transient_flags == null) {
            const flags = try self.persistent_allocator.alloc(bool, self.bitmap.capacity);
            @memset(flags, false);
            self.transient_flags = flags;
        }
        return self.transient_flags.?;
    }

    fn append(self: *MixedBitmap, key: u16, tagged: TaggedPtr, transient: bool) !void {
        try self.ensureCapacity(self.bitmap.size + 1);
        if (transient) {
            const flags = try self.ensureFlags();
            flags[self.bitmap.size] = true;
        } else if (self.transient_flags) |flags| {
            flags[self.bitmap.size] = false;
        }
        self.bitmap.keys[self.bitmap.size] = key;
        self.bitmap.containers[self.bitmap.size] = tagged;
        self.bitmap.size += 1;
    }

    fn appendPersistent(self: *MixedBitmap, key: u16, tagged: TaggedPtr) !void {
        errdefer Container.fromTagged(tagged).deinit(self.persistent_allocator);
        try self.append(key, tagged, false);
    }

    fn appendTransient(self: *MixedBitmap, key: u16, tagged: TaggedPtr) !void {
        try self.append(key, tagged, true);
    }

    fn isTransient(self: *const MixedBitmap, index: usize) bool {
        return if (self.transient_flags) |flags| flags[index] else false;
    }

    fn repair(self: *MixedBitmap) !void {
        if (self.transient_flags == null) {
            try self.bitmap.repairAfterLazy();
            return;
        }

        var total: u64 = 0;

        for (0..self.bitmap.size) |index| {
            const tagged = self.bitmap.containers[index];
            const transient = self.isTransient(index);
            const container = Container.fromTagged(tagged);

            var cardinality: u32 = 0;
            switch (container) {
                .array => |array| {
                    std.debug.assert(!transient);
                    cardinality = array.cardinality;
                    std.debug.assert(cardinality != 0);
                },
                .bitset => |bitset| {
                    cardinality = bitset.computeCardinality();
                    std.debug.assert(cardinality != 0);
                    if (cardinality <= ArrayContainer.MAX_CARDINALITY) {
                        const array = try container_ops.bitsetToArray(self.persistent_allocator, bitset);
                        if (!transient) bitset.deinit(self.persistent_allocator);
                        self.bitmap.containers[index] = TaggedPtr.initArray(array);
                        self.transient_flags.?[index] = false;
                    } else {
                        if (transient) return error.TransientBitsetSurvived;
                    }
                },
                .run => |run| {
                    std.debug.assert(!transient);
                    cardinality = run.getCardinality();
                    std.debug.assert(cardinality != 0);
                },
                .reserved => unreachable,
            }
            total += cardinality;
        }

        self.bitmap.cached_cardinality = @intCast(total);
    }

    fn deinit(self: *MixedBitmap) void {
        for (self.bitmap.containers[0..self.bitmap.size], 0..) |tagged, index| {
            if (!self.isTransient(index)) {
                Container.fromTagged(tagged).deinit(self.persistent_allocator);
            }
        }
        self.persistent_allocator.free(self.bitmap.keys[0..self.bitmap.capacity]);
        self.persistent_allocator.free(self.bitmap.containers[0..self.bitmap.capacity]);
        if (self.transient_flags) |flags| self.persistent_allocator.free(flags);
        self.* = undefined;
    }

    fn release(self: *MixedBitmap) RoaringBitmap {
        if (self.transient_flags) |flags| {
            for (flags[0..self.bitmap.size]) |transient| std.debug.assert(!transient);
            self.persistent_allocator.free(flags);
        }
        const bitmap = self.bitmap;
        self.* = undefined;
        return bitmap;
    }
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

fn knownCardinalityBound(container: Container) ?u64 {
    return switch (container) {
        .array => |array| array.cardinality,
        .bitset => |bitset| if (bitset.cardinality < 0) null else @intCast(bitset.cardinality),
        .run => |run| run.getCardinality(),
        .reserved => unreachable,
    };
}

fn pairIsEligible(a: Container, b: Container) bool {
    const a_cardinality = knownCardinalityBound(a) orelse return false;
    const b_cardinality = knownCardinalityBound(b) orelse return false;
    return a_cardinality + b_cardinality <= ArrayContainer.MAX_CARDINALITY;
}

fn lazyAccumulateOr(accumulator: *BitsetContainer, container: Container) void {
    switch (container) {
        .array => |array| accumulator.setList(array.values[0..array.cardinality]),
        .bitset => |bitset| accumulator.lazyUnionWith(bitset),
        .run => |run| {
            for (run.runs[0..run.n_runs]) |pair| {
                accumulator.setRange(pair.start, pair.end());
            }
            accumulator.cardinality = -1;
        },
        .reserved => unreachable,
    }
}

fn cloneTagged(allocator: std.mem.Allocator, tagged: TaggedPtr) !TaggedPtr {
    return (try Container.fromTagged(tagged).clone(allocator)).toTagged();
}

inline fn appendTagged(bitmap: *RoaringBitmap, key: u16, tagged: TaggedPtr) !void {
    try bitmap.ensureTotalCapacity(bitmap.size + 1);
    bitmap.keys[bitmap.size] = key;
    bitmap.containers[bitmap.size] = tagged;
    bitmap.size += 1;
}

fn constructA1(
    persistent_allocator: std.mem.Allocator,
    transient_allocator: std.mem.Allocator,
    use_transient: bool,
    a: *const RoaringBitmap,
    b: *const RoaringBitmap,
) !MixedBitmap {
    const max_size = @min(a.size + b.size, @as(u32, 1) << 16);
    var result = try MixedBitmap.initCapacity(persistent_allocator, max_size);
    errdefer result.deinit();

    var i: usize = 0;
    var j: usize = 0;
    while (i < a.size and j < b.size) {
        const key_a = a.keys[i];
        const key_b = b.keys[j];
        if (key_a < key_b) {
            try result.appendPersistent(key_a, try cloneTagged(persistent_allocator, a.containers[i]));
            i += 1;
            continue;
        }
        if (key_a > key_b) {
            try result.appendPersistent(key_b, try cloneTagged(persistent_allocator, b.containers[j]));
            j += 1;
            continue;
        }

        const container_a = Container.fromTagged(a.containers[i]);
        const container_b = Container.fromTagged(b.containers[j]);
        const transient = use_transient and pairIsEligible(container_a, container_b);
        const owner = if (transient) transient_allocator else persistent_allocator;
        const accumulator = try BitsetContainer.init(owner);
        lazyAccumulateOr(accumulator, container_a);
        lazyAccumulateOr(accumulator, container_b);
        if (transient) {
            try result.appendTransient(key_a, TaggedPtr.initBitset(accumulator));
        } else {
            try result.appendPersistent(key_a, TaggedPtr.initBitset(accumulator));
        }
        i += 1;
        j += 1;
    }

    while (i < a.size) : (i += 1) {
        try result.appendPersistent(a.keys[i], try cloneTagged(persistent_allocator, a.containers[i]));
    }
    while (j < b.size) : (j += 1) {
        try result.appendPersistent(b.keys[j], try cloneTagged(persistent_allocator, b.containers[j]));
    }

    result.bitmap.cached_cardinality = -1;
    return result;
}

fn nextNwayKey(inputs: [NWAY_OPERANDS]*const RoaringBitmap, cursors: []const usize) ?u16 {
    var min_key: ?u16 = null;
    for (inputs, cursors) |bitmap, cursor| {
        if (cursor >= bitmap.size) continue;
        const key = bitmap.keys[cursor];
        if (min_key == null or key < min_key.?) min_key = key;
    }
    return min_key;
}

fn constructA2(
    persistent_allocator: std.mem.Allocator,
    transient_allocator: std.mem.Allocator,
    use_transient: bool,
    inputs: [NWAY_OPERANDS]*const RoaringBitmap,
) !MixedBitmap {
    var result = try MixedBitmap.init(persistent_allocator);
    errdefer result.deinit();

    const cursors = try persistent_allocator.alloc(usize, NWAY_OPERANDS);
    defer persistent_allocator.free(cursors);
    @memset(cursors, 0);

    while (nextNwayKey(inputs, cursors)) |key| {
        var contributor_count: usize = 0;
        var cardinality_sum: u64 = 0;
        var bound_known = true;
        for (inputs, cursors) |bitmap, cursor| {
            if (cursor >= bitmap.size or bitmap.keys[cursor] != key) continue;
            contributor_count += 1;
            if (knownCardinalityBound(Container.fromTagged(bitmap.containers[cursor]))) |cardinality| {
                cardinality_sum +|= cardinality;
            } else {
                bound_known = false;
            }
        }

        if (contributor_count == 1) {
            for (inputs, cursors) |bitmap, *cursor| {
                if (cursor.* >= bitmap.size or bitmap.keys[cursor.*] != key) continue;
                try result.appendPersistent(
                    key,
                    try cloneTagged(persistent_allocator, bitmap.containers[cursor.*]),
                );
                cursor.* += 1;
                break;
            }
            continue;
        }

        const transient = use_transient and bound_known and
            cardinality_sum <= ArrayContainer.MAX_CARDINALITY;
        const owner = if (transient) transient_allocator else persistent_allocator;
        const accumulator = try BitsetContainer.init(owner);

        for (inputs, cursors) |bitmap, *cursor| {
            if (cursor.* >= bitmap.size or bitmap.keys[cursor.*] != key) continue;
            lazyAccumulateOr(accumulator, Container.fromTagged(bitmap.containers[cursor.*]));
            cursor.* += 1;
        }

        if (transient) {
            try result.appendTransient(key, TaggedPtr.initBitset(accumulator));
        } else {
            try result.appendPersistent(key, TaggedPtr.initBitset(accumulator));
        }
    }

    result.bitmap.cached_cardinality = -1;
    return result;
}

fn constructA2Baseline(
    allocator: std.mem.Allocator,
    inputs: [NWAY_OPERANDS]*const RoaringBitmap,
) !RoaringBitmap {
    var result = try RoaringBitmap.init(allocator);
    errdefer result.deinit();

    const cursors = try allocator.alloc(usize, NWAY_OPERANDS);
    defer allocator.free(cursors);
    @memset(cursors, 0);

    while (nextBaselineKey(&inputs, cursors)) |key| {
        if (try foldBaselineKey(allocator, &inputs, cursors, key)) |tagged| {
            appendTagged(&result, key, tagged) catch |err| {
                Container.fromTagged(tagged).deinit(allocator);
                return err;
            };
        }
    }

    result.cached_cardinality = -1;
    return result;
}

inline fn nextBaselineKey(
    inputs: []const *const RoaringBitmap,
    cursors: []const usize,
) ?u16 {
    var min_key: ?u16 = null;
    for (inputs, cursors) |bitmap, cursor| {
        if (cursor >= bitmap.size) continue;
        const key = bitmap.keys[cursor];
        if (min_key == null or key < min_key.?) min_key = key;
    }
    return min_key;
}

inline fn foldBaselineKey(
    allocator: std.mem.Allocator,
    inputs: []const *const RoaringBitmap,
    cursors: []usize,
    key: u16,
) !?TaggedPtr {
    var contributor_count: usize = 0;
    for (inputs, cursors) |bitmap, cursor| {
        if (cursor < bitmap.size and bitmap.keys[cursor] == key) contributor_count += 1;
    }

    if (contributor_count == 1) {
        for (inputs, cursors) |bitmap, *cursor| {
            if (cursor.* >= bitmap.size or bitmap.keys[cursor.*] != key) continue;
            const cloned = try cloneTagged(allocator, bitmap.containers[cursor.*]);
            cursor.* += 1;
            return cloned;
        }
    }

    const accumulator = try BitsetContainer.init(allocator);
    errdefer accumulator.deinit(allocator);
    for (inputs, cursors) |bitmap, *cursor| {
        if (cursor.* >= bitmap.size or bitmap.keys[cursor.*] != key) continue;
        lazyAccumulateOr(accumulator, Container.fromTagged(bitmap.containers[cursor.*]));
        cursor.* += 1;
    }
    return TaggedPtr.initBitset(accumulator);
}

fn measureA1Baseline(pair: *const SparsePair, phase: Phase) !Measurement {
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const allocator = counting.allocator();

    switch (phase) {
        .construction => {
            counting.resetStats();
            const start = bench_time.monotonicNanos();
            var result = try pair.a.lazyOr(allocator, &pair.b, true);
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(&result);
            const stats = counting.snapshot();
            result.deinit();
            std.debug.assert(counting.stats.live_bytes == 0);
            return .{ .elapsed_ns = elapsed, .stats = stats };
        },
        .repair => {
            var result = try pair.a.lazyOr(allocator, &pair.b, true);
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
            var result = try pair.a.lazyOr(allocator, &pair.b, true);
            try result.repairAfterLazy();
            std.mem.doNotOptimizeAway(&result);
            result.deinit();
            const elapsed = bench_time.monotonicNanos() - start;
            std.debug.assert(counting.stats.live_bytes == 0);
            return .{ .elapsed_ns = elapsed, .stats = counting.snapshot() };
        },
    }
}

fn measureA1Transient(pair: *const SparsePair, variant: RawrVariant, phase: Phase) !Measurement {
    std.debug.assert(variant != .baseline);
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const persistent_allocator = counting.allocator();

    switch (phase) {
        .construction => {
            counting.resetStats();
            const start = bench_time.monotonicNanos();
            const eligible_count = if (variant == .fba)
                classifyPair(&pair.a, &pair.b).eligible_groups
            else
                0;
            var backing = try TransientBacking.init(variant, persistent_allocator, eligible_count);
            defer backing.deinit();
            var result = try constructA1(
                persistent_allocator,
                backing.allocator(persistent_allocator),
                true,
                &pair.a,
                &pair.b,
            );
            defer result.deinit();
            backing.assertFullyConsumed();
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(&result);
            return .{
                .elapsed_ns = elapsed,
                .stats = counting.snapshot(),
                .arena_capacity = backing.capacity(),
            };
        },
        .repair => {
            const eligible_count = if (variant == .fba)
                classifyPair(&pair.a, &pair.b).eligible_groups
            else
                0;
            var backing = try TransientBacking.init(variant, persistent_allocator, eligible_count);
            defer backing.deinit();
            var result = try constructA1(
                persistent_allocator,
                backing.allocator(persistent_allocator),
                true,
                &pair.a,
                &pair.b,
            );
            defer result.deinit();
            backing.assertFullyConsumed();

            counting.resetStats();
            const start = bench_time.monotonicNanos();
            try result.repair();
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(&result);
            return .{
                .elapsed_ns = elapsed,
                .stats = counting.snapshot(),
                .arena_capacity = backing.capacity(),
            };
        },
        .combined => {
            counting.resetStats();
            const start = bench_time.monotonicNanos();
            const eligible_count = if (variant == .fba)
                classifyPair(&pair.a, &pair.b).eligible_groups
            else
                0;
            var backing = try TransientBacking.init(variant, persistent_allocator, eligible_count);
            errdefer backing.deinit();
            var result = try constructA1(
                persistent_allocator,
                backing.allocator(persistent_allocator),
                true,
                &pair.a,
                &pair.b,
            );
            errdefer result.deinit();
            backing.assertFullyConsumed();
            try result.repair();
            const arena_capacity = backing.capacity();
            backing.deinit();
            result.deinit();
            const elapsed = bench_time.monotonicNanos() - start;
            std.debug.assert(counting.stats.live_bytes == 0);
            return .{
                .elapsed_ns = elapsed,
                .stats = counting.snapshot(),
                .arena_capacity = arena_capacity,
            };
        },
    }
}

fn measureCRoaringA1(pair: *const SparsePair, phase: Phase) Measurement {
    if (phase == .construction) {
        const start = bench_time.monotonicNanos();
        const result = c.roaring_bitmap_lazy_or(pair.c_a, pair.c_b, true) orelse unreachable;
        const elapsed = bench_time.monotonicNanos() - start;
        std.mem.doNotOptimizeAway(result);
        c.roaring_bitmap_free(result);
        return .{ .elapsed_ns = elapsed };
    }

    if (phase == .repair) {
        const result = c.roaring_bitmap_lazy_or(pair.c_a, pair.c_b, true) orelse unreachable;
        const start = bench_time.monotonicNanos();
        c.roaring_bitmap_repair_after_lazy(result);
        const elapsed = bench_time.monotonicNanos() - start;
        std.mem.doNotOptimizeAway(result);
        c.roaring_bitmap_free(result);
        return .{ .elapsed_ns = elapsed };
    }

    const start = bench_time.monotonicNanos();
    const result = c.roaring_bitmap_lazy_or(pair.c_a, pair.c_b, true) orelse unreachable;
    c.roaring_bitmap_repair_after_lazy(result);
    std.mem.doNotOptimizeAway(result);
    c.roaring_bitmap_free(result);
    return .{ .elapsed_ns = bench_time.monotonicNanos() - start };
}

fn measureA2Baseline(inputs: [NWAY_OPERANDS]*const RoaringBitmap, phase: Phase) !Measurement {
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const allocator = counting.allocator();

    switch (phase) {
        .construction => {
            counting.resetStats();
            const start = bench_time.monotonicNanos();
            var result = try constructA2Baseline(allocator, inputs);
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(&result);
            const stats = counting.snapshot();
            result.deinit();
            std.debug.assert(counting.stats.live_bytes == 0);
            return .{ .elapsed_ns = elapsed, .stats = stats };
        },
        .repair => {
            var result = try constructA2Baseline(allocator, inputs);
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
            var result = try constructA2Baseline(allocator, inputs);
            try result.repairAfterLazy();
            std.mem.doNotOptimizeAway(&result);
            result.deinit();
            const elapsed = bench_time.monotonicNanos() - start;
            std.debug.assert(counting.stats.live_bytes == 0);
            return .{ .elapsed_ns = elapsed, .stats = counting.snapshot() };
        },
    }
}

fn measureA2Transient(
    inputs: [NWAY_OPERANDS]*const RoaringBitmap,
    variant: RawrVariant,
    phase: Phase,
) !Measurement {
    std.debug.assert(variant != .baseline);
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const persistent_allocator = counting.allocator();

    switch (phase) {
        .construction => {
            counting.resetStats();
            const start = bench_time.monotonicNanos();
            const eligible_count = if (variant == .fba)
                classifyNway(inputs).eligible_groups
            else
                0;
            var backing = try TransientBacking.init(variant, persistent_allocator, eligible_count);
            defer backing.deinit();
            var result = try constructA2(
                persistent_allocator,
                backing.allocator(persistent_allocator),
                true,
                inputs,
            );
            defer result.deinit();
            backing.assertFullyConsumed();
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(&result);
            return .{
                .elapsed_ns = elapsed,
                .stats = counting.snapshot(),
                .arena_capacity = backing.capacity(),
            };
        },
        .repair => {
            const eligible_count = if (variant == .fba)
                classifyNway(inputs).eligible_groups
            else
                0;
            var backing = try TransientBacking.init(variant, persistent_allocator, eligible_count);
            defer backing.deinit();
            var result = try constructA2(
                persistent_allocator,
                backing.allocator(persistent_allocator),
                true,
                inputs,
            );
            defer result.deinit();
            backing.assertFullyConsumed();

            counting.resetStats();
            const start = bench_time.monotonicNanos();
            try result.repair();
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(&result);
            return .{
                .elapsed_ns = elapsed,
                .stats = counting.snapshot(),
                .arena_capacity = backing.capacity(),
            };
        },
        .combined => {
            counting.resetStats();
            const start = bench_time.monotonicNanos();
            const eligible_count = if (variant == .fba)
                classifyNway(inputs).eligible_groups
            else
                0;
            var backing = try TransientBacking.init(variant, persistent_allocator, eligible_count);
            errdefer backing.deinit();
            var result = try constructA2(
                persistent_allocator,
                backing.allocator(persistent_allocator),
                true,
                inputs,
            );
            errdefer result.deinit();
            backing.assertFullyConsumed();
            try result.repair();
            const arena_capacity = backing.capacity();
            backing.deinit();
            result.deinit();
            const elapsed = bench_time.monotonicNanos() - start;
            std.debug.assert(counting.stats.live_bytes == 0);
            return .{
                .elapsed_ns = elapsed,
                .stats = counting.snapshot(),
                .arena_capacity = arena_capacity,
            };
        },
    }
}

fn measureA2Production(inputs: [NWAY_OPERANDS]*const RoaringBitmap) !Measurement {
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const allocator = counting.allocator();
    counting.resetStats();
    const start = bench_time.monotonicNanos();
    var result = try RoaringBitmap.orMany(allocator, &inputs);
    std.mem.doNotOptimizeAway(&result);
    result.deinit();
    const elapsed = bench_time.monotonicNanos() - start;
    std.debug.assert(counting.stats.live_bytes == 0);
    return .{ .elapsed_ns = elapsed, .stats = counting.snapshot() };
}

fn constructCRoaringNway(inputs: [NWAY_OPERANDS]*const c.roaring_bitmap_t) *c.roaring_bitmap_t {
    const result = c.roaring_bitmap_lazy_or(inputs[0], inputs[1], true) orelse unreachable;
    for (inputs[2..]) |input| c.roaring_bitmap_lazy_or_inplace(result, input, true);
    return result;
}

fn measureCRoaringA2(
    inputs: [NWAY_OPERANDS]*const c.roaring_bitmap_t,
    phase: Phase,
) Measurement {
    if (phase == .construction) {
        const start = bench_time.monotonicNanos();
        const result = constructCRoaringNway(inputs);
        const elapsed = bench_time.monotonicNanos() - start;
        std.mem.doNotOptimizeAway(result);
        c.roaring_bitmap_free(result);
        return .{ .elapsed_ns = elapsed };
    }

    if (phase == .repair) {
        const result = constructCRoaringNway(inputs);
        const start = bench_time.monotonicNanos();
        c.roaring_bitmap_repair_after_lazy(result);
        const elapsed = bench_time.monotonicNanos() - start;
        std.mem.doNotOptimizeAway(result);
        c.roaring_bitmap_free(result);
        return .{ .elapsed_ns = elapsed };
    }

    const start = bench_time.monotonicNanos();
    const result = c.roaring_bitmap_or_many(
        NWAY_OPERANDS,
        @ptrCast(@constCast(&inputs)),
    ) orelse unreachable;
    std.mem.doNotOptimizeAway(result);
    c.roaring_bitmap_free(result);
    return .{ .elapsed_ns = bench_time.monotonicNanos() - start };
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

fn buildA1Transient(
    allocator: std.mem.Allocator,
    variant: RawrVariant,
    a: *const RoaringBitmap,
    b: *const RoaringBitmap,
) !RoaringBitmap {
    const eligible_count = if (variant == .fba)
        classifyPair(a, b).eligible_groups
    else
        0;
    var backing = try TransientBacking.init(variant, allocator, eligible_count);
    defer backing.deinit();

    var result = try constructA1(
        allocator,
        backing.allocator(allocator),
        variant != .baseline,
        a,
        b,
    );
    errdefer result.deinit();
    backing.assertFullyConsumed();
    try result.repair();
    return result.release();
}

fn buildA2Transient(
    allocator: std.mem.Allocator,
    variant: RawrVariant,
    inputs: [NWAY_OPERANDS]*const RoaringBitmap,
) !RoaringBitmap {
    const eligible_count = if (variant == .fba)
        classifyNway(inputs).eligible_groups
    else
        0;
    var backing = try TransientBacking.init(variant, allocator, eligible_count);
    defer backing.deinit();

    var result = try constructA2(
        allocator,
        backing.allocator(allocator),
        variant != .baseline,
        inputs,
    );
    errdefer result.deinit();
    backing.assertFullyConsumed();
    try result.repair();
    return result.release();
}

fn validateA1(
    allocator: std.mem.Allocator,
    pair: *const SparsePair,
) !void {
    var expected = try pair.a.lazyOr(allocator, &pair.b, true);
    defer expected.deinit();
    try expected.repairAfterLazy();

    const oracle = c.roaring_bitmap_lazy_or(pair.c_a, pair.c_b, true) orelse
        return error.CRoaringAllocFailed;
    defer c.roaring_bitmap_free(oracle);
    c.roaring_bitmap_repair_after_lazy(oracle);
    try expectCRoaringLogicalEqual(allocator, &expected, oracle);

    inline for (.{ RawrVariant.arena, RawrVariant.fba }) |variant| {
        var actual = try buildA1Transient(allocator, variant, &pair.a, &pair.b);
        defer actual.deinit();
        try expectByteIdentical(allocator, &actual, &expected);
        try expectCRoaringLogicalEqual(allocator, &actual, oracle);
    }
}

fn validateA1MixedOwnership(allocator: std.mem.Allocator) !void {
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    _ = try a.add(1);
    _ = try b.add(2);
    _ = try a.addRange(1 << 16, (1 << 16) + 2_999);
    _ = try b.addRange((1 << 16) + 3_000, (1 << 16) + 5_999);
    for (0..5_000) |i| {
        _ = try a.add((2 << 16) + @as(u32, @intCast(i * 2)));
        _ = try b.add((2 << 16) + @as(u32, @intCast(i * 2 + 1)));
    }

    switch (Container.fromTagged(a.containers[2])) {
        .bitset => |bitset| bitset.cardinality = -1,
        else => return error.MixedOwnershipFixtureMismatch,
    }
    switch (Container.fromTagged(b.containers[2])) {
        .bitset => |bitset| bitset.cardinality = -1,
        else => return error.MixedOwnershipFixtureMismatch,
    }

    const classification = classifyPair(&a, &b);
    if (classification.eligible_groups != 1 or
        classification.shared_groups != 3 or
        classification.unknown_groups != 1)
    {
        return error.MixedOwnershipFixtureMismatch;
    }

    var expected = try a.lazyOr(allocator, &b, true);
    defer expected.deinit();
    try expected.repairAfterLazy();

    inline for (.{ RawrVariant.arena, RawrVariant.fba }) |variant| {
        var actual = try buildA1Transient(allocator, variant, &a, &b);
        defer actual.deinit();
        try expectByteIdentical(allocator, &actual, &expected);
    }
}

fn validateA2(
    allocator: std.mem.Allocator,
    corpus: *const NwayCorpus,
) !void {
    const inputs = corpus.rawrInputs();
    var production = try RoaringBitmap.orMany(allocator, &inputs);
    defer production.deinit();

    const c_inputs = corpus.cInputs();
    const oracle = c.roaring_bitmap_or_many(
        NWAY_OPERANDS,
        @ptrCast(@constCast(&c_inputs)),
    ) orelse
        return error.CRoaringAllocFailed;
    defer c.roaring_bitmap_free(oracle);
    try expectCRoaringLogicalEqual(allocator, &production, oracle);

    var replica = try constructA2Baseline(allocator, inputs);
    defer replica.deinit();
    try replica.repairAfterLazy();
    try expectByteIdentical(allocator, &replica, &production);

    inline for (.{ RawrVariant.arena, RawrVariant.fba }) |variant| {
        var actual = try buildA2Transient(allocator, variant, inputs);
        defer actual.deinit();
        try expectByteIdentical(allocator, &actual, &production);
        try expectCRoaringLogicalEqual(allocator, &actual, oracle);
    }
}

fn validateA2MixedOwnership(allocator: std.mem.Allocator) !void {
    var bitmaps: [NWAY_OPERANDS]?RoaringBitmap = [_]?RoaringBitmap{null} ** NWAY_OPERANDS;
    defer {
        for (&bitmaps) |*maybe_bitmap| {
            if (maybe_bitmap.*) |*bitmap| bitmap.deinit();
        }
    }

    var inputs: [NWAY_OPERANDS]*const RoaringBitmap = undefined;
    for (0..NWAY_OPERANDS) |operand| {
        var bitmap = try RoaringBitmap.init(allocator);
        errdefer bitmap.deinit();
        _ = try bitmap.add(@intCast(operand));
        const dense_start = operand * DENSE_VALUES_PER_KEY;
        for (0..DENSE_VALUES_PER_KEY) |offset| {
            _ = try bitmap.add((1 << 16) + @as(u32, @intCast(dense_start + offset)));
        }
        bitmaps[operand] = bitmap;
        inputs[operand] = &bitmaps[operand].?;
    }

    const classification = classifyNway(inputs);
    if (classification.eligible_groups != 1 or classification.shared_groups != 2) {
        return error.MixedOwnershipFixtureMismatch;
    }

    var expected = try RoaringBitmap.orMany(allocator, &inputs);
    defer expected.deinit();
    inline for (.{ RawrVariant.arena, RawrVariant.fba }) |variant| {
        var actual = try buildA2Transient(allocator, variant, inputs);
        defer actual.deinit();
        try expectByteIdentical(allocator, &actual, &expected);
    }
}

fn validateImplementations(corpora: *const Corpora) !void {
    var debug_allocator = std.heap.DebugAllocator(.{}){};
    const allocator = debug_allocator.allocator();

    {
        try validateA1(allocator, &corpora.sparse_pair);
        try validateA1MixedOwnership(allocator);
        try validateA2(allocator, &corpora.sparse_nway);
        try validateA2(allocator, &corpora.dense_nway);
        try validateA2MixedOwnership(allocator);
    }

    if (debug_allocator.deinit() != .ok) return error.MemoryLeak;
}

fn runA1(
    pair: *const SparsePair,
    classification: Classification,
) !void {
    for (phases) |phase| {
        _ = try measureA1Baseline(pair, phase);
        printMeasurement(
            .sparse_2way,
            "baseline",
            phase,
            try measureA1Baseline(pair, phase),
            classification,
        );

        inline for (.{ RawrVariant.arena, RawrVariant.fba }) |variant| {
            _ = try measureA1Transient(pair, variant, phase);
            printMeasurement(
                .sparse_2way,
                @tagName(variant),
                phase,
                try measureA1Transient(pair, variant, phase),
                classification,
            );
        }

        _ = measureCRoaringA1(pair, phase);
        printMeasurement(
            .sparse_2way,
            "croaring",
            phase,
            measureCRoaringA1(pair, phase),
            classification,
        );
    }
}

fn runA2(
    experiment: Experiment,
    corpus: *const NwayCorpus,
    classification: Classification,
) !void {
    const inputs = corpus.rawrInputs();
    const c_inputs = corpus.cInputs();

    for (phases) |phase| {
        _ = try measureA2Baseline(inputs, phase);
        printMeasurement(
            experiment,
            "baseline",
            phase,
            try measureA2Baseline(inputs, phase),
            classification,
        );

        inline for (.{ RawrVariant.arena, RawrVariant.fba }) |variant| {
            _ = try measureA2Transient(inputs, variant, phase);
            printMeasurement(
                experiment,
                @tagName(variant),
                phase,
                try measureA2Transient(inputs, variant, phase),
                classification,
            );
        }

        _ = measureCRoaringA2(c_inputs, phase);
        printMeasurement(
            experiment,
            "croaring",
            phase,
            measureCRoaringA2(c_inputs, phase),
            classification,
        );
    }

    _ = try measureA2Production(inputs);
    printMeasurement(
        experiment,
        "production",
        .combined,
        try measureA2Production(inputs),
        classification,
    );
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

    bench_time.print("Validating output parity and allocator ownership...\n", .{});
    try validateImplementations(&corpora);
    bench_time.print("validation: byte parity, CRoaring parity, and leak checks passed\n", .{});

    bench_time.print("\nPhase A experiments\n", .{});
    bench_time.print("===================\n", .{});
    try runA1(&corpora.sparse_pair, classifications[0]);
    try runA2(.sparse_nway, &corpora.sparse_nway, classifications[1]);
    try runA2(.dense_nway, &corpora.dense_nway, classifications[2]);
}
