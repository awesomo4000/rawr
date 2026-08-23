// SPDX-License-Identifier: MPL-2.0

//! Correctness and accounting setup for spec 48 tiny-bitmap measurements.
//! This executable intentionally performs no timing.

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const fixtures = @import("tiny_bench_fixtures.zig");

const RoaringBitmap = rawr.RoaringBitmap;
const allocator = std.heap.smp_allocator;

const expected_sweep_hashes = [fixtures.shapes.len][fixtures.sweep_cardinalities.len]u64{
    .{
        0x49b49b797b6b8cce,
        0x3e6136bff55b705e,
        0xa9ea73f383260c91,
        0x9381c5cd9d18da87,
        0x4bc247f5ffc156c5,
        0xc2442ca06a788f5b,
        0x7b5e42066702395f,
        0xb8861760354d85b3,
        0xb1714cd01b72c017,
        0xd937a4dbdf708ba3,
        0x587e5dc52254bb23,
        0x9407564ac450c213,
    },
    .{
        0x5e34b4bf4b53722d,
        0x78b45e1732bc6154,
        0x2ee09b0e3190d7f0,
        0xd82a2907a7e6b116,
        0xee917ab0ee45f321,
        0x6cb95c6b59e2fda8,
        0xee0c5daebafdfd90,
        0xbb19c4ce63129d85,
        0xda05a6dc6dfbb6ba,
        0xb70908907ce6a748,
        0x41fea3c0afa62980,
        0xe5022bfb787684b0,
    },
    .{
        0x5cacc00e27f02314,
        0x6407297656dfc998,
        0xd77f1358262fb977,
        0xfb1fd4cafb668d4d,
        0x3efdffff9c136d33,
        0x0601646c138ecdb1,
        0x27eb288ce9fba525,
        0x0aedba27058a5609,
        0x74d526ec29ea994d,
        0x6f0ec65eea1682a9,
        0x0a223db2e992e549,
        0xd2ae474b7eeb9a49,
    },
};
const expected_mixed_cardinality_hash: u64 = 0x0b023a26a773e913;
const expected_mixed_full_hash: u64 = 0x8f4d88269788fc3a;

const Command = enum {
    hashes,
    check,
    mutation_interleaved,
    mutation_sequential,
    mutation_structural,
};

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(allocator);
    defer args.deinit();
    _ = args.skip();

    const command_name = args.next() orelse return error.MissingCommand;
    if (args.next() != null) return error.TooManyArguments;
    const command = std.meta.stringToEnum(Command, command_name) orelse return error.UnknownCommand;

    installCRoaringHooks();
    switch (command) {
        .hashes => try printHashes(),
        .check => try runChecks(),
        .mutation_interleaved => try verifyMutation(.interleaved),
        .mutation_sequential => try verifyMutation(.sequential),
        .mutation_structural => try verifyStructuralMutations(),
    }
}

fn printHashes() !void {
    for (fixtures.shapes) |shape| {
        for (fixtures.sweep_cardinalities) |cardinality| {
            var pool = try fixtures.generateSweepPool(allocator, shape, cardinality);
            defer pool.deinit();
            std.debug.print("SWEEP\t{s}\t{d}\t0x{x:0>16}\n", .{ shape.name(), cardinality, pool.hash() });
        }
    }

    var mixed = try fixtures.generateMixedCorpus(allocator, .correct);
    defer mixed.deinit();
    std.debug.print("MIXED_CARDINALITIES\t0x{x:0>16}\n", .{mixed.cardinality_hash});
    std.debug.print("MIXED_FULL\t0x{x:0>16}\n", .{mixed.full_hash});
    std.debug.print("QUANTILES\tmedian={d}\tp99={d}\n", .{ mixed.median, mixed.p99 });
}

fn runChecks() !void {
    try verifyPinnedHashesAndFixtures();
    try verifyMutation(.interleaved);
    try verifyMutation(.sequential);
    try verifyStructuralMutations();
    try verifyAccountingAndLifecycle();
    std.debug.print("tiny setup: OK\n", .{});
}

fn verifyStructuralMutations() !void {
    try fixtures.verifyStructuralMutationGuards(allocator);
    std.debug.print("mutation structural: caught all seeded defects\n", .{});
}

fn verifyPinnedHashesAndFixtures() !void {
    for (fixtures.shapes, 0..) |shape, shape_index| {
        for (fixtures.sweep_cardinalities, 0..) |cardinality, cardinality_index| {
            var pool = try fixtures.generateSweepPool(allocator, shape, cardinality);
            defer pool.deinit();
            const expected = expected_sweep_hashes[shape_index][cardinality_index];
            if (expected == 0) return error.UnpinnedSweepHash;
            if (pool.hash() != expected) return error.SweepHashMismatch;

            try validateCrossImplementation(pool.fixture(0));
            if (pool.fixture_count > 1) try validateCrossImplementation(pool.fixture(pool.fixture_count - 1));
        }
    }

    var mixed = try fixtures.generateMixedCorpus(allocator, .correct);
    defer mixed.deinit();
    if (expected_mixed_cardinality_hash == 0 or expected_mixed_full_hash == 0) {
        return error.UnpinnedMixedHash;
    }
    if (mixed.cardinality_hash != expected_mixed_cardinality_hash) return error.MixedCardinalityHashMismatch;
    if (mixed.full_hash != expected_mixed_full_hash) return error.MixedFullHashMismatch;
    if (mixed.median < 1 or mixed.median > 2) return error.MixedMedianOutOfRange;
    if (mixed.p99 < 1000 or mixed.p99 > 20_000) return error.MixedP99OutOfRange;

    const sample_indices = [_]usize{ 0, 1, mixed.cardinalities.len / 2, mixed.cardinalities.len - 1 };
    for (sample_indices) |index| {
        const values = try allocator.alloc(u32, mixed.cardinalities[index]);
        defer allocator.free(values);
        try fixtures.fillSpread(values, fixtures.mixedValueSeed(index, mixed.cardinalities[index]));
        try validateCrossImplementation(values);
    }
    std.debug.print("quantiles: median={d} p99={d}\n", .{ mixed.median, mixed.p99 });
}

fn verifyMutation(pattern: fixtures.SharingPattern) !void {
    if (expected_mixed_cardinality_hash == 0 or expected_mixed_full_hash == 0) {
        return error.UnpinnedMixedHash;
    }
    var mixed = try fixtures.generateMixedCorpus(allocator, pattern);
    defer mixed.deinit();
    const cardinality_failed = mixed.cardinality_hash != expected_mixed_cardinality_hash;
    const full_failed = mixed.full_hash != expected_mixed_full_hash;
    if (!cardinality_failed and !full_failed) return error.SharedStreamMutationSurvived;

    switch (pattern) {
        .interleaved => if (!cardinality_failed) return error.InterleavedMutationDidNotMoveCardinalities,
        .sequential => {
            if (cardinality_failed) return error.SequentialMutationMovedCardinalities;
            if (!full_failed) return error.SequentialMutationDidNotMoveFullCorpus;
        },
        .correct => unreachable,
    }
    std.debug.print("mutation {s}: caught cardinality={any} full={any}\n", .{
        @tagName(pattern),
        cardinality_failed,
        full_failed,
    });
}

// A spread bitmap spans at most 153 containers in the pinned 10M universe.
// Source and decoded payload sizes can all differ, so leave room for both
// streams plus top-level arrays and serialization buffers without allocating
// metadata through the allocator being measured.
const max_histogram_entries = 512;

const HistogramEntry = struct {
    size: usize = 0,
    count: u64 = 0,
};

const AllocationStats = struct {
    alloc_calls: u64 = 0,
    free_calls: u64 = 0,
    resize_calls: u64 = 0,
    requested_bytes: u64 = 0,
    live_bytes: u64 = 0,
    peak_live_bytes: u64 = 0,
    histogram_len: usize = 0,
    histogram: [max_histogram_entries]HistogramEntry = @splat(.{}),

    fn recordRequest(self: *AllocationStats, size: usize) void {
        self.requested_bytes +|= @intCast(size);
        for (self.histogram[0..self.histogram_len]) |*entry| {
            if (entry.size == size) {
                entry.count += 1;
                return;
            }
        }
        if (self.histogram_len == self.histogram.len) @panic("tiny accounting histogram exhausted");
        self.histogram[self.histogram_len] = .{ .size = size, .count = 1 };
        self.histogram_len += 1;
    }

    fn addLive(self: *AllocationStats, size: usize) void {
        self.live_bytes +|= @intCast(size);
        self.peak_live_bytes = @max(self.peak_live_bytes, self.live_bytes);
    }

    fn removeLive(self: *AllocationStats, size: usize) void {
        std.debug.assert(self.live_bytes >= size);
        self.live_bytes -= @intCast(size);
    }
};

const TrackingAllocator = struct {
    backing: std.mem.Allocator,
    stats: AllocationStats = .{},

    fn init(backing: std.mem.Allocator) TrackingAllocator {
        return .{ .backing = backing };
    }

    fn allocator(self: *TrackingAllocator) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable: std.mem.Allocator.VTable = .{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    };

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.backing.rawAlloc(len, alignment, ret_addr) orelse return null;
        self.stats.alloc_calls += 1;
        self.stats.recordRequest(len);
        self.stats.addLive(len);
        return result;
    }

    fn resize(
        ctx: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        ret_addr: usize,
    ) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        self.stats.resize_calls += 1;
        if (!self.backing.rawResize(memory, alignment, new_len, ret_addr)) return false;
        self.stats.recordRequest(new_len);
        self.stats.removeLive(memory.len);
        self.stats.addLive(new_len);
        return true;
    }

    fn remap(
        ctx: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        ret_addr: usize,
    ) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        self.stats.resize_calls += 1;
        const result = self.backing.rawRemap(memory, alignment, new_len, ret_addr) orelse return null;
        self.stats.recordRequest(new_len);
        self.stats.removeLive(memory.len);
        self.stats.addLive(new_len);
        return result;
    }

    fn free(
        ctx: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        ret_addr: usize,
    ) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        self.stats.free_calls += 1;
        self.stats.removeLive(memory.len);
        self.backing.rawFree(memory, alignment, ret_addr);
    }
};

const Checkpoints = struct {
    create: AllocationStats,
    build: AllocationStats,
    serialize: AllocationStats,
    deserialize: AllocationStats,
    teardown: AllocationStats,
    cardinality: u64,
    serialized_bytes: usize,
};

fn runRawrLifecycle(backing: std.mem.Allocator, values: []const u32) !Checkpoints {
    var tracking = TrackingAllocator.init(backing);
    const tracked = tracking.allocator();

    var source = try RoaringBitmap.init(tracked);
    errdefer source.deinit();
    const create = tracking.stats;

    for (values) |value| _ = try source.add(value);
    const build = tracking.stats;

    const serialized_size = source.serializedSizeInBytes();
    const bytes = try tracked.alloc(u8, serialized_size);
    errdefer tracked.free(bytes);
    var writer = std.Io.Writer.fixed(bytes);
    try source.serializeToWriter(&writer);
    const serialize = tracking.stats;

    var decoded = try RoaringBitmap.deserialize(tracked, bytes);
    errdefer decoded.deinit();
    const deserialize = tracking.stats;
    const cardinality = decoded.cardinality();

    decoded.deinit();
    tracked.free(bytes);
    source.deinit();
    const teardown = tracking.stats;

    return .{
        .create = create,
        .build = build,
        .serialize = serialize,
        .deserialize = deserialize,
        .teardown = teardown,
        .cardinality = cardinality,
        .serialized_bytes = serialized_size,
    };
}

const PlainList = struct {
    values: []u32,

    fn deinit(self: *PlainList, list_allocator: std.mem.Allocator) void {
        list_allocator.free(self.values);
        self.* = undefined;
    }
};

fn runPlainLifecycle(backing: std.mem.Allocator, values: []const u32) !Checkpoints {
    var tracking = TrackingAllocator.init(backing);
    const tracked = tracking.allocator();
    const create = tracking.stats;

    var source = PlainList{ .values = try tracked.dupe(u32, values) };
    errdefer source.deinit(tracked);
    const build = tracking.stats;

    const serialized_size = 4 + values.len * @sizeOf(u32);
    const bytes = try tracked.alloc(u8, serialized_size);
    errdefer tracked.free(bytes);
    std.mem.writeInt(u32, bytes[0..4], @intCast(values.len), .little);
    for (values, 0..) |value, index| {
        std.mem.writeInt(u32, bytes[4 + index * 4 ..][0..4], value, .little);
    }
    const serialize = tracking.stats;

    const decoded_len = std.mem.readInt(u32, bytes[0..4], .little);
    if (decoded_len != values.len) return error.PlainListLengthMismatch;
    var decoded = PlainList{ .values = try tracked.alloc(u32, decoded_len) };
    errdefer decoded.deinit(tracked);
    for (decoded.values, 0..) |*value, index| {
        value.* = std.mem.readInt(u32, bytes[4 + index * 4 ..][0..4], .little);
    }
    const deserialize = tracking.stats;
    const cardinality = decoded.values.len;

    decoded.deinit(tracked);
    tracked.free(bytes);
    source.deinit(tracked);
    const teardown = tracking.stats;

    return .{
        .create = create,
        .build = build,
        .serialize = serialize,
        .deserialize = deserialize,
        .teardown = teardown,
        .cardinality = cardinality,
        .serialized_bytes = serialized_size,
    };
}

const CHeader = extern struct {
    base: ?*anyopaque,
    size: usize,
    magic: usize,
};

const c_header_magic: usize = 0x52415752;
var cr_stats: AllocationStats = .{};

fn installCRoaringHooks() void {
    c.roaring_init_memory_hook(.{
        .malloc = crMalloc,
        .realloc = crRealloc,
        .calloc = crCalloc,
        .free = crFree,
        .aligned_malloc = crAlignedMalloc,
        .aligned_free = crAlignedFree,
    });
}

fn crMalloc(size: usize) callconv(.c) ?*anyopaque {
    const result = crAllocateUntracked(size, @alignOf(std.c.max_align_t)) orelse return null;
    cr_stats.alloc_calls += 1;
    cr_stats.recordRequest(size);
    cr_stats.addLive(size);
    return result;
}

fn crCalloc(count: usize, element_size: usize) callconv(.c) ?*anyopaque {
    const size = std.math.mul(usize, count, element_size) catch return null;
    const result = crMalloc(size) orelse return null;
    @memset(@as([*]u8, @ptrCast(result))[0..size], 0);
    return result;
}

fn crRealloc(memory: ?*anyopaque, new_size: usize) callconv(.c) ?*anyopaque {
    if (memory == null) return crMalloc(new_size);
    if (new_size == 0) {
        crFree(memory);
        return null;
    }

    const old_header = crHeader(memory.?);
    const old_size = old_header.size;
    const result = crAllocateUntracked(new_size, @alignOf(std.c.max_align_t)) orelse return null;
    const copy_len = @min(old_size, new_size);
    @memcpy(
        @as([*]u8, @ptrCast(result))[0..copy_len],
        @as([*]const u8, @ptrCast(memory.?))[0..copy_len],
    );
    crFreeUntracked(memory.?);

    cr_stats.resize_calls += 1;
    cr_stats.recordRequest(new_size);
    cr_stats.removeLive(old_size);
    cr_stats.addLive(new_size);
    return result;
}

fn crFree(memory: ?*anyopaque) callconv(.c) void {
    const ptr = memory orelse return;
    const size = crHeader(ptr).size;
    cr_stats.free_calls += 1;
    cr_stats.removeLive(size);
    crFreeUntracked(ptr);
}

fn crAlignedMalloc(alignment: usize, size: usize) callconv(.c) ?*anyopaque {
    const result = crAllocateUntracked(size, alignment) orelse return null;
    cr_stats.alloc_calls += 1;
    cr_stats.recordRequest(size);
    cr_stats.addLive(size);
    return result;
}

fn crAlignedFree(memory: ?*anyopaque) callconv(.c) void {
    crFree(memory);
}

fn crAllocateUntracked(size: usize, requested_alignment: usize) ?*anyopaque {
    const alignment = @max(requested_alignment, @alignOf(CHeader));
    if (!std.math.isPowerOfTwo(alignment)) return null;
    const overhead = std.math.add(usize, @sizeOf(CHeader), alignment - 1) catch return null;
    const total = std.math.add(usize, @max(size, 1), overhead) catch return null;
    const base = std.c.malloc(total) orelse return null;
    const base_addr = @intFromPtr(base);
    const user_addr = std.mem.alignForward(usize, base_addr + @sizeOf(CHeader), alignment);
    const header: *CHeader = @ptrFromInt(user_addr - @sizeOf(CHeader));
    header.* = .{ .base = base, .size = size, .magic = c_header_magic };
    return @ptrFromInt(user_addr);
}

fn crHeader(memory: *anyopaque) *CHeader {
    const header: *CHeader = @ptrFromInt(@intFromPtr(memory) - @sizeOf(CHeader));
    std.debug.assert(header.magic == c_header_magic);
    return header;
}

fn crFreeUntracked(memory: *anyopaque) void {
    const header = crHeader(memory);
    const base = header.base;
    header.magic = 0;
    std.c.free(base);
}

fn runCRoaringLifecycle(values: []const u32) !Checkpoints {
    if (cr_stats.live_bytes != 0) return error.CRoaringTrackerNotDrained;
    cr_stats = .{};

    const source = c.roaring_bitmap_create() orelse return error.CRoaringAllocFailed;
    errdefer c.roaring_bitmap_free(source);
    const create = cr_stats;

    c.roaring_bitmap_add_many(source, values.len, values.ptr);
    const build = cr_stats;

    const serialized_size = c.roaring_bitmap_portable_size_in_bytes(source);
    const bytes_ptr = c.roaring_malloc(serialized_size) orelse return error.CRoaringAllocFailed;
    errdefer c.roaring_free(bytes_ptr);
    const bytes: []u8 = @as([*]u8, @ptrCast(bytes_ptr))[0..serialized_size];
    if (c.roaring_bitmap_portable_serialize(source, @ptrCast(bytes.ptr)) != serialized_size) {
        return error.CRoaringSerializeFailed;
    }
    const serialize = cr_stats;

    const decoded = c.roaring_bitmap_portable_deserialize_safe(@ptrCast(bytes.ptr), bytes.len) orelse
        return error.CRoaringDeserializeFailed;
    errdefer c.roaring_bitmap_free(decoded);
    const deserialize = cr_stats;
    const cardinality = c.roaring_bitmap_get_cardinality(decoded);

    c.roaring_bitmap_free(decoded);
    c.roaring_free(bytes_ptr);
    c.roaring_bitmap_free(source);
    const teardown = cr_stats;

    return .{
        .create = create,
        .build = build,
        .serialize = serialize,
        .deserialize = deserialize,
        .teardown = teardown,
        .cardinality = cardinality,
        .serialized_bytes = serialized_size,
    };
}

fn verifyAccountingAndLifecycle() !void {
    var pool = try fixtures.generateSweepPool(allocator, .spread, 6);
    defer pool.deinit();
    const values = pool.fixture(0);

    const rawr_smp = try runRawrLifecycle(std.heap.smp_allocator, values);
    const rawr_libc = try runRawrLifecycle(std.heap.c_allocator, values);
    const plain_smp = try runPlainLifecycle(std.heap.smp_allocator, values);
    const plain_libc = try runPlainLifecycle(std.heap.c_allocator, values);
    const croaring = try runCRoaringLifecycle(values);

    for ([_]Checkpoints{ rawr_smp, rawr_libc, plain_smp, plain_libc, croaring }) |report| {
        if (report.cardinality != values.len) return error.LifecycleCardinalityMismatch;
        if (report.teardown.live_bytes != 0) return error.LifecycleLeak;
        if (report.deserialize.live_bytes < report.serialize.live_bytes) return error.BadLifetimeCheckpoints;
        if (report.serialized_bytes == 0) return error.EmptySerialization;
        if (report.deserialize.histogram_len == 0) return error.MissingAllocationHistogram;
    }
    if (croaring.serialize.alloc_calls <= croaring.build.alloc_calls) return error.CallerBufferNotCounted;
    if (croaring.serialize.live_bytes < croaring.build.live_bytes + croaring.serialized_bytes) {
        return error.CallerBufferBytesNotCounted;
    }
}

fn validateCrossImplementation(values: []const u32) !void {
    var bitmap = try RoaringBitmap.init(allocator);
    defer bitmap.deinit();
    for (values) |value| _ = try bitmap.add(value);

    const rawr_size = bitmap.serializedSizeInBytes();
    const rawr_bytes = try allocator.alloc(u8, rawr_size);
    defer allocator.free(rawr_bytes);
    var writer = std.Io.Writer.fixed(rawr_bytes);
    try bitmap.serializeToWriter(&writer);

    const oracle = c.roaring_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring_bitmap_free(oracle);
    c.roaring_bitmap_add_many(oracle, values.len, values.ptr);
    const oracle_size = c.roaring_bitmap_portable_size_in_bytes(oracle);
    const oracle_bytes = try allocator.alloc(u8, oracle_size);
    defer allocator.free(oracle_bytes);
    if (c.roaring_bitmap_portable_serialize(oracle, @ptrCast(oracle_bytes.ptr)) != oracle_size) {
        return error.CRoaringSerializeFailed;
    }

    const c_from_rawr = c.roaring_bitmap_portable_deserialize_safe(
        @ptrCast(rawr_bytes.ptr),
        rawr_bytes.len,
    ) orelse return error.CRoaringDeserializeFailed;
    defer c.roaring_bitmap_free(c_from_rawr);
    if (c.roaring_bitmap_get_cardinality(c_from_rawr) != values.len) return error.CrossCardinalityMismatch;

    const c_values = try allocator.alloc(u32, values.len);
    defer allocator.free(c_values);
    c.roaring_bitmap_to_uint32_array(c_from_rawr, c_values.ptr);
    if (!std.mem.eql(u32, values, c_values)) return error.CrossValueMismatch;

    var rawr_from_c = try RoaringBitmap.deserializeSafe(allocator, oracle_bytes);
    defer rawr_from_c.deinit();
    if (rawr_from_c.cardinality() != values.len) return error.CrossCardinalityMismatch;
    const rawr_values = try rawr_from_c.toArrayAlloc(allocator);
    defer allocator.free(rawr_values);
    if (!std.mem.eql(u32, values, rawr_values)) return error.CrossValueMismatch;
}
