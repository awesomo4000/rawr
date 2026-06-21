const std = @import("std");
const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;
const TaggedPtr = @import("container.zig").TaggedPtr;

const Allocator = std.mem.Allocator;

pub const Profile = enum { sparse, dense, full, runs, single, boundary };

pub const ChunkProfile = struct {
    key: u16,
    profile: Profile,
};

pub const Generated = struct {
    bm: RoaringBitmap,
    values: []u32,
    allocator: Allocator,

    pub fn deinit(self: *Generated) void {
        self.bm.deinit();
        self.allocator.free(self.values);
    }
};

pub fn build(
    allocator: Allocator,
    rng: std.Random,
    chunks: []const ChunkProfile,
    run_optimize: bool,
) !Generated {
    var bm = try RoaringBitmap.init(allocator);
    errdefer bm.deinit();

    var values = std.array_list.Managed(u32).init(allocator);
    defer values.deinit();

    for (chunks) |chunk| {
        try addProfile(&bm, &values, rng, chunk.key, chunk.profile);
    }

    if (run_optimize) {
        _ = try bm.runOptimize();
    }

    std.mem.sortUnstable(u32, values.items, {}, std.sort.asc(u32));
    const unique_len = dedupeSorted(values.items);
    values.items.len = unique_len;

    return .{
        .bm = bm,
        .values = try values.toOwnedSlice(),
        .allocator = allocator,
    };
}

pub fn randomMixed(
    allocator: Allocator,
    rng: std.Random,
    max_chunks: usize,
    run_optimize: bool,
) !Generated {
    const chunk_count = rng.intRangeAtMost(usize, 0, max_chunks);
    const chunks = try allocator.alloc(ChunkProfile, chunk_count);
    defer allocator.free(chunks);

    for (chunks, 0..) |*chunk, i| {
        chunk.* = .{
            .key = @intCast(i),
            .profile = randomProfile(rng),
        };
    }

    return build(allocator, rng, chunks, run_optimize);
}

fn randomProfile(rng: std.Random) Profile {
    const weighted = [_]Profile{
        .sparse,
        .single,
        .boundary,
        .dense,
        .dense,
        .runs,
        .runs,
        .full,
    };
    return weighted[rng.uintLessThan(usize, weighted.len)];
}

fn addProfile(
    bm: *RoaringBitmap,
    values: *std.array_list.Managed(u32),
    rng: std.Random,
    key: u16,
    profile: Profile,
) !void {
    switch (profile) {
        .sparse => {
            const count = rng.intRangeAtMost(u16, 16, 160);
            const offset = rng.int(u16);
            try addScatteredValues(bm, values, key, count, offset, 521);
        },
        .dense => {
            const count = rng.intRangeAtMost(u16, 4097, 6000);
            const offset = rng.int(u16);
            try addScatteredValues(bm, values, key, count, offset, 73);
        },
        .full => {
            try addRange(bm, values, key, 0, std.math.maxInt(u16));
        },
        .runs => {
            const run_count = rng.intRangeAtMost(u16, 3, 10);
            const len = rng.intRangeAtMost(u16, 50, 200);
            const gap: u16 = 4096;
            const base = rng.intRangeAtMost(u16, 0, 512);

            for (0..run_count) |i| {
                const start: u16 = @intCast(@as(u32, base) + (@as(u32, @intCast(i)) * gap));
                const end: u16 = start + len - 1;
                try addRange(bm, values, key, start, end);
            }
        },
        .single => {
            const low = rng.int(u16);
            try addValue(bm, values, key, low);
        },
        .boundary => {
            try addValue(bm, values, key, 0);
            try addValue(bm, values, key, 1);
            try addValue(bm, values, key, 65534);
            try addValue(bm, values, key, 65535);
        },
    }
}

fn addScatteredValues(
    bm: *RoaringBitmap,
    values: *std.array_list.Managed(u32),
    key: u16,
    count: u16,
    offset: u16,
    comptime step: u32,
) !void {
    for (0..count) |i| {
        const low: u16 = @truncate((@as(u32, @intCast(i)) * step + offset) & 0xFFFF);
        try addValue(bm, values, key, low);
    }
}

fn addRange(
    bm: *RoaringBitmap,
    values: *std.array_list.Managed(u32),
    key: u16,
    start: u16,
    end: u16,
) !void {
    _ = try bm.addRange(combine(key, start), combine(key, end));

    var low = start;
    while (true) {
        try values.append(combine(key, low));
        if (low == end) break;
        low += 1;
    }
}

fn addValue(
    bm: *RoaringBitmap,
    values: *std.array_list.Managed(u32),
    key: u16,
    low: u16,
) !void {
    const value = combine(key, low);
    _ = try bm.add(value);
    try values.append(value);
}

fn combine(key: u16, low: u16) u32 {
    return (@as(u32, key) << 16) | low;
}

fn dedupeSorted(values: []u32) usize {
    if (values.len == 0) return 0;

    var write: usize = 1;
    for (values[1..]) |value| {
        if (value != values[write - 1]) {
            values[write] = value;
            write += 1;
        }
    }
    return write;
}

fn expectSortedUnique(values: []const u32) !void {
    for (values[1..], 1..) |value, i| {
        try std.testing.expect(values[i - 1] < value);
    }
}

test "generator can force mixed container types" {
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();
    defer std.testing.expectEqual(.ok, gpa.deinit()) catch @panic("leak");

    var prng = std.Random.DefaultPrng.init(0x5151);
    const rng = prng.random();

    const chunks = [_]ChunkProfile{
        .{ .key = 0, .profile = .sparse },
        .{ .key = 1, .profile = .dense },
        .{ .key = 2, .profile = .runs },
        .{ .key = 3, .profile = .full },
    };

    var generated = try build(allocator, rng, &chunks, true);
    defer generated.deinit();

    try std.testing.expectEqual(@as(u32, 4), generated.bm.size);
    try std.testing.expectEqual(TaggedPtr.ContainerType.array, generated.bm.containers[0].getType());
    try std.testing.expectEqual(TaggedPtr.ContainerType.bitset, generated.bm.containers[1].getType());
    try std.testing.expectEqual(TaggedPtr.ContainerType.run, generated.bm.containers[2].getType());
    try std.testing.expectEqual(TaggedPtr.ContainerType.run, generated.bm.containers[3].getType());

    try expectSortedUnique(generated.values);
    try std.testing.expectEqual(@as(u64, generated.values.len), generated.bm.cardinality());
}
