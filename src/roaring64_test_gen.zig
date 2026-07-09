const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Profile = enum { sparse, dense, runs, full, single, boundary };

pub const BucketProfile = struct {
    hi: u32,
    profile: Profile,
};

pub const Range = struct {
    lo: u64,
    hi: u64,
};

pub const Generated = struct {
    values: []u64,
    ranges: []Range,
    allocator: Allocator,

    pub fn deinit(self: *Generated) void {
        self.allocator.free(self.values);
        self.allocator.free(self.ranges);
    }
};

pub fn build(allocator: Allocator, rng: std.Random, buckets: []const BucketProfile) !Generated {
    var values = std.array_list.Managed(u64).init(allocator);
    defer values.deinit();

    var ranges = std.array_list.Managed(Range).init(allocator);
    defer ranges.deinit();

    for (buckets) |bucket| {
        try addProfile(&values, &ranges, rng, bucket.hi, bucket.profile);
    }

    std.mem.sortUnstable(u64, values.items, {}, std.sort.asc(u64));
    values.items.len = dedupeSorted(values.items);

    return .{
        .values = try values.toOwnedSlice(),
        .ranges = try ranges.toOwnedSlice(),
        .allocator = allocator,
    };
}

pub fn randomMixed(allocator: Allocator, rng: std.Random, max_buckets: usize) !Generated {
    const count = rng.intRangeAtMost(usize, 0, @min(max_buckets, high_key_pool.len));
    const buckets = try allocator.alloc(BucketProfile, count);
    defer allocator.free(buckets);

    var used = [_]bool{false} ** high_key_pool.len;
    for (buckets) |*bucket| {
        var pool_idx = rng.uintLessThan(usize, high_key_pool.len);
        while (used[pool_idx]) {
            pool_idx = (pool_idx + 1) % high_key_pool.len;
        }
        used[pool_idx] = true;

        bucket.* = .{
            .hi = high_key_pool[pool_idx],
            .profile = randomProfile(rng),
        };
    }

    return build(allocator, rng, buckets);
}

pub fn toBitmap(comptime Bitmap: type, allocator: Allocator, generated: *const Generated) !Bitmap {
    var bm = try Bitmap.init(allocator);
    errdefer bm.deinit();

    for (generated.ranges) |range| {
        try bm.addRange(range.lo, range.hi);
    }
    try bm.addMany(generated.values);

    return bm;
}

fn randomProfile(rng: std.Random) Profile {
    const weighted = [_]Profile{
        .sparse,
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
    values: *std.array_list.Managed(u64),
    ranges: *std.array_list.Managed(Range),
    rng: std.Random,
    hi: u32,
    profile: Profile,
) !void {
    switch (profile) {
        .sparse => {
            const low_key = rng.int(u16);
            const count = rng.intRangeAtMost(u16, 8, 120);
            const offset = rng.int(u16);
            try addScatteredValues(values, hi, low_key, count, offset, 521);
        },
        .dense => {
            const low_key = rng.int(u16);
            const count = rng.intRangeAtMost(u16, 4097, 5200);
            const offset = rng.int(u16);
            try addScatteredValues(values, hi, low_key, count, offset, 13);
        },
        .runs => {
            const low_key = rng.int(u16);
            const run_count = rng.intRangeAtMost(u16, 2, 6);
            const len = rng.intRangeAtMost(u16, 64, 192);
            const base = rng.intRangeAtMost(u16, 0, 1024);

            for (0..run_count) |i| {
                const start: u16 = @intCast(@as(u32, base) + @as(u32, @intCast(i)) * 4096);
                const end: u16 = start + len - 1;
                try addRange(values, ranges, hi, low32(low_key, start), low32(low_key, end));
            }
        },
        .full => {
            const low_key = rng.int(u16);
            try addRange(values, ranges, hi, low32(low_key, 0), low32(low_key, std.math.maxInt(u16)));
        },
        .single => {
            try addValue(values, hi, rng.int(u32));
        },
        .boundary => {
            try addValue(values, hi, 0);
            try addValue(values, hi, 1);
            try addValue(values, hi, std.math.maxInt(u16));
            try addValue(values, hi, @as(u32, 1) << 16);
            try addValue(values, hi, std.math.maxInt(u32) - 1);
            try addValue(values, hi, std.math.maxInt(u32));
        },
    }
}

fn addScatteredValues(
    values: *std.array_list.Managed(u64),
    hi: u32,
    low_key: u16,
    count: u16,
    offset: u16,
    comptime step: u32,
) !void {
    for (0..count) |i| {
        const low: u16 = @truncate(@as(u32, @intCast(i)) * step + offset);
        try addValue(values, hi, low32(low_key, low));
    }
}

fn addRange(values: *std.array_list.Managed(u64), ranges: *std.array_list.Managed(Range), hi: u32, start: u32, end: u32) !void {
    const lo_value = combine(hi, start);
    const hi_value = combine(hi, end);
    try ranges.append(.{ .lo = lo_value, .hi = hi_value });

    var low = start;
    while (true) {
        try values.append(combine(hi, low));
        if (low == end) break;
        low += 1;
    }
}

fn addValue(values: *std.array_list.Managed(u64), hi: u32, low: u32) !void {
    try values.append(combine(hi, low));
}

fn low32(key: u16, low: u16) u32 {
    return (@as(u32, key) << 16) | low;
}

fn combine(hi: u32, low: u32) u64 {
    return (@as(u64, hi) << 32) | low;
}

fn dedupeSorted(values: []u64) usize {
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

fn expectSortedUnique(values: []const u64) !void {
    for (values[1..], 1..) |value, i| {
        try std.testing.expect(values[i - 1] < value);
    }
}

const high_key_pool = [_]u32{
    0,
    1,
    2,
    17,
    18,
    0x0001_0000,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_fffe,
    0xffff_ffff,
};

test "Roaring64 generator covers edge buckets and container payload profiles" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x6400_600d);
    const rng = prng.random();

    const specs = [_]BucketProfile{
        .{ .hi = 0, .profile = .sparse },
        .{ .hi = 1, .profile = .dense },
        .{ .hi = 0x8000_0000, .profile = .runs },
        .{ .hi = 0xffff_ffff, .profile = .full },
        .{ .hi = 17, .profile = .boundary },
    };

    var generated = try build(allocator, rng, &specs);
    defer generated.deinit();

    try expectSortedUnique(generated.values);
    try std.testing.expect(generated.values[0] >> 32 == 0);
    try std.testing.expect(generated.values[generated.values.len - 1] >> 32 == std.math.maxInt(u32));
    try std.testing.expect(generated.ranges.len >= 2);

    const Roaring64Bitmap = @import("roaring64.zig").Roaring64Bitmap;
    var bm = try toBitmap(Roaring64Bitmap, allocator, &generated);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, generated.values.len), bm.cardinality());
    try std.testing.expect(bm.contains(generated.values[0]));
    try std.testing.expect(bm.contains(generated.values[generated.values.len - 1]));
}
