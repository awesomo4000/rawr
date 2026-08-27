// SPDX-License-Identifier: MPL-2.0

//! Deterministic fixture generation for the tiny-bitmap measurement harness.

const std = @import("std");

pub const sweep_cardinalities = [_]u32{ 0, 1, 2, 4, 6, 8, 12, 16, 20, 32, 64, 128 };
pub const sweep_pool_size: usize = 1024;
pub const sweep_iterations: usize = sweep_pool_size * 100;
pub const spread_universe: u32 = 10_000_000;
pub const spread_seed: u64 = 0x48_5350_2026;

pub const mixed_seed: u64 = 0x48_5A_49_50_2026;
pub const mixed_count: usize = 100_000;
pub const mixed_cap: u32 = 100_000;
pub const mixed_zipf_s: f64 = 1.48;

pub const MixedBand = enum(u8) {
    zero,
    one_two,
    three_six,
    seven_twelve,
    thirteen_thirty_two,
    thirty_three_128,
    one_twenty_nine_plus,

    pub fn name(self: MixedBand) []const u8 {
        return switch (self) {
            .zero => "0",
            .one_two => "1-2",
            .three_six => "3-6",
            .seven_twelve => "7-12",
            .thirteen_thirty_two => "13-32",
            .thirty_three_128 => "33-128",
            .one_twenty_nine_plus => "129+",
        };
    }

    pub fn contains(self: MixedBand, cardinality: u32) bool {
        return switch (self) {
            .zero => cardinality == 0,
            .one_two => cardinality >= 1 and cardinality <= 2,
            .three_six => cardinality >= 3 and cardinality <= 6,
            .seven_twelve => cardinality >= 7 and cardinality <= 12,
            .thirteen_thirty_two => cardinality >= 13 and cardinality <= 32,
            .thirty_three_128 => cardinality >= 33 and cardinality <= 128,
            .one_twenty_nine_plus => cardinality >= 129,
        };
    }
};

pub const mixed_bands = [_]MixedBand{
    .zero,
    .one_two,
    .three_six,
    .seven_twelve,
    .thirteen_thirty_two,
    .thirty_three_128,
    .one_twenty_nine_plus,
};

pub const Shape = enum(u8) {
    localized = 0,
    spread = 1,
    one_per_container = 2,

    pub fn name(self: Shape) []const u8 {
        return switch (self) {
            .localized => "localized",
            .spread => "spread",
            .one_per_container => "one-per-container",
        };
    }
};

pub const shapes = [_]Shape{ .localized, .spread, .one_per_container };

pub const FixturePool = struct {
    allocator: std.mem.Allocator,
    shape: Shape,
    cardinality: u32,
    fixture_count: usize,
    values: []u32,

    pub fn deinit(self: *FixturePool) void {
        self.allocator.free(self.values);
        self.* = undefined;
    }

    pub fn fixture(self: *const FixturePool, index: usize) []const u32 {
        std.debug.assert(index < self.fixture_count);
        if (self.cardinality == 0) return &.{};
        const card: usize = @intCast(self.cardinality);
        const start = index * card;
        return self.values[start .. start + card];
    }

    pub fn hash(self: *const FixturePool) u64 {
        var hasher = StableHasher.init();
        hasher.addByte(@intFromEnum(self.shape));
        hasher.addU32(self.cardinality);
        hasher.addU64(self.fixture_count);
        for (0..self.fixture_count) |index| {
            const values = self.fixture(index);
            hasher.addU64(values.len);
            hasher.addU32Slice(values);
        }
        return hasher.finish();
    }
};

pub fn generateSweepPool(
    allocator: std.mem.Allocator,
    shape: Shape,
    cardinality: u32,
) !FixturePool {
    const fixture_count: usize = if (cardinality == 0) 1 else sweep_pool_size;
    const value_count = try std.math.mul(usize, fixture_count, cardinality);
    const values = try allocator.alloc(u32, value_count);
    errdefer allocator.free(values);

    const card: usize = @intCast(cardinality);
    for (0..fixture_count) |fixture_index| {
        const fixture = values[fixture_index * card ..][0..card];
        switch (shape) {
            .localized => fillLocalized(fixture, fixture_index),
            .spread => try fillSpread(fixture, sweepValueSeed(shape, cardinality, fixture_index)),
            .one_per_container => fillOnePerContainer(fixture, fixture_index),
        }
        try validateFixture(shape, fixture, cardinality);
    }

    var pool = FixturePool{
        .allocator = allocator,
        .shape = shape,
        .cardinality = cardinality,
        .fixture_count = fixture_count,
        .values = values,
    };
    errdefer pool.deinit();
    try validateDistinctFixtures(allocator, &pool);
    return pool;
}

fn fillLocalized(out: []u32, fixture_index: usize) void {
    const high: u32 = @intCast(3 + fixture_index);
    const base = high << 16;
    for (out, 0..) |*value, index| value.* = base + @as(u32, @intCast(index)) * 7;
}

fn fillOnePerContainer(out: []u32, fixture_index: usize) void {
    for (out, 0..) |*value, index| {
        const high: u32 = @intCast(fixture_index + index);
        value.* = high << 16;
    }
}

pub fn fillSpread(out: []u32, seed: u64) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    try fillSpreadRandom(out, prng.random());
}

pub fn fillSpreadRandom(out: []u32, random: std.Random) !void {
    if (out.len == 0) return;

    var unique_len: usize = 0;
    while (unique_len < out.len) {
        for (out[unique_len..]) |*value| {
            value.* = random.uintLessThan(u32, spread_universe);
        }
        std.mem.sort(u32, out, {}, std.sort.asc(u32));

        var write: usize = 1;
        for (out[1..]) |value| {
            if (value == out[write - 1]) continue;
            out[write] = value;
            write += 1;
        }
        unique_len = write;
    }
}

pub fn validateFixture(shape: Shape, values: []const u32, cardinality: u32) !void {
    if (values.len != cardinality) return error.CardinalityMismatch;
    for (values, 0..) |value, index| {
        if (index != 0 and value <= values[index - 1]) return error.NotSortedUnique;
    }

    switch (shape) {
        .localized => {
            if (values.len == 0) return;
            const high = values[0] >> 16;
            for (values) |value| if (value >> 16 != high) return error.LocalizedTopologyMismatch;
        },
        .spread => for (values) |value| {
            if (value >= spread_universe) return error.SpreadValueOutOfRange;
        },
        .one_per_container => {
            for (values, 0..) |value, index| {
                if (index != 0 and value >> 16 == values[index - 1] >> 16) {
                    return error.OnePerContainerTopologyMismatch;
                }
            }
        },
    }
}

fn validateDistinctFixtures(allocator: std.mem.Allocator, pool: *const FixturePool) !void {
    if (pool.cardinality == 0) return;
    const hashes = try allocator.alloc(u64, pool.fixture_count);
    defer allocator.free(hashes);
    for (hashes, 0..) |*hash, index| {
        var hasher = StableHasher.init();
        hasher.addU32Slice(pool.fixture(index));
        hash.* = hasher.finish();
    }
    std.mem.sort(u64, hashes, {}, std.sort.asc(u64));
    for (hashes[1..], hashes[0 .. hashes.len - 1]) |current, previous| {
        if (current == previous) return error.DuplicateFixture;
    }
}

pub fn verifyStructuralMutationGuards(allocator: std.mem.Allocator) !void {
    try expectStructuralError(
        error.CardinalityMismatch,
        validateFixture(.localized, &.{1}, 2),
    );
    try expectStructuralError(
        error.NotSortedUnique,
        validateFixture(.spread, &.{ 7, 7 }, 2),
    );
    try expectStructuralError(
        error.LocalizedTopologyMismatch,
        validateFixture(.localized, &.{ 3 << 16, (3 << 16) + 70_000 }, 2),
    );
    try expectStructuralError(
        error.SpreadValueOutOfRange,
        validateFixture(.spread, &.{spread_universe}, 1),
    );
    try expectStructuralError(
        error.OnePerContainerTopologyMismatch,
        validateFixture(.one_per_container, &.{ 1 << 16, (1 << 16) + 7 }, 2),
    );

    const duplicate_values = try allocator.dupe(u32, &.{ 7, 7 });
    var duplicate_pool = FixturePool{
        .allocator = allocator,
        .shape = .spread,
        .cardinality = 1,
        .fixture_count = 2,
        .values = duplicate_values,
    };
    defer duplicate_pool.deinit();
    try expectStructuralError(
        error.DuplicateFixture,
        validateDistinctFixtures(allocator, &duplicate_pool),
    );
}

fn expectStructuralError(expected: anyerror, result: anyerror!void) !void {
    result catch |actual| {
        if (actual == expected) return;
        return error.UnexpectedStructuralError;
    };
    return error.StructuralMutationSurvived;
}

fn sweepValueSeed(shape: Shape, cardinality: u32, fixture_index: usize) u64 {
    return spread_seed ^
        (@as(u64, @intFromEnum(shape)) << 40) ^
        (@as(u64, cardinality) << 20) ^
        @as(u64, @intCast(fixture_index));
}

pub fn mixedValueSeed(corpus_index: usize, cardinality: u32) u64 {
    return mixed_seed ^
        (@as(u64, @intCast(corpus_index)) << 20) ^
        @as(u64, cardinality);
}

pub const MixedCorpus = struct {
    allocator: std.mem.Allocator,
    cardinalities: []u32,
    cardinality_hash: u64,
    full_hash: u64,
    median: u32,
    p99: u32,

    pub fn deinit(self: *MixedCorpus) void {
        self.allocator.free(self.cardinalities);
        self.* = undefined;
    }
};

pub const MixedCardinalityCorpus = struct {
    allocator: std.mem.Allocator,
    cardinalities: []u32,
    cardinality_hash: u64,
    median: u32,
    p99: u32,

    pub fn deinit(self: *MixedCardinalityCorpus) void {
        self.allocator.free(self.cardinalities);
        self.* = undefined;
    }

    pub fn bandCount(self: *const MixedCardinalityCorpus, band: MixedBand) usize {
        var count: usize = 0;
        for (self.cardinalities) |cardinality| {
            if (band.contains(cardinality)) count += 1;
        }
        return count;
    }
};

pub const SharingPattern = enum {
    correct,
    interleaved,
    sequential,
};

pub fn generateMixedCorpus(
    allocator: std.mem.Allocator,
    pattern: SharingPattern,
) !MixedCorpus {
    const cumulative = try zipfCumulative(allocator);
    defer allocator.free(cumulative);

    const cardinalities = try allocator.alloc(u32, mixed_count);
    errdefer allocator.free(cardinalities);

    var zipf_prng = std.Random.DefaultPrng.init(mixed_seed);
    const zipf_random = zipf_prng.random();

    if (pattern == .interleaved) {
        var full_hasher = StableHasher.init();
        full_hasher.addU64(mixed_count);
        for (cardinalities, 0..) |*cardinality, index| {
            cardinality.* = sampleZipf(cumulative, zipf_random);
            full_hasher.addU32(cardinality.*);
            const values = try allocator.alloc(u32, cardinality.*);
            defer allocator.free(values);
            try fillSpreadRandom(values, zipf_random);
            try validateFixture(.spread, values, cardinality.*);
            full_hasher.addU32Slice(values);
            _ = index;
        }
        return finishMixedCorpus(allocator, cardinalities, full_hasher.finish());
    }

    for (cardinalities) |*cardinality| cardinality.* = sampleZipf(cumulative, zipf_random);

    var full_hasher = StableHasher.init();
    full_hasher.addU64(mixed_count);
    for (cardinalities, 0..) |cardinality, index| {
        full_hasher.addU32(cardinality);
        const values = try allocator.alloc(u32, cardinality);
        defer allocator.free(values);
        if (pattern == .sequential) {
            try fillSpreadRandom(values, zipf_random);
        } else {
            try fillSpread(values, mixedValueSeed(index, cardinality));
        }
        try validateFixture(.spread, values, cardinality);
        full_hasher.addU32Slice(values);
    }
    return finishMixedCorpus(allocator, cardinalities, full_hasher.finish());
}

pub fn generateMixedCardinalityCorpus(allocator: std.mem.Allocator) !MixedCardinalityCorpus {
    const cumulative = try zipfCumulative(allocator);
    defer allocator.free(cumulative);

    const cardinalities = try allocator.alloc(u32, mixed_count);
    errdefer allocator.free(cardinalities);

    var zipf_prng = std.Random.DefaultPrng.init(mixed_seed);
    const random = zipf_prng.random();
    for (cardinalities) |*cardinality| cardinality.* = sampleZipf(cumulative, random);

    const summary = try summarizeMixedCardinalities(allocator, cardinalities);
    return .{
        .allocator = allocator,
        .cardinalities = cardinalities,
        .cardinality_hash = summary.hash,
        .median = summary.median,
        .p99 = summary.p99,
    };
}

fn finishMixedCorpus(
    allocator: std.mem.Allocator,
    cardinalities: []u32,
    full_hash: u64,
) !MixedCorpus {
    const summary = try summarizeMixedCardinalities(allocator, cardinalities);

    return .{
        .allocator = allocator,
        .cardinalities = cardinalities,
        .cardinality_hash = summary.hash,
        .full_hash = full_hash,
        .median = summary.median,
        .p99 = summary.p99,
    };
}

const MixedCardinalitySummary = struct {
    hash: u64,
    median: u32,
    p99: u32,
};

fn summarizeMixedCardinalities(
    allocator: std.mem.Allocator,
    cardinalities: []const u32,
) !MixedCardinalitySummary {
    var cardinality_hasher = StableHasher.init();
    cardinality_hasher.addU64(cardinalities.len);
    cardinality_hasher.addU32Slice(cardinalities);

    const sorted = try allocator.dupe(u32, cardinalities);
    defer allocator.free(sorted);
    std.mem.sort(u32, sorted, {}, std.sort.asc(u32));
    return .{
        .hash = cardinality_hasher.finish(),
        .median = sorted[(sorted.len - 1) / 2],
        .p99 = sorted[((sorted.len * 99) + 99) / 100 - 1],
    };
}

fn zipfCumulative(allocator: std.mem.Allocator) ![]f64 {
    const cumulative = try allocator.alloc(f64, @as(usize, mixed_cap) + 1);
    cumulative[0] = 0;
    var total: f64 = 0;
    for (1..cumulative.len) |index| {
        total += std.math.pow(f64, @floatFromInt(index), -mixed_zipf_s);
        cumulative[index] = total;
    }
    return cumulative;
}

fn sampleZipf(cumulative: []const f64, random: std.Random) u32 {
    const target = random.float(f64) * cumulative[cumulative.len - 1];
    var lo: usize = 1;
    var hi: usize = cumulative.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (cumulative[mid] >= target) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return @intCast(lo);
}

pub const StableHasher = struct {
    state: u64,

    const offset_basis: u64 = 0xcbf29ce484222325;
    const prime: u64 = 0x100000001b3;

    pub fn init() StableHasher {
        return .{ .state = offset_basis };
    }

    pub fn addByte(self: *StableHasher, byte: u8) void {
        self.state = (self.state ^ byte) *% prime;
    }

    pub fn addU32(self: *StableHasher, value: u32) void {
        inline for (0..4) |shift| self.addByte(@truncate(value >> (shift * 8)));
    }

    pub fn addU64(self: *StableHasher, value: u64) void {
        inline for (0..8) |shift| self.addByte(@truncate(value >> (shift * 8)));
    }

    pub fn addU32Slice(self: *StableHasher, values: []const u32) void {
        for (values) |value| self.addU32(value);
    }

    pub fn finish(self: StableHasher) u64 {
        return self.state;
    }
};

test "sweep fixtures preserve their topology" {
    for (shapes) |shape| {
        for (sweep_cardinalities) |cardinality| {
            var pool = try generateSweepPool(std.testing.allocator, shape, cardinality);
            defer pool.deinit();
            try std.testing.expectEqual(if (cardinality == 0) 1 else sweep_pool_size, pool.fixture_count);
        }
    }
}

test "structural fixture guards reject seeded defects" {
    try verifyStructuralMutationGuards(std.testing.allocator);
}
