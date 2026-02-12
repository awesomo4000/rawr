const std = @import("std");

/// Fixed 8KB bitset container for high-cardinality chunks (>4096 elements).
/// Stores the low 16 bits of values in a chunk.
pub const BitsetContainer = struct {
    /// Fixed 8KB bitset: 1024 × u64 words.
    /// Aligned to 64 bytes (cache line) for optimal SIMD access.
    words: *align(64) [1024]u64,

    /// Cardinality cache. -1 = unknown (lazy computation needed).
    cardinality: i32,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const words = try allocator.alignedAlloc(u64, .@"64", 1024);
        @memset(words, 0);

        self.* = .{
            .words = words[0..1024],
            .cardinality = 0,
        };
        return self;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        const slice: []align(64) u64 = self.words;
        allocator.free(slice);
        allocator.destroy(self);
    }

    /// Check if a value (low 16 bits) is present.
    pub fn contains(self: *const Self, value: u16) bool {
        const word_idx = value >> 6; // value / 64
        const bit_idx: u6 = @truncate(value); // value % 64
        return (self.words[word_idx] & (@as(u64, 1) << bit_idx)) != 0;
    }

    /// Add a value. Returns true if the value was newly added.
    pub fn add(self: *Self, value: u16) bool {
        const word_idx = value >> 6;
        const bit_idx: u6 = @truncate(value);
        const bit: u64 = @as(u64, 1) << bit_idx;

        const word = &self.words[word_idx];
        const was_absent = (word.* & bit) == 0;
        word.* |= bit;

        if (was_absent and self.cardinality >= 0) {
            self.cardinality += 1;
        }
        return was_absent;
    }

    /// Remove a value. Returns true if the value was present.
    pub fn remove(self: *Self, value: u16) bool {
        const word_idx = value >> 6;
        const bit_idx: u6 = @truncate(value);
        const bit: u64 = @as(u64, 1) << bit_idx;

        const word = &self.words[word_idx];
        const was_present = (word.* & bit) != 0;
        word.* &= ~bit;

        if (was_present and self.cardinality >= 0) {
            self.cardinality -= 1;
        }
        return was_present;
    }

    /// Compute cardinality by counting all set bits.
    pub fn computeCardinality(self: *Self) u32 {
        var count: u32 = 0;
        for (self.words) |word| {
            count += @popCount(word);
        }
        self.cardinality = @intCast(count);
        return count;
    }

    /// Get cardinality, computing if unknown.
    pub fn getCardinality(self: *Self) u32 {
        if (self.cardinality < 0) {
            return self.computeCardinality();
        }
        return @intCast(self.cardinality);
    }

    /// Mark cardinality as unknown (call after bulk mutations).
    pub fn invalidateCardinality(self: *Self) void {
        self.cardinality = -1;
    }

    /// SIMD-accelerated OR: dst |= src
    pub fn unionWith(dst: *Self, src: *const Self) void {
        const VEC_SIZE = 8; // 512 bits = 64 bytes = 1 cache line
        const vec_count = 1024 / VEC_SIZE;

        var card: u64 = 0;
        for (0..vec_count) |i| {
            const base = i * VEC_SIZE;
            const a: @Vector(VEC_SIZE, u64) = dst.words[base..][0..VEC_SIZE].*;
            const b: @Vector(VEC_SIZE, u64) = src.words[base..][0..VEC_SIZE].*;
            const result = a | b;
            dst.words[base..][0..VEC_SIZE].* = result;

            // Accumulate popcount
            inline for (0..VEC_SIZE) |j| {
                card += @popCount(result[j]);
            }
        }
        dst.cardinality = @intCast(card);
    }

    /// SIMD-accelerated AND: dst &= src
    pub fn intersectionWith(dst: *Self, src: *const Self) void {
        const VEC_SIZE = 8;
        const vec_count = 1024 / VEC_SIZE;

        var card: u64 = 0;
        for (0..vec_count) |i| {
            const base = i * VEC_SIZE;
            const a: @Vector(VEC_SIZE, u64) = dst.words[base..][0..VEC_SIZE].*;
            const b: @Vector(VEC_SIZE, u64) = src.words[base..][0..VEC_SIZE].*;
            const result = a & b;
            dst.words[base..][0..VEC_SIZE].* = result;

            inline for (0..VEC_SIZE) |j| {
                card += @popCount(result[j]);
            }
        }
        dst.cardinality = @intCast(card);
    }

    /// SIMD-accelerated XOR: dst ^= src
    pub fn symmetricDifferenceWith(dst: *Self, src: *const Self) void {
        const VEC_SIZE = 8;
        const vec_count = 1024 / VEC_SIZE;

        var card: u64 = 0;
        for (0..vec_count) |i| {
            const base = i * VEC_SIZE;
            const a: @Vector(VEC_SIZE, u64) = dst.words[base..][0..VEC_SIZE].*;
            const b: @Vector(VEC_SIZE, u64) = src.words[base..][0..VEC_SIZE].*;
            const result = a ^ b;
            dst.words[base..][0..VEC_SIZE].* = result;

            inline for (0..VEC_SIZE) |j| {
                card += @popCount(result[j]);
            }
        }
        dst.cardinality = @intCast(card);
    }

    /// SIMD-accelerated AND-NOT: dst &= ~src (difference)
    pub fn differenceWith(dst: *Self, src: *const Self) void {
        const VEC_SIZE = 8;
        const vec_count = 1024 / VEC_SIZE;

        var card: u64 = 0;
        for (0..vec_count) |i| {
            const base = i * VEC_SIZE;
            const a: @Vector(VEC_SIZE, u64) = dst.words[base..][0..VEC_SIZE].*;
            const b: @Vector(VEC_SIZE, u64) = src.words[base..][0..VEC_SIZE].*;
            const result = a & ~b;
            dst.words[base..][0..VEC_SIZE].* = result;

            inline for (0..VEC_SIZE) |j| {
                card += @popCount(result[j]);
            }
        }
        dst.cardinality = @intCast(card);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "init and deinit" {
    const allocator = std.testing.allocator;
    const bs = try BitsetContainer.init(allocator);
    defer bs.deinit(allocator);

    try std.testing.expectEqual(@as(i32, 0), bs.cardinality);
    try std.testing.expect(!bs.contains(0));
    try std.testing.expect(!bs.contains(65535));
}

test "add and contains" {
    const allocator = std.testing.allocator;
    const bs = try BitsetContainer.init(allocator);
    defer bs.deinit(allocator);

    // Add some values
    try std.testing.expect(bs.add(0)); // first bit
    try std.testing.expect(bs.add(63)); // last bit of first word
    try std.testing.expect(bs.add(64)); // first bit of second word
    try std.testing.expect(bs.add(65535)); // last possible value

    // Check they're present
    try std.testing.expect(bs.contains(0));
    try std.testing.expect(bs.contains(63));
    try std.testing.expect(bs.contains(64));
    try std.testing.expect(bs.contains(65535));

    // Check cardinality
    try std.testing.expectEqual(@as(i32, 4), bs.cardinality);

    // Adding again returns false
    try std.testing.expect(!bs.add(0));
    try std.testing.expectEqual(@as(i32, 4), bs.cardinality);

    // Check absent values
    try std.testing.expect(!bs.contains(1));
    try std.testing.expect(!bs.contains(62));
    try std.testing.expect(!bs.contains(65534));
}

test "remove" {
    const allocator = std.testing.allocator;
    const bs = try BitsetContainer.init(allocator);
    defer bs.deinit(allocator);

    _ = bs.add(100);
    _ = bs.add(200);
    try std.testing.expectEqual(@as(i32, 2), bs.cardinality);

    // Remove present value
    try std.testing.expect(bs.remove(100));
    try std.testing.expect(!bs.contains(100));
    try std.testing.expectEqual(@as(i32, 1), bs.cardinality);

    // Remove absent value
    try std.testing.expect(!bs.remove(100));
    try std.testing.expectEqual(@as(i32, 1), bs.cardinality);
}

test "cardinality computation" {
    const allocator = std.testing.allocator;
    const bs = try BitsetContainer.init(allocator);
    defer bs.deinit(allocator);

    for (0..1000) |i| {
        _ = bs.add(@intCast(i));
    }

    try std.testing.expectEqual(@as(u32, 1000), bs.getCardinality());

    // Invalidate and recompute
    bs.invalidateCardinality();
    try std.testing.expectEqual(@as(i32, -1), bs.cardinality);
    try std.testing.expectEqual(@as(u32, 1000), bs.getCardinality());
    try std.testing.expectEqual(@as(i32, 1000), bs.cardinality);
}

test "union" {
    const allocator = std.testing.allocator;
    const a = try BitsetContainer.init(allocator);
    defer a.deinit(allocator);
    const b = try BitsetContainer.init(allocator);
    defer b.deinit(allocator);

    // a = {0, 1, 2}
    _ = a.add(0);
    _ = a.add(1);
    _ = a.add(2);

    // b = {2, 3, 4}
    _ = b.add(2);
    _ = b.add(3);
    _ = b.add(4);

    // a |= b => a = {0, 1, 2, 3, 4}
    a.unionWith(b);

    try std.testing.expect(a.contains(0));
    try std.testing.expect(a.contains(1));
    try std.testing.expect(a.contains(2));
    try std.testing.expect(a.contains(3));
    try std.testing.expect(a.contains(4));
    try std.testing.expectEqual(@as(u32, 5), a.getCardinality());
}

test "intersection" {
    const allocator = std.testing.allocator;
    const a = try BitsetContainer.init(allocator);
    defer a.deinit(allocator);
    const b = try BitsetContainer.init(allocator);
    defer b.deinit(allocator);

    // a = {0, 1, 2, 3}
    _ = a.add(0);
    _ = a.add(1);
    _ = a.add(2);
    _ = a.add(3);

    // b = {2, 3, 4, 5}
    _ = b.add(2);
    _ = b.add(3);
    _ = b.add(4);
    _ = b.add(5);

    // a &= b => a = {2, 3}
    a.intersectionWith(b);

    try std.testing.expect(!a.contains(0));
    try std.testing.expect(!a.contains(1));
    try std.testing.expect(a.contains(2));
    try std.testing.expect(a.contains(3));
    try std.testing.expect(!a.contains(4));
    try std.testing.expect(!a.contains(5));
    try std.testing.expectEqual(@as(u32, 2), a.getCardinality());
}

test "difference" {
    const allocator = std.testing.allocator;
    const a = try BitsetContainer.init(allocator);
    defer a.deinit(allocator);
    const b = try BitsetContainer.init(allocator);
    defer b.deinit(allocator);

    // a = {0, 1, 2, 3}
    _ = a.add(0);
    _ = a.add(1);
    _ = a.add(2);
    _ = a.add(3);

    // b = {2, 3, 4}
    _ = b.add(2);
    _ = b.add(3);
    _ = b.add(4);

    // a -= b => a = {0, 1}
    a.differenceWith(b);

    try std.testing.expect(a.contains(0));
    try std.testing.expect(a.contains(1));
    try std.testing.expect(!a.contains(2));
    try std.testing.expect(!a.contains(3));
    try std.testing.expectEqual(@as(u32, 2), a.getCardinality());
}

test "symmetric difference (xor)" {
    const allocator = std.testing.allocator;
    const a = try BitsetContainer.init(allocator);
    defer a.deinit(allocator);
    const b = try BitsetContainer.init(allocator);
    defer b.deinit(allocator);

    // a = {0, 1, 2}
    _ = a.add(0);
    _ = a.add(1);
    _ = a.add(2);

    // b = {2, 3}
    _ = b.add(2);
    _ = b.add(3);

    // a ^= b => a = {0, 1, 3}
    a.symmetricDifferenceWith(b);

    try std.testing.expect(a.contains(0));
    try std.testing.expect(a.contains(1));
    try std.testing.expect(!a.contains(2));
    try std.testing.expect(a.contains(3));
    try std.testing.expectEqual(@as(u32, 3), a.getCardinality());
}
