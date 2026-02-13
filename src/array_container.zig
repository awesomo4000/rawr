const std = @import("std");

/// Sorted array container for low-cardinality chunks (≤4096 elements).
/// Stores the low 16 bits of values in a chunk as a sorted array.
pub const ArrayContainer = struct {
    /// Sorted array of low-16-bit values.
    /// Aligned to 32 bytes for SIMD binary search.
    values: []align(32) u16,

    /// Number of elements [0, 4096].
    cardinality: u16,

    /// Allocated capacity (always power of 2, minimum 4).
    capacity: u16,

    const Self = @This();

    /// Maximum cardinality before converting to bitset.
    pub const MAX_CARDINALITY: u16 = 4096;

    /// Minimum allocation capacity.
    const MIN_CAPACITY: u16 = 4;

    pub fn init(allocator: std.mem.Allocator, initial_cap: u16) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        // Round up to power of 2, minimum MIN_CAPACITY
        const cap = if (initial_cap <= MIN_CAPACITY)
            MIN_CAPACITY
        else
            std.math.ceilPowerOfTwo(u16, initial_cap) catch MAX_CARDINALITY;
        const values = try allocator.alignedAlloc(u16, .@"32", cap);

        self.* = .{
            .values = values,
            .cardinality = 0,
            .capacity = cap,
        };
        return self;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.values[0..self.capacity]);
        allocator.destroy(self);
    }

    /// Create a deep copy.
    pub fn clone(self: *const Self, allocator: std.mem.Allocator) !*Self {
        const copy = try allocator.create(Self);
        errdefer allocator.destroy(copy);

        const values = try allocator.alignedAlloc(u16, .@"32", self.capacity);
        @memcpy(values[0..self.cardinality], self.values[0..self.cardinality]);

        copy.* = .{
            .values = values,
            .cardinality = self.cardinality,
            .capacity = self.capacity,
        };
        return copy;
    }

    /// Binary search for a value. Returns index if found, null otherwise.
    fn binarySearch(self: *const Self, value: u16) ?usize {
        if (self.cardinality == 0) return null;

        var lo: usize = 0;
        var hi: usize = self.cardinality;

        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.values[mid] < value) {
                lo = mid + 1;
            } else if (self.values[mid] > value) {
                hi = mid;
            } else {
                return mid;
            }
        }
        return null;
    }

    /// Binary search returning insertion point if not found.
    fn lowerBound(self: *const Self, value: u16) usize {
        var lo: usize = 0;
        var hi: usize = self.cardinality;

        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.values[mid] < value) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    /// Check if a value is present.
    pub fn contains(self: *const Self, value: u16) bool {
        return self.binarySearch(value) != null;
    }

    /// Grow capacity if needed.
    fn ensureCapacity(self: *Self, allocator: std.mem.Allocator, needed: u16) !void {
        if (needed <= self.capacity) return;

        const new_cap = @max(
            self.capacity * 2,
            std.math.ceilPowerOfTwo(u16, needed) catch MAX_CARDINALITY,
        );
        const new_values = try allocator.alignedAlloc(u16, .@"32", new_cap);
        @memcpy(new_values[0..self.cardinality], self.values[0..self.cardinality]);
        allocator.free(self.values[0..self.capacity]);
        self.values = new_values;
        self.capacity = new_cap;
    }

    /// Add a value maintaining sorted order. Returns true if value was new.
    pub fn add(self: *Self, allocator: std.mem.Allocator, value: u16) !bool {
        const pos = self.lowerBound(value);

        // Check if already present
        if (pos < self.cardinality and self.values[pos] == value) {
            return false;
        }

        // Grow if needed
        try self.ensureCapacity(allocator, self.cardinality + 1);

        // Shift right to make room
        if (pos < self.cardinality) {
            std.mem.copyBackwards(
                u16,
                self.values[pos + 1 .. self.cardinality + 1],
                self.values[pos..self.cardinality],
            );
        }

        self.values[pos] = value;
        self.cardinality += 1;
        return true;
    }

    /// Remove a value. Returns true if value was present.
    pub fn remove(self: *Self, value: u16) bool {
        const pos = self.binarySearch(value) orelse return false;

        // Shift left to fill gap
        if (pos + 1 < self.cardinality) {
            std.mem.copyForwards(
                u16,
                self.values[pos .. self.cardinality - 1],
                self.values[pos + 1 .. self.cardinality],
            );
        }

        self.cardinality -= 1;
        return true;
    }

    /// Get cardinality.
    pub fn getCardinality(self: *const Self) u32 {
        return self.cardinality;
    }

    /// Check if container should convert to bitset.
    pub fn isFull(self: *const Self) bool {
        return self.cardinality >= MAX_CARDINALITY;
    }

    /// Shrink capacity to fit current cardinality.
    pub fn shrinkToFit(self: *Self, allocator: std.mem.Allocator) !void {
        const new_cap = if (self.cardinality <= MIN_CAPACITY)
            MIN_CAPACITY
        else
            std.math.ceilPowerOfTwo(u16, self.cardinality) catch MIN_CAPACITY;
        if (new_cap >= self.capacity) return;

        const new_values = try allocator.alignedAlloc(u16, .@"32", new_cap);
        @memcpy(new_values[0..self.cardinality], self.values[0..self.cardinality]);
        allocator.free(self.values[0..self.capacity]);
        self.values = new_values;
        self.capacity = new_cap;
    }

    /// In-place union: merge other's values into self's buffer.
    /// Returns null if result stays as array, or a new BitsetContainer if
    /// cardinality exceeds 4096 (caller must free self and use the bitset).
    ///
    /// Algorithm: Move self's values to end of buffer, then forward merge.
    /// This avoids overlap issues and requires at most one realloc.
    pub fn unionInPlace(self: *Self, allocator: std.mem.Allocator, other: *const Self) !?*@import("bitset_container.zig").BitsetContainer {
        const BitsetContainer = @import("bitset_container.zig").BitsetContainer;

        const max_card = @as(u32, self.cardinality) + other.cardinality;

        // If combined could exceed array threshold, convert to bitset
        if (max_card > MAX_CARDINALITY) {
            const bc = try BitsetContainer.init(allocator);
            errdefer bc.deinit(allocator);
            for (self.values[0..self.cardinality]) |v| _ = bc.add(v);
            for (other.values[0..other.cardinality]) |v| _ = bc.add(v);
            return bc;
        }

        // Ensure buffer has room for max_card elements (one possible realloc)
        try self.ensureCapacity(allocator, @intCast(max_card));

        // Move self's values to the END of the buffer using copyBackwards
        // Before: [A B C D . . . .]  (self has 4 values, capacity for 8)
        // After:  [. . . . A B C D]
        const self_start: usize = max_card - self.cardinality;
        std.mem.copyBackwards(
            u16,
            self.values[self_start..max_card],
            self.values[0..self.cardinality],
        );

        // Forward merge from self (now at end) and other (separate buffer)
        // Write cursor k is always <= read cursor si, so no overwrite hazard
        var si: usize = self_start; // read cursor for self (in moved region)
        var oi: usize = 0; // read cursor for other
        var k: usize = 0; // write cursor

        const self_end: usize = max_card;
        const other_end: usize = other.cardinality;

        // Branchless merge: always write the smaller value, advance contributing pointer(s).
        // On aarch64, LLVM emits csel for the output and cset for advances — no branches.
        while (si < self_end and oi < other_end) {
            const sv = self.values[si];
            const ov = other.values[oi];

            self.values[k] = if (sv <= ov) sv else ov;
            k += 1;

            si += @intFromBool(sv <= ov);
            oi += @intFromBool(ov <= sv);
        }

        // Drain remaining from self (shift left if there were duplicates)
        while (si < self_end) : (si += 1) {
            self.values[k] = self.values[si];
            k += 1;
        }

        // Drain remaining from other
        while (oi < other_end) : (oi += 1) {
            self.values[k] = other.values[oi];
            k += 1;
        }

        self.cardinality = @intCast(k);
        return null; // Stayed as array
    }
};

// ============================================================================
// Tests
// ============================================================================

test "init and deinit" {
    const allocator = std.testing.allocator;
    const ac = try ArrayContainer.init(allocator, 0);
    defer ac.deinit(allocator);

    try std.testing.expectEqual(@as(u16, 0), ac.cardinality);
    try std.testing.expectEqual(@as(u16, ArrayContainer.MIN_CAPACITY), ac.capacity);
}

test "add and contains" {
    const allocator = std.testing.allocator;
    const ac = try ArrayContainer.init(allocator, 0);
    defer ac.deinit(allocator);

    // Add values out of order
    try std.testing.expect(try ac.add(allocator, 100));
    try std.testing.expect(try ac.add(allocator, 50));
    try std.testing.expect(try ac.add(allocator, 150));
    try std.testing.expect(try ac.add(allocator, 75));

    // Check they're present
    try std.testing.expect(ac.contains(50));
    try std.testing.expect(ac.contains(75));
    try std.testing.expect(ac.contains(100));
    try std.testing.expect(ac.contains(150));

    // Check sorted order
    try std.testing.expectEqual(@as(u16, 50), ac.values[0]);
    try std.testing.expectEqual(@as(u16, 75), ac.values[1]);
    try std.testing.expectEqual(@as(u16, 100), ac.values[2]);
    try std.testing.expectEqual(@as(u16, 150), ac.values[3]);

    // Check cardinality
    try std.testing.expectEqual(@as(u16, 4), ac.cardinality);

    // Adding duplicate returns false
    try std.testing.expect(!try ac.add(allocator, 100));
    try std.testing.expectEqual(@as(u16, 4), ac.cardinality);

    // Check absent values
    try std.testing.expect(!ac.contains(0));
    try std.testing.expect(!ac.contains(51));
    try std.testing.expect(!ac.contains(65535));
}

test "remove" {
    const allocator = std.testing.allocator;
    const ac = try ArrayContainer.init(allocator, 0);
    defer ac.deinit(allocator);

    _ = try ac.add(allocator, 10);
    _ = try ac.add(allocator, 20);
    _ = try ac.add(allocator, 30);

    // Remove middle element
    try std.testing.expect(ac.remove(20));
    try std.testing.expectEqual(@as(u16, 2), ac.cardinality);
    try std.testing.expect(!ac.contains(20));
    try std.testing.expect(ac.contains(10));
    try std.testing.expect(ac.contains(30));

    // Check order preserved
    try std.testing.expectEqual(@as(u16, 10), ac.values[0]);
    try std.testing.expectEqual(@as(u16, 30), ac.values[1]);

    // Remove absent value
    try std.testing.expect(!ac.remove(20));
    try std.testing.expectEqual(@as(u16, 2), ac.cardinality);
}

test "growth" {
    const allocator = std.testing.allocator;
    const ac = try ArrayContainer.init(allocator, 0);
    defer ac.deinit(allocator);

    // Initial capacity is MIN_CAPACITY (4)
    try std.testing.expectEqual(@as(u16, 4), ac.capacity);

    // Add 5 elements to trigger growth
    for (0..5) |i| {
        _ = try ac.add(allocator, @intCast(i * 10));
    }

    try std.testing.expectEqual(@as(u16, 5), ac.cardinality);
    try std.testing.expect(ac.capacity >= 5);
    try std.testing.expectEqual(@as(u16, 8), ac.capacity); // Power of 2
}

test "shrink to fit" {
    const allocator = std.testing.allocator;
    const ac = try ArrayContainer.init(allocator, 100);
    defer ac.deinit(allocator);

    // Start with large capacity
    try std.testing.expect(ac.capacity >= 100);

    // Add only 3 elements
    _ = try ac.add(allocator, 1);
    _ = try ac.add(allocator, 2);
    _ = try ac.add(allocator, 3);

    try ac.shrinkToFit(allocator);
    try std.testing.expectEqual(@as(u16, 4), ac.capacity); // MIN_CAPACITY
    try std.testing.expectEqual(@as(u16, 3), ac.cardinality);

    // Values preserved
    try std.testing.expect(ac.contains(1));
    try std.testing.expect(ac.contains(2));
    try std.testing.expect(ac.contains(3));
}

test "edge cases" {
    const allocator = std.testing.allocator;
    const ac = try ArrayContainer.init(allocator, 0);
    defer ac.deinit(allocator);

    // Add boundary values
    _ = try ac.add(allocator, 0);
    _ = try ac.add(allocator, 65535);

    try std.testing.expect(ac.contains(0));
    try std.testing.expect(ac.contains(65535));
    try std.testing.expectEqual(@as(u16, 0), ac.values[0]);
    try std.testing.expectEqual(@as(u16, 65535), ac.values[1]);
}
