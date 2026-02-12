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
