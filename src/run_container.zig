const std = @import("std");

/// Run-length encoded container for consecutive value sequences.
/// Efficient when data has many contiguous runs.
pub const RunContainer = struct {
    /// A run represents values from start to start+length (inclusive).
    pub const RunPair = packed struct {
        start: u16,
        length: u16, // number of values AFTER start (run size = length + 1)

        pub fn end(self: RunPair) u16 {
            return self.start +| self.length;
        }

        pub fn containsValue(self: RunPair, value: u16) bool {
            return value >= self.start and value <= self.end();
        }

        pub fn size(self: RunPair) u32 {
            return @as(u32, self.length) + 1;
        }
    };

    /// Sorted array of non-overlapping, non-adjacent runs.
    runs: []RunPair,

    /// Number of runs.
    n_runs: u16,

    /// Allocated capacity.
    capacity: u16,

    const Self = @This();
    const MIN_CAPACITY: u16 = 4;

    pub fn init(allocator: std.mem.Allocator, initial_cap: u16) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const cap = @max(MIN_CAPACITY, initial_cap);
        const runs = try allocator.alloc(RunPair, cap);

        self.* = .{
            .runs = runs,
            .n_runs = 0,
            .capacity = cap,
        };
        return self;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.runs[0..self.capacity]);
        allocator.destroy(self);
    }

    /// Create a deep copy.
    pub fn clone(self: *const Self, allocator: std.mem.Allocator) !*Self {
        const copy = try allocator.create(Self);
        errdefer allocator.destroy(copy);

        const runs = try allocator.alloc(RunPair, self.capacity);
        @memcpy(runs[0..self.n_runs], self.runs[0..self.n_runs]);

        copy.* = .{
            .runs = runs,
            .n_runs = self.n_runs,
            .capacity = self.capacity,
        };
        return copy;
    }

    /// Binary search for the run containing or after value.
    /// Returns the index of the run that might contain value,
    /// or n_runs if value is greater than all runs.
    fn searchRuns(self: *const Self, value: u16) usize {
        var lo: usize = 0;
        var hi: usize = self.n_runs;

        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.runs[mid].end() < value) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    /// Check if a value is present in any run.
    pub fn contains(self: *const Self, value: u16) bool {
        if (self.n_runs == 0) return false;

        const idx = self.searchRuns(value);
        if (idx >= self.n_runs) return false;

        return self.runs[idx].containsValue(value);
    }

    /// Grow capacity if needed.
    fn ensureCapacity(self: *Self, allocator: std.mem.Allocator, needed: u16) !void {
        if (needed <= self.capacity) return;

        const new_cap = @max(self.capacity * 2, needed);
        const new_runs = try allocator.alloc(RunPair, new_cap);
        @memcpy(new_runs[0..self.n_runs], self.runs[0..self.n_runs]);
        allocator.free(self.runs[0..self.capacity]);
        self.runs = new_runs;
        self.capacity = new_cap;
    }

    /// Add a value. Returns true if value was new.
    pub fn add(self: *Self, allocator: std.mem.Allocator, value: u16) !bool {
        if (self.n_runs == 0) {
            try self.ensureCapacity(allocator, 1);
            self.runs[0] = .{ .start = value, .length = 0 };
            self.n_runs = 1;
            return true;
        }

        const idx = self.searchRuns(value);

        // Check if value is in an existing run
        if (idx < self.n_runs and self.runs[idx].containsValue(value)) {
            return false;
        }

        // Check if we can extend the previous run
        if (idx > 0 and self.runs[idx - 1].end() + 1 == value) {
            self.runs[idx - 1].length += 1;
            // Check if we need to merge with next run
            if (idx < self.n_runs and self.runs[idx - 1].end() + 1 == self.runs[idx].start) {
                self.mergeRuns(idx - 1);
            }
            return true;
        }

        // Check if we can extend the current run backwards
        if (idx < self.n_runs and value + 1 == self.runs[idx].start) {
            self.runs[idx].start -= 1;
            self.runs[idx].length += 1;
            return true;
        }

        // Need to insert a new run
        try self.ensureCapacity(allocator, self.n_runs + 1);

        // Shift runs to make room
        if (idx < self.n_runs) {
            std.mem.copyBackwards(
                RunPair,
                self.runs[idx + 1 .. self.n_runs + 1],
                self.runs[idx..self.n_runs],
            );
        }

        self.runs[idx] = .{ .start = value, .length = 0 };
        self.n_runs += 1;
        return true;
    }

    /// Merge run at idx with run at idx+1.
    fn mergeRuns(self: *Self, idx: usize) void {
        const combined_end = self.runs[idx + 1].end();
        self.runs[idx].length = combined_end - self.runs[idx].start;

        // Shift remaining runs left
        if (idx + 2 < self.n_runs) {
            std.mem.copyForwards(
                RunPair,
                self.runs[idx + 1 .. self.n_runs - 1],
                self.runs[idx + 2 .. self.n_runs],
            );
        }
        self.n_runs -= 1;
    }

    /// Remove a value. Returns true if value was present.
    pub fn remove(self: *Self, allocator: std.mem.Allocator, value: u16) !bool {
        if (self.n_runs == 0) return false;

        const idx = self.searchRuns(value);
        if (idx >= self.n_runs) return false;

        const run = &self.runs[idx];
        if (!run.containsValue(value)) return false;

        if (run.length == 0) {
            // Single element run, remove it entirely
            if (idx + 1 < self.n_runs) {
                std.mem.copyForwards(
                    RunPair,
                    self.runs[idx .. self.n_runs - 1],
                    self.runs[idx + 1 .. self.n_runs],
                );
            }
            self.n_runs -= 1;
        } else if (value == run.start) {
            // Remove from start
            run.start += 1;
            run.length -= 1;
        } else if (value == run.end()) {
            // Remove from end
            run.length -= 1;
        } else {
            // Split the run
            try self.ensureCapacity(allocator, self.n_runs + 1);
            const original_end = run.end();

            // Shrink current run
            run.length = value - run.start - 1;

            // Insert new run after
            if (idx + 1 < self.n_runs) {
                std.mem.copyBackwards(
                    RunPair,
                    self.runs[idx + 2 .. self.n_runs + 1],
                    self.runs[idx + 1 .. self.n_runs],
                );
            }
            self.runs[idx + 1] = .{
                .start = value + 1,
                .length = original_end - value - 1,
            };
            self.n_runs += 1;
        }
        return true;
    }

    /// Get total cardinality (sum of all run sizes).
    pub fn getCardinality(self: *const Self) u32 {
        var card: u32 = 0;
        for (self.runs[0..self.n_runs]) |run| {
            card += run.size();
        }
        return card;
    }

    /// Get serialized size in bytes (for comparing with other container types).
    pub fn serializedSizeInBytes(self: *const Self) u32 {
        return @as(u32, self.n_runs) * 4; // 4 bytes per RunPair
    }
};

// ============================================================================
// Tests
// ============================================================================

test "init and deinit" {
    const allocator = std.testing.allocator;
    const rc = try RunContainer.init(allocator, 0);
    defer rc.deinit(allocator);

    try std.testing.expectEqual(@as(u16, 0), rc.n_runs);
    try std.testing.expectEqual(@as(u32, 0), rc.getCardinality());
}

test "add single values" {
    const allocator = std.testing.allocator;
    const rc = try RunContainer.init(allocator, 0);
    defer rc.deinit(allocator);

    // Add isolated values
    try std.testing.expect(try rc.add(allocator, 10));
    try std.testing.expect(try rc.add(allocator, 20));
    try std.testing.expect(try rc.add(allocator, 30));

    try std.testing.expectEqual(@as(u16, 3), rc.n_runs);
    try std.testing.expectEqual(@as(u32, 3), rc.getCardinality());

    try std.testing.expect(rc.contains(10));
    try std.testing.expect(rc.contains(20));
    try std.testing.expect(rc.contains(30));
    try std.testing.expect(!rc.contains(15));
}

test "add consecutive values creates run" {
    const allocator = std.testing.allocator;
    const rc = try RunContainer.init(allocator, 0);
    defer rc.deinit(allocator);

    // Add 0, 1, 2, 3, 4 - should create single run
    _ = try rc.add(allocator, 2);
    _ = try rc.add(allocator, 3);
    _ = try rc.add(allocator, 1);
    _ = try rc.add(allocator, 4);
    _ = try rc.add(allocator, 0);

    try std.testing.expectEqual(@as(u16, 1), rc.n_runs);
    try std.testing.expectEqual(@as(u32, 5), rc.getCardinality());
    try std.testing.expectEqual(@as(u16, 0), rc.runs[0].start);
    try std.testing.expectEqual(@as(u16, 4), rc.runs[0].length);

    for (0..5) |i| {
        try std.testing.expect(rc.contains(@intCast(i)));
    }
}

test "add merges adjacent runs" {
    const allocator = std.testing.allocator;
    const rc = try RunContainer.init(allocator, 0);
    defer rc.deinit(allocator);

    // Create two separate runs
    _ = try rc.add(allocator, 0);
    _ = try rc.add(allocator, 1);
    _ = try rc.add(allocator, 5);
    _ = try rc.add(allocator, 6);

    try std.testing.expectEqual(@as(u16, 2), rc.n_runs);

    // Add value that bridges them
    _ = try rc.add(allocator, 2);
    _ = try rc.add(allocator, 3);
    _ = try rc.add(allocator, 4);

    try std.testing.expectEqual(@as(u16, 1), rc.n_runs);
    try std.testing.expectEqual(@as(u32, 7), rc.getCardinality());
}

test "add duplicate returns false" {
    const allocator = std.testing.allocator;
    const rc = try RunContainer.init(allocator, 0);
    defer rc.deinit(allocator);

    try std.testing.expect(try rc.add(allocator, 5));
    try std.testing.expect(!try rc.add(allocator, 5));
    try std.testing.expectEqual(@as(u32, 1), rc.getCardinality());
}

test "remove from single element run" {
    const allocator = std.testing.allocator;
    const rc = try RunContainer.init(allocator, 0);
    defer rc.deinit(allocator);

    _ = try rc.add(allocator, 10);
    try std.testing.expectEqual(@as(u16, 1), rc.n_runs);

    try std.testing.expect(try rc.remove(allocator, 10));
    try std.testing.expectEqual(@as(u16, 0), rc.n_runs);
    try std.testing.expect(!rc.contains(10));
}

test "remove from run start" {
    const allocator = std.testing.allocator;
    const rc = try RunContainer.init(allocator, 0);
    defer rc.deinit(allocator);

    _ = try rc.add(allocator, 10);
    _ = try rc.add(allocator, 11);
    _ = try rc.add(allocator, 12);

    try std.testing.expect(try rc.remove(allocator, 10));
    try std.testing.expectEqual(@as(u16, 1), rc.n_runs);
    try std.testing.expectEqual(@as(u16, 11), rc.runs[0].start);
    try std.testing.expect(!rc.contains(10));
    try std.testing.expect(rc.contains(11));
    try std.testing.expect(rc.contains(12));
}

test "remove from run end" {
    const allocator = std.testing.allocator;
    const rc = try RunContainer.init(allocator, 0);
    defer rc.deinit(allocator);

    _ = try rc.add(allocator, 10);
    _ = try rc.add(allocator, 11);
    _ = try rc.add(allocator, 12);

    try std.testing.expect(try rc.remove(allocator, 12));
    try std.testing.expectEqual(@as(u16, 1), rc.n_runs);
    try std.testing.expectEqual(@as(u16, 1), rc.runs[0].length);
    try std.testing.expect(rc.contains(10));
    try std.testing.expect(rc.contains(11));
    try std.testing.expect(!rc.contains(12));
}

test "remove from run middle splits" {
    const allocator = std.testing.allocator;
    const rc = try RunContainer.init(allocator, 0);
    defer rc.deinit(allocator);

    for (0..10) |i| {
        _ = try rc.add(allocator, @intCast(i));
    }
    try std.testing.expectEqual(@as(u16, 1), rc.n_runs);

    try std.testing.expect(try rc.remove(allocator, 5));
    try std.testing.expectEqual(@as(u16, 2), rc.n_runs);

    // First run: 0-4
    try std.testing.expectEqual(@as(u16, 0), rc.runs[0].start);
    try std.testing.expectEqual(@as(u16, 4), rc.runs[0].end());

    // Second run: 6-9
    try std.testing.expectEqual(@as(u16, 6), rc.runs[1].start);
    try std.testing.expectEqual(@as(u16, 9), rc.runs[1].end());

    try std.testing.expect(!rc.contains(5));
    try std.testing.expect(rc.contains(4));
    try std.testing.expect(rc.contains(6));
}

test "contains boundary values" {
    const allocator = std.testing.allocator;
    const rc = try RunContainer.init(allocator, 0);
    defer rc.deinit(allocator);

    _ = try rc.add(allocator, 0);
    _ = try rc.add(allocator, 65535);

    try std.testing.expect(rc.contains(0));
    try std.testing.expect(rc.contains(65535));
    try std.testing.expect(!rc.contains(1));
    try std.testing.expect(!rc.contains(65534));
}
