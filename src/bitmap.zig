const std = @import("std");
const ArrayContainer = @import("array_container.zig").ArrayContainer;
const BitsetContainer = @import("bitset_container.zig").BitsetContainer;
const RunContainer = @import("run_container.zig").RunContainer;
const container_mod = @import("container.zig");
const Container = container_mod.Container;
const TaggedPtr = container_mod.TaggedPtr;
const ops = @import("container_ops.zig");

/// A Roaring Bitmap: an efficient compressed bitmap for 32-bit integers.
///
/// Partitions the 32-bit space into 2^16 chunks. Each chunk is stored
/// in the optimal container type based on its cardinality.
pub const RoaringBitmap = struct {
    /// Sorted array of 16-bit chunk keys (high bits of contained values).
    keys: []u16,

    /// Array of tagged container pointers (type encoded in low 2 bits).
    containers: []TaggedPtr,

    /// Number of active containers.
    size: u32,

    /// Allocated capacity.
    capacity: u32,

    /// Allocator for all internal memory.
    allocator: std.mem.Allocator,

    const Self = @This();
    const INITIAL_CAPACITY: u32 = 4;

    pub fn init(allocator: std.mem.Allocator) !Self {
        const keys = try allocator.alloc(u16, INITIAL_CAPACITY);
        errdefer allocator.free(keys);

        const containers = try allocator.alloc(TaggedPtr, INITIAL_CAPACITY);

        return .{
            .keys = keys,
            .containers = containers,
            .size = 0,
            .capacity = INITIAL_CAPACITY,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        // Free all containers
        for (self.containers[0..self.size], self.keys[0..self.size]) |tp, _| {
            Container.fromTagged(tp).deinit(self.allocator);
        }
        self.allocator.free(self.keys[0..self.capacity]);
        self.allocator.free(self.containers[0..self.capacity]);
    }

    /// Extract high 16 bits (chunk key) from a 32-bit value.
    inline fn highBits(value: u32) u16 {
        return @truncate(value >> 16);
    }

    /// Extract low 16 bits (value within chunk) from a 32-bit value.
    inline fn lowBits(value: u32) u16 {
        return @truncate(value);
    }

    /// Combine high and low bits into a 32-bit value.
    inline fn combine(high: u16, low: u16) u32 {
        return (@as(u32, high) << 16) | low;
    }

    /// Binary search for a key. Returns index if found, null otherwise.
    fn findKey(self: *const Self, key: u16) ?usize {
        if (self.size == 0) return null;

        var lo: usize = 0;
        var hi: usize = self.size;

        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.keys[mid] < key) {
                lo = mid + 1;
            } else if (self.keys[mid] > key) {
                hi = mid;
            } else {
                return mid;
            }
        }
        return null;
    }

    /// Binary search returning insertion point if not found.
    fn lowerBound(self: *const Self, key: u16) usize {
        var lo: usize = 0;
        var hi: usize = self.size;

        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.keys[mid] < key) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    /// Grow capacity if needed.
    fn ensureCapacity(self: *Self, needed: u32) !void {
        if (needed <= self.capacity) return;

        const new_cap = @max(self.capacity * 2, needed);

        const new_keys = try self.allocator.alloc(u16, new_cap);
        @memcpy(new_keys[0..self.size], self.keys[0..self.size]);
        self.allocator.free(self.keys[0..self.capacity]);
        self.keys = new_keys;

        const new_containers = try self.allocator.alloc(TaggedPtr, new_cap);
        @memcpy(new_containers[0..self.size], self.containers[0..self.size]);
        self.allocator.free(self.containers[0..self.capacity]);
        self.containers = new_containers;

        self.capacity = new_cap;
    }

    /// Check if a value is present.
    pub fn contains(self: *const Self, value: u32) bool {
        const key = highBits(value);
        const idx = self.findKey(key) orelse return false;
        return Container.fromTagged(self.containers[idx]).contains(lowBits(value));
    }

    /// Add a value. Returns true if the value was newly added.
    pub fn add(self: *Self, value: u32) !bool {
        const key = highBits(value);
        const low = lowBits(value);

        if (self.findKey(key)) |idx| {
            // Container exists, add to it
            return self.addToContainer(idx, low);
        }

        // Need to create new container
        const idx = self.lowerBound(key);
        try self.insertContainerAt(idx, key, low);
        return true;
    }

    /// Add a range of values [lo, hi] inclusive. Returns count of newly added values.
    pub fn addRange(self: *Self, lo: u32, hi: u32) !u64 {
        if (lo > hi) return 0;

        var added: u64 = 0;
        var current = lo;

        while (current <= hi) {
            const key = highBits(current);
            const start_low = lowBits(current);

            // End of this chunk or end of range, whichever comes first
            const chunk_end = combine(key, 0xFFFF);
            const range_end_in_chunk = @min(hi, chunk_end);
            const end_low = lowBits(range_end_in_chunk);

            // Add range [start_low, end_low] to this chunk
            added += try self.addRangeToChunk(key, start_low, end_low);

            // Move to next chunk
            if (range_end_in_chunk >= hi) break;
            current = combine(key + 1, 0);
        }

        return added;
    }

    /// Add a range within a single chunk.
    fn addRangeToChunk(self: *Self, key: u16, start: u16, end: u16) !u64 {
        const range_size: u32 = @as(u32, end) - start + 1;

        if (self.findKey(key)) |idx| {
            return self.addRangeToContainer(idx, start, end);
        }

        // Need to create new container
        const insert_idx = self.lowerBound(key);
        try self.ensureCapacity(self.size + 1);

        // Shift right to make room
        if (insert_idx < self.size) {
            std.mem.copyBackwards(u16, self.keys[insert_idx + 1 .. self.size + 1], self.keys[insert_idx..self.size]);
            std.mem.copyBackwards(TaggedPtr, self.containers[insert_idx + 1 .. self.size + 1], self.containers[insert_idx..self.size]);
        }

        // Choose container type based on range size
        if (range_size > ArrayContainer.MAX_CARDINALITY) {
            // Use bitset for large ranges
            const bc = try BitsetContainer.init(self.allocator);
            bc.setRange(start, end);
            self.keys[insert_idx] = key;
            self.containers[insert_idx] = TaggedPtr.initBitset(bc);
        } else {
            // Use array for small ranges
            const ac = try ArrayContainer.init(self.allocator, @intCast(range_size));
            var i: u32 = 0;
            var v: u32 = start;
            while (v <= end) : (v += 1) {
                ac.values[i] = @intCast(v);
                i += 1;
            }
            ac.cardinality = @intCast(range_size);
            self.keys[insert_idx] = key;
            self.containers[insert_idx] = TaggedPtr.initArray(ac);
        }
        self.size += 1;
        return range_size;
    }

    /// Add a range to an existing container.
    fn addRangeToContainer(self: *Self, idx: usize, start: u16, end: u16) !u64 {
        const tp = self.containers[idx];
        const container = Container.fromTagged(tp);

        switch (container) {
            .bitset => |bc| {
                const before: u64 = if (bc.cardinality >= 0) @intCast(bc.cardinality) else bc.computeCardinality();
                bc.setRange(start, end);
                _ = bc.computeCardinality();
                return @as(u64, @intCast(bc.cardinality)) - before;
            },
            .array => |ac| {
                // Check if adding range would overflow
                const range_size: u32 = @as(u32, end) - start + 1;
                if (ac.cardinality + range_size > ArrayContainer.MAX_CARDINALITY) {
                    // Convert to bitset first
                    const bc = try self.arrayToBitset(ac);
                    const before: u64 = @intCast(bc.cardinality);
                    bc.setRange(start, end);
                    _ = bc.computeCardinality();
                    self.containers[idx] = TaggedPtr.initBitset(bc);
                    return @as(u64, @intCast(bc.cardinality)) - before;
                }
                // Add values one by one (could optimize with sorted merge)
                var added: u64 = 0;
                var v: u32 = start;
                while (v <= end) : (v += 1) {
                    if (try ac.add(self.allocator, @intCast(v))) {
                        added += 1;
                    }
                }
                return added;
            },
            .run => |rc| {
                // For run containers, add values one by one (run container handles merging)
                var added: u64 = 0;
                var v: u32 = start;
                while (v <= end) : (v += 1) {
                    if (try rc.add(self.allocator, @intCast(v))) {
                        added += 1;
                    }
                }
                return added;
            },
            .reserved => unreachable,
        }
    }

    /// Create a bitmap from a pre-sorted slice of values. O(n) construction.
    pub fn fromSorted(allocator: std.mem.Allocator, values: []const u32) !Self {
        if (values.len == 0) {
            return Self.init(allocator);
        }

        // Count containers needed
        var container_count: u32 = 1;
        var prev_key = highBits(values[0]);
        for (values[1..]) |v| {
            const key = highBits(v);
            if (key != prev_key) {
                container_count += 1;
                prev_key = key;
            }
        }

        var result = try Self.init(allocator);
        errdefer result.deinit();
        try result.ensureCapacity(container_count);

        // Process each chunk
        var chunk_start: usize = 0;
        while (chunk_start < values.len) {
            const key = highBits(values[chunk_start]);

            // Find end of this chunk
            var chunk_end = chunk_start + 1;
            while (chunk_end < values.len and highBits(values[chunk_end]) == key) {
                chunk_end += 1;
            }

            const chunk_size = chunk_end - chunk_start;

            // Choose container type
            if (chunk_size > ArrayContainer.MAX_CARDINALITY) {
                // Bitset container
                const bc = try BitsetContainer.init(allocator);
                errdefer bc.deinit(allocator);

                for (values[chunk_start..chunk_end]) |v| {
                    _ = bc.add(lowBits(v));
                }
                _ = bc.computeCardinality();

                result.keys[result.size] = key;
                result.containers[result.size] = TaggedPtr.initBitset(bc);
            } else {
                // Array container - values already sorted, just copy low bits
                const ac = try ArrayContainer.init(allocator, @intCast(chunk_size));
                errdefer ac.deinit(allocator);

                for (values[chunk_start..chunk_end], 0..) |v, i| {
                    ac.values[i] = lowBits(v);
                }
                ac.cardinality = @intCast(chunk_size);

                result.keys[result.size] = key;
                result.containers[result.size] = TaggedPtr.initArray(ac);
            }
            result.size += 1;

            chunk_start = chunk_end;
        }

        return result;
    }

    /// Add value to existing container at index, handling type conversion.
    fn addToContainer(self: *Self, idx: usize, low: u16) !bool {
        const tp = self.containers[idx];
        const container = Container.fromTagged(tp);

        switch (container) {
            .array => |ac| {
                // Check if we need to convert to bitset
                if (ac.isFull()) {
                    const bc = try self.arrayToBitset(ac);
                    _ = bc.add(low);
                    self.containers[idx] = TaggedPtr.initBitset(bc);
                    return true;
                }
                return ac.add(self.allocator, low);
            },
            .bitset => |bc| {
                return bc.add(low);
            },
            .run => |rc| {
                return rc.add(self.allocator, low);
            },
            .reserved => unreachable,
        }
    }

    /// Convert array container to bitset container.
    fn arrayToBitset(self: *Self, ac: *ArrayContainer) !*BitsetContainer {
        const bc = try BitsetContainer.init(self.allocator);
        errdefer bc.deinit(self.allocator);

        for (ac.values[0..ac.cardinality]) |v| {
            _ = bc.add(v);
        }

        ac.deinit(self.allocator);
        return bc;
    }

    /// Insert a new container at the given index.
    fn insertContainerAt(self: *Self, idx: usize, key: u16, low: u16) !void {
        try self.ensureCapacity(self.size + 1);

        // Shift right to make room
        if (idx < self.size) {
            std.mem.copyBackwards(u16, self.keys[idx + 1 .. self.size + 1], self.keys[idx..self.size]);
            std.mem.copyBackwards(TaggedPtr, self.containers[idx + 1 .. self.size + 1], self.containers[idx..self.size]);
        }

        // Create new array container with initial value
        const ac = try ArrayContainer.init(self.allocator, 0);
        _ = try ac.add(self.allocator, low);

        self.keys[idx] = key;
        self.containers[idx] = TaggedPtr.initArray(ac);
        self.size += 1;
    }

    /// Remove a value. Returns true if the value was present.
    pub fn remove(self: *Self, value: u32) !bool {
        const key = highBits(value);
        const low = lowBits(value);

        const idx = self.findKey(key) orelse return false;
        return self.removeFromContainer(idx, low);
    }

    /// Remove value from container at index.
    fn removeFromContainer(self: *Self, idx: usize, low: u16) !bool {
        const tp = self.containers[idx];
        const container = Container.fromTagged(tp);

        const was_present = switch (container) {
            .array => |ac| ac.remove(low),
            .bitset => |bc| bc.remove(low),
            .run => |rc| try rc.remove(self.allocator, low),
            .reserved => false,
        };

        if (!was_present) return false;

        // Check if container is now empty
        const card = Container.fromTagged(self.containers[idx]).getCardinality();
        if (card == 0) {
            self.removeContainerAt(idx);
        }

        return true;
    }

    /// Remove container at the given index.
    fn removeContainerAt(self: *Self, idx: usize) void {
        Container.fromTagged(self.containers[idx]).deinit(self.allocator);

        // Shift left
        if (idx + 1 < self.size) {
            std.mem.copyForwards(u16, self.keys[idx .. self.size - 1], self.keys[idx + 1 .. self.size]);
            std.mem.copyForwards(TaggedPtr, self.containers[idx .. self.size - 1], self.containers[idx + 1 .. self.size]);
        }
        self.size -= 1;
    }

    /// Get the total cardinality (number of values).
    pub fn cardinality(self: *const Self) u64 {
        var total: u64 = 0;
        for (self.containers[0..self.size]) |tp| {
            total += Container.fromTagged(tp).getCardinality();
        }
        return total;
    }

    /// Check if the bitmap is empty.
    pub fn isEmpty(self: *const Self) bool {
        return self.size == 0;
    }

    /// Get the minimum value, or null if empty.
    pub fn minimum(self: *const Self) ?u32 {
        if (self.size == 0) return null;

        const key = self.keys[0];
        const container = Container.fromTagged(self.containers[0]);

        // Find minimum in first container
        const low: ?u16 = switch (container) {
            .array => |ac| if (ac.cardinality > 0) ac.values[0] else null,
            .bitset => |bc| blk: {
                for (bc.words, 0..) |word, i| {
                    if (word != 0) {
                        break :blk @intCast(i * 64 + @ctz(word));
                    }
                }
                break :blk null;
            },
            .run => |rc| if (rc.n_runs > 0) rc.runs[0].start else null,
            .reserved => null,
        };

        return if (low) |l| combine(key, l) else null;
    }

    /// Get the maximum value, or null if empty.
    pub fn maximum(self: *const Self) ?u32 {
        if (self.size == 0) return null;

        const key = self.keys[self.size - 1];
        const container = Container.fromTagged(self.containers[self.size - 1]);

        // Find maximum in last container
        const low: ?u16 = switch (container) {
            .array => |ac| if (ac.cardinality > 0) ac.values[ac.cardinality - 1] else null,
            .bitset => |bc| blk: {
                var i: usize = 1024;
                while (i > 0) {
                    i -= 1;
                    if (bc.words[i] != 0) {
                        break :blk @intCast(i * 64 + 63 - @clz(bc.words[i]));
                    }
                }
                break :blk null;
            },
            .run => |rc| if (rc.n_runs > 0) rc.runs[rc.n_runs - 1].end() else null,
            .reserved => null,
        };

        return if (low) |l| combine(key, l) else null;
    }

    // ========================================================================
    // Set Operations
    // ========================================================================

    /// Return a new bitmap that is the union (OR) of self and other.
    pub fn bitwiseOr(self: *const Self, allocator: std.mem.Allocator, other: *const Self) !Self {
        var result = try Self.init(allocator);
        errdefer result.deinit();

        var i: usize = 0;
        var j: usize = 0;

        while (i < self.size and j < other.size) {
            const key_a = self.keys[i];
            const key_b = other.keys[j];

            if (key_a < key_b) {
                try result.appendContainer(key_a, try cloneContainer(allocator, self.containers[i]));
                i += 1;
            } else if (key_a > key_b) {
                try result.appendContainer(key_b, try cloneContainer(allocator, other.containers[j]));
                j += 1;
            } else {
                const c = try ops.containerUnion(
                    allocator,
                    Container.fromTagged(self.containers[i]),
                    Container.fromTagged(other.containers[j]),
                );
                try result.appendContainer(key_a, c.toTagged());
                i += 1;
                j += 1;
            }
        }

        while (i < self.size) : (i += 1) {
            try result.appendContainer(self.keys[i], try cloneContainer(allocator, self.containers[i]));
        }
        while (j < other.size) : (j += 1) {
            try result.appendContainer(other.keys[j], try cloneContainer(allocator, other.containers[j]));
        }

        return result;
    }

    /// Return a new bitmap that is the intersection (AND) of self and other.
    pub fn bitwiseAnd(self: *const Self, allocator: std.mem.Allocator, other: *const Self) !Self {
        var result = try Self.init(allocator);
        errdefer result.deinit();

        var i: usize = 0;
        var j: usize = 0;

        while (i < self.size and j < other.size) {
            const key_a = self.keys[i];
            const key_b = other.keys[j];

            if (key_a < key_b) {
                i += 1;
            } else if (key_a > key_b) {
                j += 1;
            } else {
                const c = try ops.containerIntersection(
                    allocator,
                    Container.fromTagged(self.containers[i]),
                    Container.fromTagged(other.containers[j]),
                );
                if (c.getCardinality() > 0) {
                    try result.appendContainer(key_a, c.toTagged());
                } else {
                    c.deinit(allocator);
                }
                i += 1;
                j += 1;
            }
        }

        return result;
    }

    /// Return a new bitmap that is the difference (AND NOT) of self and other.
    pub fn bitwiseDifference(self: *const Self, allocator: std.mem.Allocator, other: *const Self) !Self {
        var result = try Self.init(allocator);
        errdefer result.deinit();

        var i: usize = 0;
        var j: usize = 0;

        while (i < self.size) {
            const key_a = self.keys[i];

            // Advance j to key_a or past it
            while (j < other.size and other.keys[j] < key_a) : (j += 1) {}

            if (j >= other.size or other.keys[j] > key_a) {
                // No matching key in other, copy entire container
                try result.appendContainer(key_a, try cloneContainer(allocator, self.containers[i]));
            } else {
                // Matching key, compute difference
                const c = try ops.containerDifference(
                    allocator,
                    Container.fromTagged(self.containers[i]),
                    Container.fromTagged(other.containers[j]),
                );
                if (c.getCardinality() > 0) {
                    try result.appendContainer(key_a, c.toTagged());
                } else {
                    c.deinit(allocator);
                }
                j += 1;
            }
            i += 1;
        }

        return result;
    }

    /// Return a new bitmap that is the symmetric difference (XOR) of self and other.
    pub fn bitwiseXor(self: *const Self, allocator: std.mem.Allocator, other: *const Self) !Self {
        var result = try Self.init(allocator);
        errdefer result.deinit();

        var i: usize = 0;
        var j: usize = 0;

        while (i < self.size and j < other.size) {
            const key_a = self.keys[i];
            const key_b = other.keys[j];

            if (key_a < key_b) {
                try result.appendContainer(key_a, try cloneContainer(allocator, self.containers[i]));
                i += 1;
            } else if (key_a > key_b) {
                try result.appendContainer(key_b, try cloneContainer(allocator, other.containers[j]));
                j += 1;
            } else {
                const c = try ops.containerXor(
                    allocator,
                    Container.fromTagged(self.containers[i]),
                    Container.fromTagged(other.containers[j]),
                );
                if (c.getCardinality() > 0) {
                    try result.appendContainer(key_a, c.toTagged());
                } else {
                    c.deinit(allocator);
                }
                i += 1;
                j += 1;
            }
        }

        while (i < self.size) : (i += 1) {
            try result.appendContainer(self.keys[i], try cloneContainer(allocator, self.containers[i]));
        }
        while (j < other.size) : (j += 1) {
            try result.appendContainer(other.keys[j], try cloneContainer(allocator, other.containers[j]));
        }

        return result;
    }

    // ========================================================================
    // In-Place Set Operations
    // ========================================================================

    /// In-place union: self |= other. Modifies self to contain all values from both.
    pub fn bitwiseOrInPlace(self: *Self, other: *const Self) !void {
        if (other.size == 0) return;

        // We need to merge other's containers into self. This may require:
        // 1. Inserting new containers for keys only in other
        // 2. Merging containers for keys in both
        // Strategy: work backwards to avoid shifting issues, or use a temp array

        var j: usize = 0; // index into other
        var i: usize = 0; // index into self

        while (j < other.size) {
            const key_b = other.keys[j];

            // Find position in self for this key
            while (i < self.size and self.keys[i] < key_b) : (i += 1) {}

            if (i < self.size and self.keys[i] == key_b) {
                // Key exists in both - merge containers
                const old_container = Container.fromTagged(self.containers[i]);
                const other_container = Container.fromTagged(other.containers[j]);
                const merged = try ops.containerUnion(self.allocator, old_container, other_container);
                old_container.deinit(self.allocator);
                self.containers[i] = merged.toTagged();
                i += 1;
            } else {
                // Key only in other - insert cloned container
                const cloned = try cloneContainer(self.allocator, other.containers[j]);
                try self.insertTaggedContainerAt(i, key_b, cloned);
                i += 1; // skip past inserted
            }
            j += 1;
        }
    }

    /// In-place intersection: self &= other. Modifies self to contain only values in both.
    pub fn bitwiseAndInPlace(self: *Self, other: *const Self) !void {
        if (other.size == 0) {
            // Clear self
            for (self.containers[0..self.size]) |tp| {
                Container.fromTagged(tp).deinit(self.allocator);
            }
            self.size = 0;
            return;
        }

        var write_idx: usize = 0;
        var i: usize = 0;
        var j: usize = 0;

        while (i < self.size and j < other.size) {
            const key_a = self.keys[i];
            const key_b = other.keys[j];

            if (key_a < key_b) {
                // Key only in self - remove it
                Container.fromTagged(self.containers[i]).deinit(self.allocator);
                i += 1;
            } else if (key_a > key_b) {
                j += 1;
            } else {
                // Key in both - intersect containers
                const self_container = Container.fromTagged(self.containers[i]);
                const other_container = Container.fromTagged(other.containers[j]);
                const intersected = try ops.containerIntersection(self.allocator, self_container, other_container);
                self_container.deinit(self.allocator);

                if (intersected.getCardinality() > 0) {
                    self.keys[write_idx] = key_a;
                    self.containers[write_idx] = intersected.toTagged();
                    write_idx += 1;
                } else {
                    intersected.deinit(self.allocator);
                }
                i += 1;
                j += 1;
            }
        }

        // Remove remaining containers from self (not in other)
        while (i < self.size) : (i += 1) {
            Container.fromTagged(self.containers[i]).deinit(self.allocator);
        }

        self.size = @intCast(write_idx);
    }

    /// In-place difference: self -= other. Modifies self to remove values in other.
    pub fn bitwiseDifferenceInPlace(self: *Self, other: *const Self) !void {
        if (other.size == 0) return;

        var write_idx: usize = 0;
        var i: usize = 0;
        var j: usize = 0;

        while (i < self.size) {
            const key_a = self.keys[i];

            // Advance j to key_a or past it
            while (j < other.size and other.keys[j] < key_a) : (j += 1) {}

            if (j >= other.size or other.keys[j] > key_a) {
                // No matching key in other - keep container as-is
                self.keys[write_idx] = key_a;
                self.containers[write_idx] = self.containers[i];
                write_idx += 1;
            } else {
                // Matching key - compute difference
                const self_container = Container.fromTagged(self.containers[i]);
                const other_container = Container.fromTagged(other.containers[j]);
                const diff = try ops.containerDifference(self.allocator, self_container, other_container);
                self_container.deinit(self.allocator);

                if (diff.getCardinality() > 0) {
                    self.keys[write_idx] = key_a;
                    self.containers[write_idx] = diff.toTagged();
                    write_idx += 1;
                } else {
                    diff.deinit(self.allocator);
                }
                j += 1;
            }
            i += 1;
        }

        self.size = @intCast(write_idx);
    }

    /// Insert a tagged container at the given position, shifting existing containers.
    fn insertTaggedContainerAt(self: *Self, pos: usize, key: u16, tp: TaggedPtr) !void {
        try self.ensureCapacity(self.size + 1);
        // Shift elements right
        var k: usize = self.size;
        while (k > pos) : (k -= 1) {
            self.keys[k] = self.keys[k - 1];
            self.containers[k] = self.containers[k - 1];
        }
        self.keys[pos] = key;
        self.containers[pos] = tp;
        self.size += 1;
    }

    /// Check if self is a subset of other. O(n) where n is total container size.
    pub fn isSubsetOf(self: *const Self, other: *const Self) bool {
        // Fast-path: if self has more keys, it can't be a subset
        if (self.size > other.size) return false;

        var j: usize = 0;
        for (0..self.size) |i| {
            // Find matching key in other
            while (j < other.size and other.keys[j] < self.keys[i]) : (j += 1) {}

            if (j >= other.size or other.keys[j] != self.keys[i]) {
                return false; // Key not found in other
            }

            // Check container subset with O(n) algorithms per container type
            if (!containerIsSubset(self.containers[i], other.containers[j])) {
                return false;
            }
            j += 1;
        }
        return true;
    }

    /// Check if two bitmaps are equal. Single pass O(n).
    pub fn equals(self: *const Self, other: *const Self) bool {
        if (self.size != other.size) return false;

        for (0..self.size) |i| {
            if (self.keys[i] != other.keys[i]) return false;
            if (!containerEquals(self.containers[i], other.containers[i])) {
                return false;
            }
        }
        return true;
    }

    /// O(n) container subset check.
    fn containerIsSubset(a_tp: TaggedPtr, b_tp: TaggedPtr) bool {
        const a = Container.fromTagged(a_tp);
        const b = Container.fromTagged(b_tp);

        // Quick cardinality check
        if (a.getCardinality() > b.getCardinality()) return false;

        return switch (a) {
            .array => |ac| switch (b) {
                // array ⊆ array: merge-walk O(n+m)
                .array => |bc| arraySubsetArray(ac, bc),
                // array ⊆ bitset: O(n) lookups, each O(1)
                .bitset => |bc| arraySubsetBitset(ac, bc),
                // array ⊆ run: O(n) lookups
                .run => |rc| arraySubsetRun(ac, rc),
                .reserved => false,
            },
            .bitset => |ac| switch (b) {
                // bitset ⊆ array: bitset can have <4096 cardinality, check each bit
                .array => |bc| bitsetSubsetArray(ac, bc),
                // bitset ⊆ bitset: (a & ~b) == 0, O(1024) words
                .bitset => |bc| bitsetSubsetBitset(ac, bc),
                // bitset ⊆ run: check each set bit
                .run => |rc| bitsetSubsetRun(ac, rc),
                .reserved => false,
            },
            .run => |ac| switch (b) {
                // run ⊆ array: check each run value
                .array => |bc| runSubsetArray(ac, bc),
                // run ⊆ bitset: check each run value
                .bitset => |bc| runSubsetBitset(ac, bc),
                // run ⊆ run: check each run
                .run => |rc| runSubsetRun(ac, rc),
                .reserved => false,
            },
            .reserved => false,
        };
    }

    fn arraySubsetArray(a: *ArrayContainer, b: *ArrayContainer) bool {
        // Merge-walk: O(n+m)
        var i: usize = 0;
        var j: usize = 0;
        while (i < a.cardinality) {
            if (j >= b.cardinality) return false;
            if (a.values[i] < b.values[j]) return false; // a has element not in b
            if (a.values[i] == b.values[j]) i += 1;
            j += 1;
        }
        return true;
    }

    fn arraySubsetBitset(a: *ArrayContainer, b: *BitsetContainer) bool {
        for (a.values[0..a.cardinality]) |v| {
            if (!b.contains(v)) return false;
        }
        return true;
    }

    fn arraySubsetRun(a: *ArrayContainer, b: *RunContainer) bool {
        for (a.values[0..a.cardinality]) |v| {
            if (!b.contains(v)) return false;
        }
        return true;
    }

    fn bitsetSubsetBitset(a: *BitsetContainer, b: *BitsetContainer) bool {
        // (a & ~b) == 0 means all bits in a are also in b
        for (a.words, b.words) |aw, bw| {
            if ((aw & ~bw) != 0) return false;
        }
        return true;
    }

    fn bitsetSubsetArray(a: *BitsetContainer, b: *ArrayContainer) bool {
        // Check each set bit in bitset is present in array
        for (a.words, 0..) |word, word_idx| {
            var w = word;
            while (w != 0) {
                const bit = @ctz(w);
                const v: u16 = @intCast(word_idx * 64 + bit);
                if (!b.contains(v)) return false;
                w &= w - 1;
            }
        }
        return true;
    }

    fn bitsetSubsetRun(a: *BitsetContainer, b: *RunContainer) bool {
        for (a.words, 0..) |word, word_idx| {
            var w = word;
            while (w != 0) {
                const bit = @ctz(w);
                const v: u16 = @intCast(word_idx * 64 + bit);
                if (!b.contains(v)) return false;
                w &= w - 1;
            }
        }
        return true;
    }

    fn runSubsetArray(a: *RunContainer, b: *ArrayContainer) bool {
        for (a.runs[0..a.n_runs]) |run| {
            var v: u32 = run.start;
            while (v <= run.end()) : (v += 1) {
                if (!b.contains(@intCast(v))) return false;
            }
        }
        return true;
    }

    fn runSubsetBitset(a: *RunContainer, b: *BitsetContainer) bool {
        for (a.runs[0..a.n_runs]) |run| {
            var v: u32 = run.start;
            while (v <= run.end()) : (v += 1) {
                if (!b.contains(@intCast(v))) return false;
            }
        }
        return true;
    }

    fn runSubsetRun(a: *RunContainer, b: *RunContainer) bool {
        // Merge-walk: O(n_runs_a + n_runs_b) instead of O(cardinality × log(n_runs))
        var j: usize = 0;
        for (a.runs[0..a.n_runs]) |run_a| {
            const start_a = run_a.start;
            const end_a = run_a.end();

            // Skip B runs that end before A's run starts
            while (j < b.n_runs and b.runs[j].end() < start_a) : (j += 1) {}

            // Check that B's runs fully cover [start_a, end_a]
            var pos: u32 = start_a;
            while (pos <= end_a) {
                if (j >= b.n_runs) return false; // No more runs in B
                if (b.runs[j].start > pos) return false; // Gap in B's coverage
                // B.runs[j] covers up to its end
                pos = b.runs[j].end() + 1;
                if (pos <= end_a) j += 1; // Need next run in B
            }
        }
        return true;
    }

    /// O(n) container equality check.
    fn containerEquals(a_tp: TaggedPtr, b_tp: TaggedPtr) bool {
        const a = Container.fromTagged(a_tp);
        const b = Container.fromTagged(b_tp);

        // Different types can still be equal if same cardinality and same values
        if (a.getCardinality() != b.getCardinality()) return false;

        return switch (a) {
            .array => |ac| switch (b) {
                .array => |bc| std.mem.eql(u16, ac.values[0..ac.cardinality], bc.values[0..bc.cardinality]),
                .bitset => |bc| arrayEqualsBitset(ac, bc),
                .run => |rc| arrayEqualsRun(ac, rc),
                .reserved => false,
            },
            .bitset => |ac| switch (b) {
                .array => |bc| arrayEqualsBitset(bc, ac),
                .bitset => |bc| std.mem.eql(u64, ac.words, bc.words),
                .run => |rc| bitsetEqualsRun(ac, rc),
                .reserved => false,
            },
            .run => |ac| switch (b) {
                .array => |bc| arrayEqualsRun(bc, ac),
                .bitset => |bc| bitsetEqualsRun(bc, ac),
                // Same values can have different run encodings, so compare element-by-element
                .run => |rc| runEqualsRun(ac, rc),
                .reserved => false,
            },
            .reserved => false,
        };
    }

    fn arrayEqualsBitset(a: *ArrayContainer, b: *BitsetContainer) bool {
        // Cardinality already checked equal
        for (a.values[0..a.cardinality]) |v| {
            if (!b.contains(v)) return false;
        }
        return true;
    }

    fn arrayEqualsRun(a: *ArrayContainer, b: *RunContainer) bool {
        for (a.values[0..a.cardinality]) |v| {
            if (!b.contains(v)) return false;
        }
        return true;
    }

    fn bitsetEqualsRun(a: *BitsetContainer, b: *RunContainer) bool {
        for (a.words, 0..) |word, word_idx| {
            var w = word;
            while (w != 0) {
                const bit = @ctz(w);
                const v: u16 = @intCast(word_idx * 64 + bit);
                if (!b.contains(v)) return false;
                w &= w - 1;
            }
        }
        return true;
    }

    fn runEqualsRun(a: *RunContainer, b: *RunContainer) bool {
        // Cardinality already checked equal, so |A|=|B| ∧ A⊆B → A=B
        return runSubsetRun(a, b);
    }

    // ========================================================================
    // Helper Functions
    // ========================================================================

    /// Append a container (assumes keys are in sorted order).
    fn appendContainer(self: *Self, key: u16, tp: TaggedPtr) !void {
        try self.ensureCapacity(self.size + 1);
        self.keys[self.size] = key;
        self.containers[self.size] = tp;
        self.size += 1;
    }

    /// Clone a container.
    fn cloneContainer(allocator: std.mem.Allocator, tp: TaggedPtr) !TaggedPtr {
        const container = Container.fromTagged(tp);
        return switch (container) {
            .array => |ac| blk: {
                const new_ac = try ArrayContainer.init(allocator, ac.cardinality);
                @memcpy(new_ac.values[0..ac.cardinality], ac.values[0..ac.cardinality]);
                new_ac.cardinality = ac.cardinality;
                break :blk TaggedPtr.initArray(new_ac);
            },
            .bitset => |bc| blk: {
                const new_bc = try BitsetContainer.init(allocator);
                @memcpy(new_bc.words, bc.words);
                new_bc.cardinality = bc.cardinality;
                break :blk TaggedPtr.initBitset(new_bc);
            },
            .run => |rc| blk: {
                const new_rc = try RunContainer.init(allocator, rc.n_runs);
                @memcpy(new_rc.runs[0..rc.n_runs], rc.runs[0..rc.n_runs]);
                new_rc.n_runs = rc.n_runs;
                break :blk TaggedPtr.initRun(new_rc);
            },
            .reserved => unreachable,
        };
    }

    // ========================================================================
    // Iterator
    // ========================================================================

    /// Iterator over all values in the bitmap in ascending order.
    pub const Iterator = struct {
        bm: *const Self,
        container_idx: u32,
        /// Per-container iteration state
        state: ContainerState,

        const ContainerState = union(enum) {
            empty: void,
            array: ArrayState,
            bitset: BitsetState,
            run: RunState,
        };

        const ArrayState = struct {
            values: []const u16,
            pos: u32,
        };

        const BitsetState = struct {
            words: []const u64,
            word_idx: u32,
            current_word: u64,
        };

        const RunState = struct {
            runs: []const RunContainer.RunPair,
            run_idx: u32,
            pos_in_run: u16, // offset within current run
        };

        pub fn next(self: *Iterator) ?u32 {
            while (true) {
                switch (self.state) {
                    .empty => {
                        // Move to next container
                        if (self.container_idx >= self.bm.size) return null;
                        self.initContainer(self.container_idx);
                    },
                    .array => |*s| {
                        if (s.pos < s.values.len) {
                            const high: u32 = @as(u32, self.bm.keys[self.container_idx]) << 16;
                            const low: u32 = s.values[s.pos];
                            s.pos += 1;
                            return high | low;
                        }
                        self.advanceContainer();
                    },
                    .bitset => |*s| {
                        // Find next set bit
                        while (s.current_word == 0) {
                            s.word_idx += 1;
                            if (s.word_idx >= 1024) {
                                self.advanceContainer();
                                break;
                            }
                            s.current_word = s.words[s.word_idx];
                        } else {
                            const bit = @ctz(s.current_word);
                            s.current_word &= s.current_word - 1; // clear lowest bit
                            const high: u32 = @as(u32, self.bm.keys[self.container_idx]) << 16;
                            const low: u32 = @as(u32, s.word_idx) * 64 + bit;
                            return high | low;
                        }
                    },
                    .run => |*s| {
                        if (s.run_idx < s.runs.len) {
                            const run = s.runs[s.run_idx];
                            const high: u32 = @as(u32, self.bm.keys[self.container_idx]) << 16;
                            const low: u32 = @as(u32, run.start) + s.pos_in_run;

                            // run.length is the count beyond start, so {start=10, length=2} covers 10,11,12
                            if (s.pos_in_run <= run.length) {
                                const result = high | low;
                                if (s.pos_in_run < run.length) {
                                    s.pos_in_run += 1;
                                } else {
                                    // Move to next run
                                    s.run_idx += 1;
                                    s.pos_in_run = 0;
                                }
                                return result;
                            } else {
                                self.advanceContainer();
                            }
                        } else {
                            self.advanceContainer();
                        }
                    },
                }
            }
        }

        fn initContainer(self: *Iterator, idx: u32) void {
            const container = Container.fromTagged(self.bm.containers[idx]);
            switch (container) {
                .array => |ac| {
                    self.state = .{ .array = .{
                        .values = ac.values[0..ac.cardinality],
                        .pos = 0,
                    } };
                },
                .bitset => |bc| {
                    // Find first non-zero word
                    var word_idx: u32 = 0;
                    while (word_idx < 1024 and bc.words[word_idx] == 0) : (word_idx += 1) {}
                    if (word_idx < 1024) {
                        self.state = .{ .bitset = .{
                            .words = bc.words,
                            .word_idx = word_idx,
                            .current_word = bc.words[word_idx],
                        } };
                    } else {
                        self.state = .empty;
                    }
                },
                .run => |rc| {
                    if (rc.n_runs > 0) {
                        self.state = .{ .run = .{
                            .runs = rc.runs[0..rc.n_runs],
                            .run_idx = 0,
                            .pos_in_run = 0,
                        } };
                    } else {
                        self.state = .empty;
                    }
                },
                .reserved => self.state = .empty,
            }
        }

        fn advanceContainer(self: *Iterator) void {
            self.container_idx += 1;
            self.state = .empty;
        }
    };

    /// Returns an iterator over all values in the bitmap.
    pub fn iterator(self: *const Self) Iterator {
        var it = Iterator{
            .bm = self,
            .container_idx = 0,
            .state = .empty,
        };
        if (self.size > 0) {
            it.initContainer(0);
        }
        return it;
    }

    // ========================================================================
    // Serialization (RoaringFormatSpec compatible)
    // ========================================================================

    /// Cookie values for RoaringFormatSpec
    const SERIAL_COOKIE_NO_RUNCONTAINER: u32 = 12346;
    const SERIAL_COOKIE: u32 = 12347;
    const NO_OFFSET_THRESHOLD: u32 = 4; // Containers below this don't use offset header

    /// Returns true if any container is a run container.
    fn hasRunContainers(self: *const Self) bool {
        for (self.containers[0..self.size]) |tp| {
            if (TaggedPtr.getType(tp) == .run) return true;
        }
        return false;
    }

    /// Compute serialized size in bytes.
    pub fn serializedSizeInBytes(self: *const Self) usize {
        if (self.size == 0) return 8; // Just header

        const has_runs = self.hasRunContainers();
        var size: usize = 0;

        // Cookie + size (or cookie with embedded size for run format)
        if (has_runs) {
            size += 4; // cookie with size embedded
            // Run container bitset: ceil(size / 8) bytes
            size += (self.size + 7) / 8;
        } else {
            size += 8; // cookie + size
        }

        // Descriptive header: 4 bytes per container (key + cardinality-1)
        size += @as(usize, self.size) * 4;

        // Offset header for large bitmaps (>= NO_OFFSET_THRESHOLD containers with runs)
        if (has_runs and self.size >= NO_OFFSET_THRESHOLD) {
            size += @as(usize, self.size) * 4; // 4 bytes per container offset
        }

        // Container data
        for (self.containers[0..self.size]) |tp| {
            const container = Container.fromTagged(tp);
            size += switch (container) {
                .array => |ac| @as(usize, ac.cardinality) * 2,
                .bitset => 8192, // 1024 * 8 bytes
                .run => |rc| 2 + @as(usize, rc.n_runs) * 4, // n_runs prefix + pairs
                .reserved => 0,
            };
        }

        return size;
    }

    /// Serialize the bitmap to a byte slice (RoaringFormatSpec compatible).
    pub fn serialize(self: *const Self, allocator: std.mem.Allocator) ![]u8 {
        const size_bytes = self.serializedSizeInBytes();
        const buf = try allocator.alloc(u8, size_bytes);
        errdefer allocator.free(buf);

        var stream = std.io.fixedBufferStream(buf);
        const writer = stream.writer();

        try self.serializeToWriter(writer);

        return buf;
    }

    /// Serialize to any writer.
    pub fn serializeToWriter(self: *const Self, writer: anytype) !void {
        if (self.size == 0) {
            // Empty bitmap
            try writer.writeInt(u32, SERIAL_COOKIE_NO_RUNCONTAINER, .little);
            try writer.writeInt(u32, 0, .little);
            return;
        }

        const has_runs = self.hasRunContainers();

        if (has_runs) {
            // Cookie with size embedded in high 16 bits
            const cookie: u32 = SERIAL_COOKIE | (@as(u32, self.size - 1) << 16);
            try writer.writeInt(u32, cookie, .little);

            // Run container bitset (max 8KB for 65536 containers)
            const bitset_bytes = (self.size + 7) / 8;
            var run_bitset_buf: [8192]u8 = undefined;
            const run_bitset = run_bitset_buf[0..bitset_bytes];
            @memset(run_bitset, 0);

            for (self.containers[0..self.size], 0..) |tp, i| {
                if (TaggedPtr.getType(tp) == .run) {
                    run_bitset[i / 8] |= @as(u8, 1) << @intCast(i % 8);
                }
            }
            try writer.writeAll(run_bitset);
        } else {
            try writer.writeInt(u32, SERIAL_COOKIE_NO_RUNCONTAINER, .little);
            try writer.writeInt(u32, self.size, .little);
        }

        // Descriptive header: key (u16) + cardinality-1 (u16) per container
        for (self.containers[0..self.size], self.keys[0..self.size]) |tp, key| {
            try writer.writeInt(u16, key, .little);
            const card = Container.fromTagged(tp).getCardinality();
            try writer.writeInt(u16, @intCast(card - 1), .little);
        }

        // Offset header (only for runs with >= 4 containers)
        if (has_runs and self.size >= NO_OFFSET_THRESHOLD) {
            var offset: u32 = 0;
            for (self.containers[0..self.size]) |tp| {
                try writer.writeInt(u32, offset, .little);
                const container = Container.fromTagged(tp);
                offset += switch (container) {
                    .array => |ac| @as(u32, ac.cardinality) * 2,
                    .bitset => 8192,
                    .run => |rc| 2 + @as(u32, rc.n_runs) * 4, // n_runs prefix + pairs
                    .reserved => 0,
                };
            }
        }

        // Container data
        for (self.containers[0..self.size]) |tp| {
            const container = Container.fromTagged(tp);
            switch (container) {
                .array => |ac| {
                    for (ac.values[0..ac.cardinality]) |v| {
                        try writer.writeInt(u16, v, .little);
                    }
                },
                .bitset => |bc| {
                    for (bc.words) |word| {
                        try writer.writeInt(u64, word, .little);
                    }
                },
                .run => |rc| {
                    // RoaringFormatSpec: n_runs prefix followed by run pairs
                    try writer.writeInt(u16, rc.n_runs, .little);
                    for (rc.runs[0..rc.n_runs]) |run| {
                        try writer.writeInt(u16, run.start, .little);
                        try writer.writeInt(u16, run.length, .little);
                    }
                },
                .reserved => {},
            }
        }
    }

    /// Deserialize a bitmap from bytes (RoaringFormatSpec compatible).
    pub fn deserialize(allocator: std.mem.Allocator, data: []const u8) !Self {
        if (data.len < 4) return error.InvalidFormat;

        var stream = std.io.fixedBufferStream(data);
        const reader = stream.reader();

        return deserializeFromReader(allocator, reader, data.len);
    }

    /// Deserialize from any reader.
    pub fn deserializeFromReader(allocator: std.mem.Allocator, reader: anytype, data_len: usize) !Self {
        _ = data_len;

        const cookie = try reader.readInt(u32, .little);

        var size: u32 = undefined;
        var has_runs = false;
        var run_bitset: ?[]u8 = null;
        defer if (run_bitset) |rb| allocator.free(rb);

        if ((cookie & 0xFFFF) == SERIAL_COOKIE) {
            // Format with run containers
            has_runs = true;
            size = ((cookie >> 16) & 0xFFFF) + 1;

            // Read run container bitset
            const bitset_bytes = (size + 7) / 8;
            run_bitset = try allocator.alloc(u8, bitset_bytes);
            const bytes_read = try reader.readAll(run_bitset.?);
            if (bytes_read != bitset_bytes) return error.InvalidFormat;
        } else if (cookie == SERIAL_COOKIE_NO_RUNCONTAINER) {
            // Format without run containers
            size = try reader.readInt(u32, .little);
        } else {
            return error.InvalidFormat;
        }

        if (size == 0) {
            return Self.init(allocator);
        }

        var result = try Self.init(allocator);
        errdefer result.deinit();

        try result.ensureCapacity(size);

        // Read descriptive header
        var cardinalities = try allocator.alloc(u32, size);
        defer allocator.free(cardinalities);

        for (0..size) |i| {
            result.keys[i] = try reader.readInt(u16, .little);
            cardinalities[i] = @as(u32, try reader.readInt(u16, .little)) + 1;
        }

        // Skip offset header if present
        if (has_runs and size >= NO_OFFSET_THRESHOLD) {
            for (0..size) |_| {
                _ = try reader.readInt(u32, .little);
            }
        }

        // Read container data
        for (0..size) |i| {
            const is_run = if (run_bitset) |rb|
                (rb[i / 8] & (@as(u8, 1) << @intCast(i % 8))) != 0
            else
                false;

            const card = cardinalities[i];

            if (is_run) {
                // Run container: n_runs is in the data section prefix, not the header
                // (header stores cardinality-1 which is sum of run lengths, not n_runs)
                const n_runs = try reader.readInt(u16, .little);
                const rc = try RunContainer.init(allocator, n_runs);
                errdefer rc.deinit(allocator);

                for (0..n_runs) |r| {
                    rc.runs[r].start = try reader.readInt(u16, .little);
                    rc.runs[r].length = try reader.readInt(u16, .little);
                }
                rc.n_runs = n_runs;
                result.containers[i] = TaggedPtr.initRun(rc);
            } else if (card > ArrayContainer.MAX_CARDINALITY) {
                // Bitset container
                const bc = try BitsetContainer.init(allocator);
                errdefer bc.deinit(allocator);

                for (0..1024) |w| {
                    bc.words[w] = try reader.readInt(u64, .little);
                }
                bc.cardinality = @intCast(card);
                result.containers[i] = TaggedPtr.initBitset(bc);
            } else {
                // Array container
                const ac = try ArrayContainer.init(allocator, @intCast(card));
                errdefer ac.deinit(allocator);

                for (0..card) |v| {
                    ac.values[v] = try reader.readInt(u16, .little);
                }
                ac.cardinality = @intCast(card);
                result.containers[i] = TaggedPtr.initArray(ac);
            }
        }

        result.size = size;
        return result;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "init and deinit" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    try std.testing.expect(bm.isEmpty());
    try std.testing.expectEqual(@as(u64, 0), bm.cardinality());
}

test "add and contains" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Add some values
    try std.testing.expect(try bm.add(42));
    try std.testing.expect(try bm.add(1000));
    try std.testing.expect(try bm.add(100000));

    // Check they're present
    try std.testing.expect(bm.contains(42));
    try std.testing.expect(bm.contains(1000));
    try std.testing.expect(bm.contains(100000));

    // Check absent values
    try std.testing.expect(!bm.contains(0));
    try std.testing.expect(!bm.contains(43));
    try std.testing.expect(!bm.contains(999));

    // Check cardinality
    try std.testing.expectEqual(@as(u64, 3), bm.cardinality());

    // Adding duplicate returns false
    try std.testing.expect(!try bm.add(42));
    try std.testing.expectEqual(@as(u64, 3), bm.cardinality());
}

test "values in same chunk" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // All in chunk 0 (values 0-65535)
    _ = try bm.add(0);
    _ = try bm.add(100);
    _ = try bm.add(1000);
    _ = try bm.add(65535);

    try std.testing.expectEqual(@as(u32, 1), bm.size); // Single container
    try std.testing.expectEqual(@as(u64, 4), bm.cardinality());

    try std.testing.expect(bm.contains(0));
    try std.testing.expect(bm.contains(100));
    try std.testing.expect(bm.contains(1000));
    try std.testing.expect(bm.contains(65535));
}

test "values in different chunks" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Each in different chunk
    _ = try bm.add(0); // chunk 0
    _ = try bm.add(65536); // chunk 1
    _ = try bm.add(131072); // chunk 2
    _ = try bm.add(0xFFFFFFFF); // chunk 65535

    try std.testing.expectEqual(@as(u32, 4), bm.size); // Four containers
    try std.testing.expectEqual(@as(u64, 4), bm.cardinality());
}

test "remove" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(10);
    _ = try bm.add(20);
    _ = try bm.add(30);

    try std.testing.expect(try bm.remove(20));
    try std.testing.expect(!bm.contains(20));
    try std.testing.expectEqual(@as(u64, 2), bm.cardinality());

    // Remove absent value
    try std.testing.expect(!try bm.remove(20));

    // Remove last values - container should be removed
    try std.testing.expect(try bm.remove(10));
    try std.testing.expect(try bm.remove(30));
    try std.testing.expect(bm.isEmpty());
}

test "minimum and maximum" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    try std.testing.expectEqual(@as(?u32, null), bm.minimum());
    try std.testing.expectEqual(@as(?u32, null), bm.maximum());

    _ = try bm.add(100);
    try std.testing.expectEqual(@as(?u32, 100), bm.minimum());
    try std.testing.expectEqual(@as(?u32, 100), bm.maximum());

    _ = try bm.add(50);
    _ = try bm.add(200);
    try std.testing.expectEqual(@as(?u32, 50), bm.minimum());
    try std.testing.expectEqual(@as(?u32, 200), bm.maximum());

    // Add in different chunk
    _ = try bm.add(1000000);
    try std.testing.expectEqual(@as(?u32, 50), bm.minimum());
    try std.testing.expectEqual(@as(?u32, 1000000), bm.maximum());
}

test "boundary values" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(0);
    _ = try bm.add(0xFFFFFFFF);

    try std.testing.expect(bm.contains(0));
    try std.testing.expect(bm.contains(0xFFFFFFFF));
    try std.testing.expectEqual(@as(?u32, 0), bm.minimum());
    try std.testing.expectEqual(@as(?u32, 0xFFFFFFFF), bm.maximum());
}

test "many values triggers growth" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Add values in 10 different chunks
    for (0..10) |i| {
        const chunk_base: u32 = @intCast(i * 65536);
        _ = try bm.add(chunk_base + 1);
    }

    try std.testing.expectEqual(@as(u32, 10), bm.size);
    try std.testing.expect(bm.capacity >= 10);
}

// ============================================================================
// Set Operation Tests
// ============================================================================

test "bitwiseOr" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(3);
    _ = try b.add(4);
    _ = try b.add(5);

    var result = try a.bitwiseOr(allocator, &b);
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 5), result.cardinality());
    try std.testing.expect(result.contains(1));
    try std.testing.expect(result.contains(2));
    try std.testing.expect(result.contains(3));
    try std.testing.expect(result.contains(4));
    try std.testing.expect(result.contains(5));
}

test "bitwiseAnd" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(2);
    _ = try b.add(3);
    _ = try b.add(4);

    var result = try a.bitwiseAnd(allocator, &b);
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 2), result.cardinality());
    try std.testing.expect(result.contains(2));
    try std.testing.expect(result.contains(3));
    try std.testing.expect(!result.contains(1));
    try std.testing.expect(!result.contains(4));
}

test "bitwiseDifference" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(2);
    _ = try b.add(3);
    _ = try b.add(4);

    var result = try a.bitwiseDifference(allocator, &b);
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 1), result.cardinality());
    try std.testing.expect(result.contains(1));
    try std.testing.expect(!result.contains(2));
    try std.testing.expect(!result.contains(3));
}

test "bitwiseXor" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(2);
    _ = try b.add(3);
    _ = try b.add(4);

    var result = try a.bitwiseXor(allocator, &b);
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 2), result.cardinality());
    try std.testing.expect(result.contains(1));
    try std.testing.expect(result.contains(4));
    try std.testing.expect(!result.contains(2));
    try std.testing.expect(!result.contains(3));
}

test "set operations across chunks" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1); // chunk 0
    _ = try a.add(65537); // chunk 1
    _ = try a.add(131073); // chunk 2

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(65537); // chunk 1 (overlap)
    _ = try b.add(196609); // chunk 3

    var union_result = try a.bitwiseOr(allocator, &b);
    defer union_result.deinit();
    try std.testing.expectEqual(@as(u64, 4), union_result.cardinality());

    var intersect_result = try a.bitwiseAnd(allocator, &b);
    defer intersect_result.deinit();
    try std.testing.expectEqual(@as(u64, 1), intersect_result.cardinality());
    try std.testing.expect(intersect_result.contains(65537));
}

test "isSubsetOf" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(1);
    _ = try b.add(2);
    _ = try b.add(3);

    try std.testing.expect(a.isSubsetOf(&b));
    try std.testing.expect(!b.isSubsetOf(&a));

    // Self is subset of self
    try std.testing.expect(a.isSubsetOf(&a));
}

test "equals" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(1);
    _ = try b.add(2);
    _ = try b.add(3);

    var c = try RoaringBitmap.init(allocator);
    defer c.deinit();
    _ = try c.add(1);
    _ = try c.add(2);

    try std.testing.expect(a.equals(&b));
    try std.testing.expect(!a.equals(&c));
}

test "cardinality identity: |A ∪ B| + |A ∩ B| = |A| + |B|" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    for (0..100) |i| {
        _ = try a.add(@intCast(i * 3));
    }

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    for (0..100) |i| {
        _ = try b.add(@intCast(i * 5));
    }

    var union_ab = try a.bitwiseOr(allocator, &b);
    defer union_ab.deinit();

    var intersect_ab = try a.bitwiseAnd(allocator, &b);
    defer intersect_ab.deinit();

    const lhs = union_ab.cardinality() + intersect_ab.cardinality();
    const rhs = a.cardinality() + b.cardinality();

    try std.testing.expectEqual(lhs, rhs);
}

test "A − A = ∅" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(100);
    _ = try a.add(10000);

    var diff = try a.bitwiseDifference(allocator, &a);
    defer diff.deinit();

    try std.testing.expect(diff.isEmpty());
}

test "A ∪ A = A" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);

    var union_aa = try a.bitwiseOr(allocator, &a);
    defer union_aa.deinit();

    try std.testing.expect(union_aa.equals(&a));
}

// ============================================================================
// Iterator Tests
// ============================================================================

test "iterator empty bitmap" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    var it = bm.iterator();
    try std.testing.expectEqual(@as(?u32, null), it.next());
}

test "iterator single container (array)" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(5);
    _ = try bm.add(10);
    _ = try bm.add(15);

    var it = bm.iterator();
    try std.testing.expectEqual(@as(?u32, 5), it.next());
    try std.testing.expectEqual(@as(?u32, 10), it.next());
    try std.testing.expectEqual(@as(?u32, 15), it.next());
    try std.testing.expectEqual(@as(?u32, null), it.next());
}

test "iterator multiple containers" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Values in different chunks
    _ = try bm.add(100); // chunk 0
    _ = try bm.add(65536 + 200); // chunk 1
    _ = try bm.add(131072 + 300); // chunk 2

    var it = bm.iterator();
    try std.testing.expectEqual(@as(?u32, 100), it.next());
    try std.testing.expectEqual(@as(?u32, 65536 + 200), it.next());
    try std.testing.expectEqual(@as(?u32, 131072 + 300), it.next());
    try std.testing.expectEqual(@as(?u32, null), it.next());
}

test "iterator collects all values" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Add various values
    const values = [_]u32{ 0, 1, 100, 1000, 65535, 65536, 100000, 0xFFFFFFFF };
    for (values) |v| {
        _ = try bm.add(v);
    }

    // Collect via iterator
    var collected: [8]u32 = undefined;
    var count: usize = 0;
    var it = bm.iterator();
    while (it.next()) |v| {
        collected[count] = v;
        count += 1;
    }

    try std.testing.expectEqual(@as(usize, 8), count);
    // Values should be in sorted order
    try std.testing.expectEqual(@as(u32, 0), collected[0]);
    try std.testing.expectEqual(@as(u32, 1), collected[1]);
    try std.testing.expectEqual(@as(u32, 100), collected[2]);
    try std.testing.expectEqual(@as(u32, 1000), collected[3]);
    try std.testing.expectEqual(@as(u32, 65535), collected[4]);
    try std.testing.expectEqual(@as(u32, 65536), collected[5]);
    try std.testing.expectEqual(@as(u32, 100000), collected[6]);
    try std.testing.expectEqual(@as(u32, 0xFFFFFFFF), collected[7]);
}

// ============================================================================
// In-Place Operation Tests
// ============================================================================

test "bitwiseOrInPlace" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(3);
    _ = try b.add(4);
    _ = try b.add(5);

    try a.bitwiseOrInPlace(&b);

    try std.testing.expectEqual(@as(u64, 5), a.cardinality());
    try std.testing.expect(a.contains(1));
    try std.testing.expect(a.contains(2));
    try std.testing.expect(a.contains(3));
    try std.testing.expect(a.contains(4));
    try std.testing.expect(a.contains(5));
}

test "bitwiseOrInPlace with new chunk" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(100); // chunk 0

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(65536 + 100); // chunk 1

    try a.bitwiseOrInPlace(&b);

    try std.testing.expectEqual(@as(u64, 2), a.cardinality());
    try std.testing.expect(a.contains(100));
    try std.testing.expect(a.contains(65536 + 100));
    try std.testing.expectEqual(@as(u32, 2), a.size); // two containers
}

test "bitwiseAndInPlace" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);
    _ = try a.add(4);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(2);
    _ = try b.add(3);
    _ = try b.add(5);

    try a.bitwiseAndInPlace(&b);

    try std.testing.expectEqual(@as(u64, 2), a.cardinality());
    try std.testing.expect(a.contains(2));
    try std.testing.expect(a.contains(3));
    try std.testing.expect(!a.contains(1));
    try std.testing.expect(!a.contains(4));
}

test "bitwiseAndInPlace with empty other" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    try a.bitwiseAndInPlace(&b);

    try std.testing.expect(a.isEmpty());
}

test "bitwiseDifferenceInPlace" {
    const allocator = std.testing.allocator;

    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);
    _ = try a.add(4);

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();
    _ = try b.add(2);
    _ = try b.add(4);

    try a.bitwiseDifferenceInPlace(&b);

    try std.testing.expectEqual(@as(u64, 2), a.cardinality());
    try std.testing.expect(a.contains(1));
    try std.testing.expect(a.contains(3));
    try std.testing.expect(!a.contains(2));
    try std.testing.expect(!a.contains(4));
}

test "in-place operations match non-in-place" {
    const allocator = std.testing.allocator;

    // Create two bitmaps
    var a1 = try RoaringBitmap.init(allocator);
    defer a1.deinit();
    var a2 = try RoaringBitmap.init(allocator);
    defer a2.deinit();

    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    const vals_a = [_]u32{ 1, 2, 3, 65536, 65537 };
    const vals_b = [_]u32{ 2, 3, 4, 65537, 131072 };

    for (vals_a) |v| {
        _ = try a1.add(v);
        _ = try a2.add(v);
    }
    for (vals_b) |v| {
        _ = try b.add(v);
    }

    // Compare OR
    var or_result = try a1.bitwiseOr(allocator, &b);
    defer or_result.deinit();
    try a2.bitwiseOrInPlace(&b);
    try std.testing.expect(a2.equals(&or_result));
}

// ============================================================================
// Serialization Tests
// ============================================================================

test "serialize and deserialize empty bitmap" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    const bytes = try bm.serialize(allocator);
    defer allocator.free(bytes);

    var restored = try RoaringBitmap.deserialize(allocator, bytes);
    defer restored.deinit();

    try std.testing.expect(restored.isEmpty());
    try std.testing.expect(bm.equals(&restored));
}

test "serialize and deserialize array container" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(1);
    _ = try bm.add(100);
    _ = try bm.add(1000);

    const bytes = try bm.serialize(allocator);
    defer allocator.free(bytes);

    var restored = try RoaringBitmap.deserialize(allocator, bytes);
    defer restored.deinit();

    try std.testing.expectEqual(bm.cardinality(), restored.cardinality());
    try std.testing.expect(restored.contains(1));
    try std.testing.expect(restored.contains(100));
    try std.testing.expect(restored.contains(1000));
    try std.testing.expect(bm.equals(&restored));
}

test "serialize and deserialize multiple containers" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Values in different chunks
    _ = try bm.add(100); // chunk 0
    _ = try bm.add(65536 + 200); // chunk 1
    _ = try bm.add(131072 + 300); // chunk 2

    const bytes = try bm.serialize(allocator);
    defer allocator.free(bytes);

    var restored = try RoaringBitmap.deserialize(allocator, bytes);
    defer restored.deinit();

    try std.testing.expectEqual(@as(u32, 3), restored.size);
    try std.testing.expect(restored.contains(100));
    try std.testing.expect(restored.contains(65536 + 200));
    try std.testing.expect(restored.contains(131072 + 300));
    try std.testing.expect(bm.equals(&restored));
}

test "serialize round-trip preserves all values" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Add various values across chunks
    const values = [_]u32{ 0, 1, 100, 1000, 65535, 65536, 100000, 0xFFFFFFFF };
    for (values) |v| {
        _ = try bm.add(v);
    }

    const bytes = try bm.serialize(allocator);
    defer allocator.free(bytes);

    var restored = try RoaringBitmap.deserialize(allocator, bytes);
    defer restored.deinit();

    try std.testing.expectEqual(bm.cardinality(), restored.cardinality());

    // Verify all values via iterator
    var it1 = bm.iterator();
    var it2 = restored.iterator();
    while (it1.next()) |v1| {
        const v2 = it2.next();
        try std.testing.expectEqual(v1, v2.?);
    }
    try std.testing.expectEqual(@as(?u32, null), it2.next());
}

// ============================================================================
// addRange and fromSorted Tests
// ============================================================================

test "addRange single chunk" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    const added = try bm.addRange(10, 20);
    try std.testing.expectEqual(@as(u64, 11), added);
    try std.testing.expectEqual(@as(u64, 11), bm.cardinality());

    for (10..21) |i| {
        try std.testing.expect(bm.contains(@intCast(i)));
    }
    try std.testing.expect(!bm.contains(9));
    try std.testing.expect(!bm.contains(21));
}

test "addRange spanning chunks" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Range spanning chunk boundary
    const added = try bm.addRange(65530, 65545);
    try std.testing.expectEqual(@as(u64, 16), added);

    try std.testing.expect(bm.contains(65530));
    try std.testing.expect(bm.contains(65535)); // last of chunk 0
    try std.testing.expect(bm.contains(65536)); // first of chunk 1
    try std.testing.expect(bm.contains(65545));
    try std.testing.expectEqual(@as(u32, 2), bm.size); // two containers
}

test "addRange large range creates bitset" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Large range > 4096 should create bitset
    const added = try bm.addRange(0, 5000);
    try std.testing.expectEqual(@as(u64, 5001), added);
    try std.testing.expectEqual(@as(u64, 5001), bm.cardinality());
}

test "addRange to existing container" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(5);
    _ = try bm.add(15);

    const added = try bm.addRange(10, 20);
    try std.testing.expectEqual(@as(u64, 10), added); // 15 was already there
    try std.testing.expectEqual(@as(u64, 12), bm.cardinality()); // 5, 10-20
}

test "fromSorted empty" {
    const allocator = std.testing.allocator;
    const empty: []const u32 = &.{};

    var bm = try RoaringBitmap.fromSorted(allocator, empty);
    defer bm.deinit();

    try std.testing.expect(bm.isEmpty());
}

test "fromSorted single chunk" {
    const allocator = std.testing.allocator;
    const values = [_]u32{ 1, 5, 10, 100, 1000 };

    var bm = try RoaringBitmap.fromSorted(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 5), bm.cardinality());
    try std.testing.expectEqual(@as(u32, 1), bm.size); // one container

    for (values) |v| {
        try std.testing.expect(bm.contains(v));
    }
}

test "fromSorted multiple chunks" {
    const allocator = std.testing.allocator;
    const values = [_]u32{ 100, 200, 65536 + 50, 65536 + 100, 131072 + 1 };

    var bm = try RoaringBitmap.fromSorted(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 5), bm.cardinality());
    try std.testing.expectEqual(@as(u32, 3), bm.size); // three containers

    for (values) |v| {
        try std.testing.expect(bm.contains(v));
    }
}

test "fromSorted matches individual adds" {
    const allocator = std.testing.allocator;
    const values = [_]u32{ 0, 1, 100, 1000, 65535, 65536, 100000 };

    // Build with fromSorted
    var bm1 = try RoaringBitmap.fromSorted(allocator, &values);
    defer bm1.deinit();

    // Build with individual adds
    var bm2 = try RoaringBitmap.init(allocator);
    defer bm2.deinit();
    for (values) |v| {
        _ = try bm2.add(v);
    }

    try std.testing.expect(bm1.equals(&bm2));
}
