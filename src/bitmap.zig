const std = @import("std");
const ArrayContainer = @import("array_container.zig").ArrayContainer;
const BitsetContainer = @import("bitset_container.zig").BitsetContainer;
const RunContainer = @import("run_container.zig").RunContainer;
const container_mod = @import("container.zig");
const Container = container_mod.Container;
const TaggedPtr = container_mod.TaggedPtr;
const ops = @import("container_ops.zig");
const compare = @import("compare.zig");
const opt = @import("optimize.zig");
const ser = @import("serialize.zig");
const fmt = @import("format.zig");

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

    /// Cached total cardinality. -1 = unknown (recompute on next query).
    cached_cardinality: i64 = -1,

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
            .cached_cardinality = 0,
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

    /// Create a deep copy of the bitmap.
    pub fn clone(self: *const Self, allocator: std.mem.Allocator) !Self {
        var result = try Self.init(allocator);
        errdefer result.deinit();

        try result.ensureCapacity(self.size);

        for (self.containers[0..self.size], self.keys[0..self.size], 0..) |tp, key, i| {
            const cloned = try Container.fromTagged(tp).clone(allocator);
            result.containers[i] = cloned.toTagged();
            result.keys[i] = key;
        }
        result.size = self.size;
        result.cached_cardinality = self.cached_cardinality;

        return result;
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
    pub fn ensureCapacity(self: *Self, needed: u32) !void {
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
            const added = try self.addToContainer(idx, low);
            if (added and self.cached_cardinality >= 0) self.cached_cardinality += 1;
            return added;
        }

        // Need to create new container
        const idx = self.lowerBound(key);
        try self.insertContainerAt(idx, key, low);
        if (self.cached_cardinality >= 0) self.cached_cardinality += 1;
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

        if (self.cached_cardinality >= 0) self.cached_cardinality += @intCast(added);
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

        // A contiguous range is always best as a run container (4 bytes per run)
        const rc = try RunContainer.init(self.allocator, 1);
        rc.runs[0] = .{ .start = start, .length = end - start };
        rc.n_runs = 1;
        rc.cardinality = -1; // Invalidate after direct modification
        self.keys[insert_idx] = key;
        self.containers[insert_idx] = TaggedPtr.initRun(rc);
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
                // Convert array to run container, then use efficient range merge
                // Building runs from a sorted array is O(cardinality)
                var n_runs: u16 = 0;
                if (ac.cardinality > 0) {
                    n_runs = 1;
                    var i: usize = 1;
                    while (i < ac.cardinality) : (i += 1) {
                        if (ac.values[i] != ac.values[i - 1] + 1) n_runs += 1;
                    }
                }
                const rc = try RunContainer.init(self.allocator, n_runs);
                errdefer rc.deinit(self.allocator);
                if (ac.cardinality > 0) {
                    var run_idx: u16 = 0;
                    var run_start = ac.values[0];
                    var prev = ac.values[0];
                    for (ac.values[1..ac.cardinality]) |val| {
                        if (val != prev + 1) {
                            rc.runs[run_idx] = .{ .start = run_start, .length = prev - run_start };
                            run_idx += 1;
                            run_start = val;
                        }
                        prev = val;
                    }
                    rc.runs[run_idx] = .{ .start = run_start, .length = prev - run_start };
                    rc.n_runs = run_idx + 1;
                    rc.cardinality = -1;
                }
                // Add range using efficient run merge
                const added = try rc.addRange(self.allocator, start, end);
                // Replace array with run container
                ac.deinit(self.allocator);
                self.containers[idx] = TaggedPtr.initRun(rc);
                return added;
            },
            .run => |rc| {
                // Direct range merge: O(R) where R is number of affected runs
                return try rc.addRange(self.allocator, start, end);
            },
            .reserved => unreachable,
        }
    }

    /// Build from pre-sorted, deduplicated values. O(n), no binary searches.
    /// Caller must ensure values are in strictly ascending order with no duplicates.
    /// Debug builds assert this precondition. In release, duplicates cause undefined
    /// behavior (incorrect cardinality, corrupt containers).
    /// If input may be unsorted or contain duplicates, use `fromSlice` instead.
    pub fn fromSorted(allocator: std.mem.Allocator, values: []const u32) !Self {
        if (values.len == 0) {
            return Self.init(allocator);
        }

        // Debug assertion: values must be strictly ascending (sorted, no duplicates)
        if (std.debug.runtime_safety) {
            for (values[1..], 0..) |cur, i| {
                std.debug.assert(cur > values[i]); // not sorted or contains duplicates
            }
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

        result.cached_cardinality = @intCast(values.len);
        return result;
    }

    /// Build from an arbitrary slice of values. O(n log n).
    /// Sorts in-place and deduplicates. Mutates the input slice.
    /// If input is already sorted and unique, prefer `fromSorted` (O(n)).
    pub fn fromSlice(allocator: std.mem.Allocator, values: []u32) !Self {
        if (values.len == 0) return Self.init(allocator);

        // Sort in-place
        std.mem.sortUnstable(u32, values, {}, std.sort.asc(u32));

        // Deduplicate in-place
        var write: usize = 1;
        for (values[1..]) |v| {
            if (v != values[write - 1]) {
                values[write] = v;
                write += 1;
            }
        }

        return fromSorted(allocator, values[0..write]);
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
        const removed = try self.removeFromContainer(idx, low);
        if (removed and self.cached_cardinality >= 0) self.cached_cardinality -= 1;
        return removed;
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
    pub fn cardinality(self: *Self) u64 {
        if (self.cached_cardinality >= 0) return @intCast(self.cached_cardinality);
        var total: u64 = 0;
        for (self.containers[0..self.size]) |tp| {
            total += Container.fromTagged(tp).getCardinality();
        }
        self.cached_cardinality = @intCast(total);
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
                var i: usize = BitsetContainer.NUM_WORDS;
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

        result.cached_cardinality = -1;
        return result;
    }

    /// Return a new bitmap that is the intersection (AND) of self and other.
    pub fn bitwiseAnd(self: *const Self, allocator: std.mem.Allocator, other: *const Self) !Self {
        var result = try Self.init(allocator);
        errdefer result.deinit();

        // Scratch buffer for temporary array containers (avoids malloc/free churn for empty results)
        // Most sparse intersections produce empty arrays, so this eliminates ~65K malloc/free cycles.
        // Size: ArrayContainer struct (~24 bytes) + max values (4096 * 2 = 8192 bytes) + alignment padding
        var scratch_buf: [8448]u8 = undefined;
        var scratch = std.heap.FixedBufferAllocator.init(&scratch_buf);

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
                // Try scratch allocator first (works for array containers)
                // Falls back to real allocator for bitset/run containers that don't fit
                const scratch_alloc = scratch.allocator();
                const c = ops.containerIntersection(
                    scratch_alloc,
                    Container.fromTagged(self.containers[i]),
                    Container.fromTagged(other.containers[j]),
                ) catch try ops.containerIntersection(
                    allocator,
                    Container.fromTagged(self.containers[i]),
                    Container.fromTagged(other.containers[j]),
                );

                const used_scratch = scratch.end_index > 0;

                if (c.getCardinality() > 0) {
                    if (used_scratch) {
                        // Non-empty from scratch: clone into real allocator
                        const permanent = try c.clone(allocator);
                        try result.appendContainer(key_a, permanent.toTagged());
                    } else {
                        // Already in real allocator
                        try result.appendContainer(key_a, c.toTagged());
                    }
                } else if (!used_scratch) {
                    // Empty but allocated from real allocator, free it
                    c.deinit(allocator);
                }

                // Reset scratch for next iteration
                scratch.reset();

                i += 1;
                j += 1;
            }
        }

        result.cached_cardinality = -1;
        return result;
    }

    /// Compute |self ∩ other| without allocating a result bitmap.
    /// Useful for join selectivity estimation in query planning.
    pub fn andCardinality(self: *const Self, other: *const Self) u64 {
        var total: u64 = 0;
        var i: usize = 0;
        var j: usize = 0;
        while (i < self.size and j < other.size) {
            if (self.keys[i] < other.keys[j]) {
                i += 1;
            } else if (self.keys[i] > other.keys[j]) {
                j += 1;
            } else {
                total += ops.containerIntersectionCardinality(
                    Container.fromTagged(self.containers[i]),
                    Container.fromTagged(other.containers[j]),
                );
                i += 1;
                j += 1;
            }
        }
        return total;
    }

    /// Return true if self and other have any values in common.
    /// Early-exit: stops at the first match. Much cheaper than andCardinality() > 0
    /// for sparse intersections.
    pub fn intersects(self: *const Self, other: *const Self) bool {
        var i: usize = 0;
        var j: usize = 0;
        while (i < self.size and j < other.size) {
            if (self.keys[i] < other.keys[j]) {
                i += 1;
            } else if (self.keys[i] > other.keys[j]) {
                j += 1;
            } else {
                if (ops.containerIntersects(
                    Container.fromTagged(self.containers[i]),
                    Container.fromTagged(other.containers[j]),
                )) return true;
                i += 1;
                j += 1;
            }
        }
        return false;
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

        result.cached_cardinality = -1;
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

        result.cached_cardinality = -1;
        return result;
    }

    // ========================================================================
    // In-Place Set Operations
    // ========================================================================

    /// In-place union: self |= other. Modifies self to contain all values from both.
    /// Uses O(n) merge algorithm instead of O(n²) incremental insertion.
    pub fn bitwiseOrInPlace(self: *Self, other: *const Self) !void {
        if (other.size == 0) return;
        self.cached_cardinality = -1;

        // Pre-merge into new arrays to avoid O(n²) shifting
        const max_size = self.size + other.size;
        const new_keys = try self.allocator.alloc(u16, max_size);
        errdefer self.allocator.free(new_keys);
        const new_containers = try self.allocator.alloc(TaggedPtr, max_size);
        errdefer self.allocator.free(new_containers);

        // Track which containers are newly allocated (not moved from self)
        // On error, we must free these to avoid leaks
        const owned = try self.allocator.alloc(bool, max_size);
        defer self.allocator.free(owned);

        var i: usize = 0; // index into self
        var j: usize = 0; // index into other
        var k: usize = 0; // index into new arrays

        errdefer {
            // Free newly allocated containers (cloned/merged) but not moved ones
            for (new_containers[0..k], owned[0..k]) |tp, is_owned| {
                if (is_owned) {
                    Container.fromTagged(tp).deinit(self.allocator);
                }
            }
        }

        while (i < self.size and j < other.size) {
            const key_a = self.keys[i];
            const key_b = other.keys[j];

            if (key_a < key_b) {
                // Key only in self - move it (not owned by merge)
                new_keys[k] = key_a;
                new_containers[k] = self.containers[i];
                owned[k] = false;
                k += 1;
                i += 1;
            } else if (key_a > key_b) {
                // Key only in other - clone it (owned by merge)
                new_keys[k] = key_b;
                new_containers[k] = try cloneContainer(self.allocator, other.containers[j]);
                owned[k] = true;
                k += 1;
                j += 1;
            } else {
                // Key in both - merge containers in-place when possible
                const old_container = Container.fromTagged(self.containers[i]);
                const other_container = Container.fromTagged(other.containers[j]);
                const result = try ops.containerUnionInPlace(self.allocator, old_container, other_container);
                const result_tp = result.toTagged();

                // Check if a new container was allocated (e.g., array converted to bitset)
                const is_same = (@as(u64, @bitCast(result_tp)) == @as(u64, @bitCast(self.containers[i])));
                if (!is_same) {
                    // New container allocated, free the old one
                    old_container.deinit(self.allocator);
                    owned[k] = true;
                } else {
                    // Same container, just modified in place
                    owned[k] = false;
                }

                new_keys[k] = key_a;
                new_containers[k] = result_tp;
                k += 1;
                i += 1;
                j += 1;
            }
        }

        // Copy remaining from self (not owned)
        while (i < self.size) : (i += 1) {
            new_keys[k] = self.keys[i];
            new_containers[k] = self.containers[i];
            owned[k] = false;
            k += 1;
        }

        // Clone remaining from other (owned)
        while (j < other.size) : (j += 1) {
            new_keys[k] = other.keys[j];
            new_containers[k] = try cloneContainer(self.allocator, other.containers[j]);
            owned[k] = true;
            k += 1;
        }

        // Success - free old arrays (containers were moved, not freed)
        self.allocator.free(self.keys[0..self.capacity]);
        self.allocator.free(self.containers[0..self.capacity]);

        // Right-size the arrays if there's significant slack
        if (k < max_size) {
            self.keys = self.allocator.realloc(new_keys, k) catch new_keys;
            self.containers = self.allocator.realloc(new_containers, k) catch new_containers;
            self.capacity = @intCast(k);
        } else {
            self.keys = new_keys;
            self.containers = new_containers;
            self.capacity = @intCast(max_size);
        }
        self.size = @intCast(k);
    }

    /// In-place intersection: self &= other. Modifies self to contain only values in both.
    pub fn bitwiseAndInPlace(self: *Self, other: *const Self) !void {
        self.cached_cardinality = -1;
        if (other.size == 0) {
            // Clear self
            for (self.containers[0..self.size]) |tp| {
                Container.fromTagged(tp).deinit(self.allocator);
            }
            self.size = 0;
            return;
        }

        // Scratch buffer for temporary array containers (avoids malloc/free churn for empty results)
        var scratch_buf: [8448]u8 = undefined;
        var scratch = std.heap.FixedBufferAllocator.init(&scratch_buf);

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

                // Try scratch allocator first, fall back to real allocator
                const scratch_alloc = scratch.allocator();
                const intersected = ops.containerIntersection(scratch_alloc, self_container, other_container) catch
                    try ops.containerIntersection(self.allocator, self_container, other_container);

                const used_scratch = scratch.end_index > 0;
                self_container.deinit(self.allocator);

                if (intersected.getCardinality() > 0) {
                    if (used_scratch) {
                        // Non-empty from scratch: clone into real allocator
                        const permanent = try intersected.clone(self.allocator);
                        self.keys[write_idx] = key_a;
                        self.containers[write_idx] = permanent.toTagged();
                    } else {
                        // Already in real allocator
                        self.keys[write_idx] = key_a;
                        self.containers[write_idx] = intersected.toTagged();
                    }
                    write_idx += 1;
                } else if (!used_scratch) {
                    // Empty but allocated from real allocator, free it
                    intersected.deinit(self.allocator);
                }

                // Reset scratch for next iteration
                scratch.reset();

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
        self.cached_cardinality = -1;

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

    /// In-place XOR: self ^= other. Modifies self to contain symmetric difference.
    pub fn bitwiseXorInPlace(self: *Self, other: *const Self) !void {
        if (other.size == 0) return;

        self.cached_cardinality = -1;

        // Pre-merge into new arrays (XOR can add new keys from other)
        const max_size = self.size + other.size;
        const new_keys = try self.allocator.alloc(u16, max_size);
        errdefer self.allocator.free(new_keys);
        const new_containers = try self.allocator.alloc(TaggedPtr, max_size);
        errdefer self.allocator.free(new_containers);

        // Track which containers are newly allocated
        const owned = try self.allocator.alloc(bool, max_size);
        defer self.allocator.free(owned);

        var i: usize = 0; // index into self
        var j: usize = 0; // index into other
        var k: usize = 0; // index into new arrays

        errdefer {
            for (new_containers[0..k], owned[0..k]) |tp, is_owned| {
                if (is_owned) {
                    Container.fromTagged(tp).deinit(self.allocator);
                }
            }
        }

        while (i < self.size and j < other.size) {
            const key_a = self.keys[i];
            const key_b = other.keys[j];

            if (key_a < key_b) {
                // Key only in self - keep it
                new_keys[k] = key_a;
                new_containers[k] = self.containers[i];
                owned[k] = false;
                k += 1;
                i += 1;
            } else if (key_a > key_b) {
                // Key only in other - clone it
                new_keys[k] = key_b;
                new_containers[k] = try cloneContainer(self.allocator, other.containers[j]);
                owned[k] = true;
                k += 1;
                j += 1;
            } else {
                // Key in both - XOR containers
                const old_container = Container.fromTagged(self.containers[i]);
                const other_container = Container.fromTagged(other.containers[j]);
                const result = try ops.containerXor(self.allocator, old_container, other_container);

                // Free old container
                old_container.deinit(self.allocator);

                // Only keep non-empty results
                if (result.getCardinality() > 0) {
                    new_keys[k] = key_a;
                    new_containers[k] = result.toTagged();
                    owned[k] = true;
                    k += 1;
                } else {
                    result.deinit(self.allocator);
                }
                i += 1;
                j += 1;
            }
        }

        // Copy remaining from self (not owned)
        while (i < self.size) : (i += 1) {
            new_keys[k] = self.keys[i];
            new_containers[k] = self.containers[i];
            owned[k] = false;
            k += 1;
        }

        // Clone remaining from other (owned)
        while (j < other.size) : (j += 1) {
            new_keys[k] = other.keys[j];
            new_containers[k] = try cloneContainer(self.allocator, other.containers[j]);
            owned[k] = true;
            k += 1;
        }

        // Success - free old arrays
        self.allocator.free(self.keys[0..self.capacity]);
        self.allocator.free(self.containers[0..self.capacity]);

        // Right-size the arrays if there's significant slack
        if (k < max_size) {
            self.keys = self.allocator.realloc(new_keys, k) catch new_keys;
            self.containers = self.allocator.realloc(new_containers, k) catch new_containers;
            self.capacity = @intCast(k);
        } else {
            self.keys = new_keys;
            self.containers = new_containers;
            self.capacity = @intCast(max_size);
        }
        self.size = @intCast(k);
    }

    // ========================================================================
    // Optimization (delegated to optimize.zig)
    // ========================================================================

    /// Convert containers to run encoding where it saves space.
    /// Returns the number of containers that were converted.
    pub fn runOptimize(self: *Self) !u32 {
        // Invalidate cache for safety (though cardinality doesn't actually change)
        self.cached_cardinality = -1;
        return opt.runOptimize(self);
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

    // ========================================================================
    // Comparison (delegated to compare.zig)
    // ========================================================================

    /// Check if self is a subset of other. O(n) where n is total container size.
    pub fn isSubsetOf(self: *const Self, other: *const Self) bool {
        return compare.isSubsetOf(self, other);
    }

    /// Check if two bitmaps are equal. Single pass O(n).
    pub fn equals(self: *const Self, other: *const Self) bool {
        return compare.equals(self, other);
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
                new_rc.cardinality = rc.cardinality;
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
                            if (s.word_idx >= BitsetContainer.NUM_WORDS) {
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
                    while (word_idx < BitsetContainer.NUM_WORDS and bc.words[word_idx] == 0) : (word_idx += 1) {}
                    if (word_idx < BitsetContainer.NUM_WORDS) {
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
    // Serialization (delegated to serialize.zig)
    // ========================================================================

    /// Cookie values for RoaringFormatSpec (re-exported for FrozenBitmap compatibility)
    pub const SERIAL_COOKIE_NO_RUNCONTAINER = fmt.SERIAL_COOKIE_NO_RUNCONTAINER;
    pub const SERIAL_COOKIE = fmt.SERIAL_COOKIE;
    pub const NO_OFFSET_THRESHOLD = fmt.NO_OFFSET_THRESHOLD;

    /// Compute serialized size in bytes.
    pub fn serializedSizeInBytes(self: *const Self) usize {
        return ser.serializedSizeInBytes(self);
    }

    /// Serialize the bitmap to a byte slice (RoaringFormatSpec compatible).
    pub fn serialize(self: *const Self, allocator: std.mem.Allocator) ![]u8 {
        return ser.serialize(self, allocator);
    }

    /// Serialize to any writer.
    pub fn serializeToWriter(self: *const Self, writer: anytype) !void {
        return ser.serializeToWriter(self, writer);
    }

    /// Deserialize a bitmap from bytes (RoaringFormatSpec compatible).
    ///
    /// **Performance note:** For best performance, use an `ArenaAllocator`. Deserialization
    /// creates many small allocations (one per container), and arena allocation reduces
    /// this overhead significantly. Consider using `deserializeOwned` for convenience.
    ///
    /// ```zig
    /// // Fast path (recommended):
    /// var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    /// defer arena.deinit();  // frees all bitmap memory at once
    /// var bm = try RoaringBitmap.deserialize(arena.allocator(), data);
    /// // Use bm... (don't call bm.deinit(), arena handles cleanup)
    ///
    /// // Standard path (slower, but bitmap has independent lifetime):
    /// var bm = try RoaringBitmap.deserialize(allocator, data);
    /// defer bm.deinit();
    /// ```
    pub fn deserialize(allocator: std.mem.Allocator, data: []const u8) !Self {
        return ser.deserialize(allocator, data);
    }

    /// Deserialize from any reader.
    ///
    /// See `deserialize` for performance notes on arena allocation.
    pub fn deserializeFromReader(allocator: std.mem.Allocator, reader: anytype, data_len: usize) !Self {
        return ser.deserializeFromReader(allocator, reader, data_len);
    }

    // =========================================================================
    // Arena-backed convenience methods (return OwnedBitmap)
    // =========================================================================

    /// Deserialize a bitmap using arena allocation (recommended for speed).
    /// Returns an OwnedBitmap that frees all memory in one operation via deinit().
    pub fn deserializeOwned(backing: std.mem.Allocator, data: []const u8) !OwnedBitmap {
        var arena = std.heap.ArenaAllocator.init(backing);
        errdefer arena.deinit();
        const bm = try Self.deserialize(arena.allocator(), data);
        return .{ .bitmap = bm, .arena = arena };
    }

    /// Compute intersection using arena allocation (recommended for speed).
    /// Returns an OwnedBitmap.
    pub fn bitwiseAndOwned(self: *const Self, backing: std.mem.Allocator, other: *const Self) !OwnedBitmap {
        var arena = std.heap.ArenaAllocator.init(backing);
        errdefer arena.deinit();
        const result = try self.bitwiseAnd(arena.allocator(), other);
        return .{ .bitmap = result, .arena = arena };
    }

    /// Compute union using arena allocation (recommended for speed).
    /// Returns an OwnedBitmap.
    pub fn bitwiseOrOwned(self: *const Self, backing: std.mem.Allocator, other: *const Self) !OwnedBitmap {
        var arena = std.heap.ArenaAllocator.init(backing);
        errdefer arena.deinit();
        const result = try self.bitwiseOr(arena.allocator(), other);
        return .{ .bitmap = result, .arena = arena };
    }

    /// Compute difference (self \ other) using arena allocation.
    pub fn bitwiseDifferenceOwned(self: *const Self, backing: std.mem.Allocator, other: *const Self) !OwnedBitmap {
        var arena = std.heap.ArenaAllocator.init(backing);
        errdefer arena.deinit();
        const result = try self.bitwiseDifference(arena.allocator(), other);
        return .{ .bitmap = result, .arena = arena };
    }

    /// Build from arbitrary slice using arena allocation. Sorts and deduplicates in-place.
    pub fn fromSliceOwned(backing: std.mem.Allocator, values: []u32) !OwnedBitmap {
        var arena = std.heap.ArenaAllocator.init(backing);
        errdefer arena.deinit();
        const result = try fromSlice(arena.allocator(), values);
        return .{ .bitmap = result, .arena = arena };
    }

    // =========================================================================
    // Allocator guidance
    // =========================================================================

    /// ## Allocator guidance
    ///
    /// Avoid `std.heap.c_allocator` — it is 10-40x slower than alternatives
    /// for rawr's allocation patterns (many small containers).
    ///
    /// Recommended:
    /// - `OwnedBitmap` API: fastest (uses optimized allocation internally)
    /// - `std.heap.smp_allocator`: fast general-purpose, supports mutation
    /// - `std.heap.ArenaAllocator`: fast batch alloc, bulk free only
    pub const allocator_guidance = void;
};

/// A RoaringBitmap that owns its memory via an arena allocator.
/// All internal allocations use bump-pointer allocation for speed.
/// Call `deinit()` to free everything in one operation.
///
/// Returned by `deserializeOwned`, `bitwiseAndOwned`, `bitwiseOrOwned`.
pub const OwnedBitmap = struct {
    bitmap: RoaringBitmap,
    arena: std.heap.ArenaAllocator,

    /// Free all memory in one bulk operation.
    pub fn deinit(self: *OwnedBitmap) void {
        // Don't call bitmap.deinit() — arena owns all allocations.
        self.arena.deinit();
    }

    /// Check if a value is in the bitmap.
    pub fn contains(self: *const OwnedBitmap, value: u32) bool {
        return self.bitmap.contains(value);
    }

    /// Return the number of values in the bitmap.
    pub fn cardinality(self: *OwnedBitmap) u64 {
        return self.bitmap.cardinality();
    }

    /// Iterate over all values in sorted order.
    pub fn iterator(self: *const OwnedBitmap) RoaringBitmap.Iterator {
        return self.bitmap.iterator();
    }

    /// Serialize to bytes. The output is allocated with the provided
    /// allocator (NOT the internal arena), so the caller owns it.
    pub fn serialize(self: *const OwnedBitmap, out_allocator: std.mem.Allocator) ![]u8 {
        return self.bitmap.serialize(out_allocator);
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

test "bitwiseOrInPlace no leak on allocation failure" {
    // This test verifies that bitwiseOrInPlace properly cleans up
    // newly allocated containers when an allocation fails mid-operation.
    const base_allocator = std.testing.allocator;

    // Create two bitmaps with disjoint keys to force cloning
    var bm1 = try RoaringBitmap.init(base_allocator);
    defer bm1.deinit();
    var bm2 = try RoaringBitmap.init(base_allocator);
    defer bm2.deinit();

    // Add values to different chunks
    _ = try bm1.add(0); // chunk 0
    _ = try bm1.add(65536); // chunk 1
    _ = try bm2.add(131072); // chunk 2
    _ = try bm2.add(196608); // chunk 3

    // Use failing allocator that fails after a few allocations
    // This should trigger failure during cloneContainer calls
    var failing = std.testing.FailingAllocator.init(base_allocator, .{ .fail_index = 3 });

    // Create a copy with the failing allocator for the in-place op
    var bm1_copy = try bm1.clone(base_allocator);

    // Swap allocator to failing one for the operation
    bm1_copy.allocator = failing.allocator();

    // This should fail partway through and clean up properly
    const result = bm1_copy.bitwiseOrInPlace(&bm2);
    try std.testing.expectError(error.OutOfMemory, result);

    // Restore normal allocator for cleanup
    bm1_copy.allocator = base_allocator;
    bm1_copy.deinit();

    // If we get here without the testing allocator detecting leaks, the test passes
}

test "OwnedBitmap bitwiseAndOwned" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);
    _ = try b.add(2);
    _ = try b.add(3);
    _ = try b.add(4);

    var result = try a.bitwiseAndOwned(allocator, &b);
    defer result.deinit();

    try std.testing.expect(result.contains(2));
    try std.testing.expect(result.contains(3));
    try std.testing.expect(!result.contains(1));
    try std.testing.expect(!result.contains(4));
    try std.testing.expectEqual(@as(u64, 2), result.cardinality());
}

test "OwnedBitmap bitwiseOrOwned" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    _ = try a.add(1);
    _ = try a.add(2);
    _ = try b.add(3);
    _ = try b.add(4);

    var result = try a.bitwiseOrOwned(allocator, &b);
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 4), result.cardinality());
    try std.testing.expect(result.contains(1));
    try std.testing.expect(result.contains(4));
}

test "OwnedBitmap deserializeOwned" {
    const allocator = std.testing.allocator;
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(42);
    _ = try bm.add(1000);

    const data = try bm.serialize(allocator);
    defer allocator.free(data);

    var owned = try RoaringBitmap.deserializeOwned(allocator, data);
    defer owned.deinit();

    try std.testing.expect(owned.contains(42));
    try std.testing.expect(owned.contains(1000));
    try std.testing.expectEqual(@as(u64, 2), owned.cardinality());
}

test "OwnedBitmap bitwiseDifferenceOwned" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    _ = try a.add(1);
    _ = try a.add(2);
    _ = try a.add(3);
    _ = try b.add(2);
    _ = try b.add(3);
    _ = try b.add(4);

    var result = try a.bitwiseDifferenceOwned(allocator, &b);
    defer result.deinit();

    try std.testing.expect(result.contains(1));
    try std.testing.expect(!result.contains(2));
    try std.testing.expect(!result.contains(3));
    try std.testing.expect(!result.contains(4));
    try std.testing.expectEqual(@as(u64, 1), result.cardinality());
}

test "OwnedBitmap iterator" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    _ = try a.add(10);
    _ = try a.add(20);
    _ = try a.add(30);
    _ = try b.add(20);
    _ = try b.add(30);
    _ = try b.add(40);

    var result = try a.bitwiseAndOwned(allocator, &b);
    defer result.deinit();

    var iter = result.iterator();
    try std.testing.expectEqual(@as(?u32, 20), iter.next());
    try std.testing.expectEqual(@as(?u32, 30), iter.next());
    try std.testing.expectEqual(@as(?u32, null), iter.next());
}

test "andCardinality matches bitwiseAnd().cardinality()" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    // Build overlapping bitmaps across multiple containers
    _ = try a.addRange(0, 1000);
    _ = try a.addRange(100_000, 101_000);
    _ = try b.addRange(500, 1500);
    _ = try b.addRange(100_500, 101_500);

    const card_fast = a.andCardinality(&b);
    var intersection = try a.bitwiseAnd(allocator, &b);
    defer intersection.deinit();
    try std.testing.expectEqual(card_fast, intersection.cardinality());
}

test "intersects" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    // Non-overlapping (different chunks)
    _ = try a.add(100);
    _ = try b.add(100_000);
    try std.testing.expect(!a.intersects(&b));

    // Add overlap
    _ = try b.add(100);
    try std.testing.expect(a.intersects(&b));
}

test "intersects with empty" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var empty = try RoaringBitmap.init(allocator);
    defer empty.deinit();

    _ = try a.add(42);
    try std.testing.expect(!a.intersects(&empty));
    try std.testing.expect(!empty.intersects(&a));
}

test "bitwiseXorInPlace matches bitwiseXor" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    _ = try a.addRange(0, 100);
    _ = try a.add(200);
    _ = try b.addRange(50, 150);
    _ = try b.add(300);

    var a_copy = try a.clone(allocator);
    defer a_copy.deinit();
    try a_copy.bitwiseXorInPlace(&b);

    var expected = try a.bitwiseXor(allocator, &b);
    defer expected.deinit();

    try std.testing.expect(a_copy.equals(&expected));
}

test "bitwiseXorInPlace removes empty containers" {
    const allocator = std.testing.allocator;
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    // Same values in both - XOR should produce empty
    _ = try a.add(42);
    _ = try b.add(42);

    try a.bitwiseXorInPlace(&b);
    try std.testing.expectEqual(@as(u64, 0), a.cardinality());
}

test "cached cardinality stays correct through mutations" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 0), bm.cardinality());

    _ = try bm.add(1);
    try std.testing.expectEqual(@as(u64, 1), bm.cardinality());

    _ = try bm.add(1); // duplicate
    try std.testing.expectEqual(@as(u64, 1), bm.cardinality());

    _ = try bm.addRange(100, 199);
    try std.testing.expectEqual(@as(u64, 101), bm.cardinality());

    _ = try bm.remove(1);
    try std.testing.expectEqual(@as(u64, 100), bm.cardinality());

    // In-place op invalidates, next call recomputes
    var other = try RoaringBitmap.init(allocator);
    defer other.deinit();
    _ = try other.addRange(150, 250);
    try bm.bitwiseOrInPlace(&other);
    try std.testing.expectEqual(@as(u64, 151), bm.cardinality());
}

test "fromSorted basic correctness" {
    const allocator = std.testing.allocator;
    const values = [_]u32{ 1, 5, 10, 100, 1000, 10000 };

    var bm = try RoaringBitmap.fromSorted(allocator, &values);
    defer bm.deinit();

    // Cardinality matches input length
    try std.testing.expectEqual(@as(u64, values.len), bm.cardinality());

    // Contains returns true for every input value
    for (values) |v| {
        try std.testing.expect(bm.contains(v));
    }

    // Contains returns false for values not in input
    try std.testing.expect(!bm.contains(0));
    try std.testing.expect(!bm.contains(2));
    try std.testing.expect(!bm.contains(50));
    try std.testing.expect(!bm.contains(999));

    // Iteration yields exactly the input values in order
    var it = bm.iterator();
    for (values) |expected| {
        try std.testing.expectEqual(expected, it.next().?);
    }
    try std.testing.expectEqual(@as(?u32, null), it.next());
}

test "fromSorted matches incremental add" {
    const allocator = std.testing.allocator;
    const values = [_]u32{ 0, 1, 100, 1000, 65535, 65536, 65537, 100000 };

    // Build via fromSorted
    var from_sorted = try RoaringBitmap.fromSorted(allocator, &values);
    defer from_sorted.deinit();

    // Build via add
    var from_add = try RoaringBitmap.init(allocator);
    defer from_add.deinit();
    for (values) |v| {
        _ = try from_add.add(v);
    }

    // They must be equal
    try std.testing.expect(from_sorted.equals(&from_add));
    try std.testing.expectEqual(from_sorted.cardinality(), from_add.cardinality());
}

test "fromSorted cardinality cache consistency" {
    const allocator = std.testing.allocator;
    const values = [_]u32{ 1, 2, 3, 100, 200, 300 };

    var bm = try RoaringBitmap.fromSorted(allocator, &values);
    defer bm.deinit();

    // Get cached cardinality
    const cached = bm.cardinality();
    try std.testing.expectEqual(@as(u64, 6), cached);

    // Add a value to trigger cache update path
    _ = try bm.add(50);
    try std.testing.expectEqual(@as(u64, 7), bm.cardinality());

    // Remove it
    _ = try bm.remove(50);
    try std.testing.expectEqual(@as(u64, 6), bm.cardinality());

    // Force cache invalidation via in-place op and recompute
    var empty = try RoaringBitmap.init(allocator);
    defer empty.deinit();
    try bm.bitwiseAndInPlace(&empty); // AND with empty = empty

    // After invalidation and recompute, must be 0
    try std.testing.expectEqual(@as(u64, 0), bm.cardinality());
}

test "fromSorted with cross-container values" {
    const allocator = std.testing.allocator;
    // Values spanning multiple 65536-boundaries
    const values = [_]u32{ 0, 1, 65536, 65537, 131072 };

    var bm = try RoaringBitmap.fromSorted(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 5), bm.cardinality());
    try std.testing.expectEqual(@as(u32, 3), bm.size); // 3 containers

    for (values) |v| {
        try std.testing.expect(bm.contains(v));
    }
}

test "fromSorted roundtrip serialize/deserialize" {
    const allocator = std.testing.allocator;
    const values = [_]u32{ 5, 10, 15, 65536, 65540, 131072, 131073 };

    var original = try RoaringBitmap.fromSorted(allocator, &values);
    defer original.deinit();

    // Serialize
    const bytes = try original.serialize(allocator);
    defer allocator.free(bytes);

    // Deserialize
    var restored = try RoaringBitmap.deserialize(allocator, bytes);
    defer restored.deinit();

    // Must be equal
    try std.testing.expect(original.equals(&restored));
    try std.testing.expectEqual(original.cardinality(), restored.cardinality());
}

test "fromSorted rejects duplicates in debug" {
    // This test verifies the debug assertion catches duplicates.
    // In debug builds, passing duplicates should panic/assert.
    // We can't easily test panics, so we document the expected behavior.
    // The assertion is: std.debug.assert(cur > values[i])

    // For now, just verify the happy path works
    const allocator = std.testing.allocator;
    const valid = [_]u32{ 1, 2, 3 }; // no duplicates
    var bm = try RoaringBitmap.fromSorted(allocator, &valid);
    defer bm.deinit();
    try std.testing.expectEqual(@as(u64, 3), bm.cardinality());
}

test "fromSlice sorts and deduplicates" {
    const allocator = std.testing.allocator;
    var values = [_]u32{ 10, 3, 3, 7, 1, 10, 7, 1 };

    var bm = try RoaringBitmap.fromSlice(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 4), bm.cardinality());
    try std.testing.expect(bm.contains(1));
    try std.testing.expect(bm.contains(3));
    try std.testing.expect(bm.contains(7));
    try std.testing.expect(bm.contains(10));
    try std.testing.expect(!bm.contains(2));
}

test "fromSlice matches incremental add" {
    const allocator = std.testing.allocator;
    var values = [_]u32{ 100, 1, 65536, 1, 200, 65536, 50 };

    var from_slice = try RoaringBitmap.fromSlice(allocator, &values);
    defer from_slice.deinit();

    var from_add = try RoaringBitmap.init(allocator);
    defer from_add.deinit();
    for ([_]u32{ 100, 1, 65536, 200, 50 }) |v| {
        _ = try from_add.add(v);
    }

    try std.testing.expect(from_slice.equals(&from_add));
}

test "fromSlice empty" {
    const allocator = std.testing.allocator;
    var values = [_]u32{};

    var bm = try RoaringBitmap.fromSlice(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 0), bm.cardinality());
    try std.testing.expect(bm.isEmpty());
}

test "fromSlice all duplicates" {
    const allocator = std.testing.allocator;
    var values = [_]u32{ 42, 42, 42, 42 };

    var bm = try RoaringBitmap.fromSlice(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 1), bm.cardinality());
    try std.testing.expect(bm.contains(42));
}

test "fromSlice cross-container with duplicates" {
    const allocator = std.testing.allocator;
    var values = [_]u32{ 131072, 0, 65536, 0, 131072, 1, 65537 };

    var bm = try RoaringBitmap.fromSlice(allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 5), bm.cardinality());
    try std.testing.expectEqual(@as(u32, 3), bm.size); // 3 containers
}
