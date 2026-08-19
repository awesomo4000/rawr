// SPDX-License-Identifier: MPL-2.0

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
const array_kernels = @import("array_kernels.zig");
const fmt = @import("format.zig");
const range_ops = @import("range_ops.zig");

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

    pub const ValidateError = error{
        BitmapSizeRange,
        UnsortedKeys,
        DuplicateKeys,
        EmptyContainer,
        UnsortedArray,
        ArrayCardinalityRange,
        BitsetCardinalityMismatch,
        BitsetCardinalityRange,
        RunOrdering,
        RunCardinalityMismatch,
    };

    pub fn init(allocator: std.mem.Allocator) !Self {
        return initCapacity(allocator, INITIAL_CAPACITY);
    }

    /// Initialize an empty bitmap with space for exactly `container_capacity`
    /// top-level containers.
    pub fn initCapacity(allocator: std.mem.Allocator, container_capacity: u32) !Self {
        const keys = try allocator.alloc(u16, container_capacity);
        errdefer allocator.free(keys);

        const containers = try allocator.alloc(TaggedPtr, container_capacity);

        return .{
            .keys = keys,
            .containers = containers,
            .size = 0,
            .capacity = container_capacity,
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

        try result.ensureTotalCapacity(self.size);

        for (self.containers[0..self.size], self.keys[0..self.size], 0..) |tp, key, i| {
            const cloned = Container.fromTagged(tp).clone(allocator) catch |err| {
                result.size = @intCast(i);
                return err;
            };
            result.containers[i] = cloned.toTagged();
            result.keys[i] = key;
        }
        result.size = self.size;
        result.cached_cardinality = self.cached_cardinality;

        return result;
    }

    /// Verify structural invariants without mutating or repairing the bitmap.
    pub fn validate(self: *const Self) ValidateError!void {
        if (self.size == 0) return;
        if (self.size > self.keys.len or self.size > self.containers.len or
            self.capacity != self.keys.len or self.capacity != self.containers.len)
        {
            return ValidateError.BitmapSizeRange;
        }

        for (self.keys[1..self.size], 1..) |key, i| {
            const prev = self.keys[i - 1];
            if (key == prev) return ValidateError.DuplicateKeys;
            if (key < prev) return ValidateError.UnsortedKeys;
        }

        for (self.containers[0..self.size]) |tp| {
            switch (Container.fromTagged(tp)) {
                .array => |ac| try validateArrayContainer(ac),
                .bitset => |bc| try validateBitsetContainer(bc),
                .run => |rc| try validateRunContainer(rc),
                .reserved => unreachable,
            }
        }
    }

    fn validateArrayContainer(ac: *const ArrayContainer) ValidateError!void {
        if (ac.cardinality == 0) return ValidateError.EmptyContainer;
        if (ac.cardinality > ArrayContainer.MAX_CARDINALITY or ac.cardinality > ac.capacity) {
            return ValidateError.ArrayCardinalityRange;
        }

        for (ac.values[1..ac.cardinality], 1..) |value, i| {
            if (value <= ac.values[i - 1]) return ValidateError.UnsortedArray;
        }
    }

    fn validateBitsetContainer(bc: *const BitsetContainer) ValidateError!void {
        if (bc.cardinality <= ArrayContainer.MAX_CARDINALITY) {
            return ValidateError.BitsetCardinalityRange;
        }

        var actual: u32 = 0;
        for (bc.words) |word| {
            actual += @popCount(word);
        }

        if (bc.cardinality != @as(i32, @intCast(actual))) {
            return ValidateError.BitsetCardinalityMismatch;
        }
    }

    fn validateRunContainer(rc: *const RunContainer) ValidateError!void {
        if (rc.n_runs == 0) return ValidateError.EmptyContainer;
        if (rc.n_runs > rc.capacity) return ValidateError.BitmapSizeRange;

        var actual: u32 = 0;
        for (rc.runs[0..rc.n_runs], 0..) |run, i| {
            const end: u32 = @as(u32, run.start) + @as(u32, run.length);
            if (end > std.math.maxInt(u16)) return ValidateError.RunOrdering;
            actual += @as(u32, run.length) + 1;

            if (i > 0) {
                const prev = rc.runs[i - 1];
                const prev_end = @as(u32, prev.start) + @as(u32, prev.length);
                if (@as(u32, run.start) <= prev_end + 1) return ValidateError.RunOrdering;
            }
        }

        if (rc.cardinality >= 0 and rc.cardinality != @as(i32, @intCast(actual))) {
            return ValidateError.RunCardinalityMismatch;
        }
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
        return array_kernels.lowerBound(self.keys[0..self.size], key);
    }

    fn grownCapacity(current: u32, needed: u32) u32 {
        return @max(current *| 2, needed);
    }

    /// Ensure space for at least `needed` top-level containers.
    pub fn ensureTotalCapacity(self: *Self, needed: u32) !void {
        if (needed <= self.capacity) return;

        const new_cap = grownCapacity(self.capacity, needed);

        const new_keys = try self.allocator.alloc(u16, new_cap);
        errdefer self.allocator.free(new_keys);

        const new_containers = try self.allocator.alloc(TaggedPtr, new_cap);

        @memcpy(new_keys[0..self.size], self.keys[0..self.size]);
        @memcpy(new_containers[0..self.size], self.containers[0..self.size]);

        self.allocator.free(self.keys[0..self.capacity]);
        self.allocator.free(self.containers[0..self.capacity]);
        self.keys = new_keys;
        self.containers = new_containers;
        self.capacity = new_cap;
    }

    /// Remove all values while retaining the top-level container index.
    pub fn clearRetainingCapacity(self: *Self) void {
        for (self.containers[0..self.size]) |tp| {
            Container.fromTagged(tp).deinit(self.allocator);
        }
        self.size = 0;
        self.cached_cardinality = 0;
    }

    /// Shrink internal arrays and shrinkable containers to current size.
    /// Returns an approximate number of payload bytes released.
    pub fn shrinkToFit(self: *Self) !usize {
        var freed: usize = 0;

        for (self.containers[0..self.size]) |tp| {
            switch (Container.fromTagged(tp)) {
                .array => |ac| {
                    const old_cap = ac.capacity;
                    try ac.shrinkToFit(self.allocator);
                    freed += (@as(usize, old_cap) - @as(usize, ac.capacity)) * @sizeOf(u16);
                },
                .run => |rc| {
                    const old_cap = rc.capacity;
                    try rc.shrinkToFit(self.allocator);
                    freed += (@as(usize, old_cap) - @as(usize, rc.capacity)) * @sizeOf(RunContainer.RunPair);
                },
                .bitset, .reserved => {},
            }
        }

        if (self.size < self.capacity) {
            const old_cap = self.capacity;
            const new_keys = try self.allocator.alloc(u16, self.size);
            errdefer self.allocator.free(new_keys);
            const new_containers = try self.allocator.alloc(TaggedPtr, self.size);
            errdefer self.allocator.free(new_containers);

            @memcpy(new_keys[0..self.size], self.keys[0..self.size]);
            @memcpy(new_containers[0..self.size], self.containers[0..self.size]);

            self.allocator.free(self.keys[0..self.capacity]);
            self.allocator.free(self.containers[0..self.capacity]);
            self.keys = new_keys;
            self.containers = new_containers;
            self.capacity = self.size;

            freed += (@as(usize, old_cap) - @as(usize, self.capacity)) * (@sizeOf(u16) + @sizeOf(TaggedPtr));
        }

        return freed;
    }

    /// Check if a value is present.
    pub fn contains(self: *const Self, value: u32) bool {
        const key = highBits(value);
        const idx = self.findKey(key) orelse return false;
        return Container.fromTagged(self.containers[idx]).contains(lowBits(value));
    }

    /// Count values in the inclusive range [lo, hi].
    pub fn rangeCardinality(self: *const Self, lo: u32, hi: u32) u64 {
        if (lo > hi) return 0;

        const start_key = highBits(lo);
        const end_key = highBits(hi);
        var idx = self.lowerBound(start_key);
        var total: u64 = 0;

        while (idx < self.size and self.keys[idx] <= end_key) : (idx += 1) {
            const key = self.keys[idx];
            const start_low: u16 = if (key == start_key) lowBits(lo) else 0;
            const end_low: u16 = if (key == end_key) lowBits(hi) else std.math.maxInt(u16);
            const container = Container.fromTagged(self.containers[idx]);

            if (start_low == 0 and end_low == std.math.maxInt(u16)) {
                total += container.getCardinality();
            } else {
                total += ops.containerRangeCardinality(container, start_low, end_low);
            }
        }

        return total;
    }

    /// Return whether every value in the inclusive range [lo, hi] is present.
    pub fn containsRange(self: *const Self, lo: u32, hi: u32) bool {
        if (lo > hi) return true;

        const start_key = highBits(lo);
        const end_key = highBits(hi);
        var idx = self.lowerBound(start_key);
        var key_u32: u32 = start_key;
        const end_key_u32: u32 = end_key;

        while (key_u32 <= end_key_u32) : (key_u32 += 1) {
            const key: u16 = @intCast(key_u32);
            if (idx >= self.size or self.keys[idx] != key) return false;

            const start_low: u16 = if (key == start_key) lowBits(lo) else 0;
            const end_low: u16 = if (key == end_key) lowBits(hi) else std.math.maxInt(u16);
            if (!ops.containerContainsRange(Container.fromTagged(self.containers[idx]), start_low, end_low)) {
                return false;
            }

            idx += 1;
            if (key_u32 == end_key_u32) break;
        }

        return true;
    }

    /// Return whether any value in the inclusive range [lo, hi] is present.
    pub fn intersectsRange(self: *const Self, lo: u32, hi: u32) bool {
        if (lo > hi) return false;

        const start_key = highBits(lo);
        const end_key = highBits(hi);
        var idx = self.lowerBound(start_key);

        while (idx < self.size and self.keys[idx] <= end_key) : (idx += 1) {
            const key = self.keys[idx];
            const start_low: u16 = if (key == start_key) lowBits(lo) else 0;
            const end_low: u16 = if (key == end_key) lowBits(hi) else std.math.maxInt(u16);

            if (ops.containerIntersectsRange(Container.fromTagged(self.containers[idx]), start_low, end_low)) {
                return true;
            }
        }

        return false;
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

    /// Add all values in the slice. Values need not be sorted.
    pub fn addMany(self: *Self, values: []const u32) !void {
        if (values.len == 0) return;
        self.cached_cardinality = -1;

        var cursor_key: ?u16 = null;
        var cursor_idx: usize = 0;

        for (values) |value| {
            const key = highBits(value);
            const low = lowBits(value);

            if (cursor_key == null or cursor_key.? != key or cursor_idx >= self.size or self.keys[cursor_idx] != key) {
                cursor_idx = self.lowerBound(key);
                cursor_key = key;
            }

            if (cursor_idx < self.size and self.keys[cursor_idx] == key) {
                _ = try self.addToContainer(cursor_idx, low);
            } else {
                try self.insertContainerAt(cursor_idx, key, low);
            }
        }
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

    /// Remove all values in the inclusive range [lo, hi]. Returns count removed.
    pub fn removeRange(self: *Self, lo: u32, hi: u32) !u64 {
        return range_ops.removeRange(self, lo, hi);
    }

    /// Create an independently owned copy with [lo, hi] removed.
    pub fn removeRangeCopy(self: *const Self, allocator: std.mem.Allocator, lo: u32, hi: u32) !Self {
        return range_ops.removeRangeCopy(self, allocator, lo, hi);
    }

    /// Add a range within a single chunk.
    fn addRangeToChunk(self: *Self, key: u16, start: u16, end: u16) !u64 {
        const range_size: u32 = @as(u32, end) - start + 1;

        if (self.findKey(key)) |idx| {
            return self.addRangeToContainer(idx, start, end);
        }

        // Need to create new container
        const insert_idx = self.lowerBound(key);
        try self.ensureTotalCapacity(self.size + 1);

        // Shift right to make room
        if (insert_idx < self.size) {
            @memmove(self.keys[insert_idx + 1 .. self.size + 1], self.keys[insert_idx..self.size]);
            @memmove(self.containers[insert_idx + 1 .. self.size + 1], self.containers[insert_idx..self.size]);
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
        try result.ensureTotalCapacity(container_count);

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
        try self.ensureTotalCapacity(self.size + 1);

        // Shift right to make room
        if (idx < self.size) {
            @memmove(self.keys[idx + 1 .. self.size + 1], self.keys[idx..self.size]);
            @memmove(self.containers[idx + 1 .. self.size + 1], self.containers[idx..self.size]);
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

    /// Remove all values in the slice. Values need not be sorted.
    pub fn removeMany(self: *Self, values: []const u32) !void {
        if (values.len == 0 or self.size == 0) return;
        self.cached_cardinality = -1;

        var cursor_key: ?u16 = null;
        var cursor_idx: usize = 0;

        for (values) |value| {
            const key = highBits(value);
            const low = lowBits(value);

            if (cursor_key == null or cursor_key.? != key or cursor_idx >= self.size or self.keys[cursor_idx] != key) {
                cursor_idx = self.lowerBound(key);
                cursor_key = key;
            }

            if (cursor_idx >= self.size or self.keys[cursor_idx] != key) continue;

            _ = try self.removeFromContainer(cursor_idx, low);
            if (cursor_idx >= self.size or self.keys[cursor_idx] != key) {
                cursor_key = null;
            }
        }
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
        } else if (self.containers[idx].getType() == .bitset and card <= ArrayContainer.MAX_CARDINALITY) {
            const bc = self.containers[idx].getBitset();
            const ac = try ops.bitsetToArray(self.allocator, bc);
            bc.deinit(self.allocator);
            self.containers[idx] = TaggedPtr.initArray(ac);
        }

        return true;
    }

    /// Remove container at the given index.
    fn removeContainerAt(self: *Self, idx: usize) void {
        Container.fromTagged(self.containers[idx]).deinit(self.allocator);

        // Shift left
        if (idx + 1 < self.size) {
            @memmove(self.keys[idx .. self.size - 1], self.keys[idx + 1 .. self.size]);
            @memmove(self.containers[idx .. self.size - 1], self.containers[idx + 1 .. self.size]);
        }
        self.size -= 1;
    }

    /// Get the total cardinality (number of values).
    ///
    /// If the cached cardinality is invalid, this recomputes without mutating
    /// the bitmap or container caches so the method remains const-safe.
    pub fn cardinality(self: *const Self) u64 {
        return self.cardinalityConst();
    }

    /// Allocate and return all values in ascending order.
    pub fn toArrayAlloc(self: *const Self, allocator: std.mem.Allocator) ![]u32 {
        const total = self.cardinalityConst();
        const values = try allocator.alloc(u32, @intCast(total));
        errdefer allocator.free(values);
        const written = self.toArray(values);
        std.debug.assert(written == values.len);
        return values;
    }

    /// Fill a caller-provided slice with all values in ascending order.
    pub fn toArray(self: *const Self, out: []u32) usize {
        std.debug.assert(out.len >= self.cardinalityConst());

        var pos: usize = 0;
        for (self.keys[0..self.size], self.containers[0..self.size]) |key, tp| {
            const high = @as(u32, key) << 16;
            switch (Container.fromTagged(tp)) {
                .array => |ac| {
                    for (ac.values[0..ac.cardinality]) |low| {
                        out[pos] = high | low;
                        pos += 1;
                    }
                },
                .bitset => |bc| {
                    for (bc.words, 0..) |word, word_idx| {
                        var bits = word;
                        while (bits != 0) {
                            const bit = @ctz(bits);
                            bits &= bits - 1;
                            out[pos] = high | @as(u32, @intCast(word_idx * 64 + bit));
                            pos += 1;
                        }
                    }
                },
                .run => |rc| {
                    for (rc.runs[0..rc.n_runs]) |run| {
                        const end = @as(u32, run.start) + run.length;
                        var low: u32 = run.start;
                        while (low <= end) : (low += 1) {
                            out[pos] = high | low;
                            pos += 1;
                        }
                    }
                },
                .reserved => unreachable,
            }
        }
        return pos;
    }

    fn cardinalityConst(self: *const Self) u64 {
        if (self.cached_cardinality >= 0) return @intCast(self.cached_cardinality);

        var total: u64 = 0;
        for (self.containers[0..self.size]) |tp| {
            switch (Container.fromTagged(tp)) {
                .array => |ac| total += ac.cardinality,
                .bitset => |bc| total += if (bc.cardinality >= 0) @as(u32, @intCast(bc.cardinality)) else BitsetContainer.countWords(bc.words),
                .run => |rc| {
                    if (rc.cardinality >= 0) {
                        total += @as(u32, @intCast(rc.cardinality));
                    } else {
                        for (rc.runs[0..rc.n_runs]) |run| {
                            total += run.size();
                        }
                    }
                },
                .reserved => unreachable,
            }
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
        return self.twoWayAllocatingMerge(.bor, allocator, other);
    }

    /// Return a new bitmap that is the intersection (AND) of self and other.
    pub fn bitwiseAnd(self: *const Self, allocator: std.mem.Allocator, other: *const Self) !Self {
        return self.twoWayAllocatingMerge(.band, allocator, other);
    }

    /// Compute |self ∩ other| without allocating a result bitmap.
    /// Useful for join selectivity estimation in query planning.
    pub fn andCardinality(self: *const Self, other: *const Self) u64 {
        return self.twoWayCardinality(.band, other);
    }

    /// Compute |self ∪ other| without allocating a result bitmap.
    pub fn orCardinality(self: *const Self, other: *const Self) u64 {
        return self.twoWayCardinality(.bor, other);
    }

    /// Compute |self △ other| without allocating a result bitmap.
    pub fn xorCardinality(self: *const Self, other: *const Self) u64 {
        return self.twoWayCardinality(.xor, other);
    }

    /// Compute |self \ other| without allocating a result bitmap.
    pub fn differenceCardinality(self: *const Self, other: *const Self) u64 {
        return self.twoWayCardinality(.andnot, other);
    }

    /// Compute |self ∩ other| / |self ∪ other|.
    /// Matches CRoaring's undefined empty/empty behavior by returning NaN.
    pub fn jaccardIndex(self: *const Self, other: *const Self) f64 {
        const intersection = self.andCardinality(other);
        const union_cardinality = self.orCardinality(other);
        return @as(f64, @floatFromInt(intersection)) / @as(f64, @floatFromInt(union_cardinality));
    }

    /// Count values <= `value`.
    pub fn rank(self: *const Self, value: u32) u64 {
        const target_key = highBits(value);
        const target_low = lowBits(value);

        var total: u64 = 0;
        for (self.keys[0..self.size], self.containers[0..self.size]) |key, tp| {
            if (key < target_key) {
                total += Container.fromTagged(tp).getCardinality();
            } else if (key == target_key) {
                return total + ops.containerRank(Container.fromTagged(tp), target_low);
            } else {
                return total;
            }
        }
        return total;
    }

    /// Return the 0-based position of `value`, or null if absent.
    pub fn getIndex(self: *const Self, value: u32) ?u64 {
        const target_key = highBits(value);
        const target_low = lowBits(value);

        var total: u64 = 0;
        for (self.keys[0..self.size], self.containers[0..self.size]) |key, tp| {
            const container = Container.fromTagged(tp);
            if (key < target_key) {
                total += container.getCardinality();
            } else if (key == target_key) {
                if (!container.contains(target_low)) return null;
                return total + ops.containerRank(container, target_low) - 1;
            } else {
                return null;
            }
        }
        return null;
    }

    /// Return the k-th smallest value, 0-based, or null if out of range.
    pub fn select(self: *const Self, k: u64) ?u32 {
        if (k > std.math.maxInt(u32)) return null;
        var remaining: u32 = @intCast(k);
        for (self.keys[0..self.size], self.containers[0..self.size]) |key, tp| {
            switch (tp.getType()) {
                .array => {
                    const container = tp.getArray();
                    if (remaining < container.cardinality) {
                        return combine(key, container.values[@intCast(remaining)]);
                    }
                    remaining -= container.cardinality;
                },
                .bitset => {
                    const container = tp.getBitset();
                    const card = container.getCardinality();
                    if (remaining < card) {
                        const low = ops.containerSelect(.{ .bitset = container }, @intCast(remaining)) orelse return null;
                        return combine(key, low);
                    }
                    remaining -= card;
                },
                .run => {
                    const container = tp.getRun();
                    const card = container.getCardinality();
                    if (remaining < card) {
                        const low = ops.containerSelect(.{ .run = container }, remaining) orelse return null;
                        return combine(key, low);
                    }
                    remaining -= card;
                },
                .reserved => unreachable,
            }
        }
        return null;
    }

    /// Fill `out` with ranks for sorted `values`.
    /// Preconditions: `out.len == values.len` and `values` is sorted ascending.
    pub fn rankMany(self: *const Self, values: []const u32, out: []u64) void {
        std.debug.assert(out.len == values.len);
        if (std.debug.runtime_safety) {
            for (values[1..], 0..) |value, i| {
                std.debug.assert(value >= values[i]);
            }
        }

        var container_idx: usize = 0;
        var prior: u64 = 0;
        var value_idx: usize = 0;

        while (value_idx < values.len) {
            const target_key = highBits(values[value_idx]);

            while (container_idx < self.size and self.keys[container_idx] < target_key) : (container_idx += 1) {
                prior += Container.fromTagged(self.containers[container_idx]).getCardinality();
            }

            if (container_idx < self.size and self.keys[container_idx] == target_key) {
                var run_end = value_idx + 1;
                while (run_end < values.len and highBits(values[run_end]) == target_key) : (run_end += 1) {}

                const consumed = ops.containerRankMany(
                    Container.fromTagged(self.containers[container_idx]),
                    prior,
                    values[value_idx..run_end],
                    out[value_idx..run_end],
                );
                value_idx += consumed;
            } else {
                out[value_idx] = prior;
                value_idx += 1;
            }
        }
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
        return self.twoWayAllocatingMerge(.andnot, allocator, other);
    }

    /// Return a new bitmap that is the symmetric difference (XOR) of self and other.
    pub fn bitwiseXor(self: *const Self, allocator: std.mem.Allocator, other: *const Self) !Self {
        return self.twoWayAllocatingMerge(.xor, allocator, other);
    }

    /// Return a new bitmap that is the union (OR) of all inputs.
    pub fn orMany(allocator: std.mem.Allocator, bitmaps: []const *const Self) !Self {
        return manyOrMerge(allocator, bitmaps);
    }

    /// Return a new bitmap that is the union (OR) of all inputs.
    /// CRoaring exposes this as a heap-based variant; rawr's k-way lazy merge is
    /// already independent of input-size ordering, so this is a parity alias.
    pub fn orManyHeap(allocator: std.mem.Allocator, bitmaps: []const *const Self) !Self {
        return orMany(allocator, bitmaps);
    }

    /// Return a new bitmap that is the symmetric difference (XOR) of all inputs.
    pub fn xorMany(allocator: std.mem.Allocator, bitmaps: []const *const Self) !Self {
        return manyMerge(.xor, allocator, bitmaps);
    }

    /// Lazy union. Result must be repaired with `repairAfterLazy` before normal use.
    pub fn lazyOr(self: *const Self, allocator: std.mem.Allocator, other: *const Self, bitset_conversion: bool) !Self {
        return lazyMergeTwo(.bor, allocator, self, other, bitset_conversion, .baseline);
    }

    /// Lazy symmetric difference. Result must be repaired with `repairAfterLazy` before normal use.
    pub fn lazyXor(self: *const Self, allocator: std.mem.Allocator, other: *const Self) !Self {
        return lazyMergeTwo(.xor, allocator, self, other, true, .baseline);
    }

    /// Return a new bitmap with values in [lo, hi] complemented.
    pub fn flip(self: *const Self, allocator: std.mem.Allocator, lo: u32, hi: u32) !Self {
        return range_ops.flip(self, allocator, lo, hi);
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
                const is_same = result_tp.eql(self.containers[i]);
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

    /// Consuming in-place union: `self |= other`, moving right-only containers
    /// instead of cloning them.
    ///
    /// Both bitmaps must use the exact same allocator handle and must be distinct
    /// objects. Allocator mismatch returns `error.AllocatorMismatch`; aliasing
    /// returns `error.AliasedOperands`, both before mutation.
    ///
    /// On success, `other` is a valid empty bitmap whose top-level capacity is
    /// retained and may be reused or deinited normally. On allocation failure,
    /// `other` is unchanged and `self` remains valid, but `self` may contain
    /// completed unions for matched chunk keys processed before the failure.
    pub fn bitwiseOrInPlaceConsume(self: *Self, other: *Self) !void {
        if (self.allocator.ptr != other.allocator.ptr or self.allocator.vtable != other.allocator.vtable) {
            return error.AllocatorMismatch;
        }
        if (self == other) return error.AliasedOperands;

        const old_self_size = self.size;
        var i: usize = 0;
        var j: usize = 0;
        var unmatched: u32 = 0;
        while (i < old_self_size and j < other.size) {
            if (self.keys[i] < other.keys[j]) {
                i += 1;
            } else if (self.keys[i] > other.keys[j]) {
                unmatched += 1;
                j += 1;
            } else {
                i += 1;
                j += 1;
            }
        }
        unmatched += other.size - @as(u32, @intCast(j));

        const output_size = old_self_size + unmatched;
        try self.ensureTotalCapacity(output_size);

        // This is unconditional: an unmatched-only commit also changes self.
        self.cached_cardinality = -1;

        i = 0;
        j = 0;
        while (i < old_self_size and j < other.size) {
            const key_a = self.keys[i];
            const key_b = other.keys[j];
            if (key_a < key_b) {
                i += 1;
            } else if (key_a > key_b) {
                j += 1;
            } else {
                const old_container = Container.fromTagged(self.containers[i]);
                const result = try ops.containerUnionInPlace(
                    self.allocator,
                    old_container,
                    Container.fromTagged(other.containers[j]),
                );
                const result_tp = result.toTagged();
                if (!result_tp.eql(self.containers[i])) {
                    old_container.deinit(self.allocator);
                    self.containers[i] = result_tp;
                }
                i += 1;
                j += 1;
            }
        }

        // Commit starts here. All remaining operations are infallible.
        i = 0;
        j = 0;
        while (i < old_self_size and j < other.size) {
            if (self.keys[i] < other.keys[j]) {
                i += 1;
            } else if (self.keys[i] > other.keys[j]) {
                j += 1;
            } else {
                Container.fromTagged(other.containers[j]).deinit(other.allocator);
                i += 1;
                j += 1;
            }
        }

        var left_tail: usize = old_self_size;
        var right_tail: usize = other.size;
        var out_tail: usize = output_size;
        while (left_tail > 0 and right_tail > 0) {
            const key_a = self.keys[left_tail - 1];
            const key_b = other.keys[right_tail - 1];
            out_tail -= 1;
            if (key_a > key_b) {
                left_tail -= 1;
                self.keys[out_tail] = key_a;
                self.containers[out_tail] = self.containers[left_tail];
            } else if (key_a < key_b) {
                right_tail -= 1;
                self.keys[out_tail] = key_b;
                self.containers[out_tail] = other.containers[right_tail];
            } else {
                left_tail -= 1;
                right_tail -= 1;
                self.keys[out_tail] = key_a;
                self.containers[out_tail] = self.containers[left_tail];
            }
        }
        while (left_tail > 0) {
            left_tail -= 1;
            out_tail -= 1;
            self.keys[out_tail] = self.keys[left_tail];
            self.containers[out_tail] = self.containers[left_tail];
        }
        while (right_tail > 0) {
            right_tail -= 1;
            out_tail -= 1;
            self.keys[out_tail] = other.keys[right_tail];
            self.containers[out_tail] = other.containers[right_tail];
        }
        std.debug.assert(out_tail == 0);

        self.size = output_size;
        other.size = 0;
        other.cached_cardinality = 0;
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
                const IntersectionResult = struct {
                    container: Container,
                    used_scratch: bool,
                };
                const intersection: IntersectionResult = blk: {
                    const scratch_container = ops.containerIntersection(scratch_alloc, self_container, other_container) catch {
                        scratch.reset();
                        break :blk .{
                            .container = try ops.containerIntersection(self.allocator, self_container, other_container),
                            .used_scratch = false,
                        };
                    };
                    break :blk .{
                        .container = scratch_container,
                        .used_scratch = true,
                    };
                };

                const intersected = intersection.container;
                const used_scratch = intersection.used_scratch;
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
                const diff = try ops.containerDifferenceInPlace(self.allocator, self_container, other_container);
                const diff_tp = diff.toTagged();
                const is_same = diff_tp.eql(self.containers[i]);

                if (diff.getCardinality() > 0) {
                    self.keys[write_idx] = key_a;
                    self.containers[write_idx] = diff_tp;
                    write_idx += 1;
                } else {
                    diff.deinit(self.allocator);
                }
                if (!is_same) {
                    self_container.deinit(self.allocator);
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
                const result = try ops.containerXorInPlace(self.allocator, old_container, other_container);
                const result_tp = result.toTagged();
                const is_same = result_tp.eql(self.containers[i]);

                // Only keep non-empty results
                if (result.getCardinality() > 0) {
                    new_keys[k] = key_a;
                    new_containers[k] = result_tp;
                    owned[k] = !is_same;
                    k += 1;
                } else {
                    result.deinit(self.allocator);
                }
                if (!is_same) {
                    old_container.deinit(self.allocator);
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

    /// In-place lazy union. Call `repairAfterLazy` before normal use.
    pub fn lazyOrInPlace(self: *Self, other: *const Self, bitset_conversion: bool) !void {
        var result = try self.lazyOr(self.allocator, other, bitset_conversion);
        errdefer result.deinit();
        var old = self.*;
        self.* = result;
        old.deinit();
    }

    /// In-place lazy symmetric difference. Call `repairAfterLazy` before normal use.
    pub fn lazyXorInPlace(self: *Self, other: *const Self) !void {
        var result = try self.lazyXor(self.allocator, other);
        errdefer result.deinit();
        var old = self.*;
        self.* = result;
        old.deinit();
    }

    /// Complement values in [lo, hi] in place.
    pub fn flipInplace(self: *Self, lo: u32, hi: u32) !void {
        return range_ops.flipInPlace(self, lo, hi);
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

    /// Restore invariants after lazy operations.
    pub fn repairAfterLazy(self: *Self) !void {
        var write_idx: usize = 0;
        var total: u64 = 0;

        for (self.keys[0..self.size], self.containers[0..self.size]) |key, tp| {
            const container = Container.fromTagged(tp);
            switch (container) {
                .array => |ac| {
                    if (ac.cardinality == 0) {
                        ac.deinit(self.allocator);
                        continue;
                    }
                    self.keys[write_idx] = key;
                    self.containers[write_idx] = tp;
                    total += ac.cardinality;
                    write_idx += 1;
                },
                .bitset => |bc| {
                    const card = bc.computeCardinality();
                    if (card == 0) {
                        bc.deinit(self.allocator);
                        continue;
                    }
                    if (card <= ArrayContainer.MAX_CARDINALITY) {
                        const ac = try ops.bitsetToArray(self.allocator, bc);
                        bc.deinit(self.allocator);
                        self.keys[write_idx] = key;
                        self.containers[write_idx] = TaggedPtr.initArray(ac);
                    } else {
                        self.keys[write_idx] = key;
                        self.containers[write_idx] = tp;
                    }
                    total += card;
                    write_idx += 1;
                },
                .run => |rc| {
                    const card = rc.getCardinality();
                    if (card == 0) {
                        rc.deinit(self.allocator);
                        continue;
                    }
                    self.keys[write_idx] = key;
                    self.containers[write_idx] = tp;
                    total += card;
                    write_idx += 1;
                },
                .reserved => unreachable,
            }
        }

        self.size = @intCast(write_idx);
        self.cached_cardinality = @intCast(total);
    }

    pub const RepairAfterLazyOptions = struct {
        /// The caller asserts that the bitmap allocator benefits when transient
        /// bitsets are freed in descending allocation order.
        allocator_benefits_from_descending_free_order: bool = false,
    };

    /// Restore invariants after lazy operations with allocator-specific tuning.
    ///
    /// The descending-free option is intended for allocators whose reuse order
    /// benefits from reverse frees. It is not selected automatically because
    /// the effect is allocator-dependent and can regress other allocators.
    pub fn repairAfterLazyWithOptions(self: *Self, options: RepairAfterLazyOptions) !void {
        if (!options.allocator_benefits_from_descending_free_order) {
            return self.repairAfterLazy();
        }

        const deferred_bitsets = self.allocator.alloc(*BitsetContainer, self.size) catch {
            return self.repairAfterLazy();
        };
        defer self.allocator.free(deferred_bitsets);

        var deferred_count: usize = 0;
        errdefer deinitBitsetsReverse(self.allocator, deferred_bitsets[0..deferred_count]);

        const original_size: usize = self.size;
        var read_idx: usize = 0;
        var write_idx: usize = 0;
        var total: u64 = 0;

        while (read_idx < original_size) : (read_idx += 1) {
            const key = self.keys[read_idx];
            const tp = self.containers[read_idx];
            switch (Container.fromTagged(tp)) {
                .array => |ac| {
                    if (ac.cardinality == 0) {
                        ac.deinit(self.allocator);
                        continue;
                    }
                    self.keys[write_idx] = key;
                    self.containers[write_idx] = tp;
                    total += ac.cardinality;
                    write_idx += 1;
                },
                .bitset => |bc| {
                    const card = bc.computeCardinality();
                    if (card == 0) {
                        deferred_bitsets[deferred_count] = bc;
                        deferred_count += 1;
                        continue;
                    }
                    if (card <= ArrayContainer.MAX_CARDINALITY) {
                        const ac = ops.bitsetToArray(self.allocator, bc) catch |err| {
                            self.commitPartialLazyRepair(write_idx, read_idx, original_size);
                            return err;
                        };
                        deferred_bitsets[deferred_count] = bc;
                        deferred_count += 1;
                        self.keys[write_idx] = key;
                        self.containers[write_idx] = TaggedPtr.initArray(ac);
                    } else {
                        self.keys[write_idx] = key;
                        self.containers[write_idx] = tp;
                    }
                    total += card;
                    write_idx += 1;
                },
                .run => |rc| {
                    const card = rc.getCardinality();
                    if (card == 0) {
                        rc.deinit(self.allocator);
                        continue;
                    }
                    self.keys[write_idx] = key;
                    self.containers[write_idx] = tp;
                    total += card;
                    write_idx += 1;
                },
                .reserved => unreachable,
            }
        }

        self.size = @intCast(write_idx);
        self.cached_cardinality = @intCast(total);
        deinitBitsetsReverse(self.allocator, deferred_bitsets[0..deferred_count]);
    }

    fn commitPartialLazyRepair(self: *Self, write_idx: usize, read_idx: usize, original_size: usize) void {
        const tail_len = original_size - read_idx;
        if (write_idx != read_idx) {
            std.mem.copyForwards(
                u16,
                self.keys[write_idx .. write_idx + tail_len],
                self.keys[read_idx..original_size],
            );
            std.mem.copyForwards(
                TaggedPtr,
                self.containers[write_idx .. write_idx + tail_len],
                self.containers[read_idx..original_size],
            );
        }
        self.size = @intCast(write_idx + tail_len);
        self.cached_cardinality = -1;
    }

    fn deinitBitsetsReverse(allocator: std.mem.Allocator, bitsets: []const *BitsetContainer) void {
        var index = bitsets.len;
        while (index > 0) {
            index -= 1;
            bitsets[index].deinit(allocator);
        }
    }

    /// Insert a tagged container at the given position, shifting existing containers.
    fn insertTaggedContainerAt(self: *Self, pos: usize, key: u16, tp: TaggedPtr) !void {
        try self.ensureTotalCapacity(self.size + 1);
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

    /// Check if self is a proper subset of other.
    pub fn isStrictSubsetOf(self: *const Self, other: *const Self) bool {
        return self.isSubsetOf(other) and !self.equals(other);
    }

    /// Check if two bitmaps are equal. Single pass O(n).
    pub fn equals(self: *const Self, other: *const Self) bool {
        return compare.equals(self, other);
    }

    // ========================================================================
    // Helper Functions
    // ========================================================================

    const TwoWayOp = enum { bor, band, xor, andnot };

    fn twoWayCardinality(self: *const Self, comptime op: TwoWayOp, other: *const Self) u64 {
        var total: u64 = 0;
        var i: usize = 0;
        var j: usize = 0;

        while (i < self.size and j < other.size) {
            const key_a = self.keys[i];
            const key_b = other.keys[j];

            if (key_a < key_b) {
                total += aOnlyCardinality(op, Container.fromTagged(self.containers[i]));
                i += 1;
            } else if (key_a > key_b) {
                total += bOnlyCardinality(op, Container.fromTagged(other.containers[j]));
                j += 1;
            } else {
                total += bothCardinality(
                    op,
                    Container.fromTagged(self.containers[i]),
                    Container.fromTagged(other.containers[j]),
                );
                i += 1;
                j += 1;
            }
        }

        while (i < self.size) : (i += 1) {
            total += aOnlyCardinality(op, Container.fromTagged(self.containers[i]));
        }
        while (j < other.size) : (j += 1) {
            total += bOnlyCardinality(op, Container.fromTagged(other.containers[j]));
        }

        return total;
    }

    fn aOnlyCardinality(comptime op: TwoWayOp, container: Container) u64 {
        return switch (op) {
            .bor, .xor, .andnot => container.getCardinality(),
            .band => 0,
        };
    }

    fn bOnlyCardinality(comptime op: TwoWayOp, container: Container) u64 {
        return switch (op) {
            .bor, .xor => container.getCardinality(),
            .band, .andnot => 0,
        };
    }

    fn bothCardinality(comptime op: TwoWayOp, a: Container, b: Container) u64 {
        const intersection = ops.containerIntersectionCardinality(a, b);
        return switch (op) {
            .band => intersection,
            .bor => a.getCardinality() + b.getCardinality() - intersection,
            .xor => a.getCardinality() + b.getCardinality() - 2 * intersection,
            .andnot => a.getCardinality() - intersection,
        };
    }

    fn twoWayAllocatingMerge(self: *const Self, comptime op: TwoWayOp, allocator: std.mem.Allocator, other: *const Self) !Self {
        if (op == .band) {
            return self.twoWayAllocatingMergeAnd(allocator, other);
        }

        var result = try Self.init(allocator);
        errdefer result.deinit();

        var i: usize = 0;
        var j: usize = 0;

        while (i < self.size and j < other.size) {
            const key_a = self.keys[i];
            const key_b = other.keys[j];

            if (key_a < key_b) {
                try appendAOnlyAllocating(op, &result, allocator, key_a, self.containers[i]);
                i += 1;
            } else if (key_a > key_b) {
                try appendBOnlyAllocating(op, &result, allocator, key_b, other.containers[j]);
                j += 1;
            } else {
                try appendBothAllocating(
                    op,
                    &result,
                    allocator,
                    key_a,
                    Container.fromTagged(self.containers[i]),
                    Container.fromTagged(other.containers[j]),
                );
                i += 1;
                j += 1;
            }
        }

        while (i < self.size) : (i += 1) {
            try appendAOnlyAllocating(op, &result, allocator, self.keys[i], self.containers[i]);
        }
        while (j < other.size) : (j += 1) {
            try appendBOnlyAllocating(op, &result, allocator, other.keys[j], other.containers[j]);
        }

        result.cached_cardinality = -1;
        return result;
    }

    fn twoWayAllocatingMergeAnd(self: *const Self, allocator: std.mem.Allocator, other: *const Self) !Self {
        // Dense result diagnosis showed exact top-level sizing wins on both M4 and Zen 4.
        var result = try Self.initCapacity(allocator, @min(self.size, other.size));
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
                try appendIntersectionWithScratch(
                    &result,
                    allocator,
                    key_a,
                    Container.fromTagged(self.containers[i]),
                    Container.fromTagged(other.containers[j]),
                    &scratch,
                );
                i += 1;
                j += 1;
            }
        }

        result.cached_cardinality = -1;
        return result;
    }

    fn appendAOnlyAllocating(comptime op: TwoWayOp, result: *Self, allocator: std.mem.Allocator, key: u16, tp: TaggedPtr) !void {
        switch (op) {
            .bor, .xor, .andnot => try result.appendClonedContainer(allocator, key, tp),
            .band => {},
        }
    }

    fn appendBOnlyAllocating(comptime op: TwoWayOp, result: *Self, allocator: std.mem.Allocator, key: u16, tp: TaggedPtr) !void {
        switch (op) {
            .bor, .xor => try result.appendClonedContainer(allocator, key, tp),
            .band, .andnot => {},
        }
    }

    fn appendBothAllocating(
        comptime op: TwoWayOp,
        result: *Self,
        allocator: std.mem.Allocator,
        key: u16,
        a: Container,
        b: Container,
    ) !void {
        const c = switch (op) {
            .bor => try ops.containerUnion(allocator, a, b),
            .xor => try ops.containerXor(allocator, a, b),
            .andnot => try ops.containerDifference(allocator, a, b),
            .band => unreachable,
        };

        if (op == .bor or c.getCardinality() > 0) {
            try result.appendOwnedContainer(allocator, key, c.toTagged());
        } else {
            c.deinit(allocator);
        }
    }

    fn appendIntersectionWithScratch(
        result: *Self,
        allocator: std.mem.Allocator,
        key: u16,
        a: Container,
        b: Container,
        scratch: *std.heap.FixedBufferAllocator,
    ) !void {
        const scratch_alloc = scratch.allocator();
        const IntersectionResult = struct {
            container: Container,
            used_scratch: bool,
        };
        const intersection: IntersectionResult = blk: {
            const scratch_container = ops.containerIntersection(scratch_alloc, a, b) catch {
                scratch.reset();
                break :blk .{
                    .container = try ops.containerIntersection(allocator, a, b),
                    .used_scratch = false,
                };
            };
            break :blk .{
                .container = scratch_container,
                .used_scratch = true,
            };
        };

        const c = intersection.container;
        const used_scratch = intersection.used_scratch;
        if (c.getCardinality() > 0) {
            if (used_scratch) {
                const permanent = try c.clone(allocator);
                try result.appendOwnedContainer(allocator, key, permanent.toTagged());
            } else {
                try result.appendOwnedContainer(allocator, key, c.toTagged());
            }
        } else if (!used_scratch) {
            c.deinit(allocator);
        }

        scratch.reset();
    }

    /// Append a container (assumes keys are in sorted order).
    fn appendContainer(self: *Self, key: u16, tp: TaggedPtr) !void {
        try self.ensureTotalCapacity(self.size + 1);
        self.keys[self.size] = key;
        self.containers[self.size] = tp;
        self.size += 1;
    }

    fn appendOwnedContainer(self: *Self, allocator: std.mem.Allocator, key: u16, tp: TaggedPtr) !void {
        errdefer Container.fromTagged(tp).deinit(allocator);
        try self.appendContainer(key, tp);
    }

    fn appendClonedContainer(self: *Self, allocator: std.mem.Allocator, key: u16, tp: TaggedPtr) !void {
        const cloned = try cloneContainer(allocator, tp);
        try self.appendOwnedContainer(allocator, key, cloned);
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

    const ManyOp = enum { bor, xor };

    const LazyConstructionMode = enum {
        baseline,
        batched_unsorted,
        batched_sorted,
    };

    const PendingBitset = struct {
        payload_addr: usize,
        header: *BitsetContainer,
    };

    fn manyOrMerge(allocator: std.mem.Allocator, bitmaps: []const *const Self) !Self {
        if (bitmaps.len == 1) return bitmaps[0].clone(allocator);

        var capacity: usize = 0;
        for (bitmaps) |bitmap| capacity = @min(capacity +| bitmap.size, 1 << 16);
        var result = try Self.initCapacity(allocator, @intCast(capacity));
        errdefer result.deinit();
        if (bitmaps.len == 0) return result;

        const cursors = try allocator.alloc(usize, bitmaps.len);
        defer allocator.free(cursors);
        @memset(cursors, 0);

        const bitset_pointers = try allocator.alloc(*const BitsetContainer, bitmaps.len);
        defer allocator.free(bitset_pointers);

        while (nextManyKey(bitmaps, cursors)) |key| {
            const tp = try foldManyOrKey(
                allocator,
                bitmaps,
                cursors,
                key,
                bitset_pointers,
            );
            result.appendContainer(key, tp) catch |err| {
                Container.fromTagged(tp).deinit(allocator);
                return err;
            };
        }

        result.cached_cardinality = -1;
        try result.repairAfterLazy();
        return result;
    }

    fn foldManyOrKey(
        allocator: std.mem.Allocator,
        bitmaps: []const *const Self,
        cursors: []usize,
        key: u16,
        bitset_pointers: []*const BitsetContainer,
    ) !TaggedPtr {
        var source_count: usize = 0;
        for (bitmaps, cursors) |bitmap, cursor| {
            if (cursor < bitmap.size and bitmap.keys[cursor] == key) source_count += 1;
        }

        if (source_count == 1) {
            for (bitmaps, cursors) |bitmap, *cursor| {
                if (cursor.* >= bitmap.size or bitmap.keys[cursor.*] != key) continue;
                const cloned = try cloneContainer(allocator, bitmap.containers[cursor.*]);
                cursor.* += 1;
                return cloned;
            }
            unreachable;
        }

        var bitset_count: usize = 0;
        for (bitmaps, cursors) |bitmap, cursor| {
            if (cursor >= bitmap.size or bitmap.keys[cursor] != key) continue;
            switch (Container.fromTagged(bitmap.containers[cursor])) {
                .bitset => |bitset| {
                    bitset_pointers[bitset_count] = bitset;
                    bitset_count += 1;
                },
                .array, .run => {},
                .reserved => unreachable,
            }
        }

        const accumulator = if (bitset_count == 0)
            try BitsetContainer.init(allocator)
        else
            try bitset_pointers[0].clone(allocator);
        errdefer accumulator.deinit(allocator);

        if (bitset_count > 1) {
            wordMajorOr(accumulator, bitset_pointers[1..bitset_count]);
        }
        for (bitmaps, cursors) |bitmap, *cursor| {
            if (cursor.* >= bitmap.size or bitmap.keys[cursor.*] != key) continue;
            const container = Container.fromTagged(bitmap.containers[cursor.*]);
            if (!isBitsetContainer(container)) {
                lazyAccumulateIntoBitset(.bor, accumulator, container);
            }
            cursor.* += 1;
        }
        accumulator.cardinality = -1;
        return TaggedPtr.initBitset(accumulator);
    }

    fn wordMajorOr(destination: *BitsetContainer, sources: []const *const BitsetContainer) void {
        const vector_width = 8;
        var word_index: usize = 0;
        while (word_index < BitsetContainer.NUM_WORDS) : (word_index += vector_width) {
            var accumulated: @Vector(vector_width, u64) = destination.words[word_index..][0..vector_width].*;
            for (sources) |source| {
                const words: @Vector(vector_width, u64) = source.words[word_index..][0..vector_width].*;
                accumulated |= words;
            }
            destination.words[word_index..][0..vector_width].* = accumulated;
        }
        destination.cardinality = -1;
    }

    fn manyMerge(comptime op: ManyOp, allocator: std.mem.Allocator, bitmaps: []const *const Self) !Self {
        if (bitmaps.len == 1) {
            return bitmaps[0].clone(allocator);
        }

        var result = try Self.init(allocator);
        errdefer result.deinit();

        if (bitmaps.len == 0) {
            return result;
        }

        const cursors = try allocator.alloc(usize, bitmaps.len);
        defer allocator.free(cursors);
        @memset(cursors, 0);

        while (nextManyKey(bitmaps, cursors)) |key| {
            if (try foldManyKey(op, allocator, bitmaps, cursors, key)) |tp| {
                result.appendContainer(key, tp) catch |err| {
                    Container.fromTagged(tp).deinit(allocator);
                    return err;
                };
            }
        }

        result.cached_cardinality = -1;
        try result.repairAfterLazy();
        return result;
    }

    fn nextManyKey(bitmaps: []const *const Self, cursors: []const usize) ?u16 {
        var min_key: ?u16 = null;
        // Linear scan over N cursor heads: O(N * distinct_keys). A heap-based
        // cursor is deferred to the or_many_heap/lazy follow-up.
        for (bitmaps, cursors) |bm, cursor| {
            if (cursor >= bm.size) continue;
            const key = bm.keys[cursor];
            if (min_key == null or key < min_key.?) {
                min_key = key;
            }
        }
        return min_key;
    }

    fn foldManyKey(
        comptime op: ManyOp,
        allocator: std.mem.Allocator,
        bitmaps: []const *const Self,
        cursors: []usize,
        key: u16,
    ) !?TaggedPtr {
        var count: usize = 0;
        for (bitmaps, cursors) |bm, cursor| {
            if (cursor < bm.size and bm.keys[cursor] == key) {
                count += 1;
            }
        }

        if (count == 1) {
            for (bitmaps, cursors) |bm, *cursor| {
                if (cursor.* >= bm.size or bm.keys[cursor.*] != key) continue;
                const cloned = try cloneContainer(allocator, bm.containers[cursor.*]);
                cursor.* += 1;
                return cloned;
            }
        }

        const acc = try BitsetContainer.init(allocator);
        errdefer acc.deinit(allocator);

        for (bitmaps, cursors) |bm, *cursor| {
            if (cursor.* >= bm.size or bm.keys[cursor.*] != key) continue;
            lazyAccumulateIntoBitset(op, acc, Container.fromTagged(bm.containers[cursor.*]));
            cursor.* += 1;
        }

        return TaggedPtr.initBitset(acc);
    }

    fn lazyMergeTwo(
        comptime op: ManyOp,
        allocator: std.mem.Allocator,
        a: *const Self,
        b: *const Self,
        bitset_conversion: bool,
        mode: LazyConstructionMode,
    ) !Self {
        const max_result_size = @min(a.size + b.size, @as(u32, 1) << 16);
        var result = try Self.initCapacity(allocator, max_result_size);
        errdefer result.deinit();

        if (op != .bor or mode == .baseline) {
            try lazyMergeTwoBaselineInto(op, allocator, a, b, bitset_conversion, &result);
            return result;
        }

        const eligible_count = lazyBitsetPairCount(a, b, bitset_conversion);
        const pending = allocator.alloc(PendingBitset, eligible_count) catch {
            try lazyMergeTwoBaselineInto(op, allocator, a, b, bitset_conversion, &result);
            return result;
        };
        defer allocator.free(pending);

        var initialized_count: usize = 0;
        var transferred_count: usize = 0;
        errdefer {
            for (pending[transferred_count..initialized_count]) |entry| entry.header.deinit(allocator);
        }

        for (pending) |*entry| {
            const header = try initPendingBitset(allocator);
            entry.* = .{
                .payload_addr = @intFromPtr(header.words),
                .header = header,
            };
            initialized_count += 1;
        }

        if (mode == .batched_sorted) {
            std.mem.sortUnstable(PendingBitset, pending, {}, pendingBitsetLessThan);
        }
        for (pending) |entry| @memset(entry.header.words, 0);

        try lazyMergeTwoBatchedInto(
            allocator,
            a,
            b,
            bitset_conversion,
            &result,
            pending,
            &transferred_count,
        );
        if (transferred_count != pending.len) return error.LazyConstructionCountMismatch;
        result.cached_cardinality = -1;
        return result;
    }

    fn lazyMergeTwoBaselineInto(
        comptime op: ManyOp,
        allocator: std.mem.Allocator,
        a: *const Self,
        b: *const Self,
        bitset_conversion: bool,
        result: *Self,
    ) !void {
        var i: usize = 0;
        var j: usize = 0;

        while (i < a.size and j < b.size) {
            const key_a = a.keys[i];
            const key_b = b.keys[j];

            if (key_a < key_b) {
                try result.appendClonedContainer(allocator, key_a, a.containers[i]);
                i += 1;
            } else if (key_a > key_b) {
                try result.appendClonedContainer(allocator, key_b, b.containers[j]);
                j += 1;
            } else {
                const c_a = Container.fromTagged(a.containers[i]);
                const c_b = Container.fromTagged(b.containers[j]);
                const use_lazy_bitset = op == .xor or bitset_conversion or isBitsetContainer(c_a) or isBitsetContainer(c_b);

                if (use_lazy_bitset) {
                    const acc = try BitsetContainer.init(allocator);
                    lazyAccumulateIntoBitset(op, acc, c_a);
                    lazyAccumulateIntoBitset(op, acc, c_b);
                    try result.appendOwnedContainer(allocator, key_a, TaggedPtr.initBitset(acc));
                } else {
                    const merged = switch (op) {
                        .bor => try ops.containerUnion(allocator, c_a, c_b),
                        .xor => try ops.containerXor(allocator, c_a, c_b),
                    };
                    if (op == .xor and merged.getCardinality() == 0) {
                        merged.deinit(allocator);
                    } else {
                        try result.appendOwnedContainer(allocator, key_a, merged.toTagged());
                    }
                }

                i += 1;
                j += 1;
            }
        }

        while (i < a.size) : (i += 1) {
            try result.appendClonedContainer(allocator, a.keys[i], a.containers[i]);
        }
        while (j < b.size) : (j += 1) {
            try result.appendClonedContainer(allocator, b.keys[j], b.containers[j]);
        }

        result.cached_cardinality = -1;
    }

    fn lazyBitsetPairCount(a: *const Self, b: *const Self, bitset_conversion: bool) usize {
        var count: usize = 0;
        var i: usize = 0;
        var j: usize = 0;
        while (i < a.size and j < b.size) {
            const key_a = a.keys[i];
            const key_b = b.keys[j];
            if (key_a < key_b) {
                i += 1;
            } else if (key_a > key_b) {
                j += 1;
            } else {
                const c_a = Container.fromTagged(a.containers[i]);
                const c_b = Container.fromTagged(b.containers[j]);
                if (bitset_conversion or isBitsetContainer(c_a) or isBitsetContainer(c_b)) count += 1;
                i += 1;
                j += 1;
            }
        }
        return count;
    }

    fn lazyMergeTwoBatchedInto(
        allocator: std.mem.Allocator,
        a: *const Self,
        b: *const Self,
        bitset_conversion: bool,
        result: *Self,
        pending: []PendingBitset,
        transferred_count: *usize,
    ) !void {
        var i: usize = 0;
        var j: usize = 0;
        while (i < a.size and j < b.size) {
            const key_a = a.keys[i];
            const key_b = b.keys[j];
            if (key_a < key_b) {
                try result.appendClonedContainer(allocator, key_a, a.containers[i]);
                i += 1;
            } else if (key_a > key_b) {
                try result.appendClonedContainer(allocator, key_b, b.containers[j]);
                j += 1;
            } else {
                const c_a = Container.fromTagged(a.containers[i]);
                const c_b = Container.fromTagged(b.containers[j]);
                const use_lazy_bitset = bitset_conversion or isBitsetContainer(c_a) or isBitsetContainer(c_b);
                if (use_lazy_bitset) {
                    if (transferred_count.* >= pending.len) return error.LazyConstructionCountMismatch;
                    const acc = pending[transferred_count.*].header;
                    lazyAccumulateIntoBitset(.bor, acc, c_a);
                    lazyAccumulateIntoBitset(.bor, acc, c_b);
                    transferred_count.* += 1;
                    try result.appendOwnedContainer(allocator, key_a, TaggedPtr.initBitset(acc));
                } else {
                    const merged = try ops.containerUnion(allocator, c_a, c_b);
                    try result.appendOwnedContainer(allocator, key_a, merged.toTagged());
                }
                i += 1;
                j += 1;
            }
        }

        while (i < a.size) : (i += 1) {
            try result.appendClonedContainer(allocator, a.keys[i], a.containers[i]);
        }
        while (j < b.size) : (j += 1) {
            try result.appendClonedContainer(allocator, b.keys[j], b.containers[j]);
        }
    }

    fn pendingBitsetLessThan(_: void, lhs: PendingBitset, rhs: PendingBitset) bool {
        return lhs.payload_addr < rhs.payload_addr;
    }

    fn isBitsetContainer(container: Container) bool {
        return switch (container) {
            .bitset => true,
            .array, .run => false,
            .reserved => unreachable,
        };
    }

    fn lazyAccumulateIntoBitset(comptime op: ManyOp, acc: *BitsetContainer, container: Container) void {
        switch (container) {
            .array => |ac| switch (op) {
                .bor => acc.setList(ac.values[0..ac.cardinality]),
                .xor => {
                    for (ac.values[0..ac.cardinality]) |value| {
                        acc.lazyToggle(value);
                    }
                },
            },
            .bitset => |bc| switch (op) {
                .bor => acc.lazyUnionWith(bc),
                .xor => acc.lazyXorWith(bc),
            },
            .run => |rc| {
                for (rc.runs[0..rc.n_runs]) |run| {
                    switch (op) {
                        .bor => acc.setRange(run.start, run.end()),
                        .xor => acc.lazyToggleRange(run.start, run.end()),
                    }
                }
                acc.cardinality = -1;
            },
            .reserved => unreachable,
        }
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

    /// Deserialize and validate semantic invariants. Use for untrusted input.
    pub fn deserializeSafe(allocator: std.mem.Allocator, data: []const u8) !Self {
        return ser.deserializeSafe(allocator, data);
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

    /// Deserialize and validate using arena allocation.
    pub fn deserializeSafeOwned(backing: std.mem.Allocator, data: []const u8) !OwnedBitmap {
        var arena = std.heap.ArenaAllocator.init(backing);
        errdefer arena.deinit();
        const bm = try Self.deserializeSafe(arena.allocator(), data);
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

    /// Flip a range using arena allocation.
    pub fn flipOwned(self: *const Self, backing: std.mem.Allocator, lo: u32, hi: u32) !OwnedBitmap {
        var arena = std.heap.ArenaAllocator.init(backing);
        errdefer arena.deinit();
        const result = try self.flip(arena.allocator(), lo, hi);
        return .{ .bitmap = result, .arena = arena };
    }

    /// Build from arbitrary slice using arena allocation. Sorts and deduplicates in-place.
    pub fn fromSliceOwned(backing: std.mem.Allocator, values: []u32) !OwnedBitmap {
        var arena = std.heap.ArenaAllocator.init(backing);
        errdefer arena.deinit();
        const result = try fromSlice(arena.allocator(), values);
        return .{ .bitmap = result, .arena = arena };
    }

    /// Compute n-way union using arena allocation.
    pub fn orManyOwned(backing: std.mem.Allocator, bitmaps: []const *const Self) !OwnedBitmap {
        var arena = std.heap.ArenaAllocator.init(backing);
        errdefer arena.deinit();
        const result = try orMany(arena.allocator(), bitmaps);
        return .{ .bitmap = result, .arena = arena };
    }

    /// Compute n-way symmetric difference using arena allocation.
    pub fn xorManyOwned(backing: std.mem.Allocator, bitmaps: []const *const Self) !OwnedBitmap {
        var arena = std.heap.ArenaAllocator.init(backing);
        errdefer arena.deinit();
        const result = try xorMany(arena.allocator(), bitmaps);
        return .{ .bitmap = result, .arena = arena };
    }

    // =========================================================================
    // Allocator guidance
    // =========================================================================

    /// ## Allocator guidance
    ///
    /// Allocator effects are operation-dependent. On measured container-heavy
    /// operations, `std.heap.c_allocator` was roughly 1.3-1.8x slower than
    /// alternatives, while some allocation-heavy operations favored libc.
    /// See `docs/parity-measurement.md` for the current measurements.
    ///
    /// Recommended:
    /// - `OwnedBitmap` API: fastest (uses optimized allocation internally)
    /// - `std.heap.smp_allocator`: fast general-purpose, supports mutation
    /// - `std.heap.ArenaAllocator`: fast batch alloc, bulk free only
    pub const allocator_guidance = void;
};

fn initPendingBitset(allocator: std.mem.Allocator) !*BitsetContainer {
    const header = try allocator.create(BitsetContainer);
    errdefer allocator.destroy(header);

    const words = try allocator.alignedAlloc(u64, .@"64", BitsetContainer.NUM_WORDS);
    header.* = .{
        .words = words[0..BitsetContainer.NUM_WORDS],
        .cardinality = 0,
    };
    return header;
}

/// Internal benchmark dispatch for lazy-OR construction experiments.
pub const lazy_construction = struct {
    pub fn baseline(
        left: *const RoaringBitmap,
        allocator: std.mem.Allocator,
        right: *const RoaringBitmap,
        bitset_conversion: bool,
    ) !RoaringBitmap {
        return RoaringBitmap.lazyMergeTwo(.bor, allocator, left, right, bitset_conversion, .baseline);
    }

    pub fn batchedUnsorted(
        left: *const RoaringBitmap,
        allocator: std.mem.Allocator,
        right: *const RoaringBitmap,
        bitset_conversion: bool,
    ) !RoaringBitmap {
        return RoaringBitmap.lazyMergeTwo(.bor, allocator, left, right, bitset_conversion, .batched_unsorted);
    }

    pub fn batchedSorted(
        left: *const RoaringBitmap,
        allocator: std.mem.Allocator,
        right: *const RoaringBitmap,
        bitset_conversion: bool,
    ) !RoaringBitmap {
        return RoaringBitmap.lazyMergeTwo(.bor, allocator, left, right, bitset_conversion, .batched_sorted);
    }
};

test "RoaringBitmap capacity growth saturates" {
    try std.testing.expectEqual(@as(u32, 8), RoaringBitmap.grownCapacity(4, 5));
    try std.testing.expectEqual(
        std.math.maxInt(u32),
        RoaringBitmap.grownCapacity(std.math.maxInt(u32) - 1, 1),
    );
}

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

    /// Borrow the underlying bitmap for read-only queries.
    ///
    /// Use this to access the full `RoaringBitmap` read-only surface, such as
    /// `minimum`, `maximum`, `equals`, `rank`, `select`, and cardinality variants.
    pub fn asBitmap(self: *const OwnedBitmap) *const RoaringBitmap {
        return &self.bitmap;
    }

    /// Return the number of values in the bitmap.
    pub fn cardinality(self: *const OwnedBitmap) u64 {
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

test {
    _ = @import("bitmap_tests.zig");
    _ = @import("range_strategy_tests.zig");
}
