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
        // For each run in a, check it's covered by runs in b
        for (a.runs[0..a.n_runs]) |run| {
            var v: u32 = run.start;
            while (v <= run.end()) : (v += 1) {
                if (!b.contains(@intCast(v))) return false;
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
        // Same values can have different run encodings (e.g., {[0,5],[6,10]} vs {[0,10]})
        // Cardinality already checked equal, so just verify each value in a is in b
        for (a.runs[0..a.n_runs]) |run| {
            var v: u32 = run.start;
            while (v <= run.end()) : (v += 1) {
                if (!b.contains(@intCast(v))) return false;
            }
        }
        return true;
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
