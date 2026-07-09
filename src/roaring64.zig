const std = @import("std");
const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;
const ser = @import("serialize.zig");
const test_support = @import("roaring64_test_support.zig");

/// A 64-bit Roaring bitmap layered over sorted 32-bit high-key buckets.
pub const Roaring64Bitmap = struct {
    const Self = @This();
    const INITIAL_CAPACITY: u32 = 4;

    const Bucket = struct {
        hi: u32,
        bm: RoaringBitmap,
    };

    /// Sorted array of high-32-bit buckets.
    buckets: []Bucket,

    /// Number of active buckets.
    size: u32,

    /// Allocated bucket capacity.
    capacity: u32,

    /// Allocator for all internal memory.
    allocator: std.mem.Allocator,

    /// Cached total cardinality. null = unknown/invalid.
    cached_cardinality: ?u64 = 0,

    const FindBucketResult = struct {
        bucket: *Bucket,
        idx: usize,
        created: bool,
    };

    pub fn init(allocator: std.mem.Allocator) !Self {
        const buckets = try allocator.alloc(Bucket, INITIAL_CAPACITY);

        return .{
            .buckets = buckets,
            .size = 0,
            .capacity = INITIAL_CAPACITY,
            .allocator = allocator,
            .cached_cardinality = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.buckets[0..self.size]) |*bucket| {
            bucket.bm.deinit();
        }
        self.allocator.free(self.buckets[0..self.capacity]);
    }

    /// Create a deep copy of the bitmap.
    pub fn clone(self: *const Self, allocator: std.mem.Allocator) !Self {
        var result = try Self.init(allocator);
        errdefer result.deinit();

        try result.ensureCapacity(self.size);

        for (self.buckets[0..self.size]) |*bucket| {
            const cloned = try bucket.bm.clone(allocator);
            result.buckets[result.size] = .{
                .hi = bucket.hi,
                .bm = cloned,
            };
            result.size += 1;
        }
        result.cached_cardinality = self.cached_cardinality;

        return result;
    }

    pub fn isEmpty(self: *const Self) bool {
        return self.size == 0;
    }

    pub fn cardinality(self: *const Self) u64 {
        if (self.cached_cardinality) |cached| return cached;

        var total: u64 = 0;
        for (self.buckets[0..self.size]) |*bucket| {
            total += bucket.bm.cardinality();
        }
        return total;
    }

    /// Add a value. Returns true if the value was newly added.
    pub fn add(self: *Self, value: u64) !bool {
        const hi = highBits(value);
        const lo = lowBits(value);

        const found = try self.findOrCreateBucket(hi);
        errdefer if (found.created and found.bucket.bm.isEmpty()) self.dropBucket(found.idx);

        const added = try found.bucket.bm.add(lo);
        if (added) {
            if (self.cached_cardinality) |cached| {
                self.cached_cardinality = cached + 1;
            }
        }
        return added;
    }

    /// Add all values in the slice. Values need not be sorted.
    pub fn addMany(self: *Self, values: []const u64) !void {
        for (values) |value| {
            _ = try self.add(value);
        }
    }

    /// Check if a value is present.
    pub fn contains(self: *const Self, value: u64) bool {
        const idx = self.bucketIndex(highBits(value)) orelse return false;
        return self.buckets[idx].bm.contains(lowBits(value));
    }

    /// Remove a value. Returns true if the value was present.
    pub fn remove(self: *Self, value: u64) !bool {
        const idx = self.bucketIndex(highBits(value)) orelse return false;

        const removed = try self.buckets[idx].bm.remove(lowBits(value));
        if (!removed) return false;

        if (self.cached_cardinality) |cached| {
            self.cached_cardinality = cached - 1;
        }
        if (self.buckets[idx].bm.isEmpty()) {
            self.dropBucket(idx);
        }
        return true;
    }

    /// Get the minimum value, or null if empty.
    pub fn minimum(self: *const Self) ?u64 {
        if (self.size == 0) return null;

        const bucket = &self.buckets[0];
        const low = bucket.bm.minimum() orelse return null;
        return combine(bucket.hi, low);
    }

    /// Get the maximum value, or null if empty.
    pub fn maximum(self: *const Self) ?u64 {
        if (self.size == 0) return null;

        const bucket = &self.buckets[self.size - 1];
        const low = bucket.bm.maximum() orelse return null;
        return combine(bucket.hi, low);
    }

    /// Allocate and return all values in ascending order.
    pub fn toArrayAlloc(self: *const Self, allocator: std.mem.Allocator) ![]u64 {
        const total = self.cardinality();
        if (total > std.math.maxInt(usize)) return error.Overflow;

        const len: usize = @intCast(total);
        _ = std.math.mul(usize, len, @sizeOf(u64)) catch return error.Overflow;

        const values = try allocator.alloc(u64, len);
        errdefer allocator.free(values);
        const written = self.toArray(values);
        std.debug.assert(written == values.len);
        return values;
    }

    /// Fill a caller-provided slice with values in ascending order.
    pub fn toArray(self: *const Self, out: []u64) usize {
        var written: usize = 0;
        for (self.buckets[0..self.size]) |*bucket| {
            const high = @as(u64, bucket.hi) << 32;
            var it = bucket.bm.iterator();
            while (it.next()) |low| {
                if (written == out.len) return written;
                out[written] = high | low;
                written += 1;
            }
        }
        return written;
    }

    /// Iterator over all values in ascending order.
    pub const Iterator = struct {
        bm: *const Self,
        bucket_idx: u32,
        inner: ?RoaringBitmap.Iterator,

        pub fn next(self: *Iterator) ?u64 {
            while (self.bucket_idx < self.bm.size) {
                if (self.inner) |*it| {
                    if (it.next()) |low| {
                        return combine(self.bm.buckets[self.bucket_idx].hi, low);
                    }
                }

                self.bucket_idx += 1;
                self.inner = if (self.bucket_idx < self.bm.size)
                    self.bm.buckets[self.bucket_idx].bm.iterator()
                else
                    null;
            }
            return null;
        }
    };

    /// Returns an iterator over all values in ascending order.
    pub fn iterator(self: *const Self) Iterator {
        var it = Iterator{
            .bm = self,
            .bucket_idx = 0,
            .inner = null,
        };
        if (self.size > 0) {
            it.inner = self.buckets[0].bm.iterator();
        }
        return it;
    }

    /// Return a new bitmap that is the union (OR) of self and other.
    pub fn bitwiseOr(self: *const Self, allocator: std.mem.Allocator, other: *const Self) !Self {
        return self.twoWayAllocatingMerge(.bor, allocator, other);
    }

    /// Return a new bitmap that is the intersection (AND) of self and other.
    pub fn bitwiseAnd(self: *const Self, allocator: std.mem.Allocator, other: *const Self) !Self {
        return self.twoWayAllocatingMerge(.band, allocator, other);
    }

    /// Return a new bitmap that is the symmetric difference (XOR) of self and other.
    pub fn bitwiseXor(self: *const Self, allocator: std.mem.Allocator, other: *const Self) !Self {
        return self.twoWayAllocatingMerge(.xor, allocator, other);
    }

    /// Return a new bitmap that is the difference (AND NOT) of self and other.
    pub fn bitwiseDifference(self: *const Self, allocator: std.mem.Allocator, other: *const Self) !Self {
        return self.twoWayAllocatingMerge(.andnot, allocator, other);
    }

    /// Compute |self ∩ other| without allocating a result bitmap.
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

    /// Return true if self and other have any values in common.
    pub fn intersects(self: *const Self, other: *const Self) bool {
        var i: usize = 0;
        var j: usize = 0;

        while (i < self.size and j < other.size) {
            const hi_a = self.buckets[i].hi;
            const hi_b = other.buckets[j].hi;

            if (hi_a < hi_b) {
                i += 1;
            } else if (hi_a > hi_b) {
                j += 1;
            } else {
                if (self.buckets[i].bm.intersects(&other.buckets[j].bm)) return true;
                i += 1;
                j += 1;
            }
        }
        return false;
    }

    /// Check if self is a subset of other.
    pub fn isSubsetOf(self: *const Self, other: *const Self) bool {
        var i: usize = 0;
        var j: usize = 0;

        while (i < self.size) {
            if (j >= other.size) return false;

            const hi_a = self.buckets[i].hi;
            const hi_b = other.buckets[j].hi;

            if (hi_a < hi_b) {
                return false;
            } else if (hi_a > hi_b) {
                j += 1;
            } else {
                if (!self.buckets[i].bm.isSubsetOf(&other.buckets[j].bm)) return false;
                i += 1;
                j += 1;
            }
        }
        return true;
    }

    /// Check if self is a proper subset of other.
    pub fn isStrictSubsetOf(self: *const Self, other: *const Self) bool {
        return self.isSubsetOf(other) and self.cardinality() < other.cardinality();
    }

    /// Check if two bitmaps are equal.
    pub fn equals(self: *const Self, other: *const Self) bool {
        if (self.size != other.size) return false;
        for (self.buckets[0..self.size], other.buckets[0..other.size]) |*a, *b| {
            if (a.hi != b.hi) return false;
            if (!a.bm.equals(&b.bm)) return false;
        }
        return true;
    }

    /// In-place union: self |= other.
    pub fn bitwiseOrInPlace(self: *Self, other: *const Self) !void {
        var result = try self.bitwiseOr(self.allocator, other);
        self.swapWithResult(&result);
    }

    /// In-place intersection: self &= other.
    pub fn bitwiseAndInPlace(self: *Self, other: *const Self) !void {
        var result = try self.bitwiseAnd(self.allocator, other);
        self.swapWithResult(&result);
    }

    /// In-place symmetric difference: self ^= other.
    pub fn bitwiseXorInPlace(self: *Self, other: *const Self) !void {
        var result = try self.bitwiseXor(self.allocator, other);
        self.swapWithResult(&result);
    }

    /// In-place difference: self -= other.
    pub fn bitwiseDifferenceInPlace(self: *Self, other: *const Self) !void {
        var result = try self.bitwiseDifference(self.allocator, other);
        self.swapWithResult(&result);
    }

    fn swapWithResult(self: *Self, result: *Self) void {
        var old = self.*;
        self.* = result.*;
        old.deinit();
    }

    /// Count values <= `value`.
    pub fn rank(self: *const Self, value: u64) u64 {
        const target_hi = highBits(value);
        const target_low = lowBits(value);

        var total: u64 = 0;
        for (self.buckets[0..self.size]) |*bucket| {
            if (bucket.hi < target_hi) {
                total += bucket.bm.cardinality();
            } else if (bucket.hi == target_hi) {
                return total + bucket.bm.rank(target_low);
            } else {
                return total;
            }
        }
        return total;
    }

    /// Return the 0-based position of `value`, or null if absent.
    pub fn getIndex(self: *const Self, value: u64) ?u64 {
        const target_hi = highBits(value);
        const target_low = lowBits(value);

        var total: u64 = 0;
        for (self.buckets[0..self.size]) |*bucket| {
            if (bucket.hi < target_hi) {
                total += bucket.bm.cardinality();
            } else if (bucket.hi == target_hi) {
                const local = bucket.bm.getIndex(target_low) orelse return null;
                return total + local;
            } else {
                return null;
            }
        }
        return null;
    }

    /// Return the k-th smallest value, 0-based, or null if out of range.
    pub fn select(self: *const Self, k: u64) ?u64 {
        var prior: u64 = 0;
        for (self.buckets[0..self.size]) |*bucket| {
            const card = bucket.bm.cardinality();
            if (k >= prior and k - prior < card) {
                const low = bucket.bm.select(k - prior) orelse return null;
                return combine(bucket.hi, low);
            }
            prior += card;
        }
        return null;
    }

    /// Add all values in the inclusive range [lo, hi].
    /// Unlike the 32-bit API, this returns no added-count because a 64-bit range
    /// can contain more values than fit in u64.
    pub fn addRange(self: *Self, lo: u64, hi: u64) !void {
        if (lo > hi) return;

        const start_hi = highBits(lo);
        const end_hi = highBits(hi);
        const key_count = @as(u64, end_hi) - start_hi + 1;
        if (key_count > std.math.maxInt(usize)) return error.Overflow;

        const created_keys = try self.allocator.alloc(u32, @intCast(key_count));
        defer self.allocator.free(created_keys);
        var created_len: usize = 0;
        errdefer self.dropCreatedBuckets(created_keys[0..created_len]);

        self.cached_cardinality = null;

        const end_hi_u64: u64 = end_hi;
        var key: u64 = start_hi;
        while (key <= end_hi_u64) {
            const key_u32: u32 = @intCast(key);
            const start_low: u32 = if (key_u32 == start_hi) lowBits(lo) else 0;
            const end_low: u32 = if (key_u32 == end_hi) lowBits(hi) else std.math.maxInt(u32);

            const found = try self.findOrCreateBucket(key_u32);
            if (found.created) {
                created_keys[created_len] = key_u32;
                created_len += 1;
            }

            _ = try found.bucket.bm.addRange(start_low, end_low);

            if (key == end_hi_u64) break;
            key += 1;
        }
    }

    /// Remove all values in the inclusive range [lo, hi].
    /// Unlike the 32-bit API, this returns no removed-count because a 64-bit
    /// range can contain more values than fit in u64.
    pub fn removeRange(self: *Self, lo: u64, hi: u64) !void {
        if (lo > hi or self.size == 0) return;

        const start_hi = highBits(lo);
        const end_hi = highBits(hi);
        const end_hi_u64: u64 = end_hi;
        var idx = self.lowerBound(start_hi);

        self.cached_cardinality = null;

        while (idx < self.size and @as(u64, self.buckets[idx].hi) <= end_hi_u64) {
            const bucket = &self.buckets[idx];
            const start_low: u32 = if (bucket.hi == start_hi) lowBits(lo) else 0;
            const end_low: u32 = if (bucket.hi == end_hi) lowBits(hi) else std.math.maxInt(u32);

            if (start_low == 0 and end_low == std.math.maxInt(u32)) {
                self.dropBucket(idx);
                continue;
            }

            _ = try bucket.bm.removeRange(start_low, end_low);
            if (bucket.bm.isEmpty()) {
                self.dropBucket(idx);
            } else {
                idx += 1;
            }
        }
    }

    /// Count values in the inclusive range [lo, hi].
    pub fn rangeCardinality(self: *const Self, lo: u64, hi: u64) u64 {
        if (lo > hi) return 0;

        const start_hi = highBits(lo);
        const end_hi = highBits(hi);
        const end_hi_u64: u64 = end_hi;
        var idx = self.lowerBound(start_hi);
        var total: u64 = 0;

        while (idx < self.size and @as(u64, self.buckets[idx].hi) <= end_hi_u64) : (idx += 1) {
            const bucket = &self.buckets[idx];
            const start_low: u32 = if (bucket.hi == start_hi) lowBits(lo) else 0;
            const end_low: u32 = if (bucket.hi == end_hi) lowBits(hi) else std.math.maxInt(u32);

            if (start_low == 0 and end_low == std.math.maxInt(u32)) {
                total += bucket.bm.cardinality();
            } else {
                total += bucket.bm.rangeCardinality(start_low, end_low);
            }
        }

        return total;
    }

    /// Return whether every value in the inclusive range [lo, hi] is present.
    pub fn containsRange(self: *const Self, lo: u64, hi: u64) bool {
        if (lo > hi) return true;

        const start_hi = highBits(lo);
        const end_hi = highBits(hi);
        const end_hi_u64: u64 = end_hi;
        var idx = self.lowerBound(start_hi);
        var key: u64 = start_hi;

        while (key <= end_hi_u64) {
            if (idx >= self.size or @as(u64, self.buckets[idx].hi) != key) return false;

            const bucket = &self.buckets[idx];
            const start_low: u32 = if (bucket.hi == start_hi) lowBits(lo) else 0;
            const end_low: u32 = if (bucket.hi == end_hi) lowBits(hi) else std.math.maxInt(u32);
            if (!bucket.bm.containsRange(start_low, end_low)) return false;

            idx += 1;
            if (key == end_hi_u64) break;
            key += 1;
        }

        return true;
    }

    /// Compute serialized size in bytes for CRoaring's portable 64-bit format.
    pub fn serializedSizeInBytes(self: *const Self) !usize {
        var size: usize = 8;
        for (self.buckets[0..self.size]) |*bucket| {
            size = std.math.add(usize, size, 4) catch return error.Overflow;
            size = std.math.add(usize, size, bucket.bm.serializedSizeInBytes()) catch return error.Overflow;
        }
        return size;
    }

    /// Serialize to a byte slice in CRoaring's portable 64-bit format.
    pub fn serialize(self: *const Self, allocator: std.mem.Allocator) ![]u8 {
        const size_bytes = try self.serializedSizeInBytes();
        const buf = try allocator.alloc(u8, size_bytes);
        errdefer allocator.free(buf);

        var writer = std.Io.Writer.fixed(buf);
        try self.serializeToWriter(&writer);
        return buf;
    }

    /// Serialize to any writer in CRoaring's portable 64-bit format.
    pub fn serializeToWriter(self: *const Self, writer: anytype) !void {
        try writer.writeInt(u64, self.size, .little);
        for (self.buckets[0..self.size]) |*bucket| {
            try writer.writeInt(u32, bucket.hi, .little);
            try bucket.bm.serializeToWriter(writer);
        }
    }

    /// Deserialize a bitmap from CRoaring's portable 64-bit format.
    pub fn deserialize(allocator: std.mem.Allocator, data: []const u8) !Self {
        return deserializeWithMode(allocator, data, .plain);
    }

    /// Deserialize and validate embedded 32-bit bitmap invariants.
    pub fn deserializeSafe(allocator: std.mem.Allocator, data: []const u8) !Self {
        return deserializeWithMode(allocator, data, .safe);
    }

    const SetOp = enum { bor, band, xor, andnot };
    const DeserializeMode = enum { plain, safe };

    fn deserializeWithMode(allocator: std.mem.Allocator, data: []const u8, comptime mode: DeserializeMode) !Self {
        if (data.len < 8) return error.InvalidFormat;

        const bucket_count = std.mem.readInt(u64, data[0..8], .little);
        if (bucket_count > std.math.maxInt(u32)) return error.InvalidFormat;

        var result = try Self.init(allocator);
        errdefer result.deinit();

        try result.ensureCapacity(@intCast(bucket_count));

        var offset: usize = 8;
        var total: u64 = 0;
        var previous_hi: ?u32 = null;

        for (0..@as(usize, @intCast(bucket_count))) |_| {
            const key_end = std.math.add(usize, offset, 4) catch return error.InvalidFormat;
            if (key_end > data.len) return error.InvalidFormat;

            const hi = std.mem.readInt(u32, data[offset..][0..4], .little);
            offset = key_end;
            if (previous_hi) |prev| {
                if (hi <= prev) return error.InvalidFormat;
            }
            previous_hi = hi;

            const sub_len = try ser.portableSizeInBytes(data[offset..]);
            const sub_end = std.math.add(usize, offset, sub_len) catch return error.InvalidFormat;
            if (sub_end > data.len) return error.InvalidFormat;

            var sub = switch (mode) {
                .plain => try RoaringBitmap.deserialize(allocator, data[offset..sub_end]),
                .safe => try RoaringBitmap.deserializeSafe(allocator, data[offset..sub_end]),
            };
            if (sub.isEmpty()) {
                sub.deinit();
                return error.InvalidFormat;
            }

            const card = sub.cardinality();
            try result.appendOwnedBucket(hi, sub);
            total = std.math.add(u64, total, card) catch return error.Overflow;
            offset = sub_end;
        }

        if (offset != data.len) return error.InvalidFormat;
        result.cached_cardinality = total;
        return result;
    }

    fn twoWayAllocatingMerge(self: *const Self, comptime op: SetOp, allocator: std.mem.Allocator, other: *const Self) !Self {
        var result = try Self.init(allocator);
        errdefer result.deinit();

        var total: u64 = 0;
        var i: usize = 0;
        var j: usize = 0;

        while (i < self.size and j < other.size) {
            const hi_a = self.buckets[i].hi;
            const hi_b = other.buckets[j].hi;

            if (hi_a < hi_b) {
                total += try appendLeftOnly(op, &result, &self.buckets[i]);
                i += 1;
            } else if (hi_a > hi_b) {
                total += try appendRightOnly(op, &result, &other.buckets[j]);
                j += 1;
            } else {
                total += try appendBoth(op, &result, &self.buckets[i], &other.buckets[j]);
                i += 1;
                j += 1;
            }
        }

        while (i < self.size) : (i += 1) {
            total += try appendLeftOnly(op, &result, &self.buckets[i]);
        }
        while (j < other.size) : (j += 1) {
            total += try appendRightOnly(op, &result, &other.buckets[j]);
        }

        result.cached_cardinality = total;
        return result;
    }

    fn appendLeftOnly(comptime op: SetOp, result: *Self, bucket: *const Bucket) !u64 {
        return switch (op) {
            .bor, .xor, .andnot => try result.appendClonedBucket(bucket),
            .band => 0,
        };
    }

    fn appendRightOnly(comptime op: SetOp, result: *Self, bucket: *const Bucket) !u64 {
        return switch (op) {
            .bor, .xor => try result.appendClonedBucket(bucket),
            .band, .andnot => 0,
        };
    }

    fn appendBoth(comptime op: SetOp, result: *Self, a: *const Bucket, b: *const Bucket) !u64 {
        var merged = switch (op) {
            .bor => try a.bm.bitwiseOr(result.allocator, &b.bm),
            .band => try a.bm.bitwiseAnd(result.allocator, &b.bm),
            .xor => try a.bm.bitwiseXor(result.allocator, &b.bm),
            .andnot => try a.bm.bitwiseDifference(result.allocator, &b.bm),
        };

        const card = merged.cardinality();
        if (card == 0) {
            merged.deinit();
            return 0;
        }

        try result.appendOwnedBucket(a.hi, merged);
        return card;
    }

    fn twoWayCardinality(self: *const Self, comptime op: SetOp, other: *const Self) u64 {
        var total: u64 = 0;
        var i: usize = 0;
        var j: usize = 0;

        while (i < self.size and j < other.size) {
            const hi_a = self.buckets[i].hi;
            const hi_b = other.buckets[j].hi;

            if (hi_a < hi_b) {
                total += leftOnlyCardinality(op, &self.buckets[i].bm);
                i += 1;
            } else if (hi_a > hi_b) {
                total += rightOnlyCardinality(op, &other.buckets[j].bm);
                j += 1;
            } else {
                total += bothCardinality(op, &self.buckets[i].bm, &other.buckets[j].bm);
                i += 1;
                j += 1;
            }
        }

        while (i < self.size) : (i += 1) {
            total += leftOnlyCardinality(op, &self.buckets[i].bm);
        }
        while (j < other.size) : (j += 1) {
            total += rightOnlyCardinality(op, &other.buckets[j].bm);
        }

        return total;
    }

    fn leftOnlyCardinality(comptime op: SetOp, bm: *const RoaringBitmap) u64 {
        return switch (op) {
            .bor, .xor, .andnot => bm.cardinality(),
            .band => 0,
        };
    }

    fn rightOnlyCardinality(comptime op: SetOp, bm: *const RoaringBitmap) u64 {
        return switch (op) {
            .bor, .xor => bm.cardinality(),
            .band, .andnot => 0,
        };
    }

    fn bothCardinality(comptime op: SetOp, a: *const RoaringBitmap, b: *const RoaringBitmap) u64 {
        return switch (op) {
            .bor => a.orCardinality(b),
            .band => a.andCardinality(b),
            .xor => a.xorCardinality(b),
            .andnot => a.differenceCardinality(b),
        };
    }

    fn appendClonedBucket(self: *Self, bucket: *const Bucket) !u64 {
        const cloned = try bucket.bm.clone(self.allocator);
        const card = cloned.cardinality();
        try self.appendOwnedBucket(bucket.hi, cloned);
        return card;
    }

    fn appendOwnedBucket(self: *Self, hi_key: u32, bm: RoaringBitmap) !void {
        var owned = bm;
        errdefer owned.deinit();

        try self.ensureCapacity(self.size + 1);
        self.buckets[self.size] = .{
            .hi = hi_key,
            .bm = owned,
        };
        self.size += 1;
    }

    inline fn highBits(value: u64) u32 {
        return @truncate(value >> 32);
    }

    inline fn lowBits(value: u64) u32 {
        return @truncate(value);
    }

    inline fn combine(hi: u32, lo: u32) u64 {
        return (@as(u64, hi) << 32) | lo;
    }

    fn bucketIndex(self: *const Self, hi_key: u32) ?usize {
        const idx = self.lowerBound(hi_key);
        if (idx < self.size and self.buckets[idx].hi == hi_key) return idx;
        return null;
    }

    fn lowerBound(self: *const Self, hi_key: u32) usize {
        var lo: usize = 0;
        var hi: usize = self.size;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.buckets[mid].hi < hi_key) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    fn findOrCreateBucket(self: *Self, hi_key: u32) !FindBucketResult {
        const idx = self.lowerBound(hi_key);
        if (idx < self.size and self.buckets[idx].hi == hi_key) {
            return .{
                .bucket = &self.buckets[idx],
                .idx = idx,
                .created = false,
            };
        }

        try self.insertEmptyBucketAt(idx, hi_key);
        return .{
            .bucket = &self.buckets[idx],
            .idx = idx,
            .created = true,
        };
    }

    fn ensureCapacity(self: *Self, needed: u32) !void {
        if (needed <= self.capacity) return;

        const new_cap = @max(self.capacity * 2, needed);
        const new_buckets = try self.allocator.alloc(Bucket, new_cap);
        @memcpy(new_buckets[0..self.size], self.buckets[0..self.size]);
        self.allocator.free(self.buckets[0..self.capacity]);
        self.buckets = new_buckets;
        self.capacity = new_cap;
    }

    fn insertEmptyBucketAt(self: *Self, idx: usize, hi_key: u32) !void {
        try self.ensureCapacity(self.size + 1);

        var bm = try RoaringBitmap.init(self.allocator);
        errdefer bm.deinit();

        if (idx < self.size) {
            @memmove(self.buckets[idx + 1 .. self.size + 1], self.buckets[idx..self.size]);
        }

        self.buckets[idx] = .{
            .hi = hi_key,
            .bm = bm,
        };
        self.size += 1;
    }

    fn dropBucket(self: *Self, idx: usize) void {
        self.buckets[idx].bm.deinit();

        if (idx + 1 < self.size) {
            @memmove(self.buckets[idx .. self.size - 1], self.buckets[idx + 1 .. self.size]);
        }
        self.size -= 1;
    }

    fn dropCreatedBuckets(self: *Self, keys: []const u32) void {
        var i = keys.len;
        while (i > 0) {
            i -= 1;
            if (self.bucketIndex(keys[i])) |idx| {
                self.dropBucket(idx);
            }
        }
    }
};

test "Roaring64Bitmap init and deinit" {
    const allocator = std.testing.allocator;
    var bm = try Roaring64Bitmap.init(allocator);
    defer bm.deinit();

    try std.testing.expect(bm.isEmpty());
    try std.testing.expectEqual(@as(u64, 0), bm.cardinality());
}

test "Roaring64Bitmap add contains remove across high keys" {
    const allocator = std.testing.allocator;
    var bm = try Roaring64Bitmap.init(allocator);
    defer bm.deinit();

    const values = [_]u64{
        0,
        1,
        (@as(u64, 1) << 32),
        (@as(u64, 1) << 32) | 99,
        (@as(u64, 7) << 32) | std.math.maxInt(u32),
    };

    for (values) |value| {
        try std.testing.expect(try bm.add(value));
        try std.testing.expect(bm.contains(value));
    }
    try std.testing.expectEqual(@as(u64, values.len), bm.cardinality());
    try std.testing.expectEqual(@as(u32, 3), bm.size);

    try std.testing.expect(!try bm.add(values[2]));
    try std.testing.expectEqual(@as(u64, values.len), bm.cardinality());

    try std.testing.expect(try bm.remove(values[1]));
    try std.testing.expect(!bm.contains(values[1]));
    try std.testing.expectEqual(@as(u64, values.len - 1), bm.cardinality());
}

test "Roaring64Bitmap remove prunes empty bucket" {
    const allocator = std.testing.allocator;
    var bm = try Roaring64Bitmap.init(allocator);
    defer bm.deinit();

    const value = (@as(u64, 99) << 32) | 1234;
    try std.testing.expect(try bm.add(value));
    try std.testing.expectEqual(@as(u32, 1), bm.size);

    try std.testing.expect(try bm.remove(value));
    try std.testing.expectEqual(@as(u32, 0), bm.size);
    try std.testing.expect(bm.isEmpty());
    try std.testing.expectEqual(@as(u64, 0), bm.cardinality());
}

test "Roaring64Bitmap cardinality cache correctness" {
    const allocator = std.testing.allocator;
    var bm = try Roaring64Bitmap.init(allocator);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 0), bm.cardinality());
    try std.testing.expect(try bm.add(42));
    try std.testing.expectEqual(@as(u64, 1), bm.cardinality());
    try std.testing.expect(!try bm.add(42));
    try std.testing.expectEqual(@as(u64, 1), bm.cardinality());
    try std.testing.expect(try bm.add((@as(u64, 2) << 32) | 42));
    try std.testing.expectEqual(@as(u64, 2), bm.cardinality());
    try std.testing.expect(try bm.remove(42));
    try std.testing.expectEqual(@as(u64, 1), bm.cardinality());
}

test "Roaring64Bitmap min max across high keys" {
    const allocator = std.testing.allocator;
    var bm = try Roaring64Bitmap.init(allocator);
    defer bm.deinit();

    try std.testing.expectEqual(@as(?u64, null), bm.minimum());
    try std.testing.expectEqual(@as(?u64, null), bm.maximum());

    _ = try bm.add((@as(u64, 5) << 32) | 9);
    _ = try bm.add((@as(u64, 1) << 32) | 500);
    _ = try bm.add((@as(u64, 5) << 32) | 1);
    _ = try bm.add(std.math.maxInt(u64));

    try std.testing.expectEqual(@as(?u64, (@as(u64, 1) << 32) | 500), bm.minimum());
    try std.testing.expectEqual(@as(?u64, std.math.maxInt(u64)), bm.maximum());
}

test "Roaring64Bitmap iterator and toArray are ordered" {
    const allocator = std.testing.allocator;
    var bm = try Roaring64Bitmap.init(allocator);
    defer bm.deinit();

    const inserted = [_]u64{
        (@as(u64, 3) << 32) | 1,
        0,
        (@as(u64, 1) << 32) | 7,
        (@as(u64, 1) << 32) | 2,
        std.math.maxInt(u64),
    };
    for (inserted) |value| {
        _ = try bm.add(value);
    }

    const expected = [_]u64{
        0,
        (@as(u64, 1) << 32) | 2,
        (@as(u64, 1) << 32) | 7,
        (@as(u64, 3) << 32) | 1,
        std.math.maxInt(u64),
    };

    var iter_values: [expected.len]u64 = undefined;
    var it = bm.iterator();
    var i: usize = 0;
    while (it.next()) |value| : (i += 1) {
        iter_values[i] = value;
    }
    try std.testing.expectEqual(expected.len, i);
    try std.testing.expectEqualSlices(u64, &expected, &iter_values);

    var array_values: [expected.len]u64 = undefined;
    const written = bm.toArray(&array_values);
    try std.testing.expectEqual(expected.len, written);
    try std.testing.expectEqualSlices(u64, &expected, &array_values);

    var short: [3]u64 = undefined;
    const short_written = bm.toArray(&short);
    try std.testing.expectEqual(@as(usize, 3), short_written);
    try std.testing.expectEqualSlices(u64, expected[0..3], &short);
}

test "Roaring64Bitmap clone is independent" {
    const allocator = std.testing.allocator;
    var bm = try Roaring64Bitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(1);
    _ = try bm.add((@as(u64, 2) << 32) | 3);

    var cloned = try bm.clone(allocator);
    defer cloned.deinit();

    try std.testing.expectEqual(bm.cardinality(), cloned.cardinality());
    try std.testing.expect(cloned.contains(1));
    try std.testing.expect(try cloned.remove(1));
    try std.testing.expect(bm.contains(1));
    try std.testing.expect(!cloned.contains(1));
}

test "Roaring64Bitmap set ops out of place and in place" {
    const allocator = std.testing.allocator;
    const a_values = [_]u64{
        0,
        (@as(u64, 1) << 32) | 1,
        (@as(u64, 1) << 32) | 2,
        (@as(u64, 3) << 32) | 5,
        (@as(u64, 5) << 32) | 5,
    };
    const b_values = [_]u64{
        (@as(u64, 1) << 32) | 2,
        (@as(u64, 1) << 32) | 9,
        (@as(u64, 2) << 32) | 1,
        (@as(u64, 3) << 32) | 5,
        (@as(u64, 4) << 32) | 7,
    };

    var a = try test_support.fromValues(Roaring64Bitmap, allocator, &a_values);
    defer a.deinit();
    var b = try test_support.fromValues(Roaring64Bitmap, allocator, &b_values);
    defer b.deinit();

    const expected_or = [_]u64{
        0,
        (@as(u64, 1) << 32) | 1,
        (@as(u64, 1) << 32) | 2,
        (@as(u64, 1) << 32) | 9,
        (@as(u64, 2) << 32) | 1,
        (@as(u64, 3) << 32) | 5,
        (@as(u64, 4) << 32) | 7,
        (@as(u64, 5) << 32) | 5,
    };
    const expected_and = [_]u64{
        (@as(u64, 1) << 32) | 2,
        (@as(u64, 3) << 32) | 5,
    };
    const expected_xor = [_]u64{
        0,
        (@as(u64, 1) << 32) | 1,
        (@as(u64, 1) << 32) | 9,
        (@as(u64, 2) << 32) | 1,
        (@as(u64, 4) << 32) | 7,
        (@as(u64, 5) << 32) | 5,
    };
    const expected_diff = [_]u64{
        0,
        (@as(u64, 1) << 32) | 1,
        (@as(u64, 5) << 32) | 5,
    };

    var out_or = try a.bitwiseOr(allocator, &b);
    defer out_or.deinit();
    try expectRoaring64Values(&out_or, &expected_or);

    var out_and = try a.bitwiseAnd(allocator, &b);
    defer out_and.deinit();
    try expectRoaring64Values(&out_and, &expected_and);

    var out_xor = try a.bitwiseXor(allocator, &b);
    defer out_xor.deinit();
    try expectRoaring64Values(&out_xor, &expected_xor);

    var out_diff = try a.bitwiseDifference(allocator, &b);
    defer out_diff.deinit();
    try expectRoaring64Values(&out_diff, &expected_diff);

    {
        var in_place = try a.clone(allocator);
        defer in_place.deinit();
        try in_place.bitwiseOrInPlace(&b);
        try expectRoaring64Values(&in_place, &expected_or);
    }
    {
        var in_place = try a.clone(allocator);
        defer in_place.deinit();
        try in_place.bitwiseAndInPlace(&b);
        try expectRoaring64Values(&in_place, &expected_and);
    }
    {
        var in_place = try a.clone(allocator);
        defer in_place.deinit();
        try in_place.bitwiseXorInPlace(&b);
        try expectRoaring64Values(&in_place, &expected_xor);
    }
    {
        var in_place = try a.clone(allocator);
        defer in_place.deinit();
        try in_place.bitwiseDifferenceInPlace(&b);
        try expectRoaring64Values(&in_place, &expected_diff);
    }
}

test "Roaring64Bitmap cardinalities and predicates" {
    const allocator = std.testing.allocator;
    const a_values = [_]u64{
        0,
        (@as(u64, 1) << 32) | 1,
        (@as(u64, 1) << 32) | 2,
        (@as(u64, 3) << 32) | 5,
        (@as(u64, 5) << 32) | 5,
    };
    const b_values = [_]u64{
        (@as(u64, 1) << 32) | 2,
        (@as(u64, 1) << 32) | 9,
        (@as(u64, 2) << 32) | 1,
        (@as(u64, 3) << 32) | 5,
        (@as(u64, 4) << 32) | 7,
    };
    const shared_values = [_]u64{
        (@as(u64, 1) << 32) | 2,
        (@as(u64, 3) << 32) | 5,
    };

    var a = try test_support.fromValues(Roaring64Bitmap, allocator, &a_values);
    defer a.deinit();
    var b = try test_support.fromValues(Roaring64Bitmap, allocator, &b_values);
    defer b.deinit();
    var shared = try test_support.fromValues(Roaring64Bitmap, allocator, &shared_values);
    defer shared.deinit();
    var a_clone = try a.clone(allocator);
    defer a_clone.deinit();
    var empty = try Roaring64Bitmap.init(allocator);
    defer empty.deinit();

    try std.testing.expectEqual(@as(u64, 2), a.andCardinality(&b));
    try std.testing.expectEqual(@as(u64, 8), a.orCardinality(&b));
    try std.testing.expectEqual(@as(u64, 6), a.xorCardinality(&b));
    try std.testing.expectEqual(@as(u64, 3), a.differenceCardinality(&b));

    try std.testing.expect(a.intersects(&b));
    try std.testing.expect(!a.intersects(&empty));
    try std.testing.expect(shared.isSubsetOf(&a));
    try std.testing.expect(shared.isStrictSubsetOf(&a));
    try std.testing.expect(!a.isSubsetOf(&shared));
    try std.testing.expect(a.equals(&a_clone));
    try std.testing.expect(!a.equals(&b));
    try std.testing.expect(!a.isStrictSubsetOf(&a_clone));
}

test "Roaring64Bitmap set ops prune empty buckets" {
    const allocator = std.testing.allocator;

    var a = try test_support.fromValues(Roaring64Bitmap, allocator, &[_]u64{(@as(u64, 10) << 32) | 1});
    defer a.deinit();
    var b = try test_support.fromValues(Roaring64Bitmap, allocator, &[_]u64{(@as(u64, 10) << 32) | 2});
    defer b.deinit();
    var empty = try Roaring64Bitmap.init(allocator);
    defer empty.deinit();

    var and_result = try a.bitwiseAnd(allocator, &b);
    defer and_result.deinit();
    try std.testing.expect(and_result.isEmpty());
    try std.testing.expectEqual(@as(u32, 0), and_result.size);
    try std.testing.expect(and_result.equals(&empty));
    try expectNoEmptyBuckets(&and_result);

    var c = try test_support.fromValues(Roaring64Bitmap, allocator, &[_]u64{(@as(u64, 20) << 32) | 1});
    defer c.deinit();
    var d = try test_support.fromValues(Roaring64Bitmap, allocator, &[_]u64{(@as(u64, 20) << 32) | 1});
    defer d.deinit();

    var diff_result = try c.bitwiseDifference(allocator, &d);
    defer diff_result.deinit();
    try std.testing.expect(diff_result.isEmpty());
    try std.testing.expectEqual(@as(u32, 0), diff_result.size);
    try std.testing.expect(diff_result.equals(&empty));
    try expectNoEmptyBuckets(&diff_result);

    var in_place = try c.clone(allocator);
    defer in_place.deinit();
    try in_place.bitwiseDifferenceInPlace(&d);
    try std.testing.expect(in_place.isEmpty());
    try expectNoEmptyBuckets(&in_place);
}

test "Roaring64Bitmap in-place self alias xor and difference empty the bitmap" {
    const allocator = std.testing.allocator;
    const values = [_]u64{
        1,
        (@as(u64, 2) << 32) | 3,
        std.math.maxInt(u64),
    };

    var xor_self = try test_support.fromValues(Roaring64Bitmap, allocator, &values);
    defer xor_self.deinit();
    try xor_self.bitwiseXorInPlace(&xor_self);
    try std.testing.expect(xor_self.isEmpty());
    try std.testing.expectEqual(@as(u32, 0), xor_self.size);

    var diff_self = try test_support.fromValues(Roaring64Bitmap, allocator, &values);
    defer diff_self.deinit();
    try diff_self.bitwiseDifferenceInPlace(&diff_self);
    try std.testing.expect(diff_self.isEmpty());
    try std.testing.expectEqual(@as(u32, 0), diff_self.size);
}

test "Roaring64Bitmap rank select and getIndex" {
    const allocator = std.testing.allocator;
    const values = [_]u64{
        0,
        5,
        (@as(u64, 1) << 32),
        (@as(u64, 1) << 32) | 10,
        (@as(u64, 3) << 32) | 1,
    };
    var bm = try test_support.fromValues(Roaring64Bitmap, allocator, &values);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 1), bm.rank(0));
    try std.testing.expectEqual(@as(u64, 1), bm.rank(4));
    try std.testing.expectEqual(@as(u64, 2), bm.rank(5));
    try std.testing.expectEqual(@as(u64, 2), bm.rank(std.math.maxInt(u32)));
    try std.testing.expectEqual(@as(u64, 3), bm.rank(@as(u64, 1) << 32));
    try std.testing.expectEqual(@as(u64, 4), bm.rank((@as(u64, 2) << 32) | 999));
    try std.testing.expectEqual(@as(u64, 5), bm.rank(std.math.maxInt(u64)));

    try std.testing.expectEqual(@as(?u64, 0), bm.getIndex(0));
    try std.testing.expectEqual(@as(?u64, 1), bm.getIndex(5));
    try std.testing.expectEqual(@as(?u64, 2), bm.getIndex(@as(u64, 1) << 32));
    try std.testing.expectEqual(@as(?u64, 4), bm.getIndex((@as(u64, 3) << 32) | 1));
    try std.testing.expectEqual(@as(?u64, null), bm.getIndex(4));

    try std.testing.expectEqual(@as(?u64, values[0]), bm.select(0));
    try std.testing.expectEqual(@as(?u64, values[2]), bm.select(2));
    try std.testing.expectEqual(@as(?u64, values[4]), bm.select(@intCast(values.len - 1)));
    try std.testing.expectEqual(@as(?u64, null), bm.select(@intCast(values.len)));
}

test "Roaring64Bitmap addRange single key and spanning keys" {
    const allocator = std.testing.allocator;

    var single = try Roaring64Bitmap.init(allocator);
    defer single.deinit();
    try single.addRange((@as(u64, 9) << 32) | 10, (@as(u64, 9) << 32) | 12);
    try expectRoaring64Values(&single, &[_]u64{
        (@as(u64, 9) << 32) | 10,
        (@as(u64, 9) << 32) | 11,
        (@as(u64, 9) << 32) | 12,
    });

    var spanning = try Roaring64Bitmap.init(allocator);
    defer spanning.deinit();
    const lo = (@as(u64, 5) << 32) | 0xffff_fffe;
    const hi = (@as(u64, 7) << 32) | 1;
    try spanning.addRange(lo, hi);

    try std.testing.expectEqual(@as(u32, 3), spanning.size);
    try std.testing.expectEqual(@as(u64, 4_294_967_300), spanning.cardinality());
    try std.testing.expect(spanning.contains(lo));
    try std.testing.expect(spanning.contains((@as(u64, 6) << 32) | 12345));
    try std.testing.expect(spanning.contains(hi));
    try std.testing.expect(spanning.containsRange(lo, hi));
    try std.testing.expectEqual(@as(u64, 4_294_967_300), spanning.rangeCardinality(lo, hi));
    try expectNoEmptyBuckets(&spanning);
}

test "Roaring64Bitmap removeRange prunes partial and interior buckets" {
    const allocator = std.testing.allocator;

    var partial = try Roaring64Bitmap.init(allocator);
    defer partial.deinit();
    try partial.addRange((@as(u64, 10) << 32) | 1, (@as(u64, 10) << 32) | 3);
    try partial.removeRange((@as(u64, 10) << 32) | 1, (@as(u64, 10) << 32) | 3);
    try std.testing.expect(partial.isEmpty());
    try std.testing.expectEqual(@as(u32, 0), partial.size);

    var spanning = try Roaring64Bitmap.init(allocator);
    defer spanning.deinit();
    const lo = (@as(u64, 5) << 32) | 0xffff_fffe;
    const hi = (@as(u64, 7) << 32) | 1;
    try spanning.addRange(lo, hi);
    try spanning.removeRange((@as(u64, 5) << 32) | 0xffff_ffff, (@as(u64, 7) << 32));

    try std.testing.expectEqual(@as(u64, 2), spanning.cardinality());
    try std.testing.expectEqual(@as(u32, 2), spanning.size);
    try std.testing.expect(spanning.contains(lo));
    try std.testing.expect(spanning.contains(hi));
    try std.testing.expect(!spanning.contains((@as(u64, 6) << 32) | 12345));
    try expectNoEmptyBuckets(&spanning);
}

test "Roaring64Bitmap rangeCardinality and containsRange across boundaries and gaps" {
    const allocator = std.testing.allocator;
    var bm = try Roaring64Bitmap.init(allocator);
    defer bm.deinit();

    const adjacent_lo = (@as(u64, 1) << 32) | 0xffff_fffe;
    const adjacent_hi = (@as(u64, 2) << 32) | 2;
    try bm.addRange(adjacent_lo, adjacent_hi);
    _ = try bm.add((@as(u64, 4) << 32) | 10);

    try std.testing.expectEqual(@as(u64, 5), bm.rangeCardinality(adjacent_lo, adjacent_hi));
    try std.testing.expect(bm.containsRange(adjacent_lo, adjacent_hi));
    try std.testing.expectEqual(@as(u64, 1), bm.rangeCardinality((@as(u64, 4) << 32) | 9, (@as(u64, 4) << 32) | 11));
    try std.testing.expect(!bm.containsRange((@as(u64, 2) << 32) | 2, (@as(u64, 4) << 32) | 10));
    try std.testing.expect(!bm.containsRange((@as(u64, 3) << 32), (@as(u64, 3) << 32) | 1));
    try std.testing.expectEqual(@as(u64, 0), bm.rangeCardinality((@as(u64, 3) << 32), (@as(u64, 3) << 32) | 1));
}

test "Roaring64Bitmap serialize round-trips empty bitmap" {
    const allocator = std.testing.allocator;

    var bm = try Roaring64Bitmap.init(allocator);
    defer bm.deinit();

    try test_support.expectSerializationRoundTrip(allocator, &bm);
}

test "Roaring64Bitmap serialize round-trips single and many buckets" {
    const allocator = std.testing.allocator;

    var single = try test_support.fromValues(Roaring64Bitmap, allocator, &[_]u64{
        (@as(u64, 7) << 32) | 1,
        (@as(u64, 7) << 32) | 999,
    });
    defer single.deinit();
    try test_support.expectSerializationRoundTrip(allocator, &single);

    var many = try test_support.fromValues(Roaring64Bitmap, allocator, &[_]u64{
        0,
        1,
        std.math.maxInt(u32),
        (@as(u64, 1) << 32),
        (@as(u64, 2) << 32) | 3,
        (@as(u64, 17) << 32) | 42,
        std.math.maxInt(u64),
    });
    defer many.deinit();
    try test_support.expectSerializationRoundTrip(allocator, &many);
}

test "Roaring64Bitmap serialize round-trips run-bearing sub-bitmap" {
    const allocator = std.testing.allocator;

    var bm = try Roaring64Bitmap.init(allocator);
    defer bm.deinit();

    try bm.addRange((@as(u64, 11) << 32) | 10, (@as(u64, 11) << 32) | 100);
    _ = try bm.add((@as(u64, 12) << 32) | 5);

    try std.testing.expect(test_support.hasRunContainers(&bm));
    try test_support.expectSerializationRoundTrip(allocator, &bm);
}

test "Roaring64Bitmap deserializeSafe rejects malformed frames" {
    const allocator = std.testing.allocator;

    var bm = try test_support.fromValues(Roaring64Bitmap, allocator, &[_]u64{
        1,
        (@as(u64, 2) << 32) | 3,
    });
    defer bm.deinit();

    try test_support.expectMalformedFramesRejected(allocator, &bm);
}

fn expectRoaring64Values(bm: *const Roaring64Bitmap, expected: []const u64) !void {
    const allocator = std.testing.allocator;
    const actual = try bm.toArrayAlloc(allocator);
    defer allocator.free(actual);

    try std.testing.expectEqualSlices(u64, expected, actual);
    try std.testing.expectEqual(@as(u64, @intCast(expected.len)), bm.cardinality());
    try expectNoEmptyBuckets(bm);
}

fn expectNoEmptyBuckets(bm: *const Roaring64Bitmap) !void {
    for (bm.buckets[0..bm.size]) |*bucket| {
        try std.testing.expect(!bucket.bm.isEmpty());
    }
}
