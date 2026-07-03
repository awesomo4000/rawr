const std = @import("std");
const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;

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
        if (self.size == 0) return null;

        var lo: usize = 0;
        var hi: usize = self.size;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.buckets[mid].hi < hi_key) {
                lo = mid + 1;
            } else if (self.buckets[mid].hi > hi_key) {
                hi = mid;
            } else {
                return mid;
            }
        }
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
