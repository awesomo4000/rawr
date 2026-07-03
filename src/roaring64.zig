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
};

test "Roaring64Bitmap init and deinit" {
    const allocator = std.testing.allocator;
    var bm = try Roaring64Bitmap.init(allocator);
    defer bm.deinit();

    try std.testing.expect(bm.isEmpty());
    try std.testing.expectEqual(@as(u64, 0), bm.cardinality());
}
