// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const FrozenBitmap = @import("frozen.zig").FrozenBitmap;

/// A zero-copy read-only view over rawr's native 64-bit frozen layout.
///
/// This is a rawr-native format: a small 64-bit bucket table followed by
/// existing 32-bit FrozenBitmap sub-images. It is not CRoaring frozen interop.
pub const Frozen64Bitmap = struct {
    data: []const u8,
    size: u64,

    const Self = @This();
    const HEADER_SIZE: usize = 8;
    const ENTRY_SIZE: usize = 12;

    /// Create a frozen 64-bit bitmap view over borrowed bytes.
    pub fn view(data: []const u8) !Self {
        if (data.len < HEADER_SIZE) return error.InvalidFormat;

        const size = std.mem.readInt(u64, data[0..8], .little);
        if (size > std.math.maxInt(usize)) return error.InvalidFormat;

        const table_bytes = std.math.mul(usize, @intCast(size), ENTRY_SIZE) catch return error.InvalidFormat;
        const table_end = std.math.add(usize, HEADER_SIZE, table_bytes) catch return error.InvalidFormat;
        if (table_end > data.len) return error.InvalidFormat;

        const self: Self = .{
            .data = data,
            .size = size,
        };
        try self.validate(table_end);
        return self;
    }

    /// No deallocation needed; this is a view over borrowed data.
    pub fn deinit(self: *Self) void {
        _ = self;
    }

    pub fn isEmpty(self: *const Self) bool {
        return self.size == 0;
    }

    pub fn contains(self: *const Self, value: u64) bool {
        const idx = self.bucketIndex(highBits(value)) orelse return false;
        const sub = self.subView(idx);
        return sub.contains(lowBits(value));
    }

    pub fn cardinality(self: *const Self) u64 {
        var total: u64 = 0;
        for (0..@as(usize, @intCast(self.size))) |idx| {
            const sub = self.subView(idx);
            total += sub.cardinality();
        }
        return total;
    }

    pub fn minimum(self: *const Self) ?u64 {
        if (self.size == 0) return null;
        const sub = self.subView(0);
        const low = sub.minimum() orelse return null;
        return combine(self.getHi(0), low);
    }

    pub fn maximum(self: *const Self) ?u64 {
        if (self.size == 0) return null;
        const idx: usize = @intCast(self.size - 1);
        const sub = self.subView(idx);
        const low = sub.maximum() orelse return null;
        return combine(self.getHi(idx), low);
    }

    pub fn rank(self: *const Self, value: u64) u64 {
        const target_hi = highBits(value);
        const target_low = lowBits(value);

        var total: u64 = 0;
        for (0..@as(usize, @intCast(self.size))) |idx| {
            const hi = self.getHi(idx);
            const sub = self.subView(idx);
            if (hi < target_hi) {
                total += sub.cardinality();
            } else if (hi == target_hi) {
                return total + sub.rank(target_low);
            } else {
                return total;
            }
        }
        return total;
    }

    pub fn getIndex(self: *const Self, value: u64) ?u64 {
        const target_hi = highBits(value);
        const target_low = lowBits(value);

        var total: u64 = 0;
        for (0..@as(usize, @intCast(self.size))) |idx| {
            const hi = self.getHi(idx);
            const sub = self.subView(idx);
            if (hi < target_hi) {
                total += sub.cardinality();
            } else if (hi == target_hi) {
                const local = sub.getIndex(target_low) orelse return null;
                return total + local;
            } else {
                return null;
            }
        }
        return null;
    }

    pub fn select(self: *const Self, k: u64) ?u64 {
        var prior: u64 = 0;
        for (0..@as(usize, @intCast(self.size))) |idx| {
            const sub = self.subView(idx);
            const card = sub.cardinality();
            if (k - prior < card) {
                const low = sub.select(k - prior) orelse return null;
                return combine(self.getHi(idx), low);
            }
            prior += card;
        }
        return null;
    }

    pub const Iterator = struct {
        fb: *const Frozen64Bitmap,
        bucket_idx: usize,
        sub: FrozenBitmap = undefined,
        inner: ?FrozenBitmap.Iterator = null,

        pub fn next(self: *Iterator) ?u64 {
            while (self.bucket_idx < self.fb.size) {
                if (self.inner == null) {
                    self.sub = self.fb.subView(self.bucket_idx);
                    self.inner = self.sub.iterator();
                }

                if (self.inner) |*it| {
                    if (it.next()) |low| {
                        return combine(self.fb.getHi(self.bucket_idx), low);
                    }
                }

                self.bucket_idx += 1;
                self.inner = null;
            }
            return null;
        }
    };

    pub fn iterator(self: *const Self) Iterator {
        return .{
            .fb = self,
            .bucket_idx = 0,
        };
    }

    fn validate(self: *const Self, table_end: usize) !void {
        var prev_hi: ?u32 = null;
        var prev_offset = table_end;

        for (0..@as(usize, @intCast(self.size))) |idx| {
            const hi = self.getHi(idx);
            if (prev_hi) |prev| {
                if (hi <= prev) return error.InvalidFormat;
            }
            prev_hi = hi;

            const offset = self.getOffset(idx);
            if (offset < table_end or offset > self.data.len) return error.InvalidFormat;
            if (idx != 0 and offset < prev_offset) return error.InvalidFormat;
            prev_offset = offset;

            const end = self.subEnd(idx);
            if (end <= offset or end > self.data.len) return error.InvalidFormat;

            var sub = try FrozenBitmap.init(self.data[offset..end]);
            defer sub.deinit();
            if (sub.isEmpty()) return error.InvalidFormat;
        }
    }

    fn bucketIndex(self: *const Self, hi_key: u32) ?usize {
        var lo: usize = 0;
        var hi: usize = @intCast(self.size);
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const mid_key = self.getHi(mid);
            if (mid_key < hi_key) {
                lo = mid + 1;
            } else if (mid_key > hi_key) {
                hi = mid;
            } else {
                return mid;
            }
        }
        return null;
    }

    fn getHi(self: *const Self, idx: usize) u32 {
        const offset = HEADER_SIZE + idx * ENTRY_SIZE;
        return std.mem.readInt(u32, self.data[offset..][0..4], .little);
    }

    fn getOffset(self: *const Self, idx: usize) usize {
        const offset = HEADER_SIZE + idx * ENTRY_SIZE + 4;
        const value = std.mem.readInt(u64, self.data[offset..][0..8], .little);
        return @intCast(value);
    }

    fn subEnd(self: *const Self, idx: usize) usize {
        if (idx + 1 < self.size) return self.getOffset(idx + 1);
        return self.data.len;
    }

    fn subView(self: *const Self, idx: usize) FrozenBitmap {
        return FrozenBitmap.init(self.data[self.getOffset(idx)..self.subEnd(idx)]) catch unreachable;
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
};
