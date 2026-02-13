const std = @import("std");
const fmt = @import("format.zig");
const ArrayContainer = @import("array_container.zig").ArrayContainer;
const BitsetContainer = @import("bitset_container.zig").BitsetContainer;

/// A read-only bitmap view over serialized bytes. Zero-copy - no allocation for container data.
/// Use this for zero-copy reads from mmap'd LMDB values.
pub const FrozenBitmap = struct {
    data: []const u8,
    size: u32,
    has_runs: bool,
    keys_offset: usize,
    offsets_offset: usize, // 0 if no offset header
    data_offset: usize,
    run_bitset: []const u8, // empty if no runs

    const Self = @This();

    /// Create a frozen bitmap view over serialized bytes. Zero allocation.
    pub fn init(data: []const u8) !Self {
        if (data.len < 4) return error.InvalidFormat;

        const cookie = std.mem.readInt(u32, data[0..4], .little);

        var pos: usize = 4;
        var size: u32 = undefined;
        var has_runs = false;
        var run_bitset: []const u8 = &.{};

        if ((cookie & 0xFFFF) == fmt.SERIAL_COOKIE) {
            has_runs = true;
            size = ((cookie >> 16) & 0xFFFF) + 1;
            const bitset_bytes = (size + 7) / 8;
            if (pos + bitset_bytes > data.len) return error.InvalidFormat;
            run_bitset = data[pos .. pos + bitset_bytes];
            pos += bitset_bytes;
        } else if (cookie == fmt.SERIAL_COOKIE_NO_RUNCONTAINER) {
            if (data.len < 8) return error.InvalidFormat;
            size = std.mem.readInt(u32, data[4..8], .little);
            pos = 8;
        } else {
            return error.InvalidFormat;
        }

        const keys_offset = pos;
        pos += @as(usize, size) * 4; // key + cardinality-1 pairs

        // Offset header:
        // - Always for no-run format (RoaringFormatSpec requirement)
        // - For run format only when size >= NO_OFFSET_THRESHOLD
        var offsets_offset: usize = 0;
        if (!has_runs or size >= fmt.NO_OFFSET_THRESHOLD) {
            offsets_offset = pos;
            pos += @as(usize, size) * 4;
        }

        if (pos > data.len) return error.InvalidFormat;

        return .{
            .data = data,
            .size = size,
            .has_runs = has_runs,
            .keys_offset = keys_offset,
            .offsets_offset = offsets_offset,
            .data_offset = pos,
            .run_bitset = run_bitset,
        };
    }

    /// No deallocation needed - this is a view over borrowed data.
    pub fn deinit(self: *Self) void {
        _ = self;
    }

    /// Check if the bitmap is empty.
    pub fn isEmpty(self: *const Self) bool {
        return self.size == 0;
    }

    /// Get the key for container at index.
    fn getKey(self: *const Self, idx: usize) u16 {
        const offset = self.keys_offset + idx * 4;
        return std.mem.readInt(u16, self.data[offset..][0..2], .little);
    }

    /// Get the cardinality for container at index.
    fn getCardinality(self: *const Self, idx: usize) u32 {
        const offset = self.keys_offset + idx * 4 + 2;
        return @as(u32, std.mem.readInt(u16, self.data[offset..][0..2], .little)) + 1;
    }

    /// Check if container at index is a run container.
    fn isRunContainer(self: *const Self, idx: usize) bool {
        if (!self.has_runs) return false;
        return (self.run_bitset[idx / 8] & (@as(u8, 1) << @intCast(idx % 8))) != 0;
    }

    /// Binary search for a key.
    fn findKey(self: *const Self, key: u16) ?usize {
        if (self.size == 0) return null;

        var lo: usize = 0;
        var hi: usize = self.size;

        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const mid_key = self.getKey(mid);
            if (mid_key < key) {
                lo = mid + 1;
            } else if (mid_key > key) {
                hi = mid;
            } else {
                return mid;
            }
        }
        return null;
    }

    /// Get the data offset for container at index. O(1) using offset header.
    /// Fallback O(n) path only for small run-format bitmaps (size < 4 with runs).
    fn getContainerDataOffset(self: *const Self, idx: usize) usize {
        if (self.offsets_offset != 0) {
            const offset = self.offsets_offset + idx * 4;
            // Offsets are absolute positions from buffer start (per RoaringFormatSpec)
            return std.mem.readInt(u32, self.data[offset..][0..4], .little);
        }

        // Fallback for small run-format bitmaps without offset header (size < 4)
        var offset = self.data_offset;
        for (0..idx) |i| {
            offset += self.getContainerSize(i);
        }
        return offset;
    }

    /// Get the serialized size of container at index.
    fn getContainerSize(self: *const Self, idx: usize) usize {
        const card = self.getCardinality(idx);
        if (self.isRunContainer(idx)) {
            // Run: read n_runs from data prefix. For idx=0, use data_offset directly
            // to avoid mutual recursion with getContainerDataOffset.
            const data_offset = if (idx == 0) self.data_offset else self.getContainerDataOffset(idx);
            const n_runs = std.mem.readInt(u16, self.data[data_offset..][0..2], .little);
            return 2 + @as(usize, n_runs) * 4;
        } else if (card > ArrayContainer.MAX_CARDINALITY) {
            return 8192; // Bitset
        } else {
            return @as(usize, card) * 2; // Array
        }
    }

    /// Check if a value is present.
    pub fn contains(self: *const Self, value: u32) bool {
        const key: u16 = @truncate(value >> 16);
        const low: u16 = @truncate(value);

        const idx = self.findKey(key) orelse return false;
        return self.containerContains(idx, low);
    }

    /// Check if value is in container at index.
    fn containerContains(self: *const Self, idx: usize, low: u16) bool {
        const data_offset = self.getContainerDataOffset(idx);
        const card = self.getCardinality(idx);

        if (self.isRunContainer(idx)) {
            // Run container: n_runs prefix + pairs
            const n_runs = std.mem.readInt(u16, self.data[data_offset..][0..2], .little);
            const runs_data = self.data[data_offset + 2 ..];
            return self.searchRuns(runs_data, n_runs, low);
        } else if (card > ArrayContainer.MAX_CARDINALITY) {
            // Bitset container
            const word_idx = low >> 6;
            const bit_idx: u6 = @truncate(low);
            const word_offset = data_offset + @as(usize, word_idx) * 8;
            const word = std.mem.readInt(u64, self.data[word_offset..][0..8], .little);
            return (word & (@as(u64, 1) << bit_idx)) != 0;
        } else {
            // Array container: binary search
            return self.binarySearchArray(data_offset, card, low);
        }
    }

    fn binarySearchArray(self: *const Self, data_offset: usize, card: u32, value: u16) bool {
        var lo: u32 = 0;
        var hi: u32 = card;

        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const offset = data_offset + @as(usize, mid) * 2;
            const mid_val = std.mem.readInt(u16, self.data[offset..][0..2], .little);
            if (mid_val < value) {
                lo = mid + 1;
            } else if (mid_val > value) {
                hi = mid;
            } else {
                return true;
            }
        }
        return false;
    }

    fn searchRuns(self: *const Self, runs_data: []const u8, n_runs: u16, value: u16) bool {
        _ = self;
        var lo: u16 = 0;
        var hi: u16 = n_runs;

        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const offset = @as(usize, mid) * 4;
            const start = std.mem.readInt(u16, runs_data[offset..][0..2], .little);
            const length = std.mem.readInt(u16, runs_data[offset + 2 ..][0..2], .little);
            const end = start +| length;

            if (end < value) {
                lo = mid + 1;
            } else if (start > value) {
                hi = mid;
            } else {
                return true; // value in [start, end]
            }
        }
        return false;
    }

    /// Compute total cardinality by summing all containers.
    pub fn cardinality(self: *const Self) u64 {
        var total: u64 = 0;
        for (0..self.size) |i| {
            total += self.getCardinality(i);
        }
        return total;
    }

    /// Iterator over all values in the frozen bitmap.
    pub const Iterator = struct {
        fb: *const FrozenBitmap,
        container_idx: u32,
        state: State,

        const State = union(enum) {
            empty: void,
            array: ArrayState,
            bitset: BitsetState,
            run: RunState,
        };

        const ArrayState = struct {
            data_offset: usize,
            card: u32,
            pos: u32,
        };

        const BitsetState = struct {
            data_offset: usize,
            word_idx: u32,
            current_word: u64,
        };

        const RunState = struct {
            data_offset: usize,
            n_runs: u16,
            run_idx: u16,
            pos_in_run: u16,
        };

        pub fn next(self: *Iterator) ?u32 {
            while (true) {
                switch (self.state) {
                    .empty => {
                        if (self.container_idx >= self.fb.size) return null;
                        self.initContainer(self.container_idx);
                    },
                    .array => |*s| {
                        if (s.pos < s.card) {
                            const offset = s.data_offset + @as(usize, s.pos) * 2;
                            const low = std.mem.readInt(u16, self.fb.data[offset..][0..2], .little);
                            const high: u32 = @as(u32, self.fb.getKey(self.container_idx)) << 16;
                            s.pos += 1;
                            return high | low;
                        }
                        self.advanceContainer();
                    },
                    .bitset => |*s| {
                        while (s.current_word == 0) {
                            s.word_idx += 1;
                            if (s.word_idx >= BitsetContainer.NUM_WORDS) {
                                self.advanceContainer();
                                break;
                            }
                            const word_offset = s.data_offset + @as(usize, s.word_idx) * 8;
                            s.current_word = std.mem.readInt(u64, self.fb.data[word_offset..][0..8], .little);
                        } else {
                            const bit = @ctz(s.current_word);
                            s.current_word &= s.current_word - 1;
                            const high: u32 = @as(u32, self.fb.getKey(self.container_idx)) << 16;
                            const low: u32 = @as(u32, s.word_idx) * 64 + bit;
                            return high | low;
                        }
                    },
                    .run => |*s| {
                        if (s.run_idx < s.n_runs) {
                            const run_offset = s.data_offset + 2 + @as(usize, s.run_idx) * 4;
                            const start = std.mem.readInt(u16, self.fb.data[run_offset..][0..2], .little);
                            const length = std.mem.readInt(u16, self.fb.data[run_offset + 2 ..][0..2], .little);

                            const high: u32 = @as(u32, self.fb.getKey(self.container_idx)) << 16;
                            const low: u32 = @as(u32, start) + s.pos_in_run;

                            if (s.pos_in_run <= length) {
                                const result = high | low;
                                if (s.pos_in_run < length) {
                                    s.pos_in_run += 1;
                                } else {
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
            const data_offset = self.fb.getContainerDataOffset(idx);
            const card = self.fb.getCardinality(idx);

            if (self.fb.isRunContainer(idx)) {
                const n_runs = std.mem.readInt(u16, self.fb.data[data_offset..][0..2], .little);
                self.state = .{ .run = .{
                    .data_offset = data_offset,
                    .n_runs = n_runs,
                    .run_idx = 0,
                    .pos_in_run = 0,
                } };
            } else if (card > ArrayContainer.MAX_CARDINALITY) {
                // Find first non-zero word
                var word_idx: u32 = 0;
                while (word_idx < BitsetContainer.NUM_WORDS) : (word_idx += 1) {
                    const word_offset = data_offset + @as(usize, word_idx) * 8;
                    const word = std.mem.readInt(u64, self.fb.data[word_offset..][0..8], .little);
                    if (word != 0) {
                        self.state = .{ .bitset = .{
                            .data_offset = data_offset,
                            .word_idx = word_idx,
                            .current_word = word,
                        } };
                        return;
                    }
                }
                self.state = .empty;
            } else {
                self.state = .{ .array = .{
                    .data_offset = data_offset,
                    .card = card,
                    .pos = 0,
                } };
            }
        }

        fn advanceContainer(self: *Iterator) void {
            self.container_idx += 1;
            self.state = .empty;
        }
    };

    /// Returns an iterator over all values.
    pub fn iterator(self: *const Self) Iterator {
        var it = Iterator{
            .fb = self,
            .container_idx = 0,
            .state = .empty,
        };
        if (self.size > 0) {
            it.initContainer(0);
        }
        return it;
    }
};

// ============================================================================
// Tests
// ============================================================================

const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;

test "FrozenBitmap from empty bitmap" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    const serialized = try bm.serialize(allocator);
    defer allocator.free(serialized);

    var frozen = try FrozenBitmap.init(serialized);
    defer frozen.deinit();

    try std.testing.expect(frozen.isEmpty());
    try std.testing.expect(!frozen.contains(0));
}

test "FrozenBitmap contains from array container" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(100);
    _ = try bm.add(200);
    _ = try bm.add(300);

    const serialized = try bm.serialize(allocator);
    defer allocator.free(serialized);

    var frozen = try FrozenBitmap.init(serialized);
    defer frozen.deinit();

    try std.testing.expect(frozen.contains(100));
    try std.testing.expect(frozen.contains(200));
    try std.testing.expect(frozen.contains(300));
    try std.testing.expect(!frozen.contains(99));
    try std.testing.expect(!frozen.contains(101));
}

test "FrozenBitmap contains from bitset container" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Create bitset (>4096 values)
    _ = try bm.addRange(0, 5000);

    const serialized = try bm.serialize(allocator);
    defer allocator.free(serialized);

    var frozen = try FrozenBitmap.init(serialized);
    defer frozen.deinit();

    try std.testing.expect(frozen.contains(0));
    try std.testing.expect(frozen.contains(2500));
    try std.testing.expect(frozen.contains(5000));
    try std.testing.expect(!frozen.contains(5001));
}

test "FrozenBitmap contains from run container" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.addRange(100, 200);
    _ = try bm.runOptimize();

    const serialized = try bm.serialize(allocator);
    defer allocator.free(serialized);

    var frozen = try FrozenBitmap.init(serialized);
    defer frozen.deinit();

    try std.testing.expect(frozen.contains(100));
    try std.testing.expect(frozen.contains(150));
    try std.testing.expect(frozen.contains(200));
    try std.testing.expect(!frozen.contains(99));
    try std.testing.expect(!frozen.contains(201));
}

test "FrozenBitmap with multiple containers" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(100); // chunk 0
    _ = try bm.add(65536 + 50); // chunk 1
    _ = try bm.add(131072 + 25); // chunk 2

    const serialized = try bm.serialize(allocator);
    defer allocator.free(serialized);

    var frozen = try FrozenBitmap.init(serialized);
    defer frozen.deinit();

    try std.testing.expect(frozen.contains(100));
    try std.testing.expect(frozen.contains(65536 + 50));
    try std.testing.expect(frozen.contains(131072 + 25));
    try std.testing.expect(!frozen.contains(65536 + 51));
}

test "FrozenBitmap iterator" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    const values = [_]u32{ 10, 20, 30, 65536 + 5, 65536 + 15 };
    for (values) |v| {
        _ = try bm.add(v);
    }

    const serialized = try bm.serialize(allocator);
    defer allocator.free(serialized);

    var frozen = try FrozenBitmap.init(serialized);
    defer frozen.deinit();

    var iter = frozen.iterator();
    var idx: usize = 0;
    while (iter.next()) |v| {
        try std.testing.expect(idx < values.len);
        try std.testing.expectEqual(values[idx], v);
        idx += 1;
    }
    try std.testing.expectEqual(values.len, idx);
}
