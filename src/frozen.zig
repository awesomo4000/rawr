// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const fmt = @import("format.zig");
const ArrayContainer = @import("array_container.zig").ArrayContainer;
const BitsetContainer = @import("bitset_container.zig").BitsetContainer;

/// A read-only bitmap view over serialized bytes. Zero-copy - no allocation for container data.
/// Use this for zero-copy reads from mmap'd LMDB values.
///
/// The borrowed byte buffer must remain immutable for the lifetime of this view.
/// `init` validates bounds once; concurrent mutation after validation can break
/// the safety guarantees inherent to any zero-copy reader.
pub const FrozenBitmap = struct {
    data: []const u8,
    size: u32,
    has_runs: bool,
    keys_offset: usize,
    offsets_offset: usize, // 0 if no offset header
    data_offset: usize,
    run_bitset: []const u8, // empty if no runs

    const Self = @This();
    const MAX_CONTAINER_COUNT = 65_536;

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
            if (size > MAX_CONTAINER_COUNT) return error.InvalidFormat;
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

        const self: Self = .{
            .data = data,
            .size = size,
            .has_runs = has_runs,
            .keys_offset = keys_offset,
            .offsets_offset = offsets_offset,
            .data_offset = pos,
            .run_bitset = run_bitset,
        };
        try self.validateContainerBounds();
        return self;
    }

    fn validateContainerBounds(self: *const Self) !void {
        var sequential_offset = self.data_offset;

        for (0..self.size) |idx| {
            const data_offset = if (self.offsets_offset != 0) blk: {
                const offset_pos = self.offsets_offset + idx * 4;
                const value = std.mem.readInt(u32, self.data[offset_pos..][0..4], .little);
                const start: usize = value;
                if (start < self.data_offset or start > self.data.len) return error.InvalidFormat;
                break :blk start;
            } else sequential_offset;

            const container_size = try self.checkedContainerSize(idx, data_offset);
            if (container_size > self.data.len - data_offset) return error.InvalidFormat;

            if (self.offsets_offset == 0) {
                sequential_offset = data_offset + container_size;
            }
        }
    }

    fn checkedContainerSize(self: *const Self, idx: usize, data_offset: usize) !usize {
        if (data_offset > self.data.len) return error.InvalidFormat;

        const card = self.getCardinality(idx);
        if (self.isRunContainer(idx)) {
            if (2 > self.data.len - data_offset) return error.InvalidFormat;
            const n_runs = std.mem.readInt(u16, self.data[data_offset..][0..2], .little);
            const run_bytes = @as(usize, n_runs) * 4;
            if (run_bytes > self.data.len - data_offset - 2) return error.InvalidFormat;
            return 2 + run_bytes;
        }

        if (card > ArrayContainer.MAX_CARDINALITY) {
            return BitsetContainer.SIZE_BYTES;
        }

        return @as(usize, card) * 2;
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

    /// Count values less than or equal to `value`.
    /// Scans preceding container descriptors, then probes the target container.
    pub fn rank(self: *const Self, value: u32) u64 {
        const target_key: u16 = @truncate(value >> 16);
        const target_low: u16 = @truncate(value);

        var total: u64 = 0;
        for (0..self.size) |idx| {
            const key = self.getKey(idx);
            if (key < target_key) {
                total += self.getCardinality(idx);
            } else if (key == target_key) {
                return total + self.containerRank(idx, target_low);
            } else {
                return total;
            }
        }
        return total;
    }

    /// Return the 0-based position of `value`, or null if absent.
    /// Scans preceding container descriptors, then probes the target container.
    pub fn getIndex(self: *const Self, value: u32) ?u64 {
        const target_key: u16 = @truncate(value >> 16);
        const target_low: u16 = @truncate(value);

        var total: u64 = 0;
        for (0..self.size) |idx| {
            const key = self.getKey(idx);
            if (key < target_key) {
                total += self.getCardinality(idx);
            } else if (key == target_key) {
                const local = self.containerGetIndex(idx, target_low) orelse return null;
                return total + local;
            } else {
                return null;
            }
        }
        return null;
    }

    /// Return the k-th smallest value, 0-based, or null if out of range.
    /// Scans container cardinalities, then selects within the target container.
    pub fn select(self: *const Self, k: u64) ?u32 {
        if (k > std.math.maxInt(u32)) return null;
        var remaining: u32 = @intCast(k);

        for (0..self.size) |idx| {
            const card = self.getCardinality(idx);
            if (remaining < card) {
                const low = self.containerSelect(idx, remaining) orelse return null;
                return (@as(u32, self.getKey(idx)) << 16) | low;
            }
            remaining -= card;
        }
        return null;
    }

    /// Get the minimum value, or null if empty.
    /// Array and run containers use a direct read; bitsets scan up to 1,024 words.
    pub fn minimum(self: *const Self) ?u32 {
        if (self.size == 0) return null;
        const low = self.containerMinimum(0) orelse return null;
        return (@as(u32, self.getKey(0)) << 16) | low;
    }

    /// Get the maximum value, or null if empty.
    /// Array and run containers use a direct read; bitsets scan up to 1,024 words.
    pub fn maximum(self: *const Self) ?u32 {
        if (self.size == 0) return null;
        const idx: usize = self.size - 1;
        const low = self.containerMaximum(idx) orelse return null;
        return (@as(u32, self.getKey(idx)) << 16) | low;
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

    fn containerRank(self: *const Self, idx: usize, low: u16) u32 {
        const data_offset = self.getContainerDataOffset(idx);
        const card = self.getCardinality(idx);

        if (self.isRunContainer(idx)) {
            const n_runs = std.mem.readInt(u16, self.data[data_offset..][0..2], .little);
            var count: u32 = 0;
            for (0..n_runs) |run_idx| {
                const run_offset = data_offset + 2 + run_idx * 4;
                const start = std.mem.readInt(u16, self.data[run_offset..][0..2], .little);
                const length = std.mem.readInt(u16, self.data[run_offset + 2 ..][0..2], .little);
                if (low < start) return count;

                const end = start +| length;
                if (low <= end) return count + @as(u32, low - start) + 1;
                count += @as(u32, length) + 1;
            }
            return count;
        }

        if (card > ArrayContainer.MAX_CARDINALITY) {
            const word_idx: usize = low >> 6;
            const bit: u6 = @truncate(low);
            var count: u32 = 0;
            for (0..word_idx) |idx_word| {
                count += @popCount(self.readBitsetWord(data_offset, idx_word));
            }
            const mask = if (bit == 63)
                ~@as(u64, 0)
            else
                (@as(u64, 1) << (bit + 1)) - 1;
            return count + @popCount(self.readBitsetWord(data_offset, word_idx) & mask);
        }

        var lo: u32 = 0;
        var hi = card;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.readArrayValue(data_offset, mid) <= low) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    fn containerGetIndex(self: *const Self, idx: usize, value: u16) ?u32 {
        const data_offset = self.getContainerDataOffset(idx);
        const card = self.getCardinality(idx);

        if (self.isRunContainer(idx)) {
            const n_runs = std.mem.readInt(u16, self.data[data_offset..][0..2], .little);
            var prior: u32 = 0;
            for (0..n_runs) |run_idx| {
                const run_offset = data_offset + 2 + run_idx * 4;
                const start = std.mem.readInt(u16, self.data[run_offset..][0..2], .little);
                const length = std.mem.readInt(u16, self.data[run_offset + 2 ..][0..2], .little);
                if (value < start) return null;
                if (value <= start +| length) return prior + @as(u32, value - start);
                prior += @as(u32, length) + 1;
            }
            return null;
        }

        if (card > ArrayContainer.MAX_CARDINALITY) {
            const word_idx: usize = value >> 6;
            const bit: u6 = @truncate(value);
            const word = self.readBitsetWord(data_offset, word_idx);
            if (word & (@as(u64, 1) << bit) == 0) return null;
            return self.containerRank(idx, value) - 1;
        }

        var lo: u32 = 0;
        var hi = card;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const current = self.readArrayValue(data_offset, mid);
            if (current < value) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo < card and self.readArrayValue(data_offset, lo) == value) return lo;
        return null;
    }

    fn containerSelect(self: *const Self, idx: usize, k: u32) ?u16 {
        const data_offset = self.getContainerDataOffset(idx);
        const card = self.getCardinality(idx);
        if (k >= card) return null;

        if (self.isRunContainer(idx)) {
            const n_runs = std.mem.readInt(u16, self.data[data_offset..][0..2], .little);
            var remaining = k;
            for (0..n_runs) |run_idx| {
                const run_offset = data_offset + 2 + run_idx * 4;
                const start = std.mem.readInt(u16, self.data[run_offset..][0..2], .little);
                const length = std.mem.readInt(u16, self.data[run_offset + 2 ..][0..2], .little);
                const run_size = @as(u32, length) + 1;
                if (remaining < run_size) return start + @as(u16, @intCast(remaining));
                remaining -= run_size;
            }
            return null;
        }

        if (card > ArrayContainer.MAX_CARDINALITY) {
            var remaining = k;
            for (0..BitsetContainer.NUM_WORDS) |word_idx| {
                var word = self.readBitsetWord(data_offset, word_idx);
                const word_card: u32 = @popCount(word);
                if (remaining >= word_card) {
                    remaining -= word_card;
                    continue;
                }
                while (remaining > 0) : (remaining -= 1) word &= word - 1;
                return @intCast(word_idx * 64 + @ctz(word));
            }
            return null;
        }

        return self.readArrayValue(data_offset, k);
    }

    fn containerMinimum(self: *const Self, idx: usize) ?u16 {
        const data_offset = self.getContainerDataOffset(idx);
        const card = self.getCardinality(idx);

        if (self.isRunContainer(idx)) {
            const n_runs = std.mem.readInt(u16, self.data[data_offset..][0..2], .little);
            if (n_runs == 0) return null;
            return std.mem.readInt(u16, self.data[data_offset + 2 ..][0..2], .little);
        }
        if (card > ArrayContainer.MAX_CARDINALITY) {
            for (0..BitsetContainer.NUM_WORDS) |word_idx| {
                const word = self.readBitsetWord(data_offset, word_idx);
                if (word != 0) return @intCast(word_idx * 64 + @ctz(word));
            }
            return null;
        }
        return self.readArrayValue(data_offset, 0);
    }

    fn containerMaximum(self: *const Self, idx: usize) ?u16 {
        const data_offset = self.getContainerDataOffset(idx);
        const card = self.getCardinality(idx);

        if (self.isRunContainer(idx)) {
            const n_runs = std.mem.readInt(u16, self.data[data_offset..][0..2], .little);
            if (n_runs == 0) return null;
            const run_offset = data_offset + 2 + (@as(usize, n_runs) - 1) * 4;
            const start = std.mem.readInt(u16, self.data[run_offset..][0..2], .little);
            const length = std.mem.readInt(u16, self.data[run_offset + 2 ..][0..2], .little);
            return start +| length;
        }
        if (card > ArrayContainer.MAX_CARDINALITY) {
            var word_idx: usize = BitsetContainer.NUM_WORDS;
            while (word_idx > 0) {
                word_idx -= 1;
                const word = self.readBitsetWord(data_offset, word_idx);
                if (word != 0) return @intCast(word_idx * 64 + 63 - @clz(word));
            }
            return null;
        }
        return self.readArrayValue(data_offset, card - 1);
    }

    fn readArrayValue(self: *const Self, data_offset: usize, idx: u32) u16 {
        const offset = data_offset + @as(usize, idx) * 2;
        return std.mem.readInt(u16, self.data[offset..][0..2], .little);
    }

    fn readBitsetWord(self: *const Self, data_offset: usize, word_idx: usize) u64 {
        const offset = data_offset + word_idx * 8;
        return std.mem.readInt(u64, self.data[offset..][0..8], .little);
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

fn writeU16LE(data: []u8, offset: usize, value: u16) void {
    data[offset] = @truncate(value);
    data[offset + 1] = @truncate(value >> 8);
}

fn writeU32LE(data: []u8, offset: usize, value: u32) void {
    data[offset] = @truncate(value);
    data[offset + 1] = @truncate(value >> 8);
    data[offset + 2] = @truncate(value >> 16);
    data[offset + 3] = @truncate(value >> 24);
}

fn descStart(data: []const u8) usize {
    const cookie = std.mem.readInt(u32, data[0..4], .little);
    if ((cookie & 0xFFFF) == fmt.SERIAL_COOKIE) {
        const size = ((cookie >> 16) & 0xFFFF) + 1;
        return 4 + ((@as(usize, size) + 7) / 8);
    }
    return 8;
}

fn offsetTableStart(data: []const u8) usize {
    const cookie = std.mem.readInt(u32, data[0..4], .little);
    const size = if ((cookie & 0xFFFF) == fmt.SERIAL_COOKIE)
        ((cookie >> 16) & 0xFFFF) + 1
    else
        std.mem.readInt(u32, data[4..8], .little);
    return descStart(data) + @as(usize, size) * 4;
}

fn firstDataOffset(data: []const u8) usize {
    return std.mem.readInt(u32, data[offsetTableStart(data)..][0..4], .little);
}

fn buildArraySerialized(allocator: std.mem.Allocator) ![]u8 {
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(10);
    _ = try bm.add(20);
    _ = try bm.add(30);

    return bm.serialize(allocator);
}

fn buildBitsetSerialized(allocator: std.mem.Allocator) ![]u8 {
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    for (0..5000) |i| {
        _ = try bm.add(@as(u32, @intCast(i)) * 13);
    }

    return bm.serialize(allocator);
}

fn buildRunSerialized(allocator: std.mem.Allocator) ![]u8 {
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.addRange(100, 200);
    _ = try bm.runOptimize();

    return bm.serialize(allocator);
}

fn linearMinimum(frozen: *const FrozenBitmap) ?u32 {
    var iter = frozen.iterator();
    return iter.next();
}

fn linearMaximum(frozen: *const FrozenBitmap) ?u32 {
    var iter = frozen.iterator();
    var result: ?u32 = null;
    while (iter.next()) |value| result = value;
    return result;
}

fn linearRank(frozen: *const FrozenBitmap, value: u32) u64 {
    var count: u64 = 0;
    var iter = frozen.iterator();
    while (iter.next()) |current| {
        if (current > value) break;
        count += 1;
    }
    return count;
}

fn linearGetIndex(frozen: *const FrozenBitmap, value: u32) ?u64 {
    var index: u64 = 0;
    var iter = frozen.iterator();
    while (iter.next()) |current| {
        if (current == value) return index;
        if (current > value) return null;
        index += 1;
    }
    return null;
}

fn linearSelect(frozen: *const FrozenBitmap, rank: u64) ?u32 {
    var index: u64 = 0;
    var iter = frozen.iterator();
    while (iter.next()) |current| {
        if (index == rank) return current;
        index += 1;
    }
    return null;
}

const ExpectedContainerType = enum { array, bitset, run };

fn frozenContainerType(frozen: *const FrozenBitmap, idx: usize) ExpectedContainerType {
    if (frozen.isRunContainer(idx)) return .run;
    if (frozen.getCardinality(idx) > ArrayContainer.MAX_CARDINALITY) return .bitset;
    return .array;
}

fn expectCaseU64(case_name: []const u8, operation: []const u8, input: u64, expected: u64, actual: u64) !void {
    if (expected == actual) return;
    std.debug.print("frozen differential failed: case={s} operation={s} input={d} expected={d} actual={d}\n", .{
        case_name,
        operation,
        input,
        expected,
        actual,
    });
    return error.FrozenDifferentialMismatch;
}

fn expectCaseOptionalU32(
    case_name: []const u8,
    operation: []const u8,
    input: u64,
    expected: ?u32,
    actual: ?u32,
) !void {
    if (expected == actual) return;
    std.debug.print("frozen differential failed: case={s} operation={s} input={d} expected={?d} actual={?d}\n", .{
        case_name,
        operation,
        input,
        expected,
        actual,
    });
    return error.FrozenDifferentialMismatch;
}

fn expectCaseOptionalU64(
    case_name: []const u8,
    operation: []const u8,
    input: u64,
    expected: ?u64,
    actual: ?u64,
) !void {
    if (expected == actual) return;
    std.debug.print("frozen differential failed: case={s} operation={s} input={d} expected={?d} actual={?d}\n", .{
        case_name,
        operation,
        input,
        expected,
        actual,
    });
    return error.FrozenDifferentialMismatch;
}

fn expectFrozenQueryAgreement(
    allocator: std.mem.Allocator,
    case_name: []const u8,
    expected_type: ?ExpectedContainerType,
    bitmap: *const RoaringBitmap,
    probes: []const u32,
) !void {
    const serialized = try bitmap.serialize(allocator);
    defer allocator.free(serialized);
    var frozen = try FrozenBitmap.init(serialized);
    defer frozen.deinit();

    if (expected_type) |expected| {
        for (0..frozen.size) |idx| {
            const actual = frozenContainerType(&frozen, idx);
            if (actual != expected) {
                std.debug.print("frozen differential failed: case={s} container={d} expected-type={s} actual-type={s}\n", .{
                    case_name,
                    idx,
                    @tagName(expected),
                    @tagName(actual),
                });
                return error.FrozenDifferentialMismatch;
            }
        }
    }

    try expectCaseOptionalU32(case_name, "minimum/bitmap", 0, bitmap.minimum(), frozen.minimum());
    try expectCaseOptionalU32(case_name, "minimum/linear", 0, linearMinimum(&frozen), frozen.minimum());
    try expectCaseOptionalU32(case_name, "maximum/bitmap", 0, bitmap.maximum(), frozen.maximum());
    try expectCaseOptionalU32(case_name, "maximum/linear", 0, linearMaximum(&frozen), frozen.maximum());

    for (probes) |probe| {
        try expectCaseU64(case_name, "rank/bitmap", probe, bitmap.rank(probe), frozen.rank(probe));
        try expectCaseU64(case_name, "rank/linear", probe, linearRank(&frozen, probe), frozen.rank(probe));
        try expectCaseOptionalU64(case_name, "getIndex/bitmap", probe, bitmap.getIndex(probe), frozen.getIndex(probe));
        try expectCaseOptionalU64(case_name, "getIndex/linear", probe, linearGetIndex(&frozen, probe), frozen.getIndex(probe));
    }

    const card = bitmap.cardinality();
    var rank: u64 = 0;
    while (rank <= card) : (rank += 1) {
        try expectCaseOptionalU32(case_name, "select/bitmap", rank, bitmap.select(rank), frozen.select(rank));
        try expectCaseOptionalU32(case_name, "select/linear", rank, linearSelect(&frozen, rank), frozen.select(rank));
    }
}

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

test "FrozenBitmap positional queries match bitmap and linear oracle" {
    const allocator = std.testing.allocator;

    var empty = try RoaringBitmap.init(allocator);
    defer empty.deinit();
    try expectFrozenQueryAgreement(allocator, "empty", null, &empty, &.{ 0, 1, std.math.maxInt(u32) });

    var array = try RoaringBitmap.init(allocator);
    defer array.deinit();
    for ([_]u32{ 10, 20, 30, 65_535 }) |value| _ = try array.add(value);
    try expectFrozenQueryAgreement(allocator, "single-array", .array, &array, &.{ 0, 9, 10, 11, 20, 21, 65_535, 65_536 });

    var bitset = try RoaringBitmap.init(allocator);
    defer bitset.deinit();
    var value: u32 = 0;
    while (value < 10_000) : (value += 2) _ = try bitset.add(value);
    try expectFrozenQueryAgreement(allocator, "single-bitset", .bitset, &bitset, &.{ 0, 1, 5_000, 9_998, 9_999, 10_000 });

    var run = try RoaringBitmap.init(allocator);
    defer run.deinit();
    _ = try run.addRange(100, 200);
    _ = try run.runOptimize();
    try expectFrozenQueryAgreement(allocator, "single-run", .run, &run, &.{ 0, 99, 100, 150, 200, 201, 65_535 });

    var boundary_runs = try RoaringBitmap.init(allocator);
    defer boundary_runs.deinit();
    _ = try boundary_runs.addRange(65_530, 65_540);
    _ = try boundary_runs.runOptimize();
    try expectFrozenQueryAgreement(allocator, "run-container-boundary", .run, &boundary_runs, &.{
        65_529,
        65_530,
        65_535,
        65_536,
        65_540,
        65_541,
    });

    var disjoint_runs = try RoaringBitmap.init(allocator);
    defer disjoint_runs.deinit();
    _ = try disjoint_runs.addRange(10, 20);
    _ = try disjoint_runs.addRange(100, 130);
    _ = try disjoint_runs.addRange(1_000, 1_020);
    _ = try disjoint_runs.runOptimize();
    try expectFrozenQueryAgreement(allocator, "multiple-disjoint-runs", .run, &disjoint_runs, &.{
        0,
        9,
        10,
        20,
        21,
        99,
        100,
        130,
        131,
        999,
        1_000,
        1_020,
        1_021,
    });
}

test "FrozenBitmap rejects truncated array container data" {
    const allocator = std.testing.allocator;

    const serialized = try buildArraySerialized(allocator);
    defer allocator.free(serialized);

    try std.testing.expectError(error.InvalidFormat, FrozenBitmap.init(serialized[0 .. serialized.len - 1]));
}

test "FrozenBitmap rejects truncated bitset container data" {
    const allocator = std.testing.allocator;

    const serialized = try buildBitsetSerialized(allocator);
    defer allocator.free(serialized);

    try std.testing.expectError(error.InvalidFormat, FrozenBitmap.init(serialized[0 .. serialized.len - 1]));
}

test "FrozenBitmap rejects truncated run container data" {
    const allocator = std.testing.allocator;

    const serialized = try buildRunSerialized(allocator);
    defer allocator.free(serialized);

    const corrupted = try allocator.dupe(u8, serialized);
    defer allocator.free(corrupted);

    writeU16LE(corrupted, descStart(corrupted) + 4, 2);
    try std.testing.expectError(error.InvalidFormat, FrozenBitmap.init(corrupted));
}

test "FrozenBitmap rejects offset before data region" {
    const allocator = std.testing.allocator;

    const serialized = try buildArraySerialized(allocator);
    defer allocator.free(serialized);

    const corrupted = try allocator.dupe(u8, serialized);
    defer allocator.free(corrupted);

    writeU32LE(corrupted, offsetTableStart(corrupted), @intCast(firstDataOffset(corrupted) - 1));
    try std.testing.expectError(error.InvalidFormat, FrozenBitmap.init(corrupted));
}

test "FrozenBitmap rejects offset past buffer" {
    const allocator = std.testing.allocator;

    const serialized = try buildArraySerialized(allocator);
    defer allocator.free(serialized);

    const corrupted = try allocator.dupe(u8, serialized);
    defer allocator.free(corrupted);

    writeU32LE(corrupted, offsetTableStart(corrupted), @intCast(corrupted.len + 1));
    try std.testing.expectError(error.InvalidFormat, FrozenBitmap.init(corrupted));
}

test "FrozenBitmap rejects no-run size above maximum container count" {
    const allocator = std.testing.allocator;

    const size = 65_537;
    const header_len = 8 + size * 4 + size * 4;
    const corrupted = try allocator.alloc(u8, header_len);
    defer allocator.free(corrupted);
    @memset(corrupted, 0);

    writeU32LE(corrupted, 0, fmt.SERIAL_COOKIE_NO_RUNCONTAINER);
    writeU32LE(corrupted, 4, 65_537);
    try std.testing.expectError(error.InvalidFormat, FrozenBitmap.init(corrupted));
}
