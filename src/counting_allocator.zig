// SPDX-License-Identifier: MPL-2.0

const std = @import("std");

/// Single-threaded allocator instrumentation for benchmarks and tests.
pub const CountingAllocator = struct {
    backing: std.mem.Allocator,
    stats: Stats = .{},

    const Self = @This();

    pub const Stats = struct {
        alloc_calls: u64 = 0,
        free_calls: u64 = 0,
        resize_calls: u64 = 0,
        resize_successes: u64 = 0,
        resize_failures: u64 = 0,
        remap_calls: u64 = 0,
        remap_in_place: u64 = 0,
        remap_moved: u64 = 0,
        remap_failures: u64 = 0,
        cumulative_bytes: u64 = 0,
        live_bytes: u64 = 0,
        peak_live_bytes: u64 = 0,
    };

    pub fn init(backing: std.mem.Allocator) Self {
        return .{ .backing = backing };
    }

    pub fn allocator(self: *Self) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    /// Start a new measurement region without forgetting outstanding allocations.
    pub fn resetStats(self: *Self) void {
        const live_bytes = self.stats.live_bytes;
        self.stats = .{
            .live_bytes = live_bytes,
            .peak_live_bytes = live_bytes,
        };
    }

    pub fn snapshot(self: *const Self) Stats {
        return self.stats;
    }

    const vtable: std.mem.Allocator.VTable = .{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    };

    fn alloc(
        ctx: *anyopaque,
        len: usize,
        alignment: std.mem.Alignment,
        ret_addr: usize,
    ) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.stats.alloc_calls += 1;
        self.addRequested(len);

        const result = self.backing.rawAlloc(len, alignment, ret_addr) orelse return null;
        self.updateLive(0, len);
        return result;
    }

    fn resize(
        ctx: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        ret_addr: usize,
    ) bool {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.stats.resize_calls += 1;
        self.addRequested(new_len);

        if (!self.backing.rawResize(memory, alignment, new_len, ret_addr)) {
            self.stats.resize_failures += 1;
            return false;
        }

        self.stats.resize_successes += 1;
        self.updateLive(memory.len, new_len);
        return true;
    }

    fn remap(
        ctx: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        ret_addr: usize,
    ) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.stats.remap_calls += 1;
        self.addRequested(new_len);

        const result = self.backing.rawRemap(memory, alignment, new_len, ret_addr) orelse {
            self.stats.remap_failures += 1;
            return null;
        };

        if (@intFromPtr(result) == @intFromPtr(memory.ptr)) {
            self.stats.remap_in_place += 1;
        } else {
            self.stats.remap_moved += 1;
        }
        self.updateLive(memory.len, new_len);
        return result;
    }

    fn free(
        ctx: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        ret_addr: usize,
    ) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        std.debug.assert(self.stats.live_bytes >= memory.len);
        self.stats.free_calls += 1;
        self.stats.live_bytes -= memory.len;
        self.backing.rawFree(memory, alignment, ret_addr);
    }

    fn addRequested(self: *Self, len: usize) void {
        self.stats.cumulative_bytes +|= @intCast(len);
    }

    fn updateLive(self: *Self, old_len: usize, new_len: usize) void {
        if (new_len >= old_len) {
            self.stats.live_bytes +|= @intCast(new_len - old_len);
        } else {
            const decrease: u64 = @intCast(old_len - new_len);
            std.debug.assert(self.stats.live_bytes >= decrease);
            self.stats.live_bytes -= decrease;
        }
        self.stats.peak_live_bytes = @max(self.stats.peak_live_bytes, self.stats.live_bytes);
    }
};

test "counting allocator preserves live bytes across reset" {
    var storage: [4096]u8 align(64) = undefined;
    var fixed = std.heap.FixedBufferAllocator.init(&storage);
    var counting = CountingAllocator.init(fixed.allocator());
    const allocator = counting.allocator();

    const memory = try allocator.alignedAlloc(u8, .@"16", 64);
    try std.testing.expectEqual(@as(u64, 1), counting.stats.alloc_calls);
    try std.testing.expectEqual(@as(u64, 64), counting.stats.live_bytes);
    try std.testing.expectEqual(@as(u64, 64), counting.stats.peak_live_bytes);

    counting.resetStats();
    try std.testing.expectEqual(@as(u64, 0), counting.stats.alloc_calls);
    try std.testing.expectEqual(@as(u64, 64), counting.stats.live_bytes);
    try std.testing.expectEqual(@as(u64, 64), counting.stats.peak_live_bytes);

    allocator.free(memory);
    try std.testing.expectEqual(@as(u64, 1), counting.stats.free_calls);
    try std.testing.expectEqual(@as(u64, 0), counting.stats.live_bytes);
}

test "counting allocator records resize operations" {
    var storage: [4096]u8 align(64) = undefined;
    var fixed = std.heap.FixedBufferAllocator.init(&storage);
    var counting = CountingAllocator.init(fixed.allocator());
    const allocator = counting.allocator();

    var memory = try allocator.alignedAlloc(u8, .@"16", 64);
    try std.testing.expect(allocator.resize(memory, 128));
    memory = memory.ptr[0..128];

    try std.testing.expectEqual(@as(u64, 1), counting.stats.resize_calls);
    try std.testing.expectEqual(@as(u64, 1), counting.stats.resize_successes);
    try std.testing.expectEqual(@as(u64, 0), counting.stats.resize_failures);
    try std.testing.expectEqual(@as(u64, 128), counting.stats.live_bytes);

    allocator.free(memory);
}
