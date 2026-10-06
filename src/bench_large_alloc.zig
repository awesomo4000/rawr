// SPDX-License-Identifier: MPL-2.0
//! Bare allocation/write/free worker: no rawr or CRoaring implementation.
const std = @import("std");
const common = @import("bench_large_alloc_common.zig");
const Self = @This();
options: common.Options,
page_size: usize,
retained: ?[]u8 = null,
offset_min: usize = std.math.maxInt(usize),
offset_max: usize = 0,

pub fn main(init: std.process.Init) !void {
    try common.main(Self, init);
}
pub fn prepare(options: common.Options, page_size: usize) !Self {
    if (options.mode == .serialize or options.mode == .to_array_alloc) return error.BadMode;
    var self: Self = .{ .options = options, .page_size = page_size };
    if (options.mode == .retained) {
        self.retained = try options.kind.allocator().alloc(u8, options.bytes);
        @memset(self.retained.?, 0xA5);
        std.mem.doNotOptimizeAway(self.retained.?);
    }
    return self;
}
pub noinline fn operation(self: *Self) !void {
    const allocator = self.options.kind.allocator();
    const buf = self.retained orelse try allocator.alloc(u8, self.options.bytes);
    defer if (self.retained == null) allocator.free(buf);
    const offset = @intFromPtr(buf.ptr) % self.page_size;
    self.offset_min = @min(self.offset_min, offset);
    self.offset_max = @max(self.offset_max, offset);
    if (self.options.mode != .no_write) @memset(buf, 0xA5);
    std.mem.doNotOptimizeAway(buf);
}
pub fn validateAndInfo(self: *Self) !common.Info {
    if (self.retained) |buf| for (buf) |v| {
        if (v != 0xA5) return error.BadPayload;
    };
    return .{ .bytes = self.options.bytes, .alignment = 1, .offset_min = self.offset_min, .offset_max = self.offset_max };
}
pub fn deinit(self: *Self) void {
    if (self.retained) |buf| self.options.kind.allocator().free(buf);
}
