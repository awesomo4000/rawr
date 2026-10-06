// SPDX-License-Identifier: MPL-2.0
const std = @import("std");
const common = @import("bench_large_alloc_common.zig");
const dashboard = @import("bench_croaring.zig");
const Self = @This();
row: dashboard.ParityRow,
kind: dashboard.ParityAllocator,
page_size: usize,

pub fn main(init: std.process.Init) !void {
    try common.main(Self, init);
}
pub fn prepare(options: common.Options, page_size: usize) !Self {
    if (options.batch != 1) return error.CanonicalBatchMustBeOne;
    const row: dashboard.ParityRow = switch (options.mode) {
        .serialize => .serialize,
        .to_array_alloc => .to_array_alloc,
        else => return error.BadMode,
    };
    const kind: dashboard.ParityAllocator = switch (options.kind) {
        .smp => .smp,
        .libc => .libc,
        else => return error.BadAllocator,
    };
    dashboard.parityPrepare(row, .rawr);
    return .{ .row = row, .kind = kind, .page_size = page_size };
}
pub fn operation(self: *Self) !void {
    _ = dashboard.parityRun(self.row, .rawr, self.kind);
}
pub fn validateAndInfo(self: *Self) !common.Info {
    // The observed request is from a post-timing production invocation, not
    // an instrumented allocation in the authoritative timed loop.
    const observation = try dashboard.parityLargeOutputObservation(self.row, self.kind);
    try dashboard.parityValidate(self.row, self.kind);
    const offset = observation.address % self.page_size;
    return .{ .bytes = observation.bytes, .alignment = observation.alignment, .offset_min = offset, .offset_max = offset };
}
pub fn deinit(_: *Self) void {
    dashboard.parityCleanup();
}
