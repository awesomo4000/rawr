const std = @import("std");
const rawr = @import("rawr");
const c = @cImport(@cInclude("roaring.h"));

pub fn main() !void {
    std.debug.print("CRoaring Benchmark Comparison\n", .{});
    std.debug.print("=============================\n", .{});
    std.debug.print("TODO: implement comparison benchmarks\n", .{});
}
