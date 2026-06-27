const std = @import("std");

extern fn obsd_repro_mark(message: [*:0]const u8) callconv(.c) void;
extern fn obsd_repro_report_u64(label: [*:0]const u8, value: u64) callconv(.c) void;

pub fn main(init: std.process.Init) !void {
    obsd_repro_mark("before std.process.Init args");

    var args = try init.minimal.args.iterateAllocator(std.heap.smp_allocator);
    defer args.deinit();

    var count: u64 = 0;
    while (args.next()) |_| {
        count += 1;
    }

    obsd_repro_mark("after std.process.Init args");
    obsd_repro_report_u64("argc", count);
}
