const std = @import("std");

extern fn obsd_repro_mark(message: [*:0]const u8) callconv(.c) void;
extern fn obsd_repro_report_u64(label: [*:0]const u8, value: u64) callconv(.c) void;

pub fn main() void {
    obsd_repro_mark("before std.c.clock_gettime");

    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) {
        obsd_repro_mark("std.c.clock_gettime returned nonzero");
        return;
    }

    obsd_repro_mark("after std.c.clock_gettime");
    obsd_repro_report_u64("sec", @intCast(ts.sec));
    obsd_repro_report_u64("nsec", @intCast(ts.nsec));
}
