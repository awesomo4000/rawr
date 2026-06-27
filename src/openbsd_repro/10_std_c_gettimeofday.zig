const std = @import("std");

extern fn obsd_repro_mark(message: [*:0]const u8) callconv(.c) void;
extern fn obsd_repro_report_u64(label: [*:0]const u8, value: u64) callconv(.c) void;

pub fn main() void {
    obsd_repro_mark("before std.c.gettimeofday");

    var tv: std.c.timeval = undefined;
    if (std.c.gettimeofday(&tv, null) != 0) {
        obsd_repro_mark("std.c.gettimeofday returned nonzero");
        return;
    }

    obsd_repro_mark("after std.c.gettimeofday");
    obsd_repro_report_u64("sec", @intCast(tv.sec));
    obsd_repro_report_u64("usec", @intCast(tv.usec));
}
