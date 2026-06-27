const std = @import("std");

extern fn obsd_repro_mark(message: [*:0]const u8) callconv(.c) void;
extern fn obsd_repro_write(ptr: [*]const u8, len: usize) callconv(.c) void;

pub fn main() void {
    obsd_repro_mark("before std.fmt.bufPrint");

    var buffer: [128]u8 = undefined;
    const out = std.fmt.bufPrint(&buffer, "std.fmt.bufPrint value={d} text={s}\n", .{ 42, "ok" }) catch {
        obsd_repro_mark("std.fmt.bufPrint returned an error");
        return;
    };

    obsd_repro_mark("after std.fmt.bufPrint");
    obsd_repro_write(out.ptr, out.len);
}
