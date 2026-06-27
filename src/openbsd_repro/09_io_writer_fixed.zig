const std = @import("std");

extern fn obsd_repro_mark(message: [*:0]const u8) callconv(.c) void;
extern fn obsd_repro_write(ptr: [*]const u8, len: usize) callconv(.c) void;

pub fn main() void {
    obsd_repro_mark("before std.Io.Writer.fixed");

    var buffer: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    writer.print("std.Io.Writer.fixed value={d}\n", .{42}) catch {
        obsd_repro_mark("std.Io.Writer.print returned an error");
        return;
    };

    obsd_repro_mark("after std.Io.Writer.fixed");
    const out = writer.buffered();
    obsd_repro_write(out.ptr, out.len);
}
