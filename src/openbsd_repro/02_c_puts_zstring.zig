extern fn obsd_repro_mark(message: [*:0]const u8) callconv(.c) void;

pub fn main() void {
    obsd_repro_mark("zig sentinel string to C fputs");
}
