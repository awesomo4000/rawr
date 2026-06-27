extern fn obsd_repro_write(ptr: [*]const u8, len: usize) callconv(.c) void;

const message = "zig global const ptr/len to C fwrite\n";

pub fn main() void {
    obsd_repro_write(message.ptr, message.len);
}
