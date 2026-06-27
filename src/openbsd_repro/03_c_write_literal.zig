extern fn obsd_repro_write(ptr: [*]const u8, len: usize) callconv(.c) void;

pub fn main() void {
    const message = "zig comptime literal ptr/len to C fwrite\n";
    obsd_repro_write(message.ptr, message.len);
}
