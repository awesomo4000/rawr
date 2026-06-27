extern fn obsd_repro_write(ptr: [*]const u8, len: usize) callconv(.c) void;

pub fn main() void {
    var message = [_]u8{ 'z', 'i', 'g', ' ', 's', 't', 'a', 'c', 'k', ' ', 'b', 'u', 'f', '\n' };
    obsd_repro_write(message[0..].ptr, message.len);
}
