extern fn obsd_repro_hello() callconv(.c) void;

pub fn main() void {
    obsd_repro_hello();
}
