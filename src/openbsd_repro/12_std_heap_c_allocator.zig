const std = @import("std");

extern fn obsd_repro_mark(message: [*:0]const u8) callconv(.c) void;
extern fn obsd_repro_report_ptr(label: [*:0]const u8, ptr: ?*const anyopaque, len: usize) callconv(.c) void;

pub fn main() void {
    obsd_repro_mark("before std.heap.c_allocator.alloc");

    const allocator = std.heap.c_allocator;
    const memory = allocator.alloc(u8, 32) catch {
        obsd_repro_mark("std.heap.c_allocator.alloc returned OOM");
        return;
    };
    defer allocator.free(memory);
    memory[0] = 0xaa;

    obsd_repro_mark("after std.heap.c_allocator.alloc");
    obsd_repro_report_ptr("allocated", memory.ptr, memory.len);
}
