extern fn obsd_repro_mark(message: [*:0]const u8) callconv(.c) void;
extern fn obsd_repro_report_ptr(label: [*:0]const u8, ptr: ?*const anyopaque, len: usize) callconv(.c) void;
extern fn obsd_repro_malloc(size: usize) callconv(.c) ?*anyopaque;
extern fn obsd_repro_free(ptr: ?*anyopaque) callconv(.c) void;

pub fn main() void {
    obsd_repro_mark("before C malloc shim");

    const ptr = obsd_repro_malloc(32);
    if (ptr == null) {
        obsd_repro_mark("C malloc shim returned null");
        return;
    }
    defer obsd_repro_free(ptr);

    obsd_repro_mark("after C malloc shim");
    obsd_repro_report_ptr("allocated", ptr, 32);
}
