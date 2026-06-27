const c = @import("c");

extern fn rawr_bench_write_stderr(ptr: [*]const u8, len: usize) callconv(.c) void;

pub fn main() void {
    writeLiteral("direct OpenBSD bench shim write with CRoaring linked\n");

    const bm = c.roaring_bitmap_create();
    if (bm == null) {
        writeLiteral("CRoaring create returned null\n");
        return;
    }
    defer c.roaring_bitmap_free(bm);
    c.roaring_bitmap_add(bm, 42);

    writeLiteral("after CRoaring add\n");
}

fn writeLiteral(comptime bytes: []const u8) void {
    if (bytes.len != 0) rawr_bench_write_stderr(bytes.ptr, bytes.len);
}
