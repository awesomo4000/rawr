const rawr = @import("rawr");
const bench_time = @import("bench_time");
const c = @import("c");

extern fn rawr_bench_write_stderr(ptr: [*]const u8, len: usize) callconv(.c) void;

pub fn main() !void {
    writeLiteral("direct OpenBSD bench shim write with rawr and CRoaring linked\n");

    var rawr_bm = try rawr.RoaringBitmap.init(bench_time.cAllocator());
    defer rawr_bm.deinit();
    _ = try rawr_bm.add(7);

    const cr_bm = c.roaring_bitmap_create();
    if (cr_bm == null) {
        writeLiteral("CRoaring create returned null\n");
        return;
    }
    defer c.roaring_bitmap_free(cr_bm);
    c.roaring_bitmap_add(cr_bm, 7);

    writeLiteral("after rawr and CRoaring add\n");
}

fn writeLiteral(comptime bytes: []const u8) void {
    if (bytes.len != 0) rawr_bench_write_stderr(bytes.ptr, bytes.len);
}
