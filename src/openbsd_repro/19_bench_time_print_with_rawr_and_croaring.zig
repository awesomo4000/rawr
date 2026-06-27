const rawr = @import("rawr");
const bench_time = @import("bench_time");
const c = @import("c");

pub fn main() !void {
    bench_time.print("before bench_time.print with rawr and CRoaring linked\n", .{});

    var rawr_bm = try rawr.RoaringBitmap.init(bench_time.cAllocator());
    defer rawr_bm.deinit();
    _ = try rawr_bm.add(1);
    _ = try rawr_bm.add(65_537);

    const cr_bm = c.roaring_bitmap_create();
    if (cr_bm == null) {
        bench_time.print("CRoaring create returned null\n", .{});
        return;
    }
    defer c.roaring_bitmap_free(cr_bm);
    c.roaring_bitmap_add(cr_bm, 1);
    c.roaring_bitmap_add(cr_bm, 65_537);

    bench_time.print("rawr cardinality={d} CRoaring cardinality={d}\n", .{
        rawr_bm.cardinality(),
        c.roaring_bitmap_get_cardinality(cr_bm),
    });
}
