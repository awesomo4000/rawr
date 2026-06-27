const bench_time = @import("bench_time");
const c = @import("c");

pub fn main() void {
    bench_time.print("before bench_time.print with CRoaring linked\n", .{});

    const bm = c.roaring_bitmap_create();
    if (bm == null) {
        bench_time.print("CRoaring create returned null\n", .{});
        return;
    }
    defer c.roaring_bitmap_free(bm);

    c.roaring_bitmap_add(bm, 1);
    c.roaring_bitmap_add(bm, 65_537);

    bench_time.print("after CRoaring add cardinality={d}\n", .{c.roaring_bitmap_get_cardinality(bm)});
}
