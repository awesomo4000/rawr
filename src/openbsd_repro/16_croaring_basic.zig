const c = @import("c");

extern fn obsd_repro_mark(message: [*:0]const u8) callconv(.c) void;
extern fn obsd_repro_report_u64(label: [*:0]const u8, value: u64) callconv(.c) void;

pub fn main() void {
    obsd_repro_mark("before CRoaring create");

    const bm = c.roaring_bitmap_create();
    if (bm == null) {
        obsd_repro_mark("CRoaring create returned null");
        return;
    }
    defer c.roaring_bitmap_free(bm);

    c.roaring_bitmap_add(bm, 1);
    c.roaring_bitmap_add(bm, 65_537);

    obsd_repro_mark("after CRoaring add");
    obsd_repro_report_u64("cardinality", c.roaring_bitmap_get_cardinality(bm));
}
