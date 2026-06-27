const std = @import("std");
const builtin = @import("builtin");
const rawr = @import("rawr");
const bench_time = @import("bench_time");
const c = @import("c");

const allocator = if (builtin.os.tag == .openbsd) bench_time.openbsd_c_allocator else std.heap.smp_allocator;

pub fn main() !void {
    bench_time.print("bench_croaring import shape without globals\n", .{});

    var rawr_bm = try rawr.RoaringBitmap.init(allocator);
    defer rawr_bm.deinit();
    _ = try rawr_bm.add(1);

    const cr_bm = c.roaring_bitmap_create();
    if (cr_bm == null) {
        bench_time.print("CRoaring create returned null\n", .{});
        return;
    }
    defer c.roaring_bitmap_free(cr_bm);
    c.roaring_bitmap_add(cr_bm, 1);

    bench_time.print("rawr cardinality={d} CRoaring cardinality={d}\n", .{
        rawr_bm.cardinality(),
        c.roaring_bitmap_get_cardinality(cr_bm),
    });
}
