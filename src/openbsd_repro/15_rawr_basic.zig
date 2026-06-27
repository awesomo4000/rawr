const rawr = @import("rawr");
const bench_time = @import("bench_time");

pub fn main() !void {
    var bm = try rawr.RoaringBitmap.init(bench_time.cAllocator());
    defer bm.deinit();

    _ = try bm.add(1);
    _ = try bm.add(65_537);

    bench_time.print("rawr cardinality={d} contains1={} contains2={}\n", .{
        bm.cardinality(),
        bm.contains(1),
        bm.contains(65_537),
    });
}
