const bench_time = @import("bench_time");

pub fn main() void {
    bench_time.print("bench_time.print literal\n", .{});
    bench_time.print("bench_time.print formatted value={d} text={s} ratio={d:.2}\n", .{ 42, "ok", 1.25 });
}
