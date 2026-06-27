const bench_time = @import("bench_time.zig");

pub fn main() void {
    bench_time.print("Rawr vs CRoaring Benchmark Comparison\n", .{});
    bench_time.print("======================================\n", .{});
    bench_time.printRunTimestamp();
    bench_time.print("N = {d} values, {d} warmup, {d} timed runs (median)\n", .{ 1_000_000, 3, 21 });
}
