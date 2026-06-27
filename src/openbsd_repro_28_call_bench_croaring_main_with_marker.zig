const bench_croaring = @import("bench_croaring.zig");

extern fn rawr_bench_write_stderr(ptr: [*]const u8, len: usize) callconv(.c) void;

pub fn main() !void {
    writeLiteral("before bench_croaring.main\n");
    try bench_croaring.main();
    writeLiteral("after bench_croaring.main\n");
}

fn writeLiteral(comptime bytes: []const u8) void {
    if (bytes.len != 0) rawr_bench_write_stderr(bytes.ptr, bytes.len);
}
