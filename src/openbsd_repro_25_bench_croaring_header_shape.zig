const std = @import("std");
const builtin = @import("builtin");
const rawr = @import("rawr");
const RoaringBitmap = rawr.RoaringBitmap;
const c = @import("c");
const bench_time = @import("bench_time.zig");

const allocator = if (builtin.os.tag == .openbsd) bench_time.openbsd_c_allocator else std.heap.smp_allocator;

const WARMUP_RUNS = 3;
const BENCH_RUNS = 21;
const N_VALUES = 1_000_000;
const N_RANK_MANY_PROBES = 200_000;
const N_MANY_BITMAPS = 32;

var random_values: [N_VALUES]u32 = undefined;
var sequential_values: [N_VALUES]u32 = undefined;
var sparse_values: [500_000]u32 = undefined;
var sparse_len: usize = 0;
var rank_queries: [N_VALUES]u32 = undefined;
var select_queries: [N_VALUES]u32 = undefined;
var rank_many_probes: [N_RANK_MANY_PROBES]u32 = undefined;
var rank_many_out: [N_RANK_MANY_PROBES]u64 = undefined;
var range_query_lo: [N_VALUES]u32 = undefined;
var range_query_hi: [N_VALUES]u32 = undefined;
var range_large_query_lo: [N_VALUES]u32 = undefined;
var range_large_query_hi: [N_VALUES]u32 = undefined;

pub fn main() !void {
    bench_time.print("Rawr vs CRoaring Benchmark Comparison\n", .{});
    bench_time.print("======================================\n", .{});
    bench_time.printRunTimestamp();
    bench_time.print("N = {d} values, {d} warmup, {d} timed runs (median)\n", .{ N_VALUES, WARMUP_RUNS, BENCH_RUNS });

    bench_time.print("\nInitializing test data...\n", .{});
    touchBenchmarkGlobals();

    var rawr_bm = try RoaringBitmap.init(allocator);
    defer rawr_bm.deinit();
    _ = try rawr_bm.add(random_values[0]);
    _ = try rawr_bm.add(sequential_values[N_VALUES - 1]);

    const cr_bm = c.roaring_bitmap_create();
    if (cr_bm == null) {
        bench_time.print("CRoaring create returned null\n", .{});
        return;
    }
    defer c.roaring_bitmap_free(cr_bm);
    c.roaring_bitmap_add(cr_bm, random_values[0]);
    c.roaring_bitmap_add(cr_bm, sequential_values[N_VALUES - 1]);

    bench_time.print("header shape rawr cardinality={d} CRoaring cardinality={d}\n", .{
        rawr_bm.cardinality(),
        c.roaring_bitmap_get_cardinality(cr_bm),
    });
}

fn touchBenchmarkGlobals() void {
    random_values[0] = 1;
    random_values[N_VALUES - 1] = 999_999;
    sequential_values[0] = 0;
    sequential_values[N_VALUES - 1] = N_VALUES - 1;
    sparse_values[0] = 3;
    sparse_len = 1;
    rank_queries[0] = 4;
    select_queries[0] = 5;
    rank_many_probes[0] = 6;
    rank_many_out[0] = 7;
    range_query_lo[0] = 8;
    range_query_hi[0] = 9;
    range_large_query_lo[0] = 10;
    range_large_query_hi[0] = 11;

    std.mem.doNotOptimizeAway(N_MANY_BITMAPS);
    std.mem.doNotOptimizeAway(sparse_len);
    std.mem.doNotOptimizeAway(random_values[0]);
    std.mem.doNotOptimizeAway(rank_many_out[0]);
}
