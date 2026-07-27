// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const rawr = @import("rawr_range_bench");
const CountingAllocator = @import("counting_allocator.zig").CountingAllocator;

const RoaringBitmap = rawr.RoaringBitmap;
const range_ops = rawr.range_ops;

const range_lo: u32 = 100_000;
const range_hi: u32 = 650_000;

pub fn main() !void {
    std.debug.print("range strategy: direct\n", .{});
    try measureByValueFlip();
    try measureFlipInPlace();
    try measureRemoveRange();
}

fn measureByValueFlip() !void {
    var input = try buildDense(std.heap.smp_allocator);
    defer input.deinit();

    var counting = CountingAllocator.init(std.heap.smp_allocator);
    var result = try range_ops.flip(&input, counting.allocator(), range_lo, range_hi);
    const stats = counting.snapshot();
    const cardinality = result.cardinality();
    result.deinit();
    try expectDrained(&counting);
    printStats("flip", stats, cardinality);
}

fn measureFlipInPlace() !void {
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    var bitmap = try buildDense(counting.allocator());
    counting.resetStats();

    try range_ops.flipInPlace(&bitmap, range_lo, range_hi);
    const stats = counting.snapshot();
    const cardinality = bitmap.cardinality();
    bitmap.deinit();
    try expectDrained(&counting);
    printStats("flipInplace", stats, cardinality);
}

fn measureRemoveRange() !void {
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    var bitmap = try buildDense(counting.allocator());
    counting.resetStats();

    const removed = try range_ops.removeRange(&bitmap, range_lo, range_hi);
    const stats = counting.snapshot();
    bitmap.deinit();
    try expectDrained(&counting);
    printStats("removeRange", stats, removed);
}

fn buildDense(allocator: std.mem.Allocator) !RoaringBitmap {
    var bitmap = try RoaringBitmap.init(allocator);
    errdefer bitmap.deinit();
    _ = try bitmap.addRange(0, 499_999);
    return bitmap;
}

fn expectDrained(counting: *const CountingAllocator) !void {
    if (counting.stats.live_bytes != 0) return error.AllocationProbeLeak;
}

fn printStats(name: []const u8, stats: CountingAllocator.Stats, checksum: u64) void {
    std.debug.print(
        "{s}: alloc={d} free={d} resize={d} remap={d} requested={d} peak={d} checksum={d}\n",
        .{
            name,
            stats.alloc_calls,
            stats.free_calls,
            stats.resize_calls,
            stats.remap_calls,
            stats.cumulative_bytes,
            stats.peak_live_bytes,
            checksum,
        },
    );
}
