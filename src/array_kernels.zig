// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const array_simd = @import("array_simd.zig");

const skew_threshold = 64; // Scalar merge-vs-gallop crossover.
const x86_skew_threshold = 12; // Measured x86 SIMD-vs-gallop crossover.
const neon_skew_threshold = 40; // Measured AArch64 NEON-vs-gallop crossover.

pub const WriteKernel = *const fn ([]const u16, []const u16, []u16) usize;
pub const CardKernel = *const fn ([]const u16, []const u16) u64;
pub const BoolKernel = *const fn ([]const u16, []const u16) bool;

pub const WriteBenchKernel = struct {
    name: []const u8,
    func: WriteKernel,
};

pub const CardBenchKernel = struct {
    name: []const u8,
    func: CardKernel,
};

pub const BoolBenchKernel = struct {
    name: []const u8,
    func: BoolKernel,
};

pub const write_bench_kernels = if (array_simd.has_x86_simd)
    [_]WriteBenchKernel{
        .{ .name = "dispatch", .func = intersectWrite },
        .{ .name = "simd-x86", .func = array_simd.intersectWriteX86 },
        .{ .name = "gallop", .func = intersectWriteGallop },
        .{ .name = "merge", .func = intersectWriteMerge },
    }
else if (array_simd.has_neon)
    [_]WriteBenchKernel{
        .{ .name = "dispatch", .func = intersectWrite },
        .{ .name = "simd-neon", .func = array_simd.intersectWriteNeon },
        .{ .name = "gallop", .func = intersectWriteGallop },
        .{ .name = "merge", .func = intersectWriteMerge },
    }
else
    [_]WriteBenchKernel{
        .{ .name = "dispatch", .func = intersectWrite },
        .{ .name = "gallop", .func = intersectWriteGallop },
        .{ .name = "merge", .func = intersectWriteMerge },
    };

pub const card_bench_kernels = if (array_simd.has_x86_simd)
    [_]CardBenchKernel{
        .{ .name = "dispatch", .func = intersectCard },
        .{ .name = "simd-x86", .func = array_simd.intersectCardX86 },
        .{ .name = "gallop", .func = intersectCardGallop },
        .{ .name = "merge", .func = intersectCardMerge },
    }
else if (array_simd.has_neon)
    [_]CardBenchKernel{
        .{ .name = "dispatch", .func = intersectCard },
        .{ .name = "simd-neon", .func = array_simd.intersectCardNeon },
        .{ .name = "gallop", .func = intersectCardGallop },
        .{ .name = "merge", .func = intersectCardMerge },
    }
else
    [_]CardBenchKernel{
        .{ .name = "dispatch", .func = intersectCard },
        .{ .name = "gallop", .func = intersectCardGallop },
        .{ .name = "merge", .func = intersectCardMerge },
    };

pub const bool_bench_kernels = [_]BoolBenchKernel{
    .{ .name = "dispatch", .func = intersectBool },
    .{ .name = "gallop", .func = intersectBoolGallop },
    .{ .name = "merge", .func = intersectBoolMerge },
};

pub inline fn lowerBound(values: []const u16, target: u16) usize {
    var lo: usize = 0;
    var hi = values.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (values[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/// Find the first element in `arr[start..]` greater than or equal to `target`.
pub fn gallopSearch(arr: []const u16, target: u16, start: usize) usize {
    if (start >= arr.len) return arr.len;

    var step: usize = 1;
    var hi = start;
    while (hi < arr.len and arr[hi] < target) {
        hi += step;
        step *= 2;
    }
    if (hi > arr.len) hi = arr.len;

    var lo = if (step > 2) hi -| (step / 2) else start;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (arr[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

fn orderedBySize(a: []const u16, b: []const u16) struct {
    small: []const u16,
    big: []const u16,
} {
    return if (a.len <= b.len)
        .{ .small = a, .big = b }
    else
        .{ .small = b, .big = a };
}

fn shouldGallop(a_len: usize, b_len: usize) bool {
    return shouldGallopAtRatio(a_len, b_len, skew_threshold);
}

fn shouldGallopWriteOrCard(a_len: usize, b_len: usize) bool {
    const threshold = if (array_simd.has_x86_simd)
        x86_skew_threshold
    else if (array_simd.has_neon)
        neon_skew_threshold
    else
        skew_threshold;
    return shouldGallopAtRatio(a_len, b_len, threshold);
}

fn shouldGallopAtRatio(a_len: usize, b_len: usize, threshold: u32) bool {
    const small: u32 = @intCast(@min(a_len, b_len));
    const big: u32 = @intCast(@max(a_len, b_len));
    return small * threshold <= big;
}

pub fn intersectWrite(a: []const u16, b: []const u16, out: []u16) usize {
    return if (shouldGallopWriteOrCard(a.len, b.len))
        intersectWriteGallop(a, b, out)
    else if (comptime array_simd.has_x86_simd)
        array_simd.intersectWriteX86(a, b, out)
    else if (comptime array_simd.has_neon)
        array_simd.intersectWriteNeon(a, b, out)
    else
        intersectWriteMerge(a, b, out);
}

pub fn intersectWriteGallop(a: []const u16, b: []const u16, out: []u16) usize {
    const ordered = orderedBySize(a, b);
    std.debug.assert(out.len >= ordered.small.len);

    var count: usize = 0;
    var lo: usize = 0;
    for (ordered.small) |value| {
        lo = gallopSearch(ordered.big, value, lo);
        if (lo < ordered.big.len and ordered.big[lo] == value) {
            out[count] = value;
            count += 1;
            lo += 1;
        }
    }
    return count;
}

pub fn intersectWriteMerge(a: []const u16, b: []const u16, out: []u16) usize {
    std.debug.assert(out.len >= @min(a.len, b.len));

    var i: usize = 0;
    var j: usize = 0;
    var count: usize = 0;
    while (i < a.len and j < b.len) {
        const av = a[i];
        const bv = b[j];
        out[count] = av;
        count += @intFromBool(av == bv);
        i += @intFromBool(av <= bv);
        j += @intFromBool(bv <= av);
    }
    return count;
}

pub fn intersectCard(a: []const u16, b: []const u16) u64 {
    return if (shouldGallopWriteOrCard(a.len, b.len))
        intersectCardGallop(a, b)
    else if (comptime array_simd.has_x86_simd)
        array_simd.intersectCardX86(a, b)
    else if (comptime array_simd.has_neon)
        array_simd.intersectCardNeon(a, b)
    else
        intersectCardMerge(a, b);
}

pub fn intersectCardGallop(a: []const u16, b: []const u16) u64 {
    const ordered = orderedBySize(a, b);

    var count: u64 = 0;
    var lo: usize = 0;
    for (ordered.small) |value| {
        lo = gallopSearch(ordered.big, value, lo);
        if (lo < ordered.big.len and ordered.big[lo] == value) {
            count += 1;
            lo += 1;
        }
    }
    return count;
}

pub fn intersectCardMerge(a: []const u16, b: []const u16) u64 {
    var i: usize = 0;
    var j: usize = 0;
    var count: u64 = 0;
    while (i < a.len and j < b.len) {
        const av = a[i];
        const bv = b[j];
        count += @intFromBool(av == bv);
        i += @intFromBool(av <= bv);
        j += @intFromBool(bv <= av);
    }
    return count;
}

pub fn intersectBool(a: []const u16, b: []const u16) bool {
    return if (shouldGallop(a.len, b.len))
        intersectBoolGallop(a, b)
    else
        intersectBoolMerge(a, b);
}

pub fn intersectBoolGallop(a: []const u16, b: []const u16) bool {
    const ordered = orderedBySize(a, b);

    var lo: usize = 0;
    for (ordered.small) |value| {
        lo = gallopSearch(ordered.big, value, lo);
        if (lo < ordered.big.len and ordered.big[lo] == value) return true;
    }
    return false;
}

pub fn intersectBoolMerge(a: []const u16, b: []const u16) bool {
    var i: usize = 0;
    var j: usize = 0;
    while (i < a.len and j < b.len) {
        const av = a[i];
        const bv = b[j];
        if (av == bv) return true;
        i += @intFromBool(av < bv);
        j += @intFromBool(bv < av);
    }
    return false;
}

test "gallop and merge intersection kernels agree" {
    const a = [_]u16{ 1, 3, 5, 7, 9, 1000 };
    const b = [_]u16{ 0, 3, 4, 7, 10, 1000, 2000 };
    var gallop_out: [a.len]u16 = undefined;
    var merge_out: [a.len]u16 = undefined;

    const gallop_len = intersectWriteGallop(&a, &b, &gallop_out);
    const merge_len = intersectWriteMerge(&a, &b, &merge_out);
    try std.testing.expectEqual(gallop_len, merge_len);
    try std.testing.expectEqualSlices(u16, gallop_out[0..gallop_len], merge_out[0..merge_len]);
    try std.testing.expectEqual(@as(u64, gallop_len), intersectCardGallop(&a, &b));
    try std.testing.expectEqual(@as(u64, merge_len), intersectCardMerge(&a, &b));
    try std.testing.expect(intersectBoolGallop(&a, &b));
    try std.testing.expect(intersectBoolMerge(&a, &b));
}

test "intersection kernels handle empty and disjoint inputs" {
    const empty = [_]u16{};
    const a = [_]u16{ 1, 2, 3 };
    const b = [_]u16{ 4, 5, 6 };
    var out: [3]u16 = undefined;

    try std.testing.expectEqual(@as(usize, 0), intersectWriteGallop(&empty, &a, out[0..0]));
    try std.testing.expectEqual(@as(usize, 0), intersectWriteMerge(&a, &b, &out));
    try std.testing.expectEqual(@as(u64, 0), intersectCardGallop(&a, &b));
    try std.testing.expectEqual(@as(u64, 0), intersectCardMerge(&a, &b));
    try std.testing.expect(!intersectBoolGallop(&a, &b));
    try std.testing.expect(!intersectBoolMerge(&a, &b));
}

test "intersection dispatch uses gallop at inclusive skew boundary" {
    try std.testing.expect(shouldGallop(1, 64));
    try std.testing.expect(shouldGallop(64, 1));
    try std.testing.expect(shouldGallop(0, 64));
    try std.testing.expect(!shouldGallop(1, 63));
    try std.testing.expect(!shouldGallop(64, 64));
}

test "intersection dispatch shapes agree with reference kernels" {
    const a = [_]u16{ 1, 3, 5, 7, 9, 1000 };
    const b = [_]u16{ 0, 3, 4, 7, 10, 1000, 2000 };
    var dispatch_out: [a.len]u16 = undefined;
    var reference_out: [a.len]u16 = undefined;

    const dispatch_len = intersectWrite(&a, &b, &dispatch_out);
    const reference_len = intersectWriteMerge(&a, &b, &reference_out);
    try std.testing.expectEqual(reference_len, dispatch_len);
    try std.testing.expectEqualSlices(u16, reference_out[0..reference_len], dispatch_out[0..dispatch_len]);
    try std.testing.expectEqual(@as(u64, reference_len), intersectCard(&a, &b));
    try std.testing.expectEqual(reference_len != 0, intersectBool(&a, &b));
}

test "lower bound handles hits misses and insertion points" {
    const keys = [_]u16{ 2, 6, 10, 14 };

    try std.testing.expectEqual(@as(usize, 0), lowerBound(&keys, 1));
    try std.testing.expectEqual(@as(usize, 0), lowerBound(&keys, 2));
    try std.testing.expectEqual(@as(usize, 2), lowerBound(&keys, 7));
    try std.testing.expectEqual(@as(usize, 3), lowerBound(&keys, 14));
    try std.testing.expectEqual(@as(usize, keys.len), lowerBound(&keys, 15));
}
