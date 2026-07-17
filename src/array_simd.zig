// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const builtin = @import("builtin");

pub const has_x86_simd = builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx) and
    std.Target.x86.featureSetHas(builtin.cpu.features, .ssse3);
pub const has_neon = builtin.cpu.arch == .aarch64 and
    std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon);

fn genShuffleEntry(comptime mask: u8) [16]u8 {
    var entry: [16]u8 = @splat(0xff);
    var out: usize = 0;
    inline for (0..8) |lane| {
        if (mask & (1 << lane) != 0) {
            entry[out * 2] = @intCast(lane * 2);
            entry[out * 2 + 1] = @intCast(lane * 2 + 1);
            out += 1;
        }
    }
    return entry;
}

pub const shuffle_mask16: [256][16]u8 = blk: {
    @setEvalBranchQuota(10_000);
    var table: [256][16]u8 = undefined;
    for (&table, 0..) |*entry, mask| {
        entry.* = genShuffleEntry(@intCast(mask));
    }
    break :blk table;
};

inline fn compareAny(comptime movemask8: anytype, a: @Vector(8, u16), b: @Vector(8, u16)) u8 {
    var matches: @Vector(8, bool) = @splat(false);
    inline for (0..8) |lane| {
        matches = matches | (a == @as(@Vector(8, u16), @splat(b[lane])));
    }
    return movemask8(matches);
}

inline fn movemaskX86(matches: @Vector(8, bool)) u8 {
    return @bitCast(matches);
}

inline fn movemaskNeon(matches: @Vector(8, bool)) u8 {
    return @bitCast(matches);
}

inline fn shuffleX86(value: @Vector(16, u8), mask: [16]u8) @Vector(16, u8) {
    const mask_vector: @Vector(16, u8) = mask;
    return asm ("vpshufb %[mask], %[value], %[out]"
        : [out] "=x" (-> @Vector(16, u8)),
        : [value] "x" (value),
          [mask] "x" (mask_vector),
    );
}

inline fn shuffleNeon(value: @Vector(16, u8), mask: [16]u8) @Vector(16, u8) {
    const mask_vector: @Vector(16, u8) = mask;
    return asm ("tbl %[out].16b, { %[value].16b }, %[mask].16b"
        : [out] "=w" (-> @Vector(16, u8)),
        : [value] "w" (value),
          [mask] "w" (mask_vector),
    );
}

fn intersectWriteSimd(
    comptime shuffle: anytype,
    comptime movemask8: anytype,
    a: []const u16,
    b: []const u16,
    out: []u16,
) usize {
    std.debug.assert(out.len >= @min(a.len, b.len));

    var ia: usize = 0;
    var ib: usize = 0;
    var count: usize = 0;
    const block_end_a = a.len & ~@as(usize, 7);
    const block_end_b = b.len & ~@as(usize, 7);

    while (ia < block_end_a and ib < block_end_b) {
        const va: @Vector(8, u16) = a[ia..][0..8].*;
        const vb: @Vector(8, u16) = b[ib..][0..8].*;
        const mask = compareAny(movemask8, va, vb);
        if (mask != 0) {
            const packed_bytes = shuffle(@bitCast(va), shuffle_mask16[mask]);
            const scratch: [8]u16 = @bitCast(packed_bytes);
            const matched: usize = @popCount(mask);
            std.debug.assert(count + matched <= out.len);
            @memcpy(out[count..][0..matched], scratch[0..matched]);
            count += matched;
        }

        const max_a = a[ia + 7];
        const max_b = b[ib + 7];
        if (max_a <= max_b) ia += 8;
        if (max_b <= max_a) ib += 8;
    }

    count += intersectWriteTail(a[ia..], b[ib..], out[count..]);
    return count;
}

fn intersectCardSimd(comptime movemask8: anytype, a: []const u16, b: []const u16) u64 {
    var ia: usize = 0;
    var ib: usize = 0;
    var count: u64 = 0;
    const block_end_a = a.len & ~@as(usize, 7);
    const block_end_b = b.len & ~@as(usize, 7);

    while (ia < block_end_a and ib < block_end_b) {
        const va: @Vector(8, u16) = a[ia..][0..8].*;
        const vb: @Vector(8, u16) = b[ib..][0..8].*;
        count += @popCount(compareAny(movemask8, va, vb));

        const max_a = a[ia + 7];
        const max_b = b[ib + 7];
        if (max_a <= max_b) ia += 8;
        if (max_b <= max_a) ib += 8;
    }

    return count + intersectCardTail(a[ia..], b[ib..]);
}

pub fn intersectWriteX86(a: []const u16, b: []const u16, out: []u16) usize {
    if (comptime !has_x86_simd) unreachable;
    return intersectWriteSimd(shuffleX86, movemaskX86, a, b, out);
}

pub fn intersectCardX86(a: []const u16, b: []const u16) u64 {
    if (comptime !has_x86_simd) unreachable;
    return intersectCardSimd(movemaskX86, a, b);
}

pub fn intersectWriteNeon(a: []const u16, b: []const u16, out: []u16) usize {
    if (comptime !has_neon) unreachable;
    return intersectWriteSimd(shuffleNeon, movemaskNeon, a, b, out);
}

pub fn intersectCardNeon(a: []const u16, b: []const u16) u64 {
    if (comptime !has_neon) unreachable;
    return intersectCardSimd(movemaskNeon, a, b);
}

fn intersectWriteTail(a: []const u16, b: []const u16, out: []u16) usize {
    var ia: usize = 0;
    var ib: usize = 0;
    var count: usize = 0;
    while (ia < a.len and ib < b.len) {
        const av = a[ia];
        const bv = b[ib];
        if (av == bv) {
            out[count] = av;
            count += 1;
        }
        ia += @intFromBool(av <= bv);
        ib += @intFromBool(bv <= av);
    }
    return count;
}

fn intersectCardTail(a: []const u16, b: []const u16) u64 {
    var ia: usize = 0;
    var ib: usize = 0;
    var count: u64 = 0;
    while (ia < a.len and ib < b.len) {
        const av = a[ia];
        const bv = b[ib];
        count += @intFromBool(av == bv);
        ia += @intFromBool(av <= bv);
        ib += @intFromBool(bv <= av);
    }
    return count;
}

test "x86 SIMD intersection handles block boundaries and tails" {
    if (!has_x86_simd) return error.SkipZigTest;

    const a = [_]u16{ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32 };
    const b = [_]u16{ 1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30, 33 };
    const expected = [_]u16{ 2, 6, 10, 14, 18, 22, 26, 30 };
    var out: [a.len]u16 = undefined;

    const len = intersectWriteX86(&a, &b, &out);
    try std.testing.expectEqualSlices(u16, &expected, out[0..len]);
    try std.testing.expectEqual(@as(u64, expected.len), intersectCardX86(&a, &b));
}

test "x86 SIMD intersection matches scalar randomized inputs" {
    if (!has_x86_simd) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(0x11_05_51_4d_44);
    var values_a: [4096]u16 = undefined;
    var values_b: [4096]u16 = undefined;
    var expected: [4096]u16 = undefined;
    var actual: [4096]u16 = undefined;
    var present_a: [1 << 16]bool = undefined;
    var present_b: [1 << 16]bool = undefined;

    for (0..1000) |iteration| {
        const len_a = if (iteration < 80) iteration % 40 else prng.random().uintLessThan(usize, values_a.len + 1);
        const len_b = if (iteration < 80) (iteration * 7) % 40 else prng.random().uintLessThan(usize, values_b.len + 1);
        fillExactSorted(prng.random(), values_a[0..len_a], &present_a);
        fillExactSorted(prng.random(), values_b[0..len_b], &present_b);

        const expected_len = intersectWriteTail(values_a[0..len_a], values_b[0..len_b], &expected);
        const actual_len = intersectWriteX86(values_a[0..len_a], values_b[0..len_b], &actual);
        try std.testing.expectEqual(expected_len, actual_len);
        try std.testing.expectEqualSlices(u16, expected[0..expected_len], actual[0..actual_len]);
        try std.testing.expectEqual(@as(u64, expected_len), intersectCardX86(values_a[0..len_a], values_b[0..len_b]));
    }
}

test "NEON intersection handles block boundaries and tails" {
    if (!has_neon) return error.SkipZigTest;

    const a = [_]u16{ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32 };
    const b = [_]u16{ 1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30, 33 };
    const expected = [_]u16{ 2, 6, 10, 14, 18, 22, 26, 30 };
    var out: [a.len]u16 = undefined;

    const len = intersectWriteNeon(&a, &b, &out);
    try std.testing.expectEqualSlices(u16, &expected, out[0..len]);
    try std.testing.expectEqual(@as(u64, expected.len), intersectCardNeon(&a, &b));
}

test "NEON intersection matches scalar randomized inputs" {
    if (!has_neon) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(0x11_06_4e_45_4f_4e);
    var values_a: [4096]u16 = undefined;
    var values_b: [4096]u16 = undefined;
    var expected: [4096]u16 = undefined;
    var actual: [4096]u16 = undefined;
    var present_a: [1 << 16]bool = undefined;
    var present_b: [1 << 16]bool = undefined;

    for (0..1000) |iteration| {
        const len_a = if (iteration < 80) iteration % 40 else prng.random().uintLessThan(usize, values_a.len + 1);
        const len_b = if (iteration < 80) (iteration * 7) % 40 else prng.random().uintLessThan(usize, values_b.len + 1);
        fillExactSorted(prng.random(), values_a[0..len_a], &present_a);
        fillExactSorted(prng.random(), values_b[0..len_b], &present_b);

        const expected_len = intersectWriteTail(values_a[0..len_a], values_b[0..len_b], &expected);
        const actual_len = intersectWriteNeon(values_a[0..len_a], values_b[0..len_b], &actual);
        try std.testing.expectEqual(expected_len, actual_len);
        try std.testing.expectEqualSlices(u16, expected[0..expected_len], actual[0..actual_len]);
        try std.testing.expectEqual(@as(u64, expected_len), intersectCardNeon(values_a[0..len_a], values_b[0..len_b]));
    }
}

fn fillExactSorted(random: std.Random, out: []u16, present: *[1 << 16]bool) void {
    @memset(present, false);

    var count: usize = 0;
    while (count < out.len) {
        const value = random.int(u16);
        if (!present[value]) {
            present[value] = true;
            count += 1;
        }
    }

    count = 0;
    for (present, 0..) |is_present, value| {
        if (is_present) {
            out[count] = @intCast(value);
            count += 1;
        }
    }
}
