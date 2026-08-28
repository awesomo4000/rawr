// SPDX-License-Identifier: MPL-2.0

//! Benchmark-only scalar union and difference candidates for spec 51-01.

const std = @import("std");

pub const Variant = enum {
    bulk_tail,
    branchy,
    branchy_bulk_tail,
};

pub inline fn unionWrite(
    comptime variant: Variant,
    a: []const u16,
    b: []const u16,
    output: []u16,
) usize {
    return switch (variant) {
        .bulk_tail => unionBranchlessBulkTail(a, b, output),
        .branchy => unionBranchy(a, b, output, false),
        .branchy_bulk_tail => unionBranchy(a, b, output, true),
    };
}

pub inline fn differenceWrite(
    comptime variant: Variant,
    a: []const u16,
    b: []const u16,
    output: []u16,
) usize {
    return switch (variant) {
        .bulk_tail => differenceBranchlessBulkTail(a, b, output),
        .branchy => differenceBranchy(a, b, output, false),
        .branchy_bulk_tail => differenceBranchy(a, b, output, true),
    };
}

fn unionBranchlessBulkTail(a: []const u16, b: []const u16, output: []u16) usize {
    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;

    while (i < a.len and j < b.len) {
        const a_val = a[i];
        const b_val = b[j];
        output[k] = if (a_val <= b_val) a_val else b_val;
        k += 1;
        i += @intFromBool(a_val <= b_val);
        j += @intFromBool(b_val <= a_val);
    }
    return copyUnionTail(a, b, output, i, j, k);
}

fn unionBranchy(
    a: []const u16,
    b: []const u16,
    output: []u16,
    comptime bulk_tail: bool,
) usize {
    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;

    if (a.len != 0 and b.len != 0) {
        var a_val = a[0];
        var b_val = b[0];
        while (true) {
            if (a_val < b_val) {
                output[k] = a_val;
                k += 1;
                i += 1;
                if (i == a.len) break;
                a_val = a[i];
            } else if (b_val < a_val) {
                output[k] = b_val;
                k += 1;
                j += 1;
                if (j == b.len) break;
                b_val = b[j];
            } else {
                output[k] = a_val;
                k += 1;
                i += 1;
                j += 1;
                if (i == a.len or j == b.len) break;
                a_val = a[i];
                b_val = b[j];
            }
        }
    }

    if (bulk_tail) return copyUnionTail(a, b, output, i, j, k);
    while (i < a.len) : (i += 1) {
        output[k] = a[i];
        k += 1;
    }
    while (j < b.len) : (j += 1) {
        output[k] = b[j];
        k += 1;
    }
    return k;
}

fn copyUnionTail(
    a: []const u16,
    b: []const u16,
    output: []u16,
    i: usize,
    j: usize,
    start: usize,
) usize {
    // The benchmark and production array-array paths allocate output separately.
    // @memcpy is valid only while neither input aliases this destination.
    var k = start;
    if (i < a.len) {
        const tail = a[i..];
        @memcpy(output[k..][0..tail.len], tail);
        k += tail.len;
    } else if (j < b.len) {
        const tail = b[j..];
        @memcpy(output[k..][0..tail.len], tail);
        k += tail.len;
    }
    return k;
}

fn differenceBranchlessBulkTail(a: []const u16, b: []const u16, output: []u16) usize {
    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;

    while (i < a.len and j < b.len) {
        const a_val = a[i];
        const b_val = b[j];
        if (a_val < b_val) {
            output[k] = a_val;
            k += 1;
        }
        i += @intFromBool(a_val <= b_val);
        j += @intFromBool(b_val <= a_val);
    }
    return copyDifferenceTail(a, output, i, k);
}

fn differenceBranchy(
    a: []const u16,
    b: []const u16,
    output: []u16,
    comptime bulk_tail: bool,
) usize {
    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;

    if (a.len != 0 and b.len != 0) {
        var a_val = a[0];
        var b_val = b[0];
        while (true) {
            if (a_val < b_val) {
                output[k] = a_val;
                k += 1;
                i += 1;
                if (i == a.len) break;
                a_val = a[i];
            } else if (a_val == b_val) {
                i += 1;
                j += 1;
                if (i == a.len or j == b.len) break;
                a_val = a[i];
                b_val = b[j];
            } else {
                j += 1;
                if (j == b.len) break;
                b_val = b[j];
            }
        }
    }

    if (bulk_tail) return copyDifferenceTail(a, output, i, k);
    while (i < a.len) : (i += 1) {
        output[k] = a[i];
        k += 1;
    }
    return k;
}

fn copyDifferenceTail(a: []const u16, output: []u16, i: usize, start: usize) usize {
    // Output is non-aliased in every Layer A cell; preserve that precondition if
    // this candidate is ever moved into another call path.
    const tail = a[i..];
    @memcpy(output[start..][0..tail.len], tail);
    return start + tail.len;
}

fn expectVariant(
    comptime variant: Variant,
    a: []const u16,
    b: []const u16,
    expected_union: []const u16,
    expected_difference: []const u16,
) !void {
    var output: [32]u16 = undefined;
    const union_count = unionWrite(variant, a, b, &output);
    try std.testing.expectEqualSlices(u16, expected_union, output[0..union_count]);
    const difference_count = differenceWrite(variant, a, b, &output);
    try std.testing.expectEqualSlices(u16, expected_difference, output[0..difference_count]);
}

fn expectCases(comptime variant: Variant) !void {
    try expectVariant(variant, &.{}, &.{}, &.{}, &.{});
    try expectVariant(variant, &.{}, &.{ 1, 3 }, &.{ 1, 3 }, &.{});
    try expectVariant(variant, &.{ 1, 3 }, &.{}, &.{ 1, 3 }, &.{ 1, 3 });
    try expectVariant(variant, &.{1}, &.{2}, &.{ 1, 2 }, &.{1});
    try expectVariant(variant, &.{ 1, 3, 5 }, &.{ 2, 4, 6 }, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 1, 3, 5 });
    try expectVariant(variant, &.{ 1, 2, 3 }, &.{ 1, 2, 3 }, &.{ 1, 2, 3 }, &.{});
    try expectVariant(variant, &.{ 1, 2 }, &.{ 1, 2, 3, 4 }, &.{ 1, 2, 3, 4 }, &.{});
    try expectVariant(variant, &.{ 1, 2, 3, 4 }, &.{ 1, 2 }, &.{ 1, 2, 3, 4 }, &.{ 3, 4 });
}

test "bulk-copy tails cover merge edge cases" {
    try expectCases(.bulk_tail);
}

test "branchy body keeps element-wise tails correct" {
    try expectCases(.branchy);
}

test "branchy body with bulk tails covers merge edge cases" {
    try expectCases(.branchy_bulk_tail);
}
