// SPDX-License-Identifier: MPL-2.0

//! Frozen pre-spec-51-02 branchless merge loops for benchmark comparison only.

const std = @import("std");

pub inline fn legacyUnionWrite(a: []const u16, b: []const u16, output: []u16) usize {
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

pub inline fn legacyDifferenceWrite(a: []const u16, b: []const u16, output: []u16) usize {
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
    while (i < a.len) : (i += 1) {
        output[k] = a[i];
        k += 1;
    }
    return k;
}

fn expectCase(
    a: []const u16,
    b: []const u16,
    expected_union: []const u16,
    expected_difference: []const u16,
) !void {
    var output: [32]u16 = undefined;
    const union_count = legacyUnionWrite(a, b, &output);
    try std.testing.expectEqualSlices(u16, expected_union, output[0..union_count]);
    const difference_count = legacyDifferenceWrite(a, b, &output);
    try std.testing.expectEqualSlices(u16, expected_difference, output[0..difference_count]);
}

test "legacy branchless merge covers edge cases" {
    try expectCase(&.{}, &.{}, &.{}, &.{});
    try expectCase(&.{}, &.{ 1, 3 }, &.{ 1, 3 }, &.{});
    try expectCase(&.{ 1, 3 }, &.{}, &.{ 1, 3 }, &.{ 1, 3 });
    try expectCase(&.{1}, &.{2}, &.{ 1, 2 }, &.{1});
    try expectCase(&.{ 1, 3, 5 }, &.{ 2, 4, 6 }, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 1, 3, 5 });
    try expectCase(&.{ 1, 2, 3 }, &.{ 1, 2, 3 }, &.{ 1, 2, 3 }, &.{});
    try expectCase(&.{ 1, 2 }, &.{ 1, 2, 3, 4 }, &.{ 1, 2, 3, 4 }, &.{});
    try expectCase(&.{ 1, 2, 3, 4 }, &.{ 1, 2 }, &.{ 1, 2, 3, 4 }, &.{ 3, 4 });
}
