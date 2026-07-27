// SPDX-License-Identifier: MPL-2.0

//! Test-only copy of the pre-direct range-operation composition.

const std = @import("std");

pub fn removeRange(bitmap: anytype, lo: u32, hi: u32) !u64 {
    if (lo > hi) return 0;

    const before = bitmap.cardinality();
    const Bitmap = @TypeOf(bitmap.*);
    var mask = try Bitmap.init(bitmap.allocator);
    defer mask.deinit();
    _ = try mask.addRange(lo, hi);
    try bitmap.bitwiseDifferenceInPlace(&mask);
    return before - bitmap.cardinality();
}

pub fn flip(
    bitmap: anytype,
    allocator: std.mem.Allocator,
    lo: u32,
    hi: u32,
) !@TypeOf(bitmap.*) {
    var result = try bitmap.clone(allocator);
    errdefer result.deinit();
    try flipInPlace(&result, lo, hi);
    return result;
}

pub fn flipInPlace(bitmap: anytype, lo: u32, hi: u32) !void {
    if (lo > hi) return;

    const Bitmap = @TypeOf(bitmap.*);
    var mask = try Bitmap.init(bitmap.allocator);
    defer mask.deinit();
    _ = try mask.addRange(lo, hi);
    try bitmap.bitwiseXorInPlace(&mask);
}
