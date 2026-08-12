// SPDX-License-Identifier: MPL-2.0

//! Compile-only public API probe for 32-bit targets.
//!
//! The exported root forces Zig to analyze every call below even though this
//! object is never executed. Keep the allocator freestanding-safe.

const std = @import("std");
const rawr = @import("rawr");

const RoaringBitmap = rawr.RoaringBitmap;
const Roaring64Bitmap = rawr.Roaring64Bitmap;
const FrozenBitmap = rawr.FrozenBitmap;
const Frozen64Bitmap = rawr.Frozen64Bitmap;

var probe_storage: [2 * 1024 * 1024]u8 align(64) = undefined;
var probe_sink: u64 = 0;

export fn rawrCheck32Api() void {
    runProbe() catch unreachable;
}

fn runProbe() !void {
    var fba = std.heap.FixedBufferAllocator.init(&probe_storage);
    const allocator = fba.allocator();

    var left = try RoaringBitmap.init(allocator);
    defer left.deinit();
    var right = try RoaringBitmap.init(allocator);
    defer right.deinit();

    _ = try left.add(1);
    _ = try left.addRange(10, 20);
    _ = try left.remove(11);
    _ = left.contains(10);
    probe_sink +%= left.cardinality();
    probe_sink +%= left.rank(20);
    _ = left.select(0);
    _ = left.minimum();
    _ = left.maximum();
    _ = try right.add(20);

    var intersection = try left.bitwiseAnd(allocator, &right);
    defer intersection.deinit();
    var unioned = try left.bitwiseOr(allocator, &right);
    defer unioned.deinit();
    var in_place_or = try left.clone(allocator);
    defer in_place_or.deinit();
    try in_place_or.bitwiseOrInPlace(&right);
    var consumed_right = try right.clone(allocator);
    defer consumed_right.deinit();
    try in_place_or.bitwiseOrInPlaceConsume(&consumed_right);
    var in_place_difference = try left.clone(allocator);
    defer in_place_difference.deinit();
    try in_place_difference.bitwiseDifferenceInPlace(&right);
    var in_place_xor = try left.clone(allocator);
    defer in_place_xor.deinit();
    try in_place_xor.bitwiseXorInPlace(&right);
    var lazy = try left.lazyOr(allocator, &right, true);
    defer lazy.deinit();
    try lazy.repairAfterLazy();
    var tuned_lazy = try left.lazyOr(allocator, &right, true);
    defer tuned_lazy.deinit();
    try tuned_lazy.repairAfterLazyWithOptions(.{});

    var cloned = try left.clone(allocator);
    defer cloned.deinit();
    _ = try cloned.runOptimize();
    _ = try cloned.shrinkToFit();
    probe_sink +%= cloned.serializedSizeInBytes();
    const bytes = try cloned.serialize(allocator);
    defer allocator.free(bytes);
    var decoded = try RoaringBitmap.deserialize(allocator, bytes);
    defer decoded.deinit();
    var frozen = try FrozenBitmap.init(bytes);
    defer frozen.deinit();
    _ = frozen.isEmpty();
    _ = frozen.contains(10);
    probe_sink +%= frozen.cardinality();
    probe_sink +%= frozen.rank(20);
    _ = frozen.getIndex(10);
    _ = frozen.select(0);
    _ = frozen.minimum();
    _ = frozen.maximum();
    var frozen_iter = frozen.iterator();
    _ = frozen_iter.next();

    var left64 = try Roaring64Bitmap.init(allocator);
    defer left64.deinit();
    var right64 = try Roaring64Bitmap.init(allocator);
    defer right64.deinit();

    _ = try left64.add((@as(u64, 1) << 32) | 1);
    try left64.addRange((@as(u64, 2) << 32) | 10, (@as(u64, 2) << 32) | 20);
    _ = try left64.remove((@as(u64, 2) << 32) | 11);
    _ = left64.contains((@as(u64, 2) << 32) | 10);
    probe_sink +%= left64.cardinality();
    probe_sink +%= left64.rank((@as(u64, 2) << 32) | 20);
    _ = left64.select(0);
    _ = left64.minimum();
    _ = left64.maximum();
    _ = try right64.add((@as(u64, 2) << 32) | 20);

    var intersection64 = try left64.bitwiseAnd(allocator, &right64);
    defer intersection64.deinit();
    var union64 = try left64.bitwiseOr(allocator, &right64);
    defer union64.deinit();
    var difference64 = try left64.bitwiseDifference(allocator, &right64);
    defer difference64.deinit();
    var clone64 = try left64.clone(allocator);
    defer clone64.deinit();

    probe_sink +%= try clone64.serializedSizeInBytes();
    const bytes64 = try clone64.serialize(allocator);
    defer allocator.free(bytes64);
    var decoded64 = try Roaring64Bitmap.deserialize(allocator, bytes64);
    defer decoded64.deinit();
    var decoded_safe64 = try Roaring64Bitmap.deserializeSafe(allocator, bytes64);
    defer decoded_safe64.deinit();

    const frozen_size64 = try clone64.frozenSizeInBytes();
    const frozen_bytes64 = try allocator.alloc(u8, frozen_size64);
    defer allocator.free(frozen_bytes64);
    try clone64.frozenSerialize(frozen_bytes64);
    var frozen64 = try Frozen64Bitmap.view(frozen_bytes64);
    defer frozen64.deinit();
    _ = frozen64.isEmpty();
    _ = frozen64.contains((@as(u64, 1) << 32) | 1);
    probe_sink +%= frozen64.cardinality();
    probe_sink +%= frozen64.rank((@as(u64, 2) << 32) | 20);
    _ = frozen64.getIndex((@as(u64, 2) << 32) | 10);
    _ = frozen64.select(0);
    _ = frozen64.minimum();
    _ = frozen64.maximum();
    var frozen_iter64 = frozen64.iterator();
    _ = frozen_iter64.next();
}
