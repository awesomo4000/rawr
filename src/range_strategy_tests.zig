// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const range_ops = @import("range_ops.zig");
const legacy_ops = @import("range_legacy_test.zig");

const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;

const Fixture = enum {
    empty,
    array,
    bitset,
    run,
    boundary,
    mixed,
};

const Case = struct {
    name: []const u8,
    fixture: Fixture,
    lo: u32,
    hi: u32,
};

const cases = [_]Case{
    .{ .name = "empty-missing", .fixture = .empty, .lo = value(5, 10), .hi = value(5, 1000) },
    .{ .name = "array-within", .fixture = .array, .lo = value(1, 20), .hi = value(1, 3000) },
    .{ .name = "bitset-demotion", .fixture = .bitset, .lo = value(2, 0), .hi = value(2, 50_000) },
    .{ .name = "run-split", .fixture = .run, .lo = value(3, 300), .hi = value(3, 600) },
    .{ .name = "boundary-seam", .fixture = .boundary, .lo = 65_535, .hi = 65_536 },
    .{ .name = "boundary-minus-plus-one", .fixture = .boundary, .lo = 65_534, .hi = 65_537 },
    .{ .name = "mixed-cross-chunk", .fixture = .mixed, .lo = value(10, 50), .hi = value(13, 9000) },
    .{ .name = "full-edge-chunks", .fixture = .mixed, .lo = value(11, 0), .hi = value(12, 65_535) },
    .{ .name = "inverted", .fixture = .mixed, .lo = value(12, 1), .hi = value(12, 0) },
    .{ .name = "whole-universe", .fixture = .empty, .lo = 0, .hi = std.math.maxInt(u32) },
};

test "range strategies preserve legacy portable representation" {
    for (cases) |case| {
        try runCase(std.testing.allocator, case);
    }
}

test "direct removeRange remains valid across allocation failures" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        removeRangeAllocationFailureCase,
        .{},
    );
}

test "direct flipInPlace remains valid across allocation failures" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        flipInPlaceAllocationFailureCase,
        .{},
    );
}

test "direct flip keeps its input unchanged across allocation failures" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        flipAllocationFailureCase,
        .{},
    );
}

fn removeRangeAllocationFailureCase(allocator: std.mem.Allocator) !void {
    var bitmap = try RoaringBitmap.init(allocator);
    defer bitmap.deinit();

    try addArrayChunk(&bitmap, 1);
    _ = try bitmap.addRange(value(2, 0), value(2, 65_535));
    try addBitsetChunk(&bitmap, 3);
    _ = bitmap.cardinality();

    _ = range_ops.removeRange(&bitmap, value(1, 20), value(3, 50_000)) catch {
        try bitmap.validate();
        try std.testing.expectEqual(@as(i64, -1), bitmap.cached_cardinality);
        try std.testing.expectEqual(iteratorCardinality(&bitmap), bitmap.cardinality());
        return error.OutOfMemory;
    };

    try bitmap.validate();
    try std.testing.expectEqual(iteratorCardinality(&bitmap), bitmap.cardinality());
}

fn flipInPlaceAllocationFailureCase(allocator: std.mem.Allocator) !void {
    var bitmap = try buildFixture(allocator, .mixed);
    defer bitmap.deinit();
    _ = bitmap.cardinality();

    range_ops.flipInPlace(&bitmap, value(10, 50), value(15, 9000)) catch {
        try bitmap.validate();
        try std.testing.expectEqual(@as(i64, -1), bitmap.cached_cardinality);
        try std.testing.expectEqual(iteratorCardinality(&bitmap), bitmap.cardinality());
        return error.OutOfMemory;
    };

    try bitmap.validate();
    try std.testing.expectEqual(@as(i64, -1), bitmap.cached_cardinality);
    try std.testing.expectEqual(iteratorCardinality(&bitmap), bitmap.cardinality());
}

fn flipAllocationFailureCase(allocator: std.mem.Allocator) !void {
    var input = try buildFixture(allocator, .mixed);
    defer input.deinit();
    var expected = try input.clone(std.testing.allocator);
    defer expected.deinit();

    var result = range_ops.flip(
        &input,
        allocator,
        value(10, 50),
        value(15, 9000),
    ) catch {
        try input.validate();
        try std.testing.expect(input.equals(&expected));
        return error.OutOfMemory;
    };
    defer result.deinit();

    try input.validate();
    try result.validate();
    try std.testing.expect(input.equals(&expected));
}

fn runCase(backing: std.mem.Allocator, case: Case) !void {
    var arena = std.heap.ArenaAllocator.init(backing);
    defer arena.deinit();
    const allocator = arena.allocator();

    var input = try buildFixture(allocator, case.fixture);
    defer input.deinit();

    try compareRemoveRange(allocator, case, &input);
    try compareFlipInPlace(allocator, case, &input);
    try compareFlip(allocator, case, &input);
    try compareFlipOwned(allocator, case, &input);
}

fn compareRemoveRange(allocator: std.mem.Allocator, case: Case, input: *const RoaringBitmap) !void {
    var legacy = try input.clone(allocator);
    defer legacy.deinit();
    var direct = try input.clone(allocator);
    defer direct.deinit();

    const legacy_removed = try legacy_ops.removeRange(&legacy, case.lo, case.hi);
    const direct_removed = try range_ops.removeRange(&direct, case.lo, case.hi);
    try std.testing.expectEqual(legacy_removed, direct_removed);
    try expectPortableEqual(case.name, &legacy, &direct);
}

fn compareFlipInPlace(allocator: std.mem.Allocator, case: Case, input: *const RoaringBitmap) !void {
    var legacy = try input.clone(allocator);
    defer legacy.deinit();
    var direct = try input.clone(allocator);
    defer direct.deinit();

    try legacy_ops.flipInPlace(&legacy, case.lo, case.hi);
    try range_ops.flipInPlace(&direct, case.lo, case.hi);
    try expectPortableEqual(case.name, &legacy, &direct);
}

fn compareFlip(allocator: std.mem.Allocator, case: Case, input: *const RoaringBitmap) !void {
    var legacy = try legacy_ops.flip(input, allocator, case.lo, case.hi);
    defer legacy.deinit();
    var direct = try range_ops.flip(input, allocator, case.lo, case.hi);
    defer direct.deinit();

    try expectPortableEqual(case.name, &legacy, &direct);
    if (case.lo > case.hi) {
        const marker = value(60_000, 7);
        _ = try direct.add(marker);
        try std.testing.expect(!input.contains(marker));
    }
}

fn compareFlipOwned(allocator: std.mem.Allocator, case: Case, input: *const RoaringBitmap) !void {
    var expected = try legacy_ops.flip(input, allocator, case.lo, case.hi);
    defer expected.deinit();
    var owned = try input.flipOwned(allocator, case.lo, case.hi);
    defer owned.deinit();
    var stable = try owned.bitmap.clone(std.testing.allocator);
    defer stable.deinit();
    try expectPortableEqual(case.name, &expected, &stable);
}

fn expectPortableEqual(name: []const u8, legacy: *RoaringBitmap, direct: *RoaringBitmap) !void {
    try legacy.validate();
    try direct.validate();

    const legacy_bytes = try legacy.serialize(std.testing.allocator);
    defer std.testing.allocator.free(legacy_bytes);
    const direct_bytes = try direct.serialize(std.testing.allocator);
    defer std.testing.allocator.free(direct_bytes);

    if (!std.mem.eql(u8, legacy_bytes, direct_bytes)) {
        std.debug.print("range strategy bytes differ for {s}: legacy={d} direct={d}\n", .{
            name,
            legacy_bytes.len,
            direct_bytes.len,
        });
        return error.RangeStrategyByteMismatch;
    }
}

fn buildFixture(allocator: std.mem.Allocator, fixture: Fixture) !RoaringBitmap {
    var bitmap = try RoaringBitmap.init(allocator);
    errdefer bitmap.deinit();

    switch (fixture) {
        .empty => {},
        .array => try addArrayChunk(&bitmap, 1),
        .bitset => try addBitsetChunk(&bitmap, 2),
        .run => _ = try bitmap.addRange(value(3, 100), value(3, 1000)),
        .boundary => {
            try addArrayChunk(&bitmap, 0);
            _ = try bitmap.addRange(value(1, 0), value(1, 200));
        },
        .mixed => {
            try addArrayChunk(&bitmap, 10);
            try addBitsetChunk(&bitmap, 11);
            _ = try bitmap.addRange(value(13, 100), value(13, 10_000));
        },
    }

    return bitmap;
}

fn addArrayChunk(bitmap: *RoaringBitmap, key: u16) !void {
    for ([_]u16{ 1, 19, 20, 300, 2999, 3000, 65_534, 65_535 }) |low| {
        _ = try bitmap.add(value(key, low));
    }
}

fn addBitsetChunk(bitmap: *RoaringBitmap, key: u16) !void {
    for (0..5000) |index| {
        const low: u16 = @intCast((index * 13) & 0xffff);
        _ = try bitmap.add(value(key, low));
    }
}

fn value(key: u16, low: u16) u32 {
    return (@as(u32, key) << 16) | low;
}

fn iteratorCardinality(bitmap: *const RoaringBitmap) u64 {
    var count: u64 = 0;
    var iterator = bitmap.iterator();
    while (iterator.next() != null) count += 1;
    return count;
}
