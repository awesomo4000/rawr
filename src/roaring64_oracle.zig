const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");

const Roaring64Bitmap = rawr.Roaring64Bitmap;
const test_support = rawr.roaring64_test_support;

pub fn buildCRoaring(values: []const u64) !*c.roaring64_bitmap_t {
    const cr = c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
    errdefer c.roaring64_bitmap_free(cr);

    if (values.len != 0) {
        c.roaring64_bitmap_add_many(cr, values.len, @ptrCast(values.ptr));
    }
    return cr;
}

pub fn assertAgreement(
    allocator: std.mem.Allocator,
    rbm: *const Roaring64Bitmap,
    cr: *const c.roaring64_bitmap_t,
    probes: []const u64,
) !void {
    const cr_card = c.roaring64_bitmap_get_cardinality(cr);
    if (rbm.cardinality() != cr_card) return error.CardinalityMismatch;
    if (rbm.isEmpty() != c.roaring64_bitmap_is_empty(cr)) return error.EmptyMismatch;

    for (probes) |value| {
        if (rbm.contains(value) != c.roaring64_bitmap_contains(cr, value)) {
            return error.ContainsMismatch;
        }
    }

    if (rbm.isEmpty()) {
        if (rbm.minimum() != null or rbm.maximum() != null) return error.EmptyMinMaxMismatch;
    } else {
        if (rbm.minimum() != c.roaring64_bitmap_minimum(cr)) return error.MinimumMismatch;
        if (rbm.maximum() != c.roaring64_bitmap_maximum(cr)) return error.MaximumMismatch;
    }

    const rawr_values = try rbm.toArrayAlloc(allocator);
    defer allocator.free(rawr_values);
    if (rawr_values.len != cr_card) return error.ArrayCardinalityMismatch;

    const cr_values = try allocator.alloc(u64, rawr_values.len);
    defer allocator.free(cr_values);
    if (cr_values.len != 0) {
        c.roaring64_bitmap_to_uint64_array(cr, @ptrCast(cr_values.ptr));
    }
    if (!std.mem.eql(u64, rawr_values, cr_values)) return error.ArrayMismatch;

    var iter = rbm.iterator();
    for (rawr_values) |expected| {
        if (iter.next() != expected) return error.IteratorMismatch;
    }
    if (iter.next() != null) return error.IteratorExtraValue;

    try assertSerializationAgreement(allocator, rbm, cr);
}

pub fn assertPositionalAgreement(
    rbm: *const Roaring64Bitmap,
    cr: *const c.roaring64_bitmap_t,
    probes: []const u64,
) !void {
    for (probes) |value| {
        if (rbm.rank(value) != c.roaring64_bitmap_rank(cr, value)) return error.RankMismatch;

        const rawr_index = rbm.getIndex(value);
        var cr_index: u64 = undefined;
        const cr_present = c.roaring64_bitmap_get_index(cr, value, &cr_index);
        if ((rawr_index != null) != cr_present) return error.GetIndexPresenceMismatch;
        if (rawr_index) |idx| {
            if (idx != cr_index) return error.GetIndexMismatch;
        }
    }

    const card = rbm.cardinality();
    if (card == 0) {
        if (rbm.select(0) != null) return error.SelectMismatch;
        var cr_value: u64 = undefined;
        if (c.roaring64_bitmap_select(cr, 0, &cr_value)) return error.SelectPresenceMismatch;
        return;
    }

    const ranks = [_]u64{ 0, card / 2, card - 1, card };
    for (ranks) |rank| {
        const rawr_value = rbm.select(rank);
        var cr_value: u64 = undefined;
        const cr_present = c.roaring64_bitmap_select(cr, rank, &cr_value);
        if ((rawr_value != null) != cr_present) return error.SelectPresenceMismatch;
        if (rawr_value) |value| {
            if (value != cr_value) return error.SelectMismatch;
        }
    }
}

pub fn applyAddRange(
    allocator: std.mem.Allocator,
    rbm: *Roaring64Bitmap,
    cr: *c.roaring64_bitmap_t,
    lo: u64,
    hi: u64,
) !void {
    try rbm.addRange(lo, hi);
    c.roaring64_bitmap_add_range_closed(cr, lo, hi);
    const probes = [_]u64{ lo, hi };
    try assertAgreement(allocator, rbm, cr, &probes);
}

pub fn applyRemoveRange(
    allocator: std.mem.Allocator,
    rbm: *Roaring64Bitmap,
    cr: *c.roaring64_bitmap_t,
    lo: u64,
    hi: u64,
) !void {
    try rbm.removeRange(lo, hi);
    c.roaring64_bitmap_remove_range_closed(cr, lo, hi);
    const probes = [_]u64{ lo, hi };
    try assertAgreement(allocator, rbm, cr, &probes);
}

pub fn assertRangeAgreement(rbm: *const Roaring64Bitmap, cr: *const c.roaring64_bitmap_t, lo: u64, hi: u64) !void {
    if (rbm.rangeCardinality(lo, hi) != c.roaring64_bitmap_range_closed_cardinality(cr, lo, hi)) {
        return error.RangeCardinalityMismatch;
    }
    if (rbm.containsRange(lo, hi) != cContainsRangeClosed(cr, lo, hi)) {
        return error.ContainsRangeMismatch;
    }
}

pub fn cContainsRangeClosed(cr: *const c.roaring64_bitmap_t, lo: u64, hi: u64) bool {
    if (lo > hi) return true;
    if (hi == std.math.maxInt(u64)) {
        return c.roaring64_bitmap_contains_range(cr, lo, hi) and c.roaring64_bitmap_contains(cr, hi);
    }
    return c.roaring64_bitmap_contains_range(cr, lo, hi + 1);
}

pub fn assertSerializationAgreement(
    allocator: std.mem.Allocator,
    rbm: *const Roaring64Bitmap,
    cr: *const c.roaring64_bitmap_t,
) !void {
    const rawr_bytes = try rbm.serialize(allocator);
    defer allocator.free(rawr_bytes);
    if (try rbm.serializedSizeInBytes() != rawr_bytes.len) return error.SerializedSizeMismatch;

    var rawr_from_rawr = try Roaring64Bitmap.deserialize(allocator, rawr_bytes);
    defer rawr_from_rawr.deinit();
    if (!rawr_from_rawr.equals(rbm)) return error.RawrRoundTripMismatch;

    var rawr_safe = try Roaring64Bitmap.deserializeSafe(allocator, rawr_bytes);
    defer rawr_safe.deinit();
    if (!rawr_safe.equals(rbm)) return error.RawrSafeRoundTripMismatch;

    const cr_from_rawr = c.roaring64_bitmap_portable_deserialize_safe(@ptrCast(rawr_bytes.ptr), rawr_bytes.len) orelse return error.CRoaringDeserializeFailed;
    defer c.roaring64_bitmap_free(cr_from_rawr);
    if (!c.roaring64_bitmap_equals(cr_from_rawr, cr)) return error.CRoaringRoundTripMismatch;

    var comparable_owned: ?*c.roaring64_bitmap_t = null;
    const rawr_has_runs = test_support.hasRunContainers(rbm);
    const comparable_cr: *const c.roaring64_bitmap_t = if (rawr_has_runs) blk: {
        const copy = c.roaring64_bitmap_copy(cr) orelse return error.CRoaringAllocFailed;
        _ = c.roaring64_bitmap_run_optimize(copy);
        comparable_owned = copy;
        break :blk copy;
    } else cr;
    defer if (comparable_owned) |owned| c.roaring64_bitmap_free(owned);

    const cr_size = c.roaring64_bitmap_portable_size_in_bytes(comparable_cr);
    const cr_bytes = try allocator.alloc(u8, cr_size);
    defer allocator.free(cr_bytes);
    const written = c.roaring64_bitmap_portable_serialize(comparable_cr, @ptrCast(cr_bytes.ptr));
    if (written != cr_size) return error.CRoaringSerializeSizeMismatch;

    var rawr_from_cr = try Roaring64Bitmap.deserialize(allocator, cr_bytes);
    defer rawr_from_cr.deinit();
    if (!rawr_from_cr.equals(rbm)) return error.CRoaringRoundTripMismatch;

    // CRoaring run_optimize is size-driven; rawr addRange can keep tiny ranges
    // as RUN containers even when CRoaring keeps arrays. Cross-deserialize above
    // is the interop bar for those representation-only differences.
    if (cr_size != rawr_bytes.len) {
        if (rawr_has_runs) return;
        return error.SerializedSizeMismatch;
    }
    if (!std.mem.eql(u8, rawr_bytes, cr_bytes)) {
        if (rawr_has_runs) return;
        return error.SerializedBytesMismatch;
    }
}
