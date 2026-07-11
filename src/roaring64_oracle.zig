const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");

const Roaring64Bitmap = rawr.Roaring64Bitmap;
const Frozen64Bitmap = rawr.Frozen64Bitmap;
const test_support = rawr.roaring64_test_support;

pub fn buildCRoaring(values: []const u64) !*c.roaring64_bitmap_t {
    const cr = c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
    errdefer c.roaring64_bitmap_free(cr);

    if (values.len != 0) {
        c.roaring64_bitmap_add_many(cr, values.len, @ptrCast(values.ptr));
    }
    return cr;
}

pub fn buildCRoaringOfPtr(values: []const u64) !*c.roaring64_bitmap_t {
    return c.roaring64_bitmap_of_ptr(values.len, @ptrCast(values.ptr)) orelse return error.CRoaringAllocFailed;
}

pub fn buildCRoaringFromRange(min: u64, max: u64, step: u64) !*c.roaring64_bitmap_t {
    if (c.roaring64_bitmap_from_range(min, max, step)) |cr| return cr;
    return c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
}

pub fn buildCRoaring32(values: []const u32) !*c.roaring_bitmap_t {
    const cr = c.roaring_bitmap_create() orelse return error.CRoaringAllocFailed;
    errdefer c.roaring_bitmap_free(cr);

    if (values.len != 0) {
        c.roaring_bitmap_add_many(cr, values.len, @ptrCast(values.ptr));
    }
    return cr;
}

pub fn buildCRoaring64From32(values: []const u32) !*c.roaring64_bitmap_t {
    const cr32 = try buildCRoaring32(values);
    defer c.roaring_bitmap_free(cr32);

    return c.roaring64_bitmap_move_from_roaring32(cr32) orelse return error.CRoaringAllocFailed;
}

pub fn assertAgreement(
    allocator: std.mem.Allocator,
    rbm: *const Roaring64Bitmap,
    cr: *const c.roaring64_bitmap_t,
    probes: []const u64,
) !void {
    try rbm.validate();
    try assertCRoaringValid(cr);

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

pub fn assertCRoaringValid(cr: *const c.roaring64_bitmap_t) !void {
    var reason: [*c]const u8 = null;
    if (!c.roaring64_bitmap_internal_validate(cr, &reason)) {
        return error.CRoaringInvalid;
    }
}

pub fn assertStatisticsAgreement(rbm: *const Roaring64Bitmap, cr: *const c.roaring64_bitmap_t) !void {
    const rawr_stats = rbm.statistics();
    var cr_stats: c.roaring64_statistics_t = undefined;
    c.roaring64_bitmap_statistics(cr, &cr_stats);

    if (rawr_stats.n_containers != cr_stats.n_containers) return error.StatisticsMismatch;
    if (rawr_stats.n_array_containers != cr_stats.n_array_containers) return error.StatisticsMismatch;
    if (rawr_stats.n_run_containers != cr_stats.n_run_containers) return error.StatisticsMismatch;
    if (rawr_stats.n_bitset_containers != cr_stats.n_bitset_containers) return error.StatisticsMismatch;
    if (rawr_stats.n_values_array_containers != cr_stats.n_values_array_containers) return error.StatisticsMismatch;
    if (rawr_stats.n_values_run_containers != cr_stats.n_values_run_containers) return error.StatisticsMismatch;
    if (rawr_stats.n_values_bitset_containers != cr_stats.n_values_bitset_containers) return error.StatisticsMismatch;
    if (rawr_stats.cardinality != cr_stats.cardinality) return error.StatisticsMismatch;

    if (rawr_stats.cardinality != 0) {
        if (rawr_stats.min_value != cr_stats.min_value) return error.StatisticsMismatch;
        if (rawr_stats.max_value != cr_stats.max_value) return error.StatisticsMismatch;
    }
}

pub fn assertFrozenAgreement(allocator: std.mem.Allocator, rbm: *const Roaring64Bitmap) !void {
    const size = try rbm.frozenSizeInBytes();
    const bytes = try allocator.alloc(u8, size);
    defer allocator.free(bytes);

    try rbm.frozenSerialize(bytes);

    var frozen = try Frozen64Bitmap.view(bytes);
    defer frozen.deinit();

    if (frozen.cardinality() != rbm.cardinality()) return error.FrozenCardinalityMismatch;
    if (frozen.minimum() != rbm.minimum()) return error.FrozenMinimumMismatch;
    if (frozen.maximum() != rbm.maximum()) return error.FrozenMaximumMismatch;

    const values = try rbm.toArrayAlloc(allocator);
    defer allocator.free(values);

    var iter = frozen.iterator();
    for (values, 0..) |value, idx| {
        const rank: u64 = @intCast(idx);
        if (!frozen.contains(value)) return error.FrozenContainsMismatch;
        if (frozen.rank(value) != rank + 1) return error.FrozenRankMismatch;
        if (frozen.getIndex(value) != rank) return error.FrozenGetIndexMismatch;
        if (frozen.select(rank) != value) return error.FrozenSelectMismatch;
        if (iter.next() != value) return error.FrozenIteratorMismatch;
    }
    if (iter.next() != null) return error.FrozenIteratorExtraValue;
    if (frozen.select(@intCast(values.len)) != null) return error.FrozenSelectMismatch;
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
    if (rbm.intersectsRange(lo, hi) != cIntersectsRangeClosed(cr, lo, hi)) {
        return error.IntersectsRangeMismatch;
    }
}

pub fn cContainsRangeClosed(cr: *const c.roaring64_bitmap_t, lo: u64, hi: u64) bool {
    if (lo > hi) return true;
    if (hi == std.math.maxInt(u64)) {
        return c.roaring64_bitmap_contains_range(cr, lo, hi) and c.roaring64_bitmap_contains(cr, hi);
    }
    return c.roaring64_bitmap_contains_range(cr, lo, hi + 1);
}

pub fn cIntersectsRangeClosed(cr: *const c.roaring64_bitmap_t, lo: u64, hi: u64) bool {
    if (lo > hi) return false;
    if (hi == std.math.maxInt(u64)) {
        return c.roaring64_bitmap_intersect_with_range(cr, lo, hi) or c.roaring64_bitmap_contains(cr, hi);
    }
    return c.roaring64_bitmap_intersect_with_range(cr, lo, hi + 1);
}

pub fn assertJaccardAgreement(
    a: *const Roaring64Bitmap,
    b: *const Roaring64Bitmap,
    cr_a: *const c.roaring64_bitmap_t,
    cr_b: *const c.roaring64_bitmap_t,
) !void {
    const rawr_value = a.jaccardIndex(b);
    const cr_value = c.roaring64_bitmap_jaccard_index(cr_a, cr_b);
    if (std.math.isNan(rawr_value) and std.math.isNan(cr_value)) return;
    if (rawr_value == cr_value) return;

    const diff = @abs(rawr_value - cr_value);
    if (diff <= 1e-12) return;
    return error.JaccardMismatch;
}

pub fn assertFlipAgreement(
    allocator: std.mem.Allocator,
    source: *const Roaring64Bitmap,
    source_cr: *const c.roaring64_bitmap_t,
    lo: u64,
    hi: u64,
) !void {
    var rawr_result = try source.flip(allocator, lo, hi);
    defer rawr_result.deinit();

    const cr_result = c.roaring64_bitmap_flip_closed(source_cr, lo, hi) orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr_result);
    const probes = [_]u64{ lo, hi };
    try assertAgreement(allocator, &rawr_result, cr_result, &probes);

    var rawr_in_place = try source.clone(allocator);
    defer rawr_in_place.deinit();
    try rawr_in_place.flipInPlace(lo, hi);
    if (!rawr_in_place.equals(&rawr_result)) return error.InPlaceMismatch;

    const cr_in_place = c.roaring64_bitmap_copy(source_cr) orelse return error.CRoaringAllocFailed;
    defer c.roaring64_bitmap_free(cr_in_place);
    c.roaring64_bitmap_flip_closed_inplace(cr_in_place, lo, hi);
    try assertAgreement(allocator, &rawr_in_place, cr_in_place, &probes);
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
