const std = @import("std");
const rawr = @import("rawr");
const RoaringBitmap = rawr.RoaringBitmap;
const c = @cImport(@cInclude("croaring_wrapper.h"));

const allocator = std.heap.c_allocator;

var tests_passed: u32 = 0;
var tests_failed: u32 = 0;

/// Build identical bitmaps in rawr and CRoaring from a value list.
/// Serialize both, compare bytes, cross-deserialize, verify contents.
fn validateRoundTrip(name: []const u8, values: []const u32, run_optimize: bool) !void {
    // --- Build rawr bitmap ---
    var rbm = try RoaringBitmap.init(allocator);
    defer rbm.deinit();
    for (values) |v| {
        _ = try rbm.add(v);
    }
    if (run_optimize) {
        _ = try rbm.runOptimize();
    }

    // --- Build CRoaring bitmap ---
    const cr = c.roaring_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring_bitmap_free(cr);
    for (values) |v| {
        c.roaring_bitmap_add(cr, v);
    }
    if (run_optimize) {
        _ = c.roaring_bitmap_run_optimize(cr);
    }

    // --- Serialize both ---
    const rawr_bytes = try rawr.RoaringBitmap.serialize(&rbm, allocator);
    defer allocator.free(rawr_bytes);

    const cr_size = c.roaring_bitmap_portable_size_in_bytes(cr);
    const cr_buf = try allocator.alloc(u8, cr_size);
    defer allocator.free(cr_buf);
    _ = c.roaring_bitmap_portable_serialize(cr, @ptrCast(cr_buf.ptr));

    // --- Byte-level comparison ---
    if (!std.mem.eql(u8, rawr_bytes, cr_buf)) {
        std.debug.print("FAIL: {s} - bytes differ! rawr={d} bytes, croaring={d} bytes\n", .{ name, rawr_bytes.len, cr_buf.len });
        // Print first divergence point for debugging
        const min_len = @min(rawr_bytes.len, cr_buf.len);
        for (0..min_len) |i| {
            if (rawr_bytes[i] != cr_buf[i]) {
                std.debug.print("  First difference at byte {d}: rawr=0x{x:0>2} cr=0x{x:0>2}\n", .{ i, rawr_bytes[i], cr_buf[i] });
                break;
            }
        }
        tests_failed += 1;
        return error.ByteMismatch;
    }

    // --- Cross-deserialize: rawr bytes -> CRoaring ---
    const cr2 = c.roaring_bitmap_portable_deserialize_safe(@ptrCast(rawr_bytes.ptr), rawr_bytes.len) orelse {
        std.debug.print("FAIL: {s} - CRoaring failed to deserialize rawr bytes\n", .{name});
        tests_failed += 1;
        return error.CRoaringDeserializeFailed;
    };
    defer c.roaring_bitmap_free(cr2);

    if (c.roaring_bitmap_get_cardinality(cr2) != rbm.cardinality()) {
        std.debug.print("FAIL: {s} - cardinality mismatch after CRoaring deserialize\n", .{name});
        tests_failed += 1;
        return error.CardinalityMismatch;
    }
    for (values) |v| {
        if (!c.roaring_bitmap_contains(cr2, v)) {
            std.debug.print("FAIL: {s} - CRoaring missing value {d}\n", .{ name, v });
            tests_failed += 1;
            return error.MissingValue;
        }
    }

    // --- Cross-deserialize: CRoaring bytes -> rawr ---
    var rbm2 = RoaringBitmap.deserialize(allocator, cr_buf) catch |err| {
        std.debug.print("FAIL: {s} - rawr failed to deserialize CRoaring bytes: {s}\n", .{ name, @errorName(err) });
        tests_failed += 1;
        return error.RawrDeserializeFailed;
    };
    defer rbm2.deinit();

    if (rbm2.cardinality() != rbm.cardinality()) {
        std.debug.print("FAIL: {s} - cardinality mismatch after rawr deserialize\n", .{name});
        tests_failed += 1;
        return error.CardinalityMismatch;
    }
    if (!rbm2.equals(&rbm)) {
        std.debug.print("FAIL: {s} - content mismatch after rawr deserialize\n", .{name});
        tests_failed += 1;
        return error.ContentMismatch;
    }

    tests_passed += 1;
    const suffix = if (run_optimize) " [run-optimized]" else "";
    std.debug.print("  PASS: {s}{s} ({d} values, {d} bytes)\n", .{ name, suffix, values.len, rawr_bytes.len });
}

/// Validate using addRange instead of individual adds.
fn validateRangeRoundTrip(name: []const u8, start: u32, end: u32, run_optimize: bool) !void {
    // --- Build rawr bitmap ---
    var rbm = try RoaringBitmap.init(allocator);
    defer rbm.deinit();
    _ = try rbm.addRange(start, end);
    if (run_optimize) {
        _ = try rbm.runOptimize();
    }

    // --- Build CRoaring bitmap ---
    const cr = c.roaring_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring_bitmap_free(cr);
    // CRoaring uses exclusive end [start, end)
    c.roaring_bitmap_add_range(cr, start, @as(u64, end) + 1);
    if (run_optimize) {
        _ = c.roaring_bitmap_run_optimize(cr);
    }

    // --- Serialize both ---
    const rawr_bytes = try rawr.RoaringBitmap.serialize(&rbm, allocator);
    defer allocator.free(rawr_bytes);

    const cr_size = c.roaring_bitmap_portable_size_in_bytes(cr);
    const cr_buf = try allocator.alloc(u8, cr_size);
    defer allocator.free(cr_buf);
    _ = c.roaring_bitmap_portable_serialize(cr, @ptrCast(cr_buf.ptr));

    // --- Byte-level comparison ---
    if (!std.mem.eql(u8, rawr_bytes, cr_buf)) {
        std.debug.print("FAIL: {s} - bytes differ! rawr={d} bytes, croaring={d} bytes\n", .{ name, rawr_bytes.len, cr_buf.len });
        const min_len = @min(rawr_bytes.len, cr_buf.len);
        for (0..min_len) |i| {
            if (rawr_bytes[i] != cr_buf[i]) {
                std.debug.print("  First difference at byte {d}: rawr=0x{x:0>2} cr=0x{x:0>2}\n", .{ i, rawr_bytes[i], cr_buf[i] });
                break;
            }
        }
        tests_failed += 1;
        return error.ByteMismatch;
    }

    // --- Cross-deserialize: rawr bytes -> CRoaring ---
    const cr2 = c.roaring_bitmap_portable_deserialize_safe(@ptrCast(rawr_bytes.ptr), rawr_bytes.len) orelse {
        std.debug.print("FAIL: {s} - CRoaring failed to deserialize rawr bytes\n", .{name});
        tests_failed += 1;
        return error.CRoaringDeserializeFailed;
    };
    defer c.roaring_bitmap_free(cr2);

    if (c.roaring_bitmap_get_cardinality(cr2) != rbm.cardinality()) {
        std.debug.print("FAIL: {s} - cardinality mismatch\n", .{name});
        tests_failed += 1;
        return error.CardinalityMismatch;
    }

    // --- Cross-deserialize: CRoaring bytes -> rawr ---
    var rbm2 = RoaringBitmap.deserialize(allocator, cr_buf) catch |err| {
        std.debug.print("FAIL: {s} - rawr failed to deserialize CRoaring bytes: {s}\n", .{ name, @errorName(err) });
        tests_failed += 1;
        return error.RawrDeserializeFailed;
    };
    defer rbm2.deinit();

    if (!rbm2.equals(&rbm)) {
        std.debug.print("FAIL: {s} - content mismatch\n", .{name});
        tests_failed += 1;
        return error.ContentMismatch;
    }

    tests_passed += 1;
    const suffix = if (run_optimize) " [run-optimized]" else "";
    std.debug.print("  PASS: {s}{s} ({d} values, {d} bytes)\n", .{ name, suffix, end - start + 1, rawr_bytes.len });
}

pub fn main() !void {
    std.debug.print("CRoaring Interop Validation\n", .{});
    std.debug.print("===========================\n\n", .{});

    // ========== Basic tests ==========
    std.debug.print("Basic tests:\n", .{});

    // Empty bitmap
    try validateRoundTrip("empty", &.{}, false);

    // Single elements
    try validateRoundTrip("single_zero", &.{0}, false);
    try validateRoundTrip("single_max", &.{0xFFFFFFFF}, false);
    try validateRoundTrip("single_mid", &.{1000000}, false);

    // ========== Array container tests ==========
    std.debug.print("\nArray container tests:\n", .{});

    // Small array
    var arr100: [100]u32 = undefined;
    for (0..100) |i| arr100[i] = @intCast(i * 10);
    try validateRoundTrip("array_100", &arr100, false);

    // Array at threshold (4096 = max array size)
    var arr4096: [4096]u32 = undefined;
    for (0..4096) |i| arr4096[i] = @intCast(i);
    try validateRoundTrip("array_4096", &arr4096, false);

    // ========== Bitset container tests ==========
    std.debug.print("\nBitset container tests:\n", .{});

    // Just over threshold -> bitset
    var bitset5000: [5000]u32 = undefined;
    for (0..5000) |i| bitset5000[i] = @intCast(i);
    try validateRoundTrip("bitset_5000", &bitset5000, false);

    // Full chunk (65536 values) - CRoaring auto-optimizes to run, so we must too
    try validateRangeRoundTrip("bitset_full_chunk", 0, 65535, true);

    // ========== Multiple container tests ==========
    std.debug.print("\nMultiple container tests:\n", .{});

    // Values at chunk boundaries
    try validateRoundTrip("chunk_boundaries", &.{ 65535, 65536, 131071, 131072 }, false);

    // 3 containers (below NO_OFFSET_THRESHOLD for run format)
    var three_containers: [3]u32 = .{ 100, 65536 + 100, 131072 + 100 };
    try validateRoundTrip("three_containers", &three_containers, false);

    // 4 containers (at NO_OFFSET_THRESHOLD)
    var four_containers: [4]u32 = .{ 100, 65536 + 100, 131072 + 100, 196608 + 100 };
    try validateRoundTrip("four_containers", &four_containers, false);

    // 5+ containers
    var five_containers: [5]u32 = .{ 100, 65536 + 100, 131072 + 100, 196608 + 100, 262144 + 100 };
    try validateRoundTrip("five_containers", &five_containers, false);

    // ========== Run-optimized tests ==========
    std.debug.print("\nRun-optimized tests:\n", .{});

    // Range that compresses well
    try validateRangeRoundTrip("range_0_1000", 0, 1000, true);
    try validateRangeRoundTrip("range_0_10000", 0, 10000, true);

    // Multiple ranges -> multiple runs
    var multi_range: [300]u32 = undefined;
    for (0..100) |i| multi_range[i] = @intCast(i); // 0-99
    for (0..100) |i| multi_range[100 + i] = @intCast(500 + i); // 500-599
    for (0..100) |i| multi_range[200 + i] = @intCast(1000 + i); // 1000-1099
    try validateRoundTrip("multi_range_runs", &multi_range, true);

    // Alternating values (doesn't compress to runs)
    var alternating: [100]u32 = undefined;
    for (0..100) |i| alternating[i] = @intCast(i * 2); // 0, 2, 4, 6...
    try validateRoundTrip("alternating_no_runs", &alternating, true);

    // ========== Large scale tests ==========
    std.debug.print("\nLarge scale tests:\n", .{});

    // Dense range (1M values) - CRoaring auto-optimizes ranges, so we must too
    try validateRangeRoundTrip("dense_1M", 0, 999999, true);

    // Sparse random (500K values across u32 space)
    var prng = std.Random.DefaultPrng.init(12345);
    var sparse_500k: [500000]u32 = undefined;
    for (0..500000) |i| {
        sparse_500k[i] = prng.random().int(u32);
    }
    // Sort and dedupe for consistent results
    std.mem.sort(u32, &sparse_500k, {}, std.sort.asc(u32));
    var deduped_len: usize = 1;
    for (1..500000) |i| {
        if (sparse_500k[i] != sparse_500k[deduped_len - 1]) {
            sparse_500k[deduped_len] = sparse_500k[i];
            deduped_len += 1;
        }
    }
    try validateRoundTrip("sparse_500k", sparse_500k[0..deduped_len], false);

    // ========== Summary ==========
    std.debug.print("\n===========================\n", .{});
    std.debug.print("Results: {d} passed, {d} failed\n", .{ tests_passed, tests_failed });

    if (tests_failed > 0) {
        return error.TestsFailed;
    }
    std.debug.print("\nAll validation tests passed!\n", .{});
}
