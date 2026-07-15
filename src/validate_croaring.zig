// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const rawr = @import("rawr");
const RoaringBitmap = rawr.RoaringBitmap;
const FrozenBitmap = rawr.FrozenBitmap;
const test_gen = rawr.test_gen;
const c = @import("c");
const bench_time = @import("bench_time.zig");

const allocator = bench_time.cAllocator();

var tests_passed: u32 = 0;
var tests_failed: u32 = 0;

fn bitmapHasRunContainers(bm: *const RoaringBitmap) bool {
    for (bm.containers[0..bm.size]) |tp| {
        if (tp.getType() == .run) return true;
    }
    return false;
}

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

    try validateBitmapRoundTrip(name, &rbm, values, run_optimize);
}

/// Validate an already-built rawr bitmap against a CRoaring oracle built from
/// the same sorted value list.
fn validateBitmapRoundTrip(name: []const u8, rbm: *RoaringBitmap, values: []const u32, run_optimize: bool) !void {
    // --- Build CRoaring bitmap ---
    const cr = c.roaring_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring_bitmap_free(cr);
    for (values) |v| {
        c.roaring_bitmap_add(cr, v);
    }
    if (run_optimize or bitmapHasRunContainers(rbm)) {
        _ = c.roaring_bitmap_run_optimize(cr);
    }

    // --- Serialize both ---
    const rawr_bytes = try rawr.RoaringBitmap.serialize(rbm, allocator);
    defer allocator.free(rawr_bytes);

    const cr_size = c.roaring_bitmap_portable_size_in_bytes(cr);
    const cr_buf = try allocator.alloc(u8, cr_size);
    defer allocator.free(cr_buf);
    _ = c.roaring_bitmap_portable_serialize(cr, @ptrCast(cr_buf.ptr));

    // --- Byte-level comparison ---
    if (!std.mem.eql(u8, rawr_bytes, cr_buf)) {
        bench_time.print("FAIL: {s} - bytes differ! rawr={d} bytes, croaring={d} bytes\n", .{ name, rawr_bytes.len, cr_buf.len });
        // Print first divergence point for debugging
        const min_len = @min(rawr_bytes.len, cr_buf.len);
        for (0..min_len) |i| {
            if (rawr_bytes[i] != cr_buf[i]) {
                bench_time.print("  First difference at byte {d}: rawr=0x{x:0>2} cr=0x{x:0>2}\n", .{ i, rawr_bytes[i], cr_buf[i] });
                break;
            }
        }
        tests_failed += 1;
        return error.ByteMismatch;
    }

    // --- Cross-deserialize: rawr bytes -> CRoaring ---
    const cr2 = c.roaring_bitmap_portable_deserialize_safe(@ptrCast(rawr_bytes.ptr), rawr_bytes.len) orelse {
        bench_time.print("FAIL: {s} - CRoaring failed to deserialize rawr bytes\n", .{name});
        tests_failed += 1;
        return error.CRoaringDeserializeFailed;
    };
    defer c.roaring_bitmap_free(cr2);

    if (c.roaring_bitmap_get_cardinality(cr2) != rbm.cardinality()) {
        bench_time.print("FAIL: {s} - cardinality mismatch after CRoaring deserialize\n", .{name});
        tests_failed += 1;
        return error.CardinalityMismatch;
    }
    for (values) |v| {
        if (!c.roaring_bitmap_contains(cr2, v)) {
            bench_time.print("FAIL: {s} - CRoaring missing value {d}\n", .{ name, v });
            tests_failed += 1;
            return error.MissingValue;
        }
    }

    // --- Cross-deserialize: CRoaring bytes -> rawr ---
    var rbm2 = RoaringBitmap.deserialize(allocator, cr_buf) catch |err| {
        bench_time.print("FAIL: {s} - rawr failed to deserialize CRoaring bytes: {s}\n", .{ name, @errorName(err) });
        tests_failed += 1;
        return error.RawrDeserializeFailed;
    };
    defer rbm2.deinit();

    if (rbm2.cardinality() != rbm.cardinality()) {
        bench_time.print("FAIL: {s} - cardinality mismatch after rawr deserialize\n", .{name});
        tests_failed += 1;
        return error.CardinalityMismatch;
    }
    if (!rbm2.equals(rbm)) {
        bench_time.print("FAIL: {s} - content mismatch after rawr deserialize\n", .{name});
        tests_failed += 1;
        return error.ContentMismatch;
    }

    tests_passed += 1;
    const suffix = if (run_optimize) " [run-optimized]" else "";
    bench_time.print("  PASS: {s}{s} ({d} values, {d} bytes)\n", .{ name, suffix, values.len, rawr_bytes.len });
}

fn profileName(profile: test_gen.Profile) []const u8 {
    return switch (profile) {
        .sparse => "sparse",
        .dense => "dense",
        .full => "full",
        .runs => "runs",
        .single => "single",
        .boundary => "boundary",
    };
}

fn validateGeneratedRoundTrip(
    name: []const u8,
    seed: u64,
    chunks: []const test_gen.ChunkProfile,
    run_optimize: bool,
) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    var generated = try test_gen.build(allocator, prng.random(), chunks, run_optimize);
    defer generated.deinit();

    try validateBitmapRoundTrip(name, &generated.bm, generated.values, run_optimize);
}

fn validateGeneratedProfileMatrix(run_optimize: bool) !void {
    const profiles = [_]test_gen.Profile{ .sparse, .dense, .full, .runs, .single, .boundary };

    for (profiles, 0..) |a, ai| {
        for (profiles, 0..) |b, bi| {
            const chunks = [_]test_gen.ChunkProfile{
                .{ .key = 0, .profile = a },
                .{ .key = 1, .profile = b },
            };

            var name_buf: [96]u8 = undefined;
            const name = try std.fmt.bufPrint(
                &name_buf,
                "generated_{s}_{s}",
                .{ profileName(a), profileName(b) },
            );
            const seed = 0xA11C_E000 + (@as(u64, @intCast(ai)) << 8) + @as(u64, @intCast(bi));
            try validateGeneratedRoundTrip(name, seed, &chunks, run_optimize);
        }
    }
}

fn validateRandomMixedRoundTrip(name: []const u8, seed: u64, run_optimize: bool) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    var generated = try test_gen.randomMixed(allocator, prng.random(), 6, run_optimize);
    defer generated.deinit();

    try validateBitmapRoundTrip(name, &generated.bm, generated.values, run_optimize);
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
        bench_time.print("FAIL: {s} - bytes differ! rawr={d} bytes, croaring={d} bytes\n", .{ name, rawr_bytes.len, cr_buf.len });
        const min_len = @min(rawr_bytes.len, cr_buf.len);
        for (0..min_len) |i| {
            if (rawr_bytes[i] != cr_buf[i]) {
                bench_time.print("  First difference at byte {d}: rawr=0x{x:0>2} cr=0x{x:0>2}\n", .{ i, rawr_bytes[i], cr_buf[i] });
                break;
            }
        }
        tests_failed += 1;
        return error.ByteMismatch;
    }

    // --- Cross-deserialize: rawr bytes -> CRoaring ---
    const cr2 = c.roaring_bitmap_portable_deserialize_safe(@ptrCast(rawr_bytes.ptr), rawr_bytes.len) orelse {
        bench_time.print("FAIL: {s} - CRoaring failed to deserialize rawr bytes\n", .{name});
        tests_failed += 1;
        return error.CRoaringDeserializeFailed;
    };
    defer c.roaring_bitmap_free(cr2);

    if (c.roaring_bitmap_get_cardinality(cr2) != rbm.cardinality()) {
        bench_time.print("FAIL: {s} - cardinality mismatch\n", .{name});
        tests_failed += 1;
        return error.CardinalityMismatch;
    }

    // --- Cross-deserialize: CRoaring bytes -> rawr ---
    var rbm2 = RoaringBitmap.deserialize(allocator, cr_buf) catch |err| {
        bench_time.print("FAIL: {s} - rawr failed to deserialize CRoaring bytes: {s}\n", .{ name, @errorName(err) });
        tests_failed += 1;
        return error.RawrDeserializeFailed;
    };
    defer rbm2.deinit();

    if (!rbm2.equals(&rbm)) {
        bench_time.print("FAIL: {s} - content mismatch\n", .{name});
        tests_failed += 1;
        return error.ContentMismatch;
    }

    tests_passed += 1;
    const suffix = if (run_optimize) " [run-optimized]" else "";
    bench_time.print("  PASS: {s}{s} ({d} values, {d} bytes)\n", .{ name, suffix, end - start + 1, rawr_bytes.len });
}

/// Validate FrozenBitmap can read serialized bytes and contains() works correctly.
fn validateFrozenContains(name: []const u8, values: []const u32, run_optimize: bool) !void {
    // --- Build rawr bitmap and serialize ---
    var rbm = try RoaringBitmap.init(allocator);
    defer rbm.deinit();
    for (values) |v| {
        _ = try rbm.add(v);
    }
    if (run_optimize) {
        _ = try rbm.runOptimize();
    }

    const rawr_bytes = try rawr.RoaringBitmap.serialize(&rbm, allocator);
    defer allocator.free(rawr_bytes);

    // --- Wrap in FrozenBitmap and verify contains ---
    const frozen = FrozenBitmap.init(rawr_bytes) catch |err| {
        bench_time.print("FAIL: {s} - FrozenBitmap.init failed: {s}\n", .{ name, @errorName(err) });
        tests_failed += 1;
        return error.FrozenInitFailed;
    };

    // Check cardinality
    if (frozen.cardinality() != rbm.cardinality()) {
        bench_time.print("FAIL: {s} - FrozenBitmap cardinality mismatch\n", .{name});
        tests_failed += 1;
        return error.CardinalityMismatch;
    }

    // Check all values are present
    for (values) |v| {
        if (!frozen.contains(v)) {
            bench_time.print("FAIL: {s} - FrozenBitmap missing value {d}\n", .{ name, v });
            tests_failed += 1;
            return error.MissingValue;
        }
    }

    // Spot check some values that should NOT be present
    const absent_values = [_]u32{ 0xDEADBEEF, 0xCAFEBABE, 0x12345678 };
    for (absent_values) |v| {
        // Only check if the value wasn't in our input
        var found = false;
        for (values) |input_v| {
            if (input_v == v) {
                found = true;
                break;
            }
        }
        if (!found and frozen.contains(v)) {
            bench_time.print("FAIL: {s} - FrozenBitmap false positive for {d}\n", .{ name, v });
            tests_failed += 1;
            return error.FalsePositive;
        }
    }

    tests_passed += 1;
    const suffix = if (run_optimize) " [run-optimized]" else "";
    bench_time.print("  PASS: {s}{s} (FrozenBitmap, {d} values)\n", .{ name, suffix, values.len });
}

pub fn main() !void {
    bench_time.print("CRoaring Interop Validation\n", .{});
    bench_time.print("===========================\n\n", .{});

    // ========== Basic tests ==========
    bench_time.print("Basic tests:\n", .{});

    // Empty bitmap
    try validateRoundTrip("empty", &.{}, false);

    // Single elements
    try validateRoundTrip("single_zero", &.{0}, false);
    try validateRoundTrip("single_max", &.{0xFFFFFFFF}, false);
    try validateRoundTrip("single_mid", &.{1000000}, false);

    // ========== Array container tests ==========
    bench_time.print("\nArray container tests:\n", .{});

    // Small array
    var arr100: [100]u32 = undefined;
    for (0..100) |i| arr100[i] = @intCast(i * 10);
    try validateRoundTrip("array_100", &arr100, false);

    // Array at threshold (4096 = max array size)
    var arr4096: [4096]u32 = undefined;
    for (0..4096) |i| arr4096[i] = @intCast(i);
    try validateRoundTrip("array_4096", &arr4096, false);

    // ========== Bitset container tests ==========
    bench_time.print("\nBitset container tests:\n", .{});

    // Just over threshold -> bitset
    var bitset5000: [5000]u32 = undefined;
    for (0..5000) |i| bitset5000[i] = @intCast(i);
    try validateRoundTrip("bitset_5000", &bitset5000, false);

    // Full chunk as run (65536 values) - CRoaring auto-optimizes to run, so we must too
    // (This tests run serialization, not bitset - renamed to avoid confusion)
    try validateRangeRoundTrip("run_full_chunk", 0, 65535, true);

    // ========== Multiple container tests ==========
    bench_time.print("\nMultiple container tests:\n", .{});

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
    bench_time.print("\nRun-optimized tests:\n", .{});

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

    // 4+ containers with run_optimize - exercises run format WITH offset header
    // (NO_OFFSET_THRESHOLD = 4, so this triggers offset header in run format)
    var four_chunks_runs: [400]u32 = undefined;
    for (0..100) |i| four_chunks_runs[i] = @intCast(i); // chunk 0: 0-99
    for (0..100) |i| four_chunks_runs[100 + i] = @intCast(65536 + i); // chunk 1
    for (0..100) |i| four_chunks_runs[200 + i] = @intCast(131072 + i); // chunk 2
    for (0..100) |i| four_chunks_runs[300 + i] = @intCast(196608 + i); // chunk 3
    try validateRoundTrip("four_chunks_run_optimized", &four_chunks_runs, true);

    // ========== Generated profile tests ==========
    bench_time.print("\nGenerated profile matrix tests:\n", .{});
    try validateGeneratedProfileMatrix(false);
    try validateGeneratedProfileMatrix(true);

    try validateRandomMixedRoundTrip("random_mixed_seed_01", 0xD1FF_7E57_1001, false);
    try validateRandomMixedRoundTrip("random_mixed_seed_02", 0xD1FF_7E57_1002, false);
    try validateRandomMixedRoundTrip("random_mixed_seed_03", 0xD1FF_7E57_1003, false);
    try validateRandomMixedRoundTrip("random_mixed_seed_01", 0xD1FF_7E57_1001, true);
    try validateRandomMixedRoundTrip("random_mixed_seed_02", 0xD1FF_7E57_1002, true);
    try validateRandomMixedRoundTrip("random_mixed_seed_03", 0xD1FF_7E57_1003, true);

    // ========== addRange differential tests ==========
    bench_time.print("\naddRange differential tests:\n", .{});
    try validateRangeRoundTrip("addrange_run_single_chunk", 100, 10_000, true);
    try validateRangeRoundTrip("addrange_large_contiguous", 0, 5_000, false);
    try validateRangeRoundTrip("addrange_cross_chunks", 65_530, 196_620, true);

    // ========== Large scale tests ==========
    bench_time.print("\nLarge scale tests:\n", .{});

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

    // ========== FrozenBitmap tests ==========
    // Gap 1 fix: validate FrozenBitmap can read serialized bytes correctly
    bench_time.print("\nFrozenBitmap tests:\n", .{});

    // Array container
    try validateFrozenContains("frozen_array", &arr100, false);

    // Bitset container
    try validateFrozenContains("frozen_bitset", &bitset5000, false);

    // Run container (single chunk)
    try validateFrozenContains("frozen_run_single", &multi_range, true);

    // Run container with offset header (4+ chunks)
    try validateFrozenContains("frozen_run_with_offsets", &four_chunks_runs, true);

    // Multiple containers without run optimize
    try validateFrozenContains("frozen_multi_container", &five_containers, false);

    // ========== Summary ==========
    bench_time.print("\n===========================\n", .{});
    bench_time.print("Results: {d} passed, {d} failed\n", .{ tests_passed, tests_failed });

    if (tests_failed > 0) {
        return error.TestsFailed;
    }
    bench_time.print("\nAll validation tests passed!\n", .{});
}
