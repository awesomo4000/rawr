const std = @import("std");
const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;
const Container = @import("container.zig").Container;
const TaggedPtr = @import("container.zig").TaggedPtr;
const ArrayContainer = @import("array_container.zig").ArrayContainer;
const BitsetContainer = @import("bitset_container.zig").BitsetContainer;
const RunContainer = @import("run_container.zig").RunContainer;

/// Convert containers to run encoding where it saves space.
/// Returns the number of containers that were converted.
pub fn runOptimize(bm: *RoaringBitmap) !u32 {
    var converted: u32 = 0;

    for (0..bm.size) |i| {
        const tp = bm.containers[i];
        const container = Container.fromTagged(tp);

        switch (container) {
            .array => |ac| {
                // Count runs in array
                const n_runs = countRunsInArray(ac);
                // Array: cardinality * 2 bytes, Run: n_runs * 4 bytes
                if (n_runs * 4 < @as(u32, ac.cardinality) * 2) {
                    const rc = try arrayToRunContainer(bm.allocator, ac);
                    ac.deinit(bm.allocator);
                    bm.containers[i] = TaggedPtr.initRun(rc);
                    converted += 1;
                }
            },
            .bitset => |bc| {
                // Count runs in bitset
                const n_runs = countRunsInBitset(bc);
                // Bitset: 8192 bytes, Run: n_runs * 4 bytes
                if (n_runs * 4 < 8192) {
                    const rc = try bitsetToRunContainer(bm.allocator, bc);
                    bc.deinit(bm.allocator);
                    bm.containers[i] = TaggedPtr.initRun(rc);
                    converted += 1;
                }
            },
            .run => {}, // Already a run container
            .reserved => {},
        }
    }

    return converted;
}

/// Count the number of runs that would be needed to represent an array container.
fn countRunsInArray(ac: *ArrayContainer) u32 {
    if (ac.cardinality == 0) return 0;

    var n_runs: u32 = 1;
    var prev: u16 = ac.values[0];

    for (ac.values[1..ac.cardinality]) |v| {
        if (v != prev + 1) {
            n_runs += 1;
        }
        prev = v;
    }

    return n_runs;
}

/// Count the number of runs in a bitset container.
/// Uses bit-parallel run-start detection: a run starts where bit=1 and previous bit=0.
pub fn countRunsInBitset(bc: *BitsetContainer) u32 {
    var n_runs: u32 = 0;
    var prev_high_bit: u64 = 0; // MSB of previous word carried forward

    for (bc.words) |word| {
        // Shift word left by 1, filling bit 0 with the MSB of the previous word.
        // This gives us the "previous bit" for each position.
        const prev_bits = (word << 1) | prev_high_bit;
        // A run starts wherever current=1 and previous=0.
        const run_starts = word & ~prev_bits;
        n_runs += @popCount(run_starts);
        // Carry the MSB to the next word.
        prev_high_bit = word >> 63;
    }

    return n_runs;
}

/// Convert array container to run container.
fn arrayToRunContainer(allocator: std.mem.Allocator, ac: *ArrayContainer) !*RunContainer {
    if (ac.cardinality == 0) {
        return RunContainer.init(allocator, 0);
    }

    const n_runs = countRunsInArray(ac);
    const rc = try RunContainer.init(allocator, @intCast(n_runs));
    errdefer rc.deinit(allocator);

    var run_idx: u16 = 0;
    var run_start: u16 = ac.values[0];
    var run_len: u16 = 0;

    for (ac.values[1..ac.cardinality]) |v| {
        if (v == run_start + run_len + 1) {
            run_len += 1;
        } else {
            rc.runs[run_idx] = .{ .start = run_start, .length = run_len };
            run_idx += 1;
            run_start = v;
            run_len = 0;
        }
    }
    // Last run
    rc.runs[run_idx] = .{ .start = run_start, .length = run_len };
    rc.n_runs = @intCast(n_runs);

    return rc;
}

/// Convert bitset container to run container.
fn bitsetToRunContainer(allocator: std.mem.Allocator, bc: *BitsetContainer) !*RunContainer {
    const n_runs = countRunsInBitset(bc);
    const rc = try RunContainer.init(allocator, @intCast(n_runs));
    errdefer rc.deinit(allocator);

    var run_idx: u16 = 0;
    var in_run = false;
    var run_start: u16 = 0;

    for (bc.words, 0..) |word, word_idx| {
        const base: u16 = @intCast(word_idx * 64);
        var w = word;
        var bit_idx: u6 = 0;

        while (w != 0 or in_run) {
            const bit: u1 = @truncate(w);
            const pos = base + bit_idx;

            if (bit == 1 and !in_run) {
                // Start new run
                run_start = pos;
                in_run = true;
            } else if (bit == 0 and in_run) {
                // End run
                rc.runs[run_idx] = .{ .start = run_start, .length = pos - run_start - 1 };
                run_idx += 1;
                in_run = false;
            }

            if (bit_idx == 63) break;
            w >>= 1;
            bit_idx += 1;
        }
    }

    // Handle run that extends to end
    if (in_run) {
        rc.runs[run_idx] = .{ .start = run_start, .length = 65535 - run_start };
        run_idx += 1;
    }

    rc.n_runs = run_idx;
    return rc;
}

// ============================================================================
// Tests
// ============================================================================

test "runOptimize converts array with runs" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Add a range - creates array container with consecutive values
    _ = try bm.addRange(100, 200);
    try std.testing.expectEqual(@as(u64, 101), bm.cardinality());

    // Initially should be array
    const container_before = Container.fromTagged(bm.containers[0]);
    try std.testing.expect(container_before == .array);

    // runOptimize should convert to run (101 values as 1 run = 4 bytes vs 202 bytes)
    const converted = try bm.runOptimize();
    try std.testing.expectEqual(@as(u32, 1), converted);

    // Now should be run
    const container_after = Container.fromTagged(bm.containers[0]);
    try std.testing.expect(container_after == .run);

    // Data should be preserved
    try std.testing.expectEqual(@as(u64, 101), bm.cardinality());
    try std.testing.expect(bm.contains(100));
    try std.testing.expect(bm.contains(200));
    try std.testing.expect(!bm.contains(99));
    try std.testing.expect(!bm.contains(201));
}

test "runOptimize keeps sparse array" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Add sparse values - each becomes its own run
    _ = try bm.add(100);
    _ = try bm.add(200);
    _ = try bm.add(300);
    _ = try bm.add(400);

    // 4 values = 8 bytes array, 4 runs = 16 bytes, so array is better
    const converted = try bm.runOptimize();
    try std.testing.expectEqual(@as(u32, 0), converted);

    const container = Container.fromTagged(bm.containers[0]);
    try std.testing.expect(container == .array);
}

test "runOptimize converts bitset with long run" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Add large range to create bitset (>4096 values)
    _ = try bm.addRange(0, 10000);
    try std.testing.expectEqual(@as(u64, 10001), bm.cardinality());

    // Should be bitset (8192 bytes) which is > 1 run * 4 bytes
    const container_before = Container.fromTagged(bm.containers[0]);
    try std.testing.expect(container_before == .bitset);

    const converted = try bm.runOptimize();
    try std.testing.expectEqual(@as(u32, 1), converted);

    const container_after = Container.fromTagged(bm.containers[0]);
    try std.testing.expect(container_after == .run);

    // Data preserved
    try std.testing.expectEqual(@as(u64, 10001), bm.cardinality());
    try std.testing.expect(bm.contains(0));
    try std.testing.expect(bm.contains(10000));
}

test "runOptimize counts runs correctly across word boundaries" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Create bitset with runs spanning word boundaries
    // Word 0 has 64 bits (0-63), word 1 starts at bit 64
    // Run 1: bits 0-65 (spans word boundary between words 0 and 1)
    // Run 2: bits 5000-5065 (another run spanning a word boundary)
    // Total: 132 values in 2 runs, plus we need >4096 to stay as bitset
    _ = try bm.addRange(0, 65); // 66 values spanning word 0-1 boundary
    _ = try bm.addRange(5000, 5065); // 66 values spanning another word boundary
    _ = try bm.addRange(10000, 14000); // 4001 values to force bitset

    const container = Container.fromTagged(bm.containers[0]);
    try std.testing.expect(container == .bitset);

    // This should count as 3 runs. The word boundary bug would overcount.
    // 3 runs * 4 bytes = 12 bytes << 8192 bytes, so should convert.
    const converted = try bm.runOptimize();
    try std.testing.expectEqual(@as(u32, 1), converted);

    // Verify it's now a run container with correct data
    const container_after = Container.fromTagged(bm.containers[0]);
    try std.testing.expect(container_after == .run);

    // Data preserved
    try std.testing.expect(bm.contains(0));
    try std.testing.expect(bm.contains(65));
    try std.testing.expect(!bm.contains(66));
    try std.testing.expect(bm.contains(5000));
    try std.testing.expect(bm.contains(10000));
    try std.testing.expect(bm.contains(14000));
}

test "runOptimize no-op on run containers" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.addRange(100, 200);
    _ = try bm.runOptimize();

    // Second call should convert nothing
    const converted = try bm.runOptimize();
    try std.testing.expectEqual(@as(u32, 0), converted);
}
