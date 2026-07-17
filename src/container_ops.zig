// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const ArrayContainer = @import("array_container.zig").ArrayContainer;
const BitsetContainer = @import("bitset_container.zig").BitsetContainer;
const RunContainer = @import("run_container.zig").RunContainer;
const array_kernels = @import("array_kernels.zig");
const container_mod = @import("container.zig");
const Container = container_mod.Container;

/// Cross-container operations: all 9 pairwise combinations for each set operation.
/// Returns newly allocated containers.

// ============================================================================
// Helpers
// ============================================================================

/// Count values <= `low` in a container.
pub fn containerRank(c: Container, low: u16) u32 {
    return switch (c) {
        .array => |ac| arrayRank(ac, low),
        .bitset => |bc| bitsetRank(bc, low),
        .run => |rc| runRank(rc, low),
        .reserved => unreachable,
    };
}

/// Rank sorted probes that all target this container in one forward sweep.
/// `base` is the rank accumulated from prior containers.
pub fn containerRankMany(c: Container, base: u64, values: []const u32, out: []u64) usize {
    std.debug.assert(out.len >= values.len);
    switch (c) {
        .array => |ac| arrayRankMany(ac, base, values, out),
        .bitset => |bc| bitsetRankMany(bc, base, values, out),
        .run => |rc| runRankMany(rc, base, values, out),
        .reserved => unreachable,
    }
    return values.len;
}

/// Return the k-th value in a container, 0-based.
pub fn containerSelect(c: Container, k: u32) ?u16 {
    return switch (c) {
        .array => |ac| arraySelect(ac, k),
        .bitset => |bc| bitsetSelect(bc, k),
        .run => |rc| runSelect(rc, k),
        .reserved => unreachable,
    };
}

/// Count values in the inclusive low-16 range [start, end].
pub fn containerRangeCardinality(c: Container, start: u16, end: u16) u32 {
    if (start > end) return 0;
    return switch (c) {
        .array, .run => blk: {
            const hi = containerRank(c, end);
            const lo = if (start == 0) 0 else containerRank(c, start - 1);
            break :blk hi - lo;
        },
        .bitset => |bc| bitsetRangeCardinality(bc, start, end),
        .reserved => unreachable,
    };
}

fn arrayRankMany(ac: *const ArrayContainer, base: u64, values: []const u32, out: []u64) void {
    var cursor: usize = 0;
    const array_values = ac.values[0..ac.cardinality];
    for (values, out[0..values.len]) |value, *rank_out| {
        const low: u16 = @truncate(value);
        while (cursor < array_values.len and array_values[cursor] <= low) : (cursor += 1) {}
        rank_out.* = base + cursor;
    }
}

fn bitsetRankMany(bc: *const BitsetContainer, base: u64, values: []const u32, out: []u64) void {
    var word_idx: usize = 0;
    var running: u32 = 0;

    for (values, out[0..values.len]) |value, *rank_out| {
        const low: u16 = @truncate(value);
        const target_word: usize = low >> 6;
        const bit: u6 = @truncate(low);

        while (word_idx < target_word) : (word_idx += 1) {
            running += @popCount(bc.words[word_idx]);
        }

        const mask = if (bit == 63)
            ~@as(u64, 0)
        else
            (@as(u64, 1) << (bit + 1)) - 1;
        rank_out.* = base + running + @popCount(bc.words[target_word] & mask);
    }
}

fn runRankMany(rc: *const RunContainer, base: u64, values: []const u32, out: []u64) void {
    var run_idx: usize = 0;
    var running: u32 = 0;

    for (values, out[0..values.len]) |value, *rank_out| {
        const low: u16 = @truncate(value);

        while (run_idx < rc.n_runs and rc.runs[run_idx].end() < low) : (run_idx += 1) {
            running += rc.runs[run_idx].size();
        }

        if (run_idx >= rc.n_runs or low < rc.runs[run_idx].start) {
            rank_out.* = base + running;
        } else {
            rank_out.* = base + running + @as(u32, low - rc.runs[run_idx].start) + 1;
        }
    }
}

/// Return whether every value in inclusive low-16 range [start, end] is present.
pub fn containerContainsRange(c: Container, start: u16, end: u16) bool {
    if (start > end) return true;
    return switch (c) {
        .array => |ac| arrayContainsRange(ac, start, end),
        .bitset => |bc| bitsetContainsRange(bc, start, end),
        .run => |rc| runContainsRange(rc, start, end),
        .reserved => unreachable,
    };
}

/// Return whether any value in inclusive low-16 range [start, end] is present.
pub fn containerIntersectsRange(c: Container, start: u16, end: u16) bool {
    if (start > end) return false;
    return switch (c) {
        .array => |ac| arrayIntersectsRange(ac, start, end),
        .bitset => |bc| bitsetIntersectsRange(bc, start, end),
        .run => |rc| runIntersectsRange(rc, start, end),
        .reserved => unreachable,
    };
}

fn arrayRank(ac: *const ArrayContainer, low: u16) u32 {
    var lo: usize = 0;
    var hi: usize = ac.cardinality;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (ac.values[mid] <= low) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return @intCast(lo);
}

fn bitsetRank(bc: *const BitsetContainer, low: u16) u32 {
    const word_idx: usize = low >> 6;
    const bit: u6 = @truncate(low);

    var count: u32 = 0;
    for (bc.words[0..word_idx]) |word| {
        count += @popCount(word);
    }

    const mask = if (bit == 63)
        ~@as(u64, 0)
    else
        (@as(u64, 1) << (bit + 1)) - 1;
    count += @popCount(bc.words[word_idx] & mask);
    return count;
}

fn bitsetRangeCardinality(bc: *const BitsetContainer, start: u16, end: u16) u32 {
    const first_word: usize = start >> 6;
    const last_word: usize = end >> 6;
    const start_bit: u6 = @truncate(start);
    const end_bit: u6 = @truncate(end);

    if (first_word == last_word) {
        return @popCount(bc.words[first_word] & bitRangeMask(start_bit, end_bit));
    }

    var count: u32 = @popCount(bc.words[first_word] & bitRangeMask(start_bit, 63));
    count += BitsetContainer.countWords(bc.words[first_word + 1 .. last_word]);
    count += @popCount(bc.words[last_word] & bitRangeMask(0, end_bit));
    return count;
}

fn runRank(rc: *const RunContainer, low: u16) u32 {
    var count: u32 = 0;
    for (rc.runs[0..rc.n_runs]) |run| {
        if (low < run.start) return count;

        const end = run.end();
        if (low <= end) {
            count += @as(u32, low - run.start) + 1;
            return count;
        }

        count += run.size();
    }
    return count;
}

fn arraySelect(ac: *const ArrayContainer, k: u32) ?u16 {
    if (k >= ac.cardinality) return null;
    return ac.values[k];
}

fn bitsetSelect(bc: *const BitsetContainer, k: u32) ?u16 {
    var remaining = k;
    for (bc.words, 0..) |word, word_idx| {
        const word_card: u32 = @popCount(word);
        if (remaining >= word_card) {
            remaining -= word_card;
            continue;
        }

        var bits = word;
        while (remaining > 0) : (remaining -= 1) {
            bits &= bits - 1;
        }
        return @intCast(word_idx * 64 + @ctz(bits));
    }
    return null;
}

fn runSelect(rc: *const RunContainer, k: u32) ?u16 {
    var remaining = k;
    for (rc.runs[0..rc.n_runs]) |run| {
        const size = run.size();
        if (remaining < size) {
            return run.start + @as(u16, @intCast(remaining));
        }
        remaining -= size;
    }
    return null;
}

fn arrayContainsRange(ac: *const ArrayContainer, start: u16, end: u16) bool {
    const range_size = @as(u32, end) - start + 1;
    const below_start = if (start == 0) 0 else arrayRank(ac, start - 1);
    return arrayRank(ac, end) - below_start == range_size;
}

fn bitsetContainsRange(bc: *const BitsetContainer, start: u16, end: u16) bool {
    const first_word: usize = start >> 6;
    const last_word: usize = end >> 6;
    const start_bit: u6 = @truncate(start);
    const end_bit: u6 = @truncate(end);

    if (first_word == last_word) {
        const mask = bitRangeMask(start_bit, end_bit);
        return (bc.words[first_word] & mask) == mask;
    }

    const first_mask = bitRangeMask(start_bit, 63);
    if ((bc.words[first_word] & first_mask) != first_mask) return false;

    for (bc.words[first_word + 1 .. last_word]) |word| {
        if (word != ~@as(u64, 0)) return false;
    }

    const last_mask = bitRangeMask(0, end_bit);
    return (bc.words[last_word] & last_mask) == last_mask;
}

fn runContainsRange(rc: *const RunContainer, start: u16, end: u16) bool {
    for (rc.runs[0..rc.n_runs]) |run| {
        if (end < run.start) return false;
        if (start >= run.start and end <= run.end()) return true;
    }
    return false;
}

fn arrayIntersectsRange(ac: *const ArrayContainer, start: u16, end: u16) bool {
    const idx = array_kernels.gallopSearch(ac.values[0..ac.cardinality], start, 0);
    return idx < ac.cardinality and ac.values[idx] <= end;
}

fn bitsetIntersectsRange(bc: *const BitsetContainer, start: u16, end: u16) bool {
    const first_word: usize = start >> 6;
    const last_word: usize = end >> 6;
    const start_bit: u6 = @truncate(start);
    const end_bit: u6 = @truncate(end);

    if (first_word == last_word) {
        return (bc.words[first_word] & bitRangeMask(start_bit, end_bit)) != 0;
    }

    if ((bc.words[first_word] & bitRangeMask(start_bit, 63)) != 0) return true;

    for (bc.words[first_word + 1 .. last_word]) |word| {
        if (word != 0) return true;
    }

    return (bc.words[last_word] & bitRangeMask(0, end_bit)) != 0;
}

fn runIntersectsRange(rc: *const RunContainer, start: u16, end: u16) bool {
    for (rc.runs[0..rc.n_runs]) |run| {
        if (end < run.start) return false;
        if (start <= run.end()) return true;
    }
    return false;
}

fn bitRangeMask(start_bit: u6, end_bit: u6) u64 {
    const end_mask = if (end_bit == 63)
        ~@as(u64, 0)
    else
        (@as(u64, 1) << (end_bit + 1)) - 1;
    const start_mask = (@as(u64, 1) << start_bit) - 1;
    return end_mask & ~start_mask;
}

fn copyBitsetRange(dst: *BitsetContainer, src: *const BitsetContainer, start: u16, end: u16) void {
    if (start > end) return;

    const start_word = start >> 6;
    const end_word = end >> 6;
    const start_bit: u6 = @truncate(start);
    const end_bit: u6 = @truncate(end);

    dst.cardinality = -1;

    if (start_word == end_word) {
        dst.words[start_word] |= src.words[start_word] & bitRangeMask(start_bit, end_bit);
    } else {
        dst.words[start_word] |= src.words[start_word] & bitRangeMask(start_bit, 63);
        for (dst.words[start_word + 1 .. end_word], src.words[start_word + 1 .. end_word]) |*dst_word, src_word| {
            dst_word.* |= src_word;
        }
        dst.words[end_word] |= src.words[end_word] & bitRangeMask(0, end_bit);
    }
}

// ============================================================================
// Union (OR)
// ============================================================================

/// In-place union: a |= b. Modifies a's container directly when possible.
/// Returns the (possibly different) container to use. Caller should NOT free
/// the original if it was modified in place (check pointer equality).
/// For array∪array, this avoids allocating a new container entirely.
pub fn containerUnionInPlace(allocator: std.mem.Allocator, a: Container, b: Container) !Container {
    return switch (a) {
        .array => |ac| switch (b) {
            .array => |bc| arrayUnionArrayInPlace(allocator, ac, bc),
            .bitset => |bc| arrayUnionBitsetInPlace(allocator, ac, bc),
            .run => |rc| arrayUnionRun(allocator, ac, rc), // TODO: in-place version
            .reserved => unreachable,
        },
        .bitset => |ac| switch (b) {
            .array => |bc| bitsetUnionArrayInPlace(ac, bc),
            .bitset => |bc| bitsetUnionBitsetInPlace(ac, bc),
            .run => |rc| bitsetUnionRun(allocator, ac, rc),
            .reserved => unreachable,
        },
        .run => |ac| switch (b) {
            // Run containers convert to bitset/array, use non-in-place for now
            .array => |bc| arrayUnionRun(allocator, bc, ac),
            .bitset => |bc| bitsetUnionRun(allocator, bc, ac),
            .run => |rc| runUnionRun(allocator, ac, rc),
            .reserved => unreachable,
        },
        .reserved => unreachable,
    };
}

fn arrayUnionArrayInPlace(allocator: std.mem.Allocator, a: *ArrayContainer, b: *ArrayContainer) !Container {
    // Use ArrayContainer's in-place union
    const maybe_bitset = try a.unionInPlace(allocator, b);
    if (maybe_bitset) |bc| {
        // Converted to bitset - caller must free the array
        return .{ .bitset = bc };
    }
    // Stayed as array, same pointer
    return .{ .array = a };
}

/// Non-lazy union; the returned bitset always has a valid cardinality.
fn arrayUnionBitsetInPlace(allocator: std.mem.Allocator, ac: *ArrayContainer, bc: *BitsetContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);
    @memcpy(result.words, bc.words);
    result.setList(ac.values[0..ac.cardinality]);
    _ = result.computeCardinality();
    return .{ .bitset = result };
}

/// Non-lazy union; repairs the destination cardinality before returning.
fn bitsetUnionArrayInPlace(bc: *BitsetContainer, ac: *ArrayContainer) Container {
    bc.setList(ac.values[0..ac.cardinality]);
    _ = bc.computeCardinality();
    return .{ .bitset = bc };
}

fn bitsetUnionBitsetInPlace(a: *BitsetContainer, b: *BitsetContainer) Container {
    // OR words directly - no allocation
    a.unionWith(b);
    return .{ .bitset = a };
}

fn bitsetUnionRunInPlace(bc: *BitsetContainer, rc: *RunContainer) Container {
    // Use setRange for efficient word-level fills instead of element-by-element
    for (rc.runs[0..rc.n_runs]) |run| {
        bc.setRange(run.start, run.end());
    }
    bc.cardinality = -1; // setRange doesn't track cardinality, so invalidate here
    return .{ .bitset = bc };
}

pub fn containerDifferenceInPlace(allocator: std.mem.Allocator, a: Container, b: Container) !Container {
    return switch (a) {
        .array => containerDifference(allocator, a, b),
        .bitset => |bc| switch (b) {
            .array => |ac| bitsetDifferenceArrayInPlace(allocator, bc, ac),
            .bitset => |other_bc| bitsetDifferenceBitsetInPlace(allocator, bc, other_bc),
            .run => |rc| bitsetDifferenceRunInPlace(allocator, bc, rc),
            .reserved => unreachable,
        },
        .run => containerDifference(allocator, a, b),
        .reserved => unreachable,
    };
}

fn bitsetDifferenceArrayInPlace(allocator: std.mem.Allocator, bc: *BitsetContainer, ac: *ArrayContainer) !Container {
    for (ac.values[0..ac.cardinality]) |value| {
        _ = bc.remove(value);
    }
    return demoteBitsetIfSmall(allocator, bc);
}

fn bitsetDifferenceBitsetInPlace(allocator: std.mem.Allocator, a: *BitsetContainer, b: *BitsetContainer) !Container {
    a.differenceWith(b);
    return demoteBitsetIfSmall(allocator, a);
}

fn bitsetDifferenceRunInPlace(allocator: std.mem.Allocator, bc: *BitsetContainer, rc: *RunContainer) !Container {
    for (rc.runs[0..rc.n_runs]) |run| {
        bc.clearRange(run.start, run.end());
    }
    _ = bc.computeCardinality();
    return demoteBitsetIfSmall(allocator, bc);
}

pub fn containerXorInPlace(allocator: std.mem.Allocator, a: Container, b: Container) !Container {
    return switch (a) {
        .array => containerXor(allocator, a, b),
        .bitset => |bc| switch (b) {
            .array => |ac| bitsetXorArrayInPlace(allocator, bc, ac),
            .bitset => |other_bc| bitsetXorBitsetInPlace(allocator, bc, other_bc),
            .run => |rc| bitsetXorRunInPlace(allocator, bc, rc),
            .reserved => unreachable,
        },
        .run => containerXor(allocator, a, b),
        .reserved => unreachable,
    };
}

/// Non-lazy XOR; repairs cardinality before demotion or return.
fn bitsetXorArrayInPlace(allocator: std.mem.Allocator, bc: *BitsetContainer, ac: *ArrayContainer) !Container {
    bc.toggleList(ac.values[0..ac.cardinality]);
    _ = bc.computeCardinality();
    return demoteBitsetIfSmall(allocator, bc);
}

fn bitsetXorBitsetInPlace(allocator: std.mem.Allocator, a: *BitsetContainer, b: *BitsetContainer) !Container {
    a.symmetricDifferenceWith(b);
    return demoteBitsetIfSmall(allocator, a);
}

fn bitsetXorRunInPlace(allocator: std.mem.Allocator, bc: *BitsetContainer, rc: *RunContainer) !Container {
    for (rc.runs[0..rc.n_runs]) |run| {
        bc.toggleRange(run.start, run.end());
    }
    _ = bc.computeCardinality();
    return demoteBitsetIfSmall(allocator, bc);
}

fn demoteBitsetIfSmall(allocator: std.mem.Allocator, bc: *BitsetContainer) !Container {
    if (bc.getCardinality() <= ArrayContainer.MAX_CARDINALITY) {
        return .{ .array = try bitsetToArray(allocator, bc) };
    }
    return .{ .bitset = bc };
}

pub fn containerUnion(allocator: std.mem.Allocator, a: Container, b: Container) !Container {
    return switch (a) {
        .array => |ac| switch (b) {
            .array => |bc| arrayUnionArray(allocator, ac, bc),
            .bitset => |bc| arrayUnionBitset(allocator, ac, bc),
            .run => |rc| arrayUnionRun(allocator, ac, rc),
            .reserved => unreachable,
        },
        .bitset => |ac| switch (b) {
            .array => |bc| arrayUnionBitset(allocator, bc, ac), // commutative
            .bitset => |bc| bitsetUnionBitset(allocator, ac, bc),
            .run => |rc| bitsetUnionRun(allocator, ac, rc),
            .reserved => unreachable,
        },
        .run => |ac| switch (b) {
            .array => |bc| arrayUnionRun(allocator, bc, ac), // commutative
            .bitset => |bc| bitsetUnionRun(allocator, bc, ac), // commutative
            .run => |rc| runUnionRun(allocator, ac, rc),
            .reserved => unreachable,
        },
        .reserved => unreachable,
    };
}

/// Non-lazy union; computes the actual cardinality and demotes small results.
fn arrayUnionArray(allocator: std.mem.Allocator, a: *ArrayContainer, b: *ArrayContainer) !Container {
    const max_card = @as(u32, a.cardinality) + b.cardinality;

    // If combined could exceed array threshold, use bitset
    if (max_card > ArrayContainer.MAX_CARDINALITY) {
        const bc = try BitsetContainer.init(allocator);
        errdefer bc.deinit(allocator);
        bc.setList(a.values[0..a.cardinality]);
        bc.setList(b.values[0..b.cardinality]);
        const actual_cardinality = bc.computeCardinality();
        if (actual_cardinality <= ArrayContainer.MAX_CARDINALITY) {
            const arr = try bitsetToArray(allocator, bc);
            bc.deinit(allocator);
            return .{ .array = arr };
        }
        return .{ .bitset = bc };
    }

    // Merge two sorted arrays
    const result = try ArrayContainer.init(allocator, @intCast(@min(max_card, ArrayContainer.MAX_CARDINALITY)));
    errdefer result.deinit(allocator);

    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;
    const sa = a.values[0..a.cardinality];
    const sb = b.values[0..b.cardinality];

    // Branchless merge: always write the smaller value, advance contributing pointer(s).
    // On aarch64, LLVM emits csel for the output and cset for advances — no branches.
    while (i < sa.len and j < sb.len) {
        const a_val = sa[i];
        const b_val = sb[j];

        result.values[k] = if (a_val <= b_val) a_val else b_val;
        k += 1;

        i += @intFromBool(a_val <= b_val);
        j += @intFromBool(b_val <= a_val);
    }
    // Drain remaining elements
    while (i < sa.len) : (i += 1) {
        result.values[k] = sa[i];
        k += 1;
    }
    while (j < sb.len) : (j += 1) {
        result.values[k] = sb[j];
        k += 1;
    }
    result.cardinality = @intCast(k);
    return arrayToArrayOrRun(allocator, result);
}

/// Non-lazy union; the returned bitset always has a valid cardinality.
fn arrayUnionBitset(allocator: std.mem.Allocator, ac: *ArrayContainer, bc: *BitsetContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);

    // Copy bitset
    @memcpy(result.words, bc.words);

    result.setList(ac.values[0..ac.cardinality]);
    _ = result.computeCardinality();
    return .{ .bitset = result };
}

/// Non-lazy union; repairs cardinality before representation selection.
fn arrayUnionRun(allocator: std.mem.Allocator, ac: *ArrayContainer, rc: *RunContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);

    for (rc.runs[0..rc.n_runs]) |run| {
        result.setRange(run.start, run.end());
    }
    result.setList(ac.values[0..ac.cardinality]);
    _ = result.computeCardinality();
    return bitsetToArrayOrRun(allocator, result);
}

fn bitsetUnionBitset(allocator: std.mem.Allocator, a: *BitsetContainer, b: *BitsetContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    @memcpy(result.words, a.words);
    result.unionWith(b);
    return .{ .bitset = result };
}

fn bitsetUnionRun(allocator: std.mem.Allocator, bc: *BitsetContainer, rc: *RunContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);
    @memcpy(result.words, bc.words);

    for (rc.runs[0..rc.n_runs]) |run| {
        result.setRange(run.start, run.end());
    }

    _ = result.computeCardinality();
    return bitsetToArrayOrRun(allocator, result);
}

fn runUnionRun(allocator: std.mem.Allocator, a: *RunContainer, b: *RunContainer) !Container {
    // Merge runs directly - O(n_runs) instead of O(cardinality)
    const max_runs = @as(usize, a.n_runs) + b.n_runs;
    const result = try RunContainer.init(allocator, @intCast(@min(max_runs, 65535)));
    errdefer result.deinit(allocator);

    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;

    while (i < a.n_runs or j < b.n_runs) {
        // Pick the run that starts first (or only remaining)
        const use_a = if (i >= a.n_runs) false else if (j >= b.n_runs) true else a.runs[i].start <= b.runs[j].start;

        const run = if (use_a) a.runs[i] else b.runs[j];
        if (use_a) i += 1 else j += 1;

        // Merge with previous run if adjacent or overlapping
        if (k > 0 and result.runs[k - 1].end() +| 1 >= run.start) {
            // Extend previous run
            result.runs[k - 1].length = @max(result.runs[k - 1].end(), run.end()) - result.runs[k - 1].start;
        } else {
            // Add new run
            result.runs[k] = run;
            k += 1;
        }
    }
    result.n_runs = @intCast(k);
    result.cardinality = -1;
    return .{ .run = result };
}

// ============================================================================
// Intersection (AND)
// ============================================================================

pub fn containerIntersection(allocator: std.mem.Allocator, a: Container, b: Container) !Container {
    return switch (a) {
        .array => |ac| switch (b) {
            .array => |bc| arrayIntersectArray(allocator, ac, bc),
            .bitset => |bc| arrayIntersectBitset(allocator, ac, bc),
            .run => |rc| arrayIntersectRun(allocator, ac, rc),
            .reserved => unreachable,
        },
        .bitset => |ac| switch (b) {
            .array => |bc| arrayIntersectBitset(allocator, bc, ac), // commutative
            .bitset => |bc| bitsetIntersectBitset(allocator, ac, bc),
            .run => |rc| bitsetIntersectRun(allocator, ac, rc),
            .reserved => unreachable,
        },
        .run => |ac| switch (b) {
            .array => |bc| arrayIntersectRun(allocator, bc, ac), // commutative
            .bitset => |bc| bitsetIntersectRun(allocator, bc, ac), // commutative
            .run => |rc| runIntersectRun(allocator, ac, rc),
            .reserved => unreachable,
        },
        .reserved => unreachable,
    };
}

fn arrayIntersectArray(allocator: std.mem.Allocator, a: *ArrayContainer, b: *ArrayContainer) !Container {
    const result = try ArrayContainer.init(allocator, @min(a.cardinality, b.cardinality));
    errdefer result.deinit(allocator);

    result.cardinality = @intCast(array_kernels.intersectWrite(
        a.values[0..a.cardinality],
        b.values[0..b.cardinality],
        result.values,
    ));
    return .{ .array = result };
}

fn arrayIntersectBitset(allocator: std.mem.Allocator, ac: *ArrayContainer, bc: *BitsetContainer) !Container {
    const result = try ArrayContainer.init(allocator, ac.cardinality);
    errdefer result.deinit(allocator);

    var k: usize = 0;
    for (ac.values[0..ac.cardinality]) |v| {
        if (bc.contains(v)) {
            result.values[k] = v;
            k += 1;
        }
    }
    result.cardinality = @intCast(k);
    return .{ .array = result };
}

fn arrayIntersectRun(allocator: std.mem.Allocator, ac: *ArrayContainer, rc: *RunContainer) !Container {
    const result = try ArrayContainer.init(allocator, ac.cardinality);
    errdefer result.deinit(allocator);

    var k: usize = 0;
    for (ac.values[0..ac.cardinality]) |v| {
        if (rc.contains(v)) {
            result.values[k] = v;
            k += 1;
        }
    }
    result.cardinality = @intCast(k);
    return .{ .array = result };
}

fn bitsetIntersectBitset(allocator: std.mem.Allocator, a: *BitsetContainer, b: *BitsetContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);
    @memcpy(result.words, a.words);
    result.intersectionWith(b);

    // Convert to array if cardinality is low
    if (result.getCardinality() <= ArrayContainer.MAX_CARDINALITY) {
        const arr = try bitsetToArray(allocator, result);
        result.deinit(allocator);
        return .{ .array = arr };
    }
    return .{ .bitset = result };
}

fn bitsetIntersectRun(allocator: std.mem.Allocator, bc: *BitsetContainer, rc: *RunContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);

    for (rc.runs[0..rc.n_runs]) |run| {
        copyBitsetRange(result, bc, run.start, run.end());
    }

    _ = result.computeCardinality();
    if (result.getCardinality() <= ArrayContainer.MAX_CARDINALITY) {
        const arr = try bitsetToArray(allocator, result);
        result.deinit(allocator);
        return .{ .array = arr };
    }
    return .{ .bitset = result };
}

fn runIntersectRun(allocator: std.mem.Allocator, a: *RunContainer, b: *RunContainer) !Container {
    // Intersect runs directly - find overlapping regions
    const max_result_runs = @as(usize, a.n_runs) + b.n_runs;
    const result = try RunContainer.init(allocator, @intCast(@min(max_result_runs, 65535)));
    errdefer result.deinit(allocator);

    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;

    while (i < a.n_runs and j < b.n_runs) {
        const ra = a.runs[i];
        const rb = b.runs[j];

        // Check if runs overlap
        if (ra.start <= rb.end() and rb.start <= ra.end()) {
            // Overlapping - create intersection run
            const start = @max(ra.start, rb.start);
            const end = @min(ra.end(), rb.end());
            result.runs[k] = .{ .start = start, .length = end - start };
            k += 1;
        }

        // Advance the run that ends first
        if (ra.end() < rb.end()) {
            i += 1;
        } else {
            j += 1;
        }
    }
    result.n_runs = @intCast(k);
    result.cardinality = -1;
    return .{ .run = result };
}

// ============================================================================
// Intersection Cardinality (no allocation)
// ============================================================================

/// Compute |a ∩ b| without allocating a result container.
pub fn containerIntersectionCardinality(a: Container, b: Container) u64 {
    return switch (a) {
        .array => |ac| switch (b) {
            .array => |bc| arrayIntersectArrayCard(ac, bc),
            .bitset => |bc| arrayIntersectBitsetCard(ac, bc),
            .run => |rc| arrayIntersectRunCard(ac, rc),
            .reserved => unreachable,
        },
        .bitset => |ac| switch (b) {
            .array => |bc| arrayIntersectBitsetCard(bc, ac),
            .bitset => |bc| bitsetIntersectBitsetCard(ac, bc),
            .run => |rc| bitsetIntersectRunCard(ac, rc),
            .reserved => unreachable,
        },
        .run => |ac| switch (b) {
            .array => |bc| arrayIntersectRunCard(bc, ac),
            .bitset => |bc| bitsetIntersectRunCard(bc, ac),
            .run => |rc| runIntersectRunCard(ac, rc),
            .reserved => unreachable,
        },
        .reserved => unreachable,
    };
}

/// Return true if a ∩ b is non-empty. Early exit on first match.
pub fn containerIntersects(a: Container, b: Container) bool {
    return switch (a) {
        .array => |ac| switch (b) {
            .array => |bc| arrayIntersectsArray(ac, bc),
            .bitset => |bc| arrayIntersectsBitset(ac, bc),
            .run => |rc| arrayIntersectsRun(ac, rc),
            .reserved => unreachable,
        },
        .bitset => |ac| switch (b) {
            .array => |bc| arrayIntersectsBitset(bc, ac),
            .bitset => |bc| bitsetIntersectsBitset(ac, bc),
            .run => |rc| bitsetIntersectsRun(ac, rc),
            .reserved => unreachable,
        },
        .run => |ac| switch (b) {
            .array => |bc| arrayIntersectsRun(bc, ac),
            .bitset => |bc| bitsetIntersectsRun(bc, ac),
            .run => |rc| runIntersectsRun(ac, rc),
            .reserved => unreachable,
        },
        .reserved => unreachable,
    };
}

fn arrayIntersectArrayCard(a: *ArrayContainer, b: *ArrayContainer) u64 {
    return array_kernels.intersectCard(
        a.values[0..a.cardinality],
        b.values[0..b.cardinality],
    );
}

fn arrayIntersectBitsetCard(ac: *ArrayContainer, bc: *BitsetContainer) u64 {
    var count: u64 = 0;
    for (ac.values[0..ac.cardinality]) |v| {
        if (bc.contains(v)) count += 1;
    }
    return count;
}

fn arrayIntersectRunCard(ac: *ArrayContainer, rc: *RunContainer) u64 {
    var count: u64 = 0;
    for (ac.values[0..ac.cardinality]) |v| {
        if (rc.contains(v)) count += 1;
    }
    return count;
}

fn bitsetIntersectBitsetCard(a: *BitsetContainer, b: *BitsetContainer) u64 {
    const VEC_SIZE = 8;
    const vec_count = 1024 / VEC_SIZE;
    var card_vec: @Vector(VEC_SIZE, u64) = @splat(0);
    for (0..vec_count) |i| {
        const base = i * VEC_SIZE;
        const va: @Vector(VEC_SIZE, u64) = a.words[base..][0..VEC_SIZE].*;
        const vb: @Vector(VEC_SIZE, u64) = b.words[base..][0..VEC_SIZE].*;
        const result = va & vb;
        card_vec += @popCount(result);
    }
    return @reduce(.Add, card_vec);
}

fn bitsetIntersectRunCard(bc: *BitsetContainer, rc: *RunContainer) u64 {
    var count: u64 = 0;
    for (rc.runs[0..rc.n_runs]) |run| {
        var v: u32 = run.start;
        while (v <= run.end()) : (v += 1) {
            if (bc.contains(@intCast(v))) count += 1;
        }
    }
    return count;
}

fn runIntersectRunCard(a: *RunContainer, b: *RunContainer) u64 {
    var i: usize = 0;
    var j: usize = 0;
    var count: u64 = 0;
    while (i < a.n_runs and j < b.n_runs) {
        const a_start = a.runs[i].start;
        const a_end = a.runs[i].end();
        const b_start = b.runs[j].start;
        const b_end = b.runs[j].end();

        if (a_start <= b_end and b_start <= a_end) {
            // Overlap
            const lo = @max(a_start, b_start);
            const hi = @min(a_end, b_end);
            count += @as(u64, hi - lo) + 1;
        }

        if (a_end <= b_end) i += 1 else j += 1;
    }
    return count;
}

// Intersects (early-exit) implementations

fn arrayIntersectsArray(a: *ArrayContainer, b: *ArrayContainer) bool {
    return array_kernels.intersectBool(
        a.values[0..a.cardinality],
        b.values[0..b.cardinality],
    );
}

fn arrayIntersectsBitset(ac: *ArrayContainer, bc: *BitsetContainer) bool {
    for (ac.values[0..ac.cardinality]) |v| {
        if (bc.contains(v)) return true;
    }
    return false;
}

fn arrayIntersectsRun(ac: *ArrayContainer, rc: *RunContainer) bool {
    for (ac.values[0..ac.cardinality]) |v| {
        if (rc.contains(v)) return true;
    }
    return false;
}

fn bitsetIntersectsBitset(a: *BitsetContainer, b: *BitsetContainer) bool {
    for (a.words[0..1024], b.words[0..1024]) |wa, wb| {
        if (wa & wb != 0) return true;
    }
    return false;
}

fn bitsetIntersectsRun(bc: *BitsetContainer, rc: *RunContainer) bool {
    for (rc.runs[0..rc.n_runs]) |run| {
        var v: u32 = run.start;
        while (v <= run.end()) : (v += 1) {
            if (bc.contains(@intCast(v))) return true;
        }
    }
    return false;
}

fn runIntersectsRun(a: *RunContainer, b: *RunContainer) bool {
    var i: usize = 0;
    var j: usize = 0;
    while (i < a.n_runs and j < b.n_runs) {
        const a_start = a.runs[i].start;
        const a_end = a.runs[i].end();
        const b_start = b.runs[j].start;
        const b_end = b.runs[j].end();

        if (a_start <= b_end and b_start <= a_end) {
            return true; // Overlap found
        }

        if (a_end <= b_end) i += 1 else j += 1;
    }
    return false;
}

// ============================================================================
// Difference (AND NOT)
// ============================================================================

pub fn containerDifference(allocator: std.mem.Allocator, a: Container, b: Container) !Container {
    return switch (a) {
        .array => |ac| switch (b) {
            .array => |bc| arrayDifferenceArray(allocator, ac, bc),
            .bitset => |bc| arrayDifferenceBitset(allocator, ac, bc),
            .run => |rc| arrayDifferenceRun(allocator, ac, rc),
            .reserved => unreachable,
        },
        .bitset => |ac| switch (b) {
            .array => |bc| bitsetDifferenceArray(allocator, ac, bc),
            .bitset => |bc| bitsetDifferenceBitset(allocator, ac, bc),
            .run => |rc| bitsetDifferenceRun(allocator, ac, rc),
            .reserved => unreachable,
        },
        .run => |ac| switch (b) {
            .array => |bc| runDifferenceArray(allocator, ac, bc),
            .bitset => |bc| runDifferenceBitset(allocator, ac, bc),
            .run => |rc| runDifferenceRun(allocator, ac, rc),
            .reserved => unreachable,
        },
        .reserved => unreachable,
    };
}

fn arrayDifferenceArray(allocator: std.mem.Allocator, a: *ArrayContainer, b: *ArrayContainer) !Container {
    const result = try ArrayContainer.init(allocator, a.cardinality);
    errdefer result.deinit(allocator);

    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;
    const sa = a.values[0..a.cardinality];
    const sb = b.values[0..b.cardinality];

    // Branchless merge: keep element from A only when A < B (not in B).
    while (i < sa.len and j < sb.len) {
        const a_val = sa[i];
        const b_val = sb[j];

        // Write a_val only when strictly less than b_val (not in B).
        if (a_val < b_val) {
            result.values[k] = a_val;
            k += 1;
        }

        // Advance pointers branchlessly.
        i += @intFromBool(a_val <= b_val);
        j += @intFromBool(b_val <= a_val);
    }
    // Drain remaining from A (all not in B since B is exhausted).
    while (i < sa.len) : (i += 1) {
        result.values[k] = sa[i];
        k += 1;
    }
    result.cardinality = @intCast(k);
    return .{ .array = result };
}

fn arrayDifferenceBitset(allocator: std.mem.Allocator, ac: *ArrayContainer, bc: *BitsetContainer) !Container {
    const result = try ArrayContainer.init(allocator, ac.cardinality);
    errdefer result.deinit(allocator);

    var k: usize = 0;
    for (ac.values[0..ac.cardinality]) |v| {
        if (!bc.contains(v)) {
            result.values[k] = v;
            k += 1;
        }
    }
    result.cardinality = @intCast(k);
    return .{ .array = result };
}

fn arrayDifferenceRun(allocator: std.mem.Allocator, ac: *ArrayContainer, rc: *RunContainer) !Container {
    const result = try ArrayContainer.init(allocator, ac.cardinality);
    errdefer result.deinit(allocator);

    var k: usize = 0;
    for (ac.values[0..ac.cardinality]) |v| {
        if (!rc.contains(v)) {
            result.values[k] = v;
            k += 1;
        }
    }
    result.cardinality = @intCast(k);
    return arrayToArrayOrRun(allocator, result);
}

fn bitsetDifferenceArray(allocator: std.mem.Allocator, bc: *BitsetContainer, ac: *ArrayContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);
    @memcpy(result.words, bc.words);

    for (ac.values[0..ac.cardinality]) |v| {
        _ = result.remove(v);
    }

    const card = result.computeCardinality();
    if (card <= ArrayContainer.MAX_CARDINALITY) {
        const arr = try bitsetToArray(allocator, result);
        result.deinit(allocator);
        return .{ .array = arr };
    }
    return .{ .bitset = result };
}

fn bitsetDifferenceBitset(allocator: std.mem.Allocator, a: *BitsetContainer, b: *BitsetContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);
    @memcpy(result.words, a.words);
    result.differenceWith(b);

    const card = result.getCardinality();
    if (card <= ArrayContainer.MAX_CARDINALITY) {
        const arr = try bitsetToArray(allocator, result);
        result.deinit(allocator);
        return .{ .array = arr };
    }
    return .{ .bitset = result };
}

fn bitsetDifferenceRun(allocator: std.mem.Allocator, bc: *BitsetContainer, rc: *RunContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);
    @memcpy(result.words, bc.words);

    for (rc.runs[0..rc.n_runs]) |run| {
        result.clearRange(run.start, run.end());
    }

    const card = result.computeCardinality();
    _ = card;
    return bitsetToArrayOrRun(allocator, result);
}

fn runDifferenceArray(allocator: std.mem.Allocator, rc: *RunContainer, ac: *ArrayContainer) !Container {
    // Convert run to bitset, remove array elements
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);

    for (rc.runs[0..rc.n_runs]) |run| {
        var v: u32 = run.start;
        while (v <= run.end()) : (v += 1) {
            _ = result.add(@intCast(v));
        }
    }

    for (ac.values[0..ac.cardinality]) |v| {
        _ = result.remove(v);
    }

    const card = result.computeCardinality();
    _ = card;
    return bitsetToArrayOrRun(allocator, result);
}

fn runDifferenceBitset(allocator: std.mem.Allocator, rc: *RunContainer, bc: *BitsetContainer) !Container {
    const result = try ArrayContainer.init(allocator, @intCast(@min(rc.getCardinality(), ArrayContainer.MAX_CARDINALITY)));
    errdefer result.deinit(allocator);

    var k: usize = 0;
    for (rc.runs[0..rc.n_runs], 0..) |run, run_idx| {
        var v: u32 = run.start;
        while (v <= run.end()) : (v += 1) {
            if (!bc.contains(@intCast(v))) {
                if (k >= ArrayContainer.MAX_CARDINALITY) {
                    // Need to convert to bitset
                    result.cardinality = @intCast(k);
                    const bs = try arrayToBitset(allocator, result);
                    result.deinit(allocator);
                    // Finish current run
                    while (v <= run.end()) : (v += 1) {
                        if (!bc.contains(@intCast(v))) {
                            _ = bs.add(@intCast(v));
                        }
                    }
                    // Process remaining runs
                    for (rc.runs[run_idx + 1 .. rc.n_runs]) |remaining_run| {
                        var rv: u32 = remaining_run.start;
                        while (rv <= remaining_run.end()) : (rv += 1) {
                            if (!bc.contains(@intCast(rv))) {
                                _ = bs.add(@intCast(rv));
                            }
                        }
                    }
                    _ = bs.computeCardinality();
                    return bitsetToArrayOrRun(allocator, bs);
                }
                result.values[k] = @intCast(v);
                k += 1;
            }
        }
    }
    result.cardinality = @intCast(k);
    return arrayToArrayOrRun(allocator, result);
}

fn runDifferenceRun(allocator: std.mem.Allocator, a: *RunContainer, b: *RunContainer) !Container {
    const result = try RunContainer.init(allocator, @intCast(@min(@as(usize, a.n_runs) + b.n_runs, 65535)));
    errdefer result.deinit(allocator);

    var j: usize = 0;
    for (a.runs[0..a.n_runs]) |run| {
        const run_end: u32 = run.end();
        var keep_start: u32 = run.start;

        while (j < b.n_runs and @as(u32, b.runs[j].end()) < keep_start) : (j += 1) {}

        var scan = j;
        while (scan < b.n_runs and @as(u32, b.runs[scan].start) <= run_end) : (scan += 1) {
            const blocker = b.runs[scan];
            const blocker_start: u32 = blocker.start;
            const blocker_end: u32 = blocker.end();

            if (blocker_start > keep_start) {
                _ = try result.addRange(allocator, @intCast(keep_start), @intCast(@min(blocker_start - 1, run_end)));
            }
            if (blocker_end >= run_end) {
                keep_start = run_end + 1;
                break;
            }
            keep_start = blocker_end + 1;
        }

        if (keep_start <= run_end) {
            _ = try result.addRange(allocator, @intCast(keep_start), run.end());
        }
    }

    if (result.n_runs == 0) {
        const empty = try ArrayContainer.init(allocator, 0);
        result.deinit(allocator);
        return .{ .array = empty };
    }
    result.cardinality = -1;
    return .{ .run = result };
}

// ============================================================================
// Symmetric Difference (XOR)
// ============================================================================

pub fn containerXor(allocator: std.mem.Allocator, a: Container, b: Container) !Container {
    return switch (a) {
        .array => |ac| switch (b) {
            .array => |bc| arrayXorArray(allocator, ac, bc),
            .bitset => |bc| arrayXorBitset(allocator, ac, bc),
            .run => |rc| arrayXorRun(allocator, ac, rc),
            .reserved => unreachable,
        },
        .bitset => |ac| switch (b) {
            .array => |bc| arrayXorBitset(allocator, bc, ac), // commutative
            .bitset => |bc| bitsetXorBitset(allocator, ac, bc),
            .run => |rc| bitsetXorRun(allocator, ac, rc),
            .reserved => unreachable,
        },
        .run => |ac| switch (b) {
            .array => |bc| arrayXorRun(allocator, bc, ac), // commutative
            .bitset => |bc| bitsetXorRun(allocator, bc, ac), // commutative
            .run => |rc| runXorRun(allocator, ac, rc),
            .reserved => unreachable,
        },
        .reserved => unreachable,
    };
}

/// Non-lazy XOR; repairs cardinality before demotion or return.
fn arrayXorArray(allocator: std.mem.Allocator, a: *ArrayContainer, b: *ArrayContainer) !Container {
    const max_card = @as(u32, a.cardinality) + b.cardinality;

    if (max_card > ArrayContainer.MAX_CARDINALITY) {
        // Use bitset
        const result = try BitsetContainer.init(allocator);
        errdefer result.deinit(allocator);
        result.setList(a.values[0..a.cardinality]);
        result.toggleList(b.values[0..b.cardinality]);
        const card = result.computeCardinality();
        if (card <= ArrayContainer.MAX_CARDINALITY) {
            const arr = try bitsetToArray(allocator, result);
            result.deinit(allocator);
            return .{ .array = arr };
        }
        return .{ .bitset = result };
    }

    // Merge with XOR logic
    const result = try ArrayContainer.init(allocator, @intCast(max_card));
    errdefer result.deinit(allocator);

    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;
    const sa = a.values[0..a.cardinality];
    const sb = b.values[0..b.cardinality];

    while (i < sa.len and j < sb.len) {
        if (sa[i] < sb[j]) {
            result.values[k] = sa[i];
            i += 1;
            k += 1;
        } else if (sa[i] > sb[j]) {
            result.values[k] = sb[j];
            j += 1;
            k += 1;
        } else {
            // Equal - skip both (XOR removes common elements)
            i += 1;
            j += 1;
        }
    }
    while (i < sa.len) : (i += 1) {
        result.values[k] = sa[i];
        k += 1;
    }
    while (j < sb.len) : (j += 1) {
        result.values[k] = sb[j];
        k += 1;
    }
    result.cardinality = @intCast(k);
    return .{ .array = result };
}

/// Non-lazy XOR; repairs cardinality before demotion or return.
fn arrayXorBitset(allocator: std.mem.Allocator, ac: *ArrayContainer, bc: *BitsetContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);
    @memcpy(result.words, bc.words);

    result.toggleList(ac.values[0..ac.cardinality]);
    const card = result.computeCardinality();
    if (card <= ArrayContainer.MAX_CARDINALITY) {
        const arr = try bitsetToArray(allocator, result);
        result.deinit(allocator);
        return .{ .array = arr };
    }
    return .{ .bitset = result };
}

/// Non-lazy XOR; repairs cardinality before representation selection.
fn arrayXorRun(allocator: std.mem.Allocator, ac: *ArrayContainer, rc: *RunContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);

    for (rc.runs[0..rc.n_runs]) |run| {
        result.setRange(run.start, run.end());
    }
    result.toggleList(ac.values[0..ac.cardinality]);
    _ = result.computeCardinality();
    return bitsetToArrayOrRun(allocator, result);
}

fn bitsetXorBitset(allocator: std.mem.Allocator, a: *BitsetContainer, b: *BitsetContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);
    @memcpy(result.words, a.words);
    result.symmetricDifferenceWith(b);

    const card = result.getCardinality();
    if (card <= ArrayContainer.MAX_CARDINALITY) {
        const arr = try bitsetToArray(allocator, result);
        result.deinit(allocator);
        return .{ .array = arr };
    }
    return .{ .bitset = result };
}

fn bitsetXorRun(allocator: std.mem.Allocator, bc: *BitsetContainer, rc: *RunContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);
    @memcpy(result.words, bc.words);

    for (rc.runs[0..rc.n_runs]) |run| {
        result.toggleRange(run.start, run.end());
    }

    const card = result.computeCardinality();
    _ = card;
    return bitsetToArrayOrRun(allocator, result);
}

fn runXorRun(allocator: std.mem.Allocator, a: *RunContainer, b: *RunContainer) !Container {
    const max_runs = @min(2 * (@as(usize, a.n_runs) + @as(usize, b.n_runs)), 65535);
    const result = try RunContainer.init(allocator, @intCast(max_runs));
    errdefer result.deinit(allocator);

    var a_boundary: usize = 0;
    var b_boundary: usize = 0;
    var in_a = false;
    var in_b = false;
    var prev: u32 = 0;

    while (a_boundary < @as(usize, a.n_runs) * 2 or b_boundary < @as(usize, b.n_runs) * 2) {
        const next_a = nextRunBoundary(a, a_boundary);
        const next_b = nextRunBoundary(b, b_boundary);
        const boundary = @min(next_a, next_b);

        if (boundary > prev and (in_a != in_b) and prev <= 65535) {
            _ = try result.addRange(allocator, @intCast(prev), @intCast(@min(boundary - 1, 65535)));
        }

        while (a_boundary < @as(usize, a.n_runs) * 2 and nextRunBoundary(a, a_boundary) == boundary) : (a_boundary += 1) {
            in_a = !in_a;
        }
        while (b_boundary < @as(usize, b.n_runs) * 2 and nextRunBoundary(b, b_boundary) == boundary) : (b_boundary += 1) {
            in_b = !in_b;
        }
        prev = boundary;
    }

    if (result.n_runs == 0) {
        const empty = try ArrayContainer.init(allocator, 0);
        result.deinit(allocator);
        return .{ .array = empty };
    }
    result.cardinality = -1;
    return .{ .run = result };
}

fn nextRunBoundary(rc: *RunContainer, boundary_idx: usize) u32 {
    if (boundary_idx >= @as(usize, rc.n_runs) * 2) return std.math.maxInt(u32);
    const run = rc.runs[boundary_idx / 2];
    if (boundary_idx % 2 == 0) {
        return run.start;
    }
    return @as(u32, run.end()) + 1;
}

// ============================================================================
// Container Type Conversions
// ============================================================================

pub fn bitsetToArray(allocator: std.mem.Allocator, bc: *BitsetContainer) !*ArrayContainer {
    const card = bc.getCardinality();
    const result = try ArrayContainer.init(allocator, @intCast(@min(card, ArrayContainer.MAX_CARDINALITY)));
    errdefer result.deinit(allocator);

    var k: usize = 0;
    for (bc.words, 0..) |word, word_idx| {
        var w = word;
        while (w != 0) {
            const bit = @ctz(w);
            result.values[k] = @intCast(word_idx * 64 + bit);
            k += 1;
            w &= w - 1; // clear lowest set bit
        }
    }
    result.cardinality = @intCast(k);
    return result;
}

/// Exact conversion from a unique array; assigns cardinality without rescanning.
pub fn arrayToBitset(allocator: std.mem.Allocator, ac: *ArrayContainer) !*BitsetContainer {
    const result = try BitsetContainer.init(allocator);
    result.setList(ac.values[0..ac.cardinality]);
    result.cardinality = @intCast(ac.cardinality);
    return result;
}

fn bitsetToArrayOrRun(allocator: std.mem.Allocator, bc: *BitsetContainer) !Container {
    const card = bc.getCardinality();
    if (card > ArrayContainer.MAX_CARDINALITY) {
        const run_count = countRunsInBitset(bc);
        if (run_count * 4 < BitsetContainer.SIZE_BYTES) {
            const rc = try bitsetToRun(allocator, bc, run_count);
            bc.deinit(allocator);
            return .{ .run = rc };
        }
        return .{ .bitset = bc };
    }

    const arr = try bitsetToArray(allocator, bc);
    errdefer arr.deinit(allocator);
    const container = try arrayToArrayOrRun(allocator, arr);
    bc.deinit(allocator);
    return container;
}

fn arrayToArrayOrRun(allocator: std.mem.Allocator, arr: *ArrayContainer) !Container {
    const run_count = countRunsInArray(arr);
    if (run_count * 4 < @as(u32, arr.cardinality) * 2) {
        const rc = try arrayToRun(allocator, arr, run_count);
        arr.deinit(allocator);
        return .{ .run = rc };
    }

    return .{ .array = arr };
}

fn countRunsInArray(ac: *ArrayContainer) u32 {
    if (ac.cardinality == 0) return 0;

    var count: u32 = 1;
    var previous = ac.values[0];
    for (ac.values[1..ac.cardinality]) |value| {
        if (value != previous + 1) {
            count += 1;
        }
        previous = value;
    }
    return count;
}

fn countRunsInBitset(bc: *BitsetContainer) u32 {
    var count: u32 = 0;
    var previous_high_bit: u64 = 0;

    for (bc.words) |word| {
        const previous_bits = (word << 1) | previous_high_bit;
        const run_starts = word & ~previous_bits;
        count += @popCount(run_starts);
        previous_high_bit = word >> 63;
    }

    return count;
}

fn arrayToRun(allocator: std.mem.Allocator, ac: *ArrayContainer, run_count: u32) !*RunContainer {
    const rc = try RunContainer.init(allocator, @intCast(run_count));
    errdefer rc.deinit(allocator);

    if (ac.cardinality == 0) {
        return rc;
    }

    var run_idx: usize = 0;
    var run_start = ac.values[0];
    var run_len: u16 = 0;

    for (ac.values[1..ac.cardinality]) |value| {
        if (value == run_start + run_len + 1) {
            run_len += 1;
        } else {
            rc.runs[run_idx] = .{ .start = run_start, .length = run_len };
            run_idx += 1;
            run_start = value;
            run_len = 0;
        }
    }

    rc.runs[run_idx] = .{ .start = run_start, .length = run_len };
    rc.n_runs = @intCast(run_count);
    rc.cardinality = -1;
    return rc;
}

fn bitsetToRun(allocator: std.mem.Allocator, bc: *BitsetContainer, run_count: u32) !*RunContainer {
    const rc = try RunContainer.init(allocator, @intCast(run_count));
    errdefer rc.deinit(allocator);

    var run_idx: u16 = 0;
    var in_run = false;
    var run_start: u16 = 0;

    for (bc.words, 0..) |word, word_idx| {
        const base: u16 = @intCast(word_idx * 64);
        var bits = word;
        var bit_idx: u6 = 0;

        while (bits != 0 or in_run) {
            const bit: u1 = @truncate(bits);
            const pos = base + bit_idx;

            if (bit == 1 and !in_run) {
                run_start = pos;
                in_run = true;
            } else if (bit == 0 and in_run) {
                rc.runs[run_idx] = .{ .start = run_start, .length = pos - run_start - 1 };
                run_idx += 1;
                in_run = false;
            }

            if (bit_idx == 63) break;
            bits >>= 1;
            bit_idx += 1;
        }
    }

    if (in_run) {
        rc.runs[run_idx] = .{ .start = run_start, .length = 65535 - run_start };
        run_idx += 1;
    }

    rc.n_runs = run_idx;
    rc.cardinality = -1;
    return rc;
}

// ============================================================================
// Tests
// ============================================================================

test "array union array" {
    const allocator = std.testing.allocator;

    const a = try ArrayContainer.init(allocator, 0);
    defer a.deinit(allocator);
    _ = try a.add(allocator, 1);
    _ = try a.add(allocator, 2);
    _ = try a.add(allocator, 3);

    const b = try ArrayContainer.init(allocator, 0);
    defer b.deinit(allocator);
    _ = try b.add(allocator, 3);
    _ = try b.add(allocator, 4);
    _ = try b.add(allocator, 5);

    const result = try containerUnion(allocator, .{ .array = a }, .{ .array = b });
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 5), result.getCardinality());
    try std.testing.expect(result.contains(1));
    try std.testing.expect(result.contains(2));
    try std.testing.expect(result.contains(3));
    try std.testing.expect(result.contains(4));
    try std.testing.expect(result.contains(5));
}

test "array intersect array" {
    const allocator = std.testing.allocator;

    const a = try ArrayContainer.init(allocator, 0);
    defer a.deinit(allocator);
    _ = try a.add(allocator, 1);
    _ = try a.add(allocator, 2);
    _ = try a.add(allocator, 3);

    const b = try ArrayContainer.init(allocator, 0);
    defer b.deinit(allocator);
    _ = try b.add(allocator, 2);
    _ = try b.add(allocator, 3);
    _ = try b.add(allocator, 4);

    const result = try containerIntersection(allocator, .{ .array = a }, .{ .array = b });
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 2), result.getCardinality());
    try std.testing.expect(result.contains(2));
    try std.testing.expect(result.contains(3));
}

test "array difference array" {
    const allocator = std.testing.allocator;

    const a = try ArrayContainer.init(allocator, 0);
    defer a.deinit(allocator);
    _ = try a.add(allocator, 1);
    _ = try a.add(allocator, 2);
    _ = try a.add(allocator, 3);

    const b = try ArrayContainer.init(allocator, 0);
    defer b.deinit(allocator);
    _ = try b.add(allocator, 2);
    _ = try b.add(allocator, 3);
    _ = try b.add(allocator, 4);

    const result = try containerDifference(allocator, .{ .array = a }, .{ .array = b });
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 1), result.getCardinality());
    try std.testing.expect(result.contains(1));
}

test "array xor array" {
    const allocator = std.testing.allocator;

    const a = try ArrayContainer.init(allocator, 0);
    defer a.deinit(allocator);
    _ = try a.add(allocator, 1);
    _ = try a.add(allocator, 2);
    _ = try a.add(allocator, 3);

    const b = try ArrayContainer.init(allocator, 0);
    defer b.deinit(allocator);
    _ = try b.add(allocator, 2);
    _ = try b.add(allocator, 3);
    _ = try b.add(allocator, 4);

    const result = try containerXor(allocator, .{ .array = a }, .{ .array = b });
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 2), result.getCardinality());
    try std.testing.expect(result.contains(1));
    try std.testing.expect(result.contains(4));
}

test "bitset union bitset" {
    const allocator = std.testing.allocator;

    const a = try BitsetContainer.init(allocator);
    defer a.deinit(allocator);
    _ = a.add(100);
    _ = a.add(200);

    const b = try BitsetContainer.init(allocator);
    defer b.deinit(allocator);
    _ = b.add(200);
    _ = b.add(300);

    const result = try containerUnion(allocator, .{ .bitset = a }, .{ .bitset = b });
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 3), result.getCardinality());
    try std.testing.expect(result.contains(100));
    try std.testing.expect(result.contains(200));
    try std.testing.expect(result.contains(300));
}

test "bitset to array conversion on small intersection" {
    const allocator = std.testing.allocator;

    const a = try BitsetContainer.init(allocator);
    defer a.deinit(allocator);
    _ = a.add(1);
    _ = a.add(2);
    _ = a.add(3);

    const b = try BitsetContainer.init(allocator);
    defer b.deinit(allocator);
    _ = b.add(2);
    _ = b.add(3);
    _ = b.add(4);

    const result = try containerIntersection(allocator, .{ .bitset = a }, .{ .bitset = b });
    defer result.deinit(allocator);

    // Result should be array since cardinality is small
    try std.testing.expectEqual(Container.array, std.meta.activeTag(result));
    try std.testing.expectEqual(@as(u32, 2), result.getCardinality());
}

test "bitsetToArray with full word (regression: u6 overflow)" {
    const allocator = std.testing.allocator;

    const bc = try BitsetContainer.init(allocator);
    defer bc.deinit(allocator);

    // Set all 64 bits in word 0 (values 0-63)
    bc.words[0] = 0xFFFFFFFFFFFFFFFF;
    bc.cardinality = 64;

    const ac = try bitsetToArray(allocator, bc);
    defer ac.deinit(allocator);

    try std.testing.expectEqual(@as(u16, 64), ac.cardinality);

    // Verify all values present and in order
    for (0..64) |i| {
        try std.testing.expectEqual(@as(u16, @intCast(i)), ac.values[i]);
    }
}

test "galloping: skewed array intersection" {
    const allocator = std.testing.allocator;

    // Big array: 0, 1, 2, ..., 3999 (4000 elements)
    const big = try ArrayContainer.init(allocator, 4000);
    defer big.deinit(allocator);
    for (0..4000) |i| {
        big.values[i] = @intCast(i);
    }
    big.cardinality = 4000;

    // Small array: 100, 500, 999, 2000, 5000 (5 elements, one outside big's range)
    const small = try ArrayContainer.init(allocator, 5);
    defer small.deinit(allocator);
    small.values[0] = 100;
    small.values[1] = 500;
    small.values[2] = 999;
    small.values[3] = 2000;
    small.values[4] = 5000; // not in big
    small.cardinality = 5;

    const result = try arrayIntersectArray(allocator, small, big);
    defer result.array.deinit(allocator);

    // Should find 4 matches (100, 500, 999, 2000), not 5000
    try std.testing.expectEqual(@as(u16, 4), result.array.cardinality);
    try std.testing.expectEqual(@as(u16, 100), result.array.values[0]);
    try std.testing.expectEqual(@as(u16, 500), result.array.values[1]);
    try std.testing.expectEqual(@as(u16, 999), result.array.values[2]);
    try std.testing.expectEqual(@as(u16, 2000), result.array.values[3]);
}
