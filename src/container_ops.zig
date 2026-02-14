const std = @import("std");
const ArrayContainer = @import("array_container.zig").ArrayContainer;
const BitsetContainer = @import("bitset_container.zig").BitsetContainer;
const RunContainer = @import("run_container.zig").RunContainer;
const container_mod = @import("container.zig");
const Container = container_mod.Container;

/// Cross-container operations: all 9 pairwise combinations for each set operation.
/// Returns newly allocated containers.

// ============================================================================
// Helpers
// ============================================================================

/// Exponential search for `target` in sorted `arr[start..]`.
/// Returns the index of the first element >= target.
/// O(log(distance_to_target)) — fast when target is nearby, degrades
/// gracefully to O(log n) when target is far.
fn gallopSearch(arr: []const u16, target: u16, start: usize) usize {
    if (start >= arr.len) return arr.len;

    // Phase 1: exponential gallop to find bracket
    var step: usize = 1;
    var hi = start;
    while (hi < arr.len and arr[hi] < target) {
        hi += step;
        step *= 2;
    }
    // Clamp hi
    if (hi > arr.len) hi = arr.len;

    // Phase 2: binary search within [lo, hi)
    var lo = if (step > 2) hi -| (step / 2) else start;

    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (arr[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
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
            .run => |rc| bitsetUnionRunInPlace(ac, rc),
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

fn arrayUnionBitsetInPlace(allocator: std.mem.Allocator, ac: *ArrayContainer, bc: *BitsetContainer) !Container {
    // Result is always a bitset - must allocate new one and copy
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);
    @memcpy(result.words, bc.words);
    for (ac.values[0..ac.cardinality]) |v| {
        _ = result.add(v);
    }
    _ = result.computeCardinality();
    return .{ .bitset = result };
}

fn bitsetUnionArrayInPlace(bc: *BitsetContainer, ac: *ArrayContainer) Container {
    // Add array elements to existing bitset - no allocation
    // bc.add() already maintains cardinality correctly
    for (ac.values[0..ac.cardinality]) |v| {
        _ = bc.add(v);
    }
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

fn arrayUnionArray(allocator: std.mem.Allocator, a: *ArrayContainer, b: *ArrayContainer) !Container {
    const max_card = @as(u32, a.cardinality) + b.cardinality;

    // If combined could exceed array threshold, use bitset
    if (max_card > ArrayContainer.MAX_CARDINALITY) {
        const bc = try BitsetContainer.init(allocator);
        errdefer bc.deinit(allocator);
        for (a.values[0..a.cardinality]) |v| _ = bc.add(v);
        for (b.values[0..b.cardinality]) |v| _ = bc.add(v);
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
    return .{ .array = result };
}

fn arrayUnionBitset(allocator: std.mem.Allocator, ac: *ArrayContainer, bc: *BitsetContainer) !Container {
    // Result is always a bitset (bitset cardinality >= array threshold)
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);

    // Copy bitset
    @memcpy(result.words, bc.words);

    // Add array elements
    for (ac.values[0..ac.cardinality]) |v| {
        _ = result.add(v);
    }
    _ = result.computeCardinality();
    return .{ .bitset = result };
}

fn arrayUnionRun(allocator: std.mem.Allocator, ac: *ArrayContainer, rc: *RunContainer) !Container {
    // Convert both to bitset for simplicity, then optimize
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);

    // Add run elements
    for (rc.runs[0..rc.n_runs]) |run| {
        var v: u32 = run.start;
        while (v <= run.end()) : (v += 1) {
            _ = result.add(@intCast(v));
        }
    }

    // Add array elements
    for (ac.values[0..ac.cardinality]) |v| {
        _ = result.add(v);
    }

    const card = result.computeCardinality();
    if (card <= ArrayContainer.MAX_CARDINALITY) {
        // Convert to array
        const arr = try bitsetToArray(allocator, result);
        result.deinit(allocator);
        return .{ .array = arr };
    }
    return .{ .bitset = result };
}

fn bitsetUnionBitset(allocator: std.mem.Allocator, a: *BitsetContainer, b: *BitsetContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    @memcpy(result.words, a.words);
    result.unionWith(b);
    return .{ .bitset = result };
}

fn bitsetUnionRun(allocator: std.mem.Allocator, bc: *BitsetContainer, rc: *RunContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    @memcpy(result.words, bc.words);

    // Add run elements
    for (rc.runs[0..rc.n_runs]) |run| {
        var v: u32 = run.start;
        while (v <= run.end()) : (v += 1) {
            _ = result.add(@intCast(v));
        }
    }
    _ = result.computeCardinality();
    return .{ .bitset = result };
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

    // Walk the smaller array, gallop into the larger.
    // O(small × log big) — much faster than O(n+m) when sizes differ significantly.
    const small = if (a.cardinality <= b.cardinality)
        a.values[0..a.cardinality]
    else
        b.values[0..b.cardinality];
    const big = if (a.cardinality <= b.cardinality)
        b.values[0..b.cardinality]
    else
        a.values[0..a.cardinality];

    var k: usize = 0;
    var lo: usize = 0; // search start in big, advances monotonically

    for (small) |val| {
        lo = gallopSearch(big, val, lo);
        if (lo < big.len and big[lo] == val) {
            result.values[k] = val;
            k += 1;
            lo += 1; // past this match for next search
        }
    }

    result.cardinality = @intCast(k);
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
    // Result is at most the run's cardinality
    const result = try ArrayContainer.init(allocator, @intCast(@min(rc.getCardinality(), ArrayContainer.MAX_CARDINALITY)));
    errdefer result.deinit(allocator);

    var k: usize = 0;
    for (rc.runs[0..rc.n_runs]) |run| {
        var v: u32 = run.start;
        while (v <= run.end()) : (v += 1) {
            if (bc.contains(@intCast(v))) {
                result.values[k] = @intCast(v);
                k += 1;
            }
        }
    }
    result.cardinality = @intCast(k);

    // If too large for array, convert to bitset
    if (result.cardinality > ArrayContainer.MAX_CARDINALITY) {
        const bs = try arrayToBitset(allocator, result);
        result.deinit(allocator);
        return .{ .bitset = bs };
    }
    return .{ .array = result };
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
    const small = if (a.cardinality <= b.cardinality)
        a.values[0..a.cardinality]
    else
        b.values[0..b.cardinality];
    const big = if (a.cardinality <= b.cardinality)
        b.values[0..b.cardinality]
    else
        a.values[0..a.cardinality];

    var count: u64 = 0;
    var lo: usize = 0;
    for (small) |val| {
        lo = gallopSearch(big, val, lo);
        if (lo < big.len and big[lo] == val) {
            count += 1;
            lo += 1;
        }
    }
    return count;
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
    var card: u64 = 0;
    for (0..vec_count) |i| {
        const base = i * VEC_SIZE;
        const va: @Vector(VEC_SIZE, u64) = a.words[base..][0..VEC_SIZE].*;
        const vb: @Vector(VEC_SIZE, u64) = b.words[base..][0..VEC_SIZE].*;
        const result = va & vb;
        inline for (0..VEC_SIZE) |j| {
            card += @popCount(result[j]);
        }
    }
    return card;
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
    const small = if (a.cardinality <= b.cardinality)
        a.values[0..a.cardinality]
    else
        b.values[0..b.cardinality];
    const big = if (a.cardinality <= b.cardinality)
        b.values[0..b.cardinality]
    else
        a.values[0..a.cardinality];

    var lo: usize = 0;
    for (small) |val| {
        lo = gallopSearch(big, val, lo);
        if (lo < big.len and big[lo] == val) {
            return true;
        }
    }
    return false;
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
    return .{ .array = result };
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
        var v: u32 = run.start;
        while (v <= run.end()) : (v += 1) {
            _ = result.remove(@intCast(v));
        }
    }

    const card = result.computeCardinality();
    if (card <= ArrayContainer.MAX_CARDINALITY) {
        const arr = try bitsetToArray(allocator, result);
        result.deinit(allocator);
        return .{ .array = arr };
    }
    return .{ .bitset = result };
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
    if (card <= ArrayContainer.MAX_CARDINALITY) {
        const arr = try bitsetToArray(allocator, result);
        result.deinit(allocator);
        return .{ .array = arr };
    }
    return .{ .bitset = result };
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
                    return .{ .bitset = bs };
                }
                result.values[k] = @intCast(v);
                k += 1;
            }
        }
    }
    result.cardinality = @intCast(k);
    return .{ .array = result };
}

fn runDifferenceRun(allocator: std.mem.Allocator, a: *RunContainer, b: *RunContainer) !Container {
    const result = try ArrayContainer.init(allocator, @intCast(@min(a.getCardinality(), ArrayContainer.MAX_CARDINALITY)));
    errdefer result.deinit(allocator);

    var k: usize = 0;
    for (a.runs[0..a.n_runs], 0..) |run, run_idx| {
        var v: u32 = run.start;
        while (v <= run.end()) : (v += 1) {
            if (!b.contains(@intCast(v))) {
                if (k >= ArrayContainer.MAX_CARDINALITY) {
                    // Convert to bitset and continue
                    const bs = try arrayToBitset(allocator, result);
                    result.deinit(allocator);
                    // Finish current run
                    while (v <= run.end()) : (v += 1) {
                        if (!b.contains(@intCast(v))) {
                            _ = bs.add(@intCast(v));
                        }
                    }
                    // Process remaining runs
                    for (a.runs[run_idx + 1 .. a.n_runs]) |remaining_run| {
                        var rv: u32 = remaining_run.start;
                        while (rv <= remaining_run.end()) : (rv += 1) {
                            if (!b.contains(@intCast(rv))) {
                                _ = bs.add(@intCast(rv));
                            }
                        }
                    }
                    _ = bs.computeCardinality();
                    return .{ .bitset = bs };
                }
                result.values[k] = @intCast(v);
                k += 1;
            }
        }
    }
    result.cardinality = @intCast(k);
    return .{ .array = result };
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

fn arrayXorArray(allocator: std.mem.Allocator, a: *ArrayContainer, b: *ArrayContainer) !Container {
    const max_card = @as(u32, a.cardinality) + b.cardinality;

    if (max_card > ArrayContainer.MAX_CARDINALITY) {
        // Use bitset
        const result = try BitsetContainer.init(allocator);
        errdefer result.deinit(allocator);
        for (a.values[0..a.cardinality]) |v| _ = result.add(v);
        for (b.values[0..b.cardinality]) |v| {
            if (result.contains(v)) {
                _ = result.remove(v);
            } else {
                _ = result.add(v);
            }
        }
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

fn arrayXorBitset(allocator: std.mem.Allocator, ac: *ArrayContainer, bc: *BitsetContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);
    @memcpy(result.words, bc.words);

    for (ac.values[0..ac.cardinality]) |v| {
        if (result.contains(v)) {
            _ = result.remove(v);
        } else {
            _ = result.add(v);
        }
    }

    const card = result.computeCardinality();
    if (card <= ArrayContainer.MAX_CARDINALITY) {
        const arr = try bitsetToArray(allocator, result);
        result.deinit(allocator);
        return .{ .array = arr };
    }
    return .{ .bitset = result };
}

fn arrayXorRun(allocator: std.mem.Allocator, ac: *ArrayContainer, rc: *RunContainer) !Container {
    // Convert to bitset and XOR
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);

    for (rc.runs[0..rc.n_runs]) |run| {
        var v: u32 = run.start;
        while (v <= run.end()) : (v += 1) {
            _ = result.add(@intCast(v));
        }
    }

    for (ac.values[0..ac.cardinality]) |v| {
        if (result.contains(v)) {
            _ = result.remove(v);
        } else {
            _ = result.add(v);
        }
    }

    const card = result.computeCardinality();
    if (card <= ArrayContainer.MAX_CARDINALITY) {
        const arr = try bitsetToArray(allocator, result);
        result.deinit(allocator);
        return .{ .array = arr };
    }
    return .{ .bitset = result };
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
        var v: u32 = run.start;
        while (v <= run.end()) : (v += 1) {
            const val: u16 = @intCast(v);
            if (result.contains(val)) {
                _ = result.remove(val);
            } else {
                _ = result.add(val);
            }
        }
    }

    const card = result.computeCardinality();
    if (card <= ArrayContainer.MAX_CARDINALITY) {
        const arr = try bitsetToArray(allocator, result);
        result.deinit(allocator);
        return .{ .array = arr };
    }
    return .{ .bitset = result };
}

fn runXorRun(allocator: std.mem.Allocator, a: *RunContainer, b: *RunContainer) !Container {
    const result = try BitsetContainer.init(allocator);
    errdefer result.deinit(allocator);

    for (a.runs[0..a.n_runs]) |run| {
        var v: u32 = run.start;
        while (v <= run.end()) : (v += 1) {
            _ = result.add(@intCast(v));
        }
    }

    for (b.runs[0..b.n_runs]) |run| {
        var v: u32 = run.start;
        while (v <= run.end()) : (v += 1) {
            const val: u16 = @intCast(v);
            if (result.contains(val)) {
                _ = result.remove(val);
            } else {
                _ = result.add(val);
            }
        }
    }

    const card = result.computeCardinality();
    if (card <= ArrayContainer.MAX_CARDINALITY) {
        const arr = try bitsetToArray(allocator, result);
        result.deinit(allocator);
        return .{ .array = arr };
    }
    return .{ .bitset = result };
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

pub fn arrayToBitset(allocator: std.mem.Allocator, ac: *ArrayContainer) !*BitsetContainer {
    const result = try BitsetContainer.init(allocator);
    for (ac.values[0..ac.cardinality]) |v| {
        _ = result.add(v);
    }
    return result;
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
