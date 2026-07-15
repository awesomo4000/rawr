<!-- SPDX-License-Identifier: MPL-2.0 -->

# Final Rawr Features: Galloping, andCardinality, intersects, xorInPlace

Four additions to call rawr complete for Datalog evaluation use.

## Dependency order

```
1. Galloping intersection      (arrayIntersectArray gets faster)
2. andCardinality + intersects (new functions, array path reuses galloping)
3. bitwiseXorInPlace           (trivial, independent)
```

Galloping first because the andCardinality array-array path should use it too.

---

## 1. Galloping intersection (src/container_ops.zig)

When two sorted arrays have very different sizes, galloping (exponential search)
beats linear merge. Linear merge is O(n + m). Galloping is O(small × log(big)).
For 50 elements vs 4000 elements: 4050 comparisons → ~300.

The standard approach: walk the smaller array, for each element do an exponential
search into the larger array. Falls back to linear-merge behavior when arrays
are similarly sized (gallop step of 1 degrades to linear scan).

### Changes to arrayIntersectArray (~line 287)

Replace the inner loop. The function signature and output logic stay identical.

```zig
fn arrayIntersectArray(allocator: std.mem.Allocator, a: *ArrayContainer, b: *ArrayContainer) !Container {
    const result = try ArrayContainer.init(allocator, @min(a.cardinality, b.cardinality));
    errdefer result.deinit(allocator);

    // Always walk the smaller array, gallop into the larger
    const small = if (a.cardinality <= b.cardinality)
        a.values[0..a.cardinality] else b.values[0..b.cardinality];
    const big = if (a.cardinality <= b.cardinality)
        b.values[0..b.cardinality] else a.values[0..a.cardinality];

    var k: usize = 0;
    var lo: usize = 0; // search start in big, advances monotonically

    for (small) |val| {
        // Gallop: find val in big[lo..] using exponential search
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
```

### gallopSearch helper (same file, top of file or inline)

```zig
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
    var lo = if (step > 2) hi - step / 2 else start;
    // Make sure lo is valid
    if (lo < start) lo = start;

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
```

### Also update arrayUnionArray

Same pattern applies. Walk smaller array galloping into larger, outputting
non-matched elements from both sides. This is slightly more involved because
union outputs all elements from both arrays. The current branchless merge is
fine for union — galloping helps less there because you must emit every element
anyway. **Skip for now**, only do intersection.

### Tests

Existing tests cover this (arrayIntersectArray is called by bitwiseAnd).
Add one targeted test for the skewed case:

```zig
test "galloping: skewed intersection" {
    // 10 elements vs 4000 elements, ~5 matches
    var small = try RoaringBitmap.init(allocator);
    defer small.deinit();
    var big = try RoaringBitmap.init(allocator);
    defer big.deinit();

    // Big: 0, 1, 2, ..., 3999 (all in one chunk)
    _ = try big.addRange(0, 3999);

    // Small: 100, 500, 999, 2000, 5000 (5000 is in different chunk)
    for ([_]u32{ 100, 500, 999, 2000, 5000 }) |v| {
        _ = try small.add(v);
    }

    var result = try small.bitwiseAnd(allocator, &big);
    defer result.deinit();
    try std.testing.expectEqual(@as(u64, 4), result.cardinality());
    try std.testing.expect(result.contains(100));
    try std.testing.expect(result.contains(2000));
    try std.testing.expect(!result.contains(5000));
}
```

Also run `zig build validate` to confirm CRoaring interop still passes.

---

## 2. andCardinality + intersects (src/container_ops.zig + src/bitmap.zig)

### Why these exist

`andCardinality` computes |A ∩ B| without allocating a result bitmap. The
Datalog evaluator needs this for join ordering — estimate selectivity of
"relation_a ∩ relation_b" without materializing the intersection.

`intersects` answers "is A ∩ B non-empty?" with early exit on first match.
Useful for quick predicate checks: "does any member of this group have admin?"

Both avoid all output allocation. Per-container cardinality IS stored already
(array.cardinality, bitset.cardinality, run knows its total), but that's
the cardinality of each container in isolation. The intersection cardinality
requires looking at both containers' data.

### container_ops.zig — new functions

```zig
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
    // Same dispatch structure, each impl returns on first match.
    // ... same 9-way dispatch ...
}
```

### The 6 cardinality implementations

These mirror the existing intersection functions but count instead of building.

**arrayIntersectArrayCard** — same merge walk (use galloping from step 1),
increment counter instead of writing to output:

```zig
fn arrayIntersectArrayCard(a: *ArrayContainer, b: *ArrayContainer) u64 {
    const small = if (a.cardinality <= b.cardinality)
        a.values[0..a.cardinality] else b.values[0..b.cardinality];
    const big = if (a.cardinality <= b.cardinality)
        b.values[0..b.cardinality] else a.values[0..a.cardinality];

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
```

**arrayIntersectBitsetCard** — check each array element in bitset:

```zig
fn arrayIntersectBitsetCard(ac: *ArrayContainer, bc: *BitsetContainer) u64 {
    var count: u64 = 0;
    for (ac.values[0..ac.cardinality]) |v| {
        if (bc.contains(v)) count += 1;
    }
    return count;
}
```

**bitsetIntersectBitsetCard** — SIMD popcount, no output allocation. This is
the big win — the current code allocates an 8KB result bitset just to count:

```zig
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
```

**arrayIntersectRunCard** — check each array element against runs:

```zig
fn arrayIntersectRunCard(ac: *ArrayContainer, rc: *RunContainer) u64 {
    var count: u64 = 0;
    for (ac.values[0..ac.cardinality]) |v| {
        if (rc.contains(v)) count += 1;
    }
    return count;
}
```

**bitsetIntersectRunCard** — walk runs, check bitset:

```zig
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
```

**runIntersectRunCard** — same overlap walk as runIntersectRun, sum overlap
lengths instead of building runs:

```zig
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
```

### intersects implementations

Same 6 functions but return `bool` and early-exit on first match. Most are
trivial rewrites — replace `count += 1` with `return true`, add `return false`
at end. `bitsetIntersectBitsetIntersects` can early-exit per word:

```zig
fn bitsetIntersectBitsetIntersects(a: *BitsetContainer, b: *BitsetContainer) bool {
    for (a.words[0..1024], b.words[0..1024]) |wa, wb| {
        if (wa & wb != 0) return true;
    }
    return false;
}
```

### bitmap.zig — public API

Add to RoaringBitmap:

```zig
/// Compute |self ∩ other| without allocating a result bitmap.
/// Useful for join selectivity estimation in query planning.
pub fn andCardinality(self: *const Self, other: *const Self) u64 {
    var total: u64 = 0;
    var i: usize = 0;
    var j: usize = 0;
    while (i < self.size and j < other.size) {
        if (self.keys[i] < other.keys[j]) {
            i += 1;
        } else if (self.keys[i] > other.keys[j]) {
            j += 1;
        } else {
            total += container_ops.containerIntersectionCardinality(
                Container.fromTagged(self.containers[i]),
                Container.fromTagged(other.containers[j]),
            );
            i += 1;
            j += 1;
        }
    }
    return total;
}

/// Return true if self and other have any values in common.
/// Early-exit: stops at the first match. Much cheaper than andCardinality() > 0
/// for sparse intersections.
pub fn intersects(self: *const Self, other: *const Self) bool {
    var i: usize = 0;
    var j: usize = 0;
    while (i < self.size and j < other.size) {
        if (self.keys[i] < other.keys[j]) {
            i += 1;
        } else if (self.keys[i] > other.keys[j]) {
            j += 1;
        } else {
            if (container_ops.containerIntersects(
                Container.fromTagged(self.containers[i]),
                Container.fromTagged(other.containers[j]),
            )) return true;
            i += 1;
            j += 1;
        }
    }
    return false;
}
```

### Tests

```zig
test "andCardinality matches bitwiseAnd().cardinality()" {
    // Build two overlapping bitmaps
    // ... add values ...
    const card_fast = a.andCardinality(&b);
    var intersection = try a.bitwiseAnd(allocator, &b);
    defer intersection.deinit();
    try std.testing.expectEqual(card_fast, intersection.cardinality());
}

test "intersects" {
    // Non-overlapping
    try std.testing.expect(!a.intersects(&b));
    // Add overlap
    _ = try b.add(some_value_in_a);
    try std.testing.expect(a.intersects(&b));
}
```

Also add to CRoaring validation if CRoaring exposes andCardinality (it does:
`roaring_bitmap_and_cardinality`).

---

## 3. bitwiseXorInPlace (src/bitmap.zig + src/container_ops.zig)

Follows the exact same pattern as the existing `bitwiseOrInPlace`,
`bitwiseAndInPlace`, and `bitwiseDifferenceInPlace`. The 9 container-pair
cases mirror `bitwiseXor` but modify self in-place.

### bitmap.zig

```zig
pub fn bitwiseXorInPlace(self: *Self, other: *const Self) !void {
    // Same structure as bitwiseOrInPlace / bitwiseAndInPlace.
    // Walk both key arrays, for matching keys: xor containers in-place.
    // For keys only in other: clone and insert.
    // For keys only in self: keep as-is.
    // ... (follow bitwiseOrInPlace pattern exactly) ...
}
```

Look at the existing `bitwiseOrInPlace` and copy its structure — it handles
the key merge, container replacement, and growth logic. XOR is symmetric
difference so the key-merge logic is identical to OR (keys present in either
side appear in output).

### Tests

```zig
test "bitwiseXorInPlace matches bitwiseXor" {
    var a_copy = try a.clone(allocator);
    defer a_copy.deinit();
    try a_copy.bitwiseXorInPlace(&b);

    var expected = try a.bitwiseXor(allocator, &b);
    defer expected.deinit();

    try std.testing.expect(a_copy.equals(&expected));
}
```

---

## Summary

| Feature | Lines (est.) | Depends on | Why |
|---------|-------------|------------|-----|
| Galloping | ~30 | Nothing | Skewed intersection O(small × log big) |
| andCardinality | ~100 | Galloping (for array path) | Join selectivity estimation |
| intersects | ~80 | Nothing (but shares structure) | Quick predicate checks |
| bitwiseXorInPlace | ~80 | Nothing | API completeness |

Total: ~290 lines. After this, rawr has everything needed for Datalog evaluation.
