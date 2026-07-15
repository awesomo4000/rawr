<!-- SPDX-License-Identifier: MPL-2.0 -->

# Optimization: Fast addRange

**Applies to:** `src/bitmap.zig` (`addRange`, `addRangeToChunk`, `addRangeToContainer`)
**Current:** 10x slower than CRoaring (both measure as sub-millisecond for 1M, ratio is noisy but the implementation is clearly suboptimal)
**Depends on:** Nothing. Standalone change.

## Problem

Three issues in the current addRange implementation:

### 1. New containers never use run encoding (lines 213-231)

When creating a new container for a contiguous range, the code chooses between
array and bitset based on cardinality. But a contiguous range is the *ideal*
case for a run container: one `RunPair` = 4 bytes, vs up to 8KB for bitset
or 8KB for a full array.

Current:
```zig
if (range_size > ArrayContainer.MAX_CARDINALITY) {
    // Use bitset for large ranges — allocates 8KB, sets bits one region at a time
    const bc = try BitsetContainer.init(self.allocator);
    bc.setRange(start, end);
    ...
} else {
    // Use array for small ranges — fills values in a loop
    const ac = try ArrayContainer.init(self.allocator, @intCast(range_size));
    var v: u32 = start;
    while (v <= end) : (v += 1) { ac.values[i] = @intCast(v); i += 1; }
    ...
}
```

Should: create a `RunContainer` with a single run pair `{start, length}`.
One allocation (4 bytes of payload), O(1).

### 2. Existing array containers: per-element add (lines 260-268)

```zig
// Add values one by one (could optimize with sorted merge)
var v: u32 = start;
while (v <= end) : (v += 1) {
    if (try ac.add(self.allocator, @intCast(v))) {  // binary search + shift!
        added += 1;
    }
}
```

Each `ac.add()` does a binary search to find insertion point + memmove to
shift elements right. For a range of N values into an array of M existing
elements, this is O(N × M) — effectively quadratic.

### 3. Existing run containers: per-element add (lines 271-279)

```zig
var v: u32 = start;
while (v <= end) : (v += 1) {
    if (try rc.add(self.allocator, @intCast(v))) { added += 1; }
}
```

A contiguous range should merge with existing runs in O(R) where R is the
number of overlapping runs. Instead it does N individual insertions.

## Fix

### New container: always use RunContainer (addRangeToChunk, ~line 212)

Replace the array/bitset choice with:

```zig
// A contiguous range is always best as a run container (4 bytes per run pair)
const rc = try RunContainer.init(self.allocator, 1);
rc.runs[0] = .{ .start = start, .length = @intCast(end - start) };
rc.n_runs = 1;
self.keys[insert_idx] = key;
self.containers[insert_idx] = TaggedPtr.initRun(rc);
```

One allocation, one assignment. After ingest, `runOptimize` can convert to
array or bitset if the container gets mixed (non-contiguous) data later.

### Existing array container: bulk insert (addRangeToContainer, ~line 248)

Two options depending on whether the range overlaps existing values:

**If array is empty or range is entirely after all existing values** (common
during sequential ingest — values arrive in order):

```zig
// Fast path: append range to end
try ac.ensureCapacity(self.allocator, ac.cardinality + range_size);
var v: u16 = start;
for (ac.values[ac.cardinality..][0..range_size]) |*slot| {
    slot.* = v;
    v += 1;
}
ac.cardinality += @intCast(range_size);
```

**General case: sorted merge.** Build the range as a temporary sorted array,
merge with existing values. O(M + N) instead of O(M × N):

```zig
// Build range array
var range_vals: [ArrayContainer.MAX_CARDINALITY]u16 = undefined;
var count: u16 = 0;
var v: u16 = start;
while (v <= end) : (v += 1) {
    range_vals[count] = v;
    count += 1;
}
// Merge ac.values[0..ac.cardinality] with range_vals[0..count]
// into a new buffer, then copy back. Standard sorted merge.
```

Or simpler: convert to run container, add the run, then let optimize decide:

```zig
// Convert array to run, add range as a new run, re-optimize
var rc = try arrayToRun(ac);
try rc.addRun(start, end);  // merge with existing runs: O(n_runs)
// Check if run container is still optimal, convert back if needed
```

### Existing run container: run merge (addRangeToContainer, ~line 270)

Replace per-element loop with a direct run merge. The new range `[start, end]`
needs to be merged with existing runs:

```zig
fn addRunRange(rc: *RunContainer, allocator: std.mem.Allocator, start: u16, end: u16) !u64 {
    // Find first run that could overlap or be adjacent to [start, end]
    // Binary search for start in runs array
    var lo: usize = 0;
    var hi: usize = rc.n_runs;

    // Find insertion point
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (rc.runs[mid].end() < start -| 1) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // Merge with overlapping/adjacent runs
    var new_start = start;
    var new_end = end;
    var merge_start = lo;
    var merge_end = lo;

    while (merge_end < rc.n_runs) {
        const run = rc.runs[merge_end];
        if (run.start > new_end +| 1) break;  // gap, no more merging
        new_start = @min(new_start, run.start);
        new_end = @max(new_end, run.end());
        merge_end += 1;
    }

    // Count values before
    var before: u64 = 0;
    for (rc.runs[merge_start..merge_end]) |run| {
        before += @as(u64, run.length) + 1;
    }

    // Replace merged runs with single new run
    const new_run = RunPair{ .start = new_start, .length = @intCast(new_end - new_start) };
    const runs_removed = merge_end - merge_start;

    if (runs_removed == 0) {
        // Insert new run at merge_start
        // Shift runs right, insert
        try rc.ensureCapacity(allocator, rc.n_runs + 1);
        std.mem.copyBackwards(RunPair, rc.runs[merge_start + 1 ..][0..rc.n_runs - merge_start], rc.runs[merge_start..rc.n_runs]);
        rc.runs[merge_start] = new_run;
        rc.n_runs += 1;
    } else {
        // Replace first merged run, shift remaining left
        rc.runs[merge_start] = new_run;
        if (runs_removed > 1) {
            const remaining = rc.n_runs - merge_end;
            std.mem.copyForwards(RunPair, rc.runs[merge_start + 1 ..][0..remaining], rc.runs[merge_end..rc.n_runs]);
        }
        rc.n_runs -= @intCast(runs_removed - 1);
    }

    const after: u64 = @as(u64, new_run.length) + 1;
    return after - before;
}
```

### Existing bitset container: already fast

The current code uses `bc.setRange(start, end)` which does word-level fills.
This is already the right approach. The only issue is the `computeCardinality()`
call after, which walks all 1024 words. Could be optimized to only recount
the affected word range, but this is minor.

## Summary of changes

```
addRangeToChunk (new container):
  BEFORE: array (element loop) or bitset (8KB alloc + setRange)
  AFTER:  run container (4 bytes, O(1))

addRangeToContainer, array path:
  BEFORE: per-element ac.add() — O(N × M) per chunk
  AFTER:  bulk append or sorted merge — O(N + M) per chunk

addRangeToContainer, run path:
  BEFORE: per-element rc.add() — O(N × R) per chunk
  AFTER:  run merge — O(R) per chunk (R = number of existing runs)

addRangeToContainer, bitset path:
  UNCHANGED: already uses setRange (word-level fills)
```

## Verification

`zig build test` — existing addRange tests cover correctness.
`zig build validate` — CRoaring byte-identity (may need runOptimize after
addRange since CRoaring may choose different container types).
`zig build bench-compare` — addRange should drop from ~10x to ~1x.

## Expected impact

addRange(0, 1M) creates 16 chunks. Currently: 16 × (8KB bitset alloc +
setRange + computeCardinality) for chunks > 4096, element-by-element for
smaller chunks. After: 16 × (4-byte run container init). Should match or
beat CRoaring.
