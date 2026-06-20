# Optimization: Arena Allocator for Set Operations

**Applies to:** `src/bench_croaring.zig`
**Depends on:** Nothing. The arena pattern is already proven (commit c28e1b7
reduced deserialize from 2.63x to 0.47x vs CRoaring). This adds the same
pattern for set operations.

## Problem

Sparse set operations allocate a fresh result container for every container
pair in the output:

```
bitwiseAnd (sparse):  1.87ms vs 0.64ms  (2.91x slower)
bitwiseOr (sparse):   7.45ms vs 2.32ms  (3.21x slower)
```

Same root cause as deserialize was: ~65K containers × 2 allocations each ≈
130K allocator calls per operation.

## Current code (src/bench_croaring.zig, lines 167-181)

```zig
fn benchRawrAndSparse() void {
    const a = &rawr_sparse_a.?;
    const b = &rawr_sparse_b.?;
    var result = a.bitwiseAnd(allocator, b) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrOrSparse() void {
    const a = &rawr_sparse_a.?;
    const b = &rawr_sparse_b.?;
    var result = a.bitwiseOr(allocator, b) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
}
```

The module-level `allocator` is `std.heap.c_allocator` (line 6).

## Change: add arena-backed benchmark variants

### New functions

Add after the existing `benchRawrOrSparse` (after line 181):

```zig
fn benchRawrAndSparseArena() void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = &rawr_sparse_a.?;
    const b = &rawr_sparse_b.?;
    var result = a.bitwiseAnd(arena.allocator(), b) catch unreachable;
    // Don't call result.deinit() — arena.deinit() handles cleanup
    std.mem.doNotOptimizeAway(&result);
}

fn benchRawrOrSparseArena() void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = &rawr_sparse_a.?;
    const b = &rawr_sparse_b.?;
    var result = a.bitwiseOr(arena.allocator(), b) catch unreachable;
    std.mem.doNotOptimizeAway(&result);
}
```

Same pattern as the existing `benchRawrDeserializeArena` (line 247).

### Output section

In the main function, after each existing sparse set op result, add the arena
variant. Find the SET OPERATIONS section (~line 490):

```zig
// After line 498:
r = benchmark(benchRawrAndSparseArena, .{});
printResult("bitwiseAnd (sparse, arena)", r.median_ns, cr.median_ns);

// After line 506:
r = benchmark(benchRawrOrSparseArena, .{});
printResult("bitwiseOr (sparse, arena)", r.median_ns, cr.median_ns);
```

Note: reuse the `cr` value from the preceding CRoaring benchmark for the same
operation — the CRoaring number doesn't change, we're just comparing rawr's
arena path against the same baseline.

## Expected output

```
SET OPERATIONS (new bitmap)
bitwiseAnd (sparse)                1.87         0.64     2.91x
bitwiseAnd (sparse, arena)         ?.??         0.64     ?.??x
bitwiseAnd (dense)                 0.00         0.00    15.47x
bitwiseOr (sparse)                 7.45         2.32     3.21x
bitwiseOr (sparse, arena)          ?.??         2.32     ?.??x
bitwiseOr (dense)                  0.00         0.00    12.51x
```

## Expected impact

Based on the deserialize result (6.07ms → 1.09ms, 5.6x improvement):

```
                          Current    Expected (arena)    CRoaring
bitwiseAnd (sparse)        1.87ms      ~0.5-0.8ms        0.64ms
bitwiseOr (sparse)         7.45ms      ~1.5-2.5ms        2.32ms
```

bitwiseAnd may approach or beat CRoaring parity. bitwiseOr involves more
output containers (union produces larger results than intersection), so the
remaining merge-walk cost will be a bigger fraction.

## No library changes

This is purely a benchmark harness addition. The arena allocator pattern is
a caller-side optimization — pass `arena.allocator()` instead of
`std.heap.c_allocator`. All existing rawr APIs accept `std.mem.Allocator`
and work unchanged.

## Verification

`zig build test` and `zig build validate` should pass unchanged (no library
code modified). Run `zig build bench-compare` to see the new arena lines.
