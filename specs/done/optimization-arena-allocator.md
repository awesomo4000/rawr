# Optimization: Arena Allocator for Deserialize and Set Operations

## Context

After bulk I/O (commit b0c8fd6) and descriptive header bulk read/write
(in progress), the serialize path is near parity (~1.19x). The deserialize
path remains at 2.63x. Sparse set operations (bitwiseAnd 2.59x, bitwiseOr
3.25x) have the same root cause.

## Root cause: per-container allocation

Deserialize of 1M sparse values creates ~65K containers. Each container
requires 2 allocations:

1. `allocator.create(ArrayContainer)` — the struct itself
2. `allocator.alloc(u16, capacity)` — the backing values array

That's ~130K individual allocator calls. At ~40-60ns per call on macOS M4,
that's 5-8ms of pure allocation overhead — nearly the entire gap vs CRoaring's
2.30ms.

Sparse set operations have the same pattern: `bitwiseAnd` and `bitwiseOr`
allocate a fresh result container for every container pair in the output.

## Fix: arena allocator (zero code changes)

Zig's `std.heap.ArenaAllocator` turns N individual allocations into a handful
of page-level allocations with bump-pointer advancement. Each `alloc` call
becomes a pointer increment instead of a `malloc`.

The caller passes an arena allocator to `deserialize` instead of the general-
purpose allocator:

```zig
// Fast deserialization:
var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
defer arena.deinit();  // frees everything at once
var bm = try RoaringBitmap.deserialize(arena.allocator(), data);
// Use bm...
// bm.deinit() is a no-op on arena-backed memory
// arena.deinit() handles cleanup
```

**No changes to `serialize.zig` or any container code.** Every existing
`allocator.create()` and `allocator.alloc()` call inside `deserializeFromReader`
works unchanged — the arena just services them faster.

Same approach for set operations:

```zig
var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
defer arena.deinit();
const result = try bm_a.bitwiseAnd(arena.allocator(), &bm_b);
```

## What to implement

### 1. Benchmark harness change (src/bench_croaring.zig)

Add an arena-backed variant for deserialize and sparse set operations in the
benchmark to measure the actual impact:

```zig
// Deserialize with arena
{
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var bm = try RoaringBitmap.deserialize(arena.allocator(), serialized_data);
    _ = bm;  // don't deinit — arena handles it
}
```

Report both `deserialize` and `deserialize (arena)` so we can see the
allocation overhead in isolation.

### 2. deinit correctness with arena

Verify that `RoaringBitmap.deinit()` works correctly when backed by an arena.
It calls `allocator.free()` on each container's backing storage and then
`allocator.destroy()` on the struct. With `ArenaAllocator`, `free()` is a
no-op (memory freed in bulk by `arena.deinit()`). This should work without
changes, but confirm with the existing test suite:

```bash
# Run tests with arena allocator threaded through
zig build test
zig build validate  # CRoaring byte-identity
```

If any container's `deinit` does arithmetic on the allocator's state (e.g.
tracking allocation size for resize), that would break. Review `ArrayContainer.deinit`,
`BitsetContainer.deinit`, `RunContainer.deinit` to confirm they only call
`allocator.free()` and `allocator.destroy()`.

### 3. Document the pattern

Add a usage note to the public API (README or doc comment on `deserialize`)
that arena allocation is the recommended fast path for deserialize and set
operations when the bitmap has bounded lifetime.

## Expected impact

```
                     Current     With arena    CRoaring
deserialize (ms)      6.04        ~2.5-3.0       2.30
bitwiseAnd sparse     1.71        ~0.8-1.0       0.66
bitwiseOr sparse      7.60        ~3.0-4.0       2.34
```

Deserialize should approach CRoaring parity since the remaining work (bulk
memcpy of container data) is already equivalent. Set operations will improve
but may still have a gap from merge-walk efficiency (separate from allocation).

## Non-goals

This doc does NOT cover:
- Descriptive header bulk I/O (already in progress with coder)
- Branchless merge for set operations (separate spec)
- Galloping intersection (separate spec)
