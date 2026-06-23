# Spec 07-07: Bulk add/remove + array extract

Seventh piece of the [CRoaring parity effort](07-parity-inventory.md). Batch
mutate from a `[]u32` and bulk-extract to a `[]u32`. All Tier-2, all "clean" — the
work is making them faster than the equivalent `add`/`remove`/iterator loop by
amortizing the per-container lookup.

## Features

| rawr (new) | CRoaring | Semantics |
|---|---|---|
| `addMany(values)` | `roaring_bitmap_add_many` | add every value in the slice |
| `removeMany(values)` | `roaring_bitmap_remove_many` | remove every value in the slice |
| `toArrayAlloc(allocator)` | `roaring_bitmap_to_uint32_array` | all values, ascending, into a fresh `[]u32` |
| `toArray(out)` | (same, caller buffer) | fill a caller-provided slice (`out.len >= cardinality`), return count |

`addMany`/`removeMany` are `*Self` (mutating, errorable — may allocate/convert/
drop containers). The extract methods are `*const Self`. Inputs to `addMany`/
`removeMany` need **not** be sorted.

## Task 0 — Wrapper decls

```c
void roaring_bitmap_add_many(roaring_bitmap_t*, size_t n_args, const uint32_t* vals);
void roaring_bitmap_remove_many(roaring_bitmap_t*, size_t n_args, const uint32_t* vals);
void roaring_bitmap_to_uint32_array(const roaring_bitmap_t*, uint32_t* ans); // ans sized to cardinality
```

## Task 1 — `addMany` / `removeMany`

```zig
pub fn addMany(self: *Self, values: []const u32) !void
pub fn removeMany(self: *Self, values: []const u32) !void
```

Correct baseline is just a loop over `add`/`remove`, but the point of these is to
**amortize the container lookup**: consecutive values that share a high-16 key
hit the same container, so a naive per-value `findKey` is wasteful. Implement a
cursor that reuses the located container while the high key is unchanged, only
re-locating on a key change (CRoaring's `add_many` does exactly this). Keep
`cached_cardinality` correct (incremental like `add`/`remove`, or invalidate to
`-1`). `removeMany` must drop containers that empty out (ghost-container
discipline) and may demote bitset→array.

Note: inputs aren't required sorted, so the lookup-reuse only helps on runs of
equal keys; that's fine — it's the common case for bulk loads and still correct
for arbitrary order.

## Task 2 — `toArrayAlloc` / `toArray`

```zig
pub fn toArrayAlloc(self: *const Self, allocator: std.mem.Allocator) ![]u32
pub fn toArray(self: *const Self, out: []u32) usize   // out.len >= cardinality
```

`toArrayAlloc` allocates a `[]u32` of `cardinality()` and fills it; `toArray`
fills a caller buffer and returns the count (debug-assert `out.len >=
cardinality`). Both write values **ascending**.

The win over the existing iterator is **bulk per-container fill** instead of
per-value `next()`:
- array: add the high bits to each low-16 and copy the block,
- bitset: walk words, emit set bits (`@ctz` / clear-lowest),
- run: expand each run range.

Track a write cursor across containers. (`toArrayAlloc` = `cardinality()` +
`alloc` + `toArray`.)

## Task 3 — Differential checks (`diff_test.zig`)

Over the mixed generator, both `run_optimize` states:

1. **addMany:** start from an empty bitmap and from a populated one; add a
   `[]u32` (unsorted, with duplicates, spanning multiple chunks and container
   types) in rawr and via `roaring_bitmap_add_many`; `assertSameValues` (byte may
   differ — addMany can land a different valid representation). Also a pure-rawr
   cross-check: `addMany(vals)` equals looping `add`.
2. **removeMany:** from a populated bitmap, remove a `[]u32` (some present, some
   absent, including whole-container clears) vs `roaring_bitmap_remove_many`;
   `assertSameValues`; pure-rawr cross-check vs looping `remove`.
3. **toArray:** for generated bitmaps, compare rawr's output slice to
   `roaring_bitmap_to_uint32_array` element-for-element (both ascending, equal
   length = cardinality). Include empty bitmap (length 0). Also cross-check rawr
   `toArray` equals draining the rawr `iterator` (pure rawr).
4. Edge cases: empty input slice (no-op for add/remove); `addMany` with all
   duplicates; `toArray` on an empty bitmap.

## Task 4 — Benchmark

Extend `bench_croaring.zig`: `addMany` (bulk load of a large `[]u32`) vs
`roaring_bitmap_add_many`, and `toArrayAlloc`/`toArray` vs
`roaring_bitmap_to_uint32_array`. Record ratios. These are throughput APIs; the
lookup-reuse (add) and per-container bulk fill (extract) are what should keep them
near CRoaring. (Watch the inlined-vs-opaque-C asymmetry on small inputs — use
large inputs so the work dominates, per the rangeCardinality lesson.)

## Acceptance criteria

1. `addMany`/`removeMany`/`toArrayAlloc`/`toArray` exist with the signatures and
   contracts above; inputs need not be sorted; `removeMany` drops emptied
   containers; `toArray` debug-asserts buffer size.
2. All match CRoaring (`add_many`/`remove_many`/`to_uint32_array`) across the
   mixed generator + edges, both run-optimized and not — mutations by
   `assertSameValues`, extract element-for-element — plus the pure-rawr
   cross-checks (vs looping add/remove and vs the iterator).
3. `addMany` reuses the container across runs of equal high keys (not per-value
   `findKey`); `toArray` fills per-container in bulk (not per-value `next`).
4. Benched vs CRoaring with ratios recorded.
5. No leaks; `zig build test`, `validate`, `difftest`, `bench-compare` pass.

## Notes

- `range_uint32_array` (`roaring_bitmap_range_uint32_array`, values in a range
  with offset/limit) is **out of scope** here — a Tier-2 one-off for later if a
  caller needs it.
- Mark `add_many`/`remove_many`/`to_uint32_array` ✅ in the
  [inventory](07-parity-inventory.md) when done.
- After this, the only remaining `07` work is the **merge-join refactor** (and
  optional Tier-3 one-offs).
