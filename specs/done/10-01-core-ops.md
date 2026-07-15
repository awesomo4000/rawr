<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 10-01: `Roaring64Bitmap` core per-value operations

First feature piece of [64-bit Roaring](10-roaring64.md). Builds on the
[harness scaffold](10-00-harness-scaffold.md) — the empty type, the three build
steps (`test64`/`validate64`/`difftest64`), and the lifecycle `roaring64_*`
wrapper decls already exist and are green. This chunk fills in the **per-value
operations** and the bucket infrastructure they need, and gives the
validate64/difftest64 stubs their first real assertions.

## Deliverable

Extends `src/roaring64.zig` with the sorted-bucket helpers and the basic
per-value API. No set ops, no positional queries, no serialization yet (those are
10-02 … 10-04). Build steps and wrapper-header scaffolding came in 10-00; this
chunk only *appends* the few extra `roaring64_*` accessor decls its agreement
checks need (see Task 4).

## Task 1 — Bucket infrastructure + `clone`

The backing struct (`buckets: []Bucket` of `{ hi: u32, bm: RoaringBitmap }`,
`size`, `capacity`, `allocator`, `cached_cardinality`) and `init`/`deinit`/
`isEmpty`/`cardinality` already exist from 10-00. Add:

- `clone(self, allocator) !Self` — deep clone (clone each sub-bitmap).

Helpers (private):
- `bucketIndex(hi) ?usize` — binary search the sorted `buckets` for `hi`.
- `findOrCreateBucket(hi) !*Bucket` — binary search; on miss, insert a new
  `{ hi, RoaringBitmap.init(allocator) }` at the sorted position (shift right,
  grow if needed). Returns a pointer valid until the next structural mutation.
- `dropBucket(idx)` — deinit the sub-bitmap and remove it from the slice
  (the prune primitive; used heavily from 10-02 on).

**OOM rollback (the zombie-bucket dual of the prune invariant):** a caller that
creates a bucket and *then* delegates a fallible op into it must undo the
creation on failure. If `findOrCreateBucket` inserted a fresh empty bucket and
the subsequent `bm.add` / `bm.addRange` returns `error.OutOfMemory`, `dropBucket`
it (via `errdefer`) before propagating — otherwise a failed op leaves a zombie
empty bucket and breaks the "buckets are never empty" invariant. A caller that
hit an *existing* bucket leaves it alone (it was non-empty before). Applies to
`add`/`addMany` here and to every create-then-delegate op in later chunks.

## Task 2 — Per-value operations

All split `value` into `hi = @truncate(value >> 32)`, `lo = @truncate(value)`.

- `add(self, value: u64) !bool` — `findOrCreateBucket(hi)`, delegate
  `bm.add(lo)`. Returns whether newly added. Invalidate cardinality cache.
- `addMany(self, values: []const u64) !void` — loop `add` (a key-sorted bulk
  path is a later optimization, not this chunk).
- `contains(self, value: u64) bool` — `bucketIndex(hi)` then `bm.contains(lo)`;
  `false` if no bucket.
- `remove(self, value: u64) !bool` — `bucketIndex(hi)`, delegate `bm.remove(lo)`;
  **if the sub-bitmap is now empty, `dropBucket`.** Invalidate cache.
- `cardinality(self) u64` — cached sum of `bucket.bm.cardinality()`.
- `isEmpty(self) bool` — `size == 0` (buckets are never left empty, by the prune
  invariant).
- `minimum(self) ?u64` — first bucket: `(hi << 32) | bm.minimum().?`. `null` if
  empty.
- `maximum(self) ?u64` — last bucket: `(hi << 32) | bm.maximum().?`. `null` if
  empty.

> **Oracle caveat for empty:** rawr returns `null` for min/max of an empty
> bitmap, but CRoaring returns sentinels — `roaring64_bitmap_minimum` yields
> `UINT64_MAX` and `roaring64_bitmap_maximum` yields `0`. The validate/difftest
> comparison must special-case empty (assert rawr → `null` and skip the raw CRoaring
> value), not compare the sentinels directly.

## Task 3 — Bulk extract + iterator

- `toArrayAlloc(self, allocator) ![]u64` / `toArray(self, out: []u64) usize` —
  concat sub-bitmap extractions, re-attaching `hi` to the high 32 bits, in key
  order. Per the toplevel **overflow policy**, `toArrayAlloc` returns
  `error.Overflow` when the element count (or `count * @sizeOf(u64)`) exceeds
  `maxInt(usize)` — use checked arithmetic, never wrap. `toArray(out)` writes
  `min(cardinality, out.len)` values and **returns the number written**; it can't
  overflow (bounded by `out.len`). A caller wanting a full extraction compares the
  return to `out.len` — a short fill means the set was larger than the buffer
  (and, when cardinality exceeds `usize`, full extraction is impossible by
  construction).
- `iterator(self) Iterator` — ordered walk: outer cursor over buckets, inner
  `RoaringBitmap.Iterator` over the current sub-bitmap; `next()` yields
  `(hi << 32) | lo` and advances to the next bucket when the inner iterator is
  exhausted. Empty bitmap → first `next()` returns `null`.

## Task 4 — Wrapper header: accessor decls

10-00 already declared the lifecycle `roaring64_*` functions
(`create`/`free`/`copy`/`add`/`get_cardinality`/`is_empty`). Append the few
accessors this chunk's agreement checks need:

```c
void roaring64_bitmap_add_many(roaring64_bitmap_t *r, size_t n, const uint64_t *vals);
bool roaring64_bitmap_remove_checked(roaring64_bitmap_t *r, uint64_t x);
bool roaring64_bitmap_contains(const roaring64_bitmap_t *r, uint64_t x);
uint64_t roaring64_bitmap_minimum(const roaring64_bitmap_t *r);
uint64_t roaring64_bitmap_maximum(const roaring64_bitmap_t *r);
void roaring64_bitmap_to_uint64_array(const roaring64_bitmap_t *r, uint64_t *out);
bool roaring64_bitmap_equals(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
```

(Later chunks append their own: set ops in 10-02, rank/select/range in 10-03,
portable serialize in 10-04.)

## Task 5 — Fill in the validate64 / difftest64 stubs

Replace the empty-vs-empty stubs from 10-00 with real assertions over a generated
64-bit corpus, for the ops built in this chunk:

- **`validate64`** — round-trip generated sets through CRoaring `roaring64` for
  cardinality / membership / min / max / `to_uint64_array` agreement. (Full
  portable serialize round-trip arrives in 10-04.)
- **`difftest64`** — assert per-value-op agreement (add/contains/remove,
  cardinality, min/max, toArray) over a mixed generator. Later chunks extend it
  with set ops and positional queries.

## Acceptance

- Bucket helpers + `clone` + the per-value ops (Tasks 1–3) implemented,
  delegating to the 32-bit core.
- Inline `test {}` coverage for: add/contains/remove round-trips across multiple
  high-keys; **remove-empties-bucket prune** (add one value, remove it, assert
  `size == 0` and `isEmpty`); cardinality cache correctness; min/max with values
  in different buckets; iterator yields all values in ascending `u64` order;
  `toArray` matches the iterator.
- `validate64` and `difftest64` pass over a generated corpus for the ops
  implemented here (cardinality / contains / min / max / to-array agreement vs
  CRoaring `roaring64`).
- `zig build test test64 validate64 difftest64` green; no 32-bit regression.
