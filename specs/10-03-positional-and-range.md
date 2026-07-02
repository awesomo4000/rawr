# Spec 10-03: `Roaring64Bitmap` positional + range ops

Third piece of [64-bit Roaring](10-roaring64.md). The two operation families
with **genuinely new 64-bit logic** (everything before this was delegation):
positional queries (`rank`/`select`/`getIndex`) via a prefix-sum over buckets,
and the multi-key-spanning range ops (`addRange`/`removeRange`).

## Features

| rawr 64-bit (new) | CRoaring | Semantics |
|---|---|---|
| `rank(value)` | `roaring64_bitmap_rank` | count of set elements `≤ value` (`u64`) |
| `select(k)` | `roaring64_bitmap_select` | `k`-th smallest element, 0-based; `null` if `k ≥ cardinality` |
| `getIndex(value)` | `roaring64_bitmap_get_index` | 0-based position of `value` if present, else `null` |
| `addRange(lo, hi)` | `roaring64_bitmap_add_range_closed` | add `[lo, hi]` **inclusive** |
| `removeRange(lo, hi)` | `roaring64_bitmap_remove_range_closed` | remove `[lo, hi]` inclusive |
| `rangeCardinality(lo, hi)` | `roaring64_bitmap_range_closed_cardinality` | count in `[lo, hi]` **inclusive** |
| `containsRange(lo, hi)` | `roaring64_bitmap_contains_range` (half-open — see Task 5) | all of `[lo, hi]` present |

`rank`/`select`/`getIndex`/`rangeCardinality`/`containsRange` are `*const`,
alloc-free. `addRange`/`removeRange` mutate `self` and may create/drop buckets.

**Return types — intentional divergence from the 32-bit API.** The 32-bit
`addRange`/`removeRange` return `!u64` (count added/removed). The 64-bit ops
return **`!void`**: a full-domain range spans up to 2⁶⁴ elements, so an exact
delta can overflow `u64` (the same cardinality edge documented in the toplevel),
and CRoaring's `roaring64_bitmap_add_range_closed` itself returns `void`. We drop
the count deliberately rather than return a saturating/wrong number. Call this
out in the doc comment so the width difference is not mistaken for an oversight.

## Task 0 — Wrapper decls

Append to `vendor/croaring_wrapper.h` (note CRoaring's range APIs are
**half-open**; the `_closed` variants are inclusive — bind the closed ones to
match rawr's inclusive semantics):

```c
uint64_t roaring64_bitmap_rank(const roaring64_bitmap_t*, uint64_t x);
bool roaring64_bitmap_select(const roaring64_bitmap_t*, uint64_t rank, uint64_t *element);
bool roaring64_bitmap_get_index(const roaring64_bitmap_t*, uint64_t x, uint64_t *out_index);
void roaring64_bitmap_add_range_closed(roaring64_bitmap_t*, uint64_t min, uint64_t max);
void roaring64_bitmap_remove_range_closed(roaring64_bitmap_t*, uint64_t min, uint64_t max);
uint64_t roaring64_bitmap_range_closed_cardinality(const roaring64_bitmap_t*, uint64_t min, uint64_t max);
bool roaring64_bitmap_contains_range(const roaring64_bitmap_t*, uint64_t min, uint64_t max);  // half-open [min,max)
```

> There is no `contains_range_closed` in the vendored 64-bit API — only the
> half-open `contains_range`. To oracle rawr's inclusive `containsRange(lo, hi)`,
> the difftest calls `roaring64_bitmap_contains_range(r, lo, hi + 1)` — which
> **overflows when `hi == maxInt(u64)`.** Special-case it: when `hi` is the max
> value, split into `contains_range(r, lo, hi)` AND `contains(r, hi)`, or assert
> that top case rawr-only. `range_closed_cardinality` has no such issue (it takes
> inclusive bounds directly).

> Confirm exact `get_index` signature against `vendor/roaring.h` during
> implementation — CRoaring 64-bit returns presence via bool + out-param in the
> modern API; if the vendored amalgam differs, bind what is actually exported and
> note it.

## Task 1 — `rank` / `getIndex`

`rank(self, value) u64` — let `key = high32(value)`, `lo = low32(value)`:
- for every bucket with `bucket.hi < key`: add `bucket.bm.cardinality()`.
- for the bucket with `bucket.hi == key` (if present): add `bm.rank(lo)`.
- stop at the first `bucket.hi > key`.

`getIndex(self, value) ?u64` — `null` if not present; otherwise
`rank(value) - 1` (rank counts `≤ value`, inclusive when present). Reuse the
bucket scan; delegate the in-bucket presence/index to the 32-bit `getIndex`.

Per-bucket cardinality is read from each sub-bitmap's own cache, so the prefix
sum is cheap; no separate cache required for v1.

## Task 2 — `select`

`select(self, k: u64) ?u64` — walk buckets in key order accumulating
`bucket.bm.cardinality()`; when the running sum would exceed `k`, the target is
in this bucket at local rank `k - sum_before` → `(bucket.hi << 32) |
bucket.bm.select(local).?`. If the loop ends with `k ≥ cardinality`, `null`.

## Task 3 — `addRange` (multi-key spanning)

`addRange(self, lo: u64, hi: u64) !void`, inclusive both ends. With
`lo_key = high32(lo)`, `hi_key = high32(hi)`:
- **Single key** (`lo_key == hi_key`): `findOrCreateBucket(lo_key)`,
  `bm.addRange(low32(lo), low32(hi))`.
- **Spanning** keys `lo_key … hi_key`:
  - first key: `bm.addRange(low32(lo), 0xFFFFFFFF)`.
  - interior keys `lo_key+1 … hi_key-1`: each `bm.addRange(0, 0xFFFFFFFF)` (fully
    materialized — find-or-create each).
  - last key: `bm.addRange(0, low32(hi))`.

Inclusive carries from the 32-bit `addRange` (already inclusive both ends).
Reject `lo > hi` the same way the 32-bit API does (empty/no-op or error — match
`RoaringBitmap.addRange`). Invalidate cardinality cache.

Per the 10-01 OOM-rollback rule, a wide range creates several interior buckets
before delegating — if any `bm.addRange` fails partway, `dropBucket` every bucket
this call newly created (track the created keys, or `errdefer` a rollback) so a
failed `addRange` leaves no zombie empty buckets.

> Materializing every interior key for a very wide range is O(#keys) buckets —
> acceptable and correct; matches CRoaring's behavior. Not a perf concern for v1.

## Task 4 — `removeRange`

`removeRange(self, lo, hi) !void` — symmetric: first/last partial keys delegate
to `bm.removeRange`; interior keys are dropped entirely (`dropBucket`) rather
than emptied one element at a time. **Any partial key whose sub-bitmap becomes
empty must be pruned** (the core invariant). Invalidate cache.

## Task 5 — `rangeCardinality` / `containsRange`

Same key-spanning decomposition as `addRange`, but read-only. Both are inclusive
`[lo, hi]`; the oracle is `roaring64_bitmap_range_closed_cardinality` (inclusive,
no overflow) and `roaring64_bitmap_contains_range` (half-open — apply the
`hi + 1` / max-value handling from Task 0's note).
- `rangeCardinality` — sum of per-key `bm.rangeCardinality(...)` (full
  `cardinality()` for fully-covered interior keys; `0` for absent keys).
- `containsRange` — every key in `[lo_key, hi_key]` must exist and its covered
  sub-range be fully present (`bm.containsRange`); interior keys need the full
  `[0, 0xFFFFFFFF]`. Early-exit `false` on the first gap (including an absent
  interior key).

## Acceptance

- All seven ops implemented; positional ops are alloc-free `*const`.
- Inline tests with values/ranges that **span multiple high-keys**, including:
  - `rank` at values below/at/above set elements and across bucket boundaries;
  - `select(0)`, `select(card-1)`, `select(card)` → `null`, and selects that land
    in interior buckets;
  - `getIndex` present/absent;
  - `addRange` spanning ≥3 keys (materializing interior buckets) and a
    single-key range; round-trip cardinality matches the inclusive width;
  - `removeRange` that empties a partial key (asserts prune) and that drops
    interior keys;
  - `rangeCardinality`/`containsRange` across boundaries and across gaps.
- `difftest64`/`validate64` extended: rank/select/getIndex (scalar agreement) and
  addRange/removeRange/rangeCardinality/containsRange (bitmap + scalar agreement)
  vs CRoaring `roaring64` over the generator.
- `zig build test64 difftest64 validate64` green; no 32-bit regression.
