<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 10-11: `Roaring64Bitmap` intersectsRange

Phase-2 parity chunk of [64-bit Roaring](10-roaring64.md). Whether the bitmap has
**any** value inside a range (cheaper than materializing the intersection).

## Feature

| rawr 64-bit (new) | CRoaring | Semantics |
|---|---|---|
| `intersectsRange(lo, hi) bool` | `roaring64_bitmap_intersect_with_range` | any value in **inclusive** `[lo, hi]`? |

`*const`, allocation-free, early-exit. Mirrors 32-bit `intersectsRange`.

## Semantics — pin down the half-open/inclusive mismatch

- rawr's range ops are **inclusive** (`[lo, hi]`), house style. Expose
  `intersectsRange(lo, hi)` inclusive.
- CRoaring's `intersect_with_range(min, max)` is **half-open** `[min, max)`.
- **Distinct from `containsRange`** (10-03, "are *all* values present") — this is
  "is *any* value present". Keep the names/semantics explicit so the two range
  predicates are never conflated.

## Implementation

Same key-span decomposition as `rangeCardinality` (10-03), but early-exit: walk
buckets from `lowerBound(start_hi)` while `hi <= end_hi_u64`; for each covered
bucket call the 32-bit `bm.intersectsRange(start_low, end_low)` (or, for a fully-
covered interior key with a non-empty bucket, `true` immediately). Return `true`
on the first hit; `false` if the walk finds nothing. Widened `u64` cursor.

## Wrapper decl

```c
bool roaring64_bitmap_intersect_with_range(const roaring64_bitmap_t *r, uint64_t min, uint64_t max);  // half-open
```

## Tests / oracle

- Inline: hit in first/interior/last key of a span; miss (range falls in a gap);
  range fully inside one bucket; `maxInt(u64)` upper bound.
- `difftest64`/`validate64`: oracle the inclusive rawr call against the half-open
  CRoaring one with `hi + 1`, applying the **`hi == maxInt(u64)` overflow
  special-case** from 10-03's `cContainsRangeClosed` (split into `[lo, hi)` OR
  `contains(hi)`), reusing that helper's shape from `roaring64_oracle.zig`.

## Acceptance

- `intersectsRange` inclusive, early-exit, allocation-free; not conflated with
  `containsRange`.
- Oracled against half-open `intersect_with_range` with the max-value handling.
- Green; no 32-bit regression.
