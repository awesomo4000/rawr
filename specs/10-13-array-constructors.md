# Spec 10-13: `Roaring64Bitmap` array constructors

Phase-2 parity chunk of [64-bit Roaring](10-roaring64.md). Efficient construction
from a `u64` array, mirroring rawr's 32-bit `fromSorted`/`fromSlice`.

## Features

| rawr 64-bit (new) | CRoaring | Semantics |
|---|---|---|
| `fromSortedSlice(allocator, values) !Self` | `roaring64_bitmap_of_ptr` (sorted input) | build from **sorted, deduped** `[]const u64` |
| `fromSlice(allocator, values) !Self` | `roaring64_bitmap_of_ptr` / `from_array` | build from arbitrary `[]u64` (may sort/dedupe in place) |

Match the 32-bit naming (`fromSorted`/`fromSlice`) so the API reads the same at
both widths. CRoaring's `of_ptr(n, vals)` / `from_array` take an arbitrary array;
the "sorted" variant is a rawr optimization.

## Implementation

Group the input by high-32 key and bulk-build each sub-bitmap, rather than N
individual `add`s:

- **`fromSortedSlice`** — input is sorted, so runs of equal high-32 keys are
  contiguous. Walk the slice, and for each maximal run sharing a `hi`, slice out
  the low-32 values and hand them to the 32-bit `RoaringBitmap.fromSorted`
  (already sorted within the run). Append one bucket per key in order — no binary
  search, no per-value bucket lookup. Set the cardinality cache to the input
  count (after dedupe).
- **`fromSlice`** — sort + dedupe the `u64` slice (in place is fine, it's the
  caller's), then delegate to the `fromSortedSlice` path. Document that `fromSlice`
  may reorder the caller's buffer (same contract as 32-bit `fromSlice`).

Empty input → empty bitmap.

## Wrapper decl

```c
roaring64_bitmap_t *roaring64_bitmap_of_ptr(size_t n_args, const uint64_t *vals);
```

## Tests / oracle

- Inline: empty; single key; many keys (contiguous runs); unsorted input to
  `fromSlice` (assert result equals the sorted construction); duplicate values
  (deduped); values spanning edge keys and `maxInt(u64)`. Assert `equals` a
  reference built via `addMany`.
- `difftest64`/`validate64`: build via `fromSortedSlice`/`fromSlice` and assert
  `assertAgreement` (+ serialization) vs `roaring64_bitmap_of_ptr` on the same
  array.

## Acceptance

- Both constructors group-by-key and bulk-build sub-bitmaps (no per-value bucket
  lookup); cardinality cache set; `fromSlice` reorder contract documented.
- Oracled against `of_ptr`; green; no 32-bit regression.
