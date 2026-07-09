# Spec 10-12: `Roaring64Bitmap` fromRange

Phase-2 parity chunk of [64-bit Roaring](10-roaring64.md). Construct a bitmap
from an arithmetic sequence over a range.

## Feature — it is stepped and half-open (not a plain range)

| rawr 64-bit (new) | CRoaring | Semantics |
|---|---|---|
| `fromRange(allocator, min, max, step) !Self` | `roaring64_bitmap_from_range` | values `min + k*step` in **half-open** `[min, max)` |

**The `step` parameter is required.** CRoaring's `from_range(min, max, step)`
adds `min, min+step, min+2*step, …` while `< max` (half-open). This is **not**
the same as rawr's inclusive contiguous `addRange` — do not conflate them.

## Implementation

- Validate: `step >= 1` (reject/return-empty on `step == 0`); `min >= max` →
  empty bitmap.
- **`step == 1` fast path:** contiguous `[min, max)` → build via the inclusive
  `addRange(min, max - 1)` machinery (10-03) — but guard `max == 0` /
  `max - 1` underflow, and handle `max == 0` (empty). This reuses the efficient
  range materialization (interior keys fully filled) rather than adding
  value-by-value.
- **`step > 1`:** iterate `v = min; while (v < max) : (v += step)` adding each
  value; watch for `v += step` **overflow** near `maxInt(u64)` (use
  `@addWithOverflow` and stop on overflow). Add via the normal `add`/bucket path.

## Wrapper decl

```c
roaring64_bitmap_t *roaring64_bitmap_from_range(uint64_t min, uint64_t max, uint64_t step);
```

## Tests / oracle

- Inline: `step == 1` contiguous (spanning keys); `step > 1` sparse; `min >= max`
  → empty; `step` that lands exactly on `max` (excluded, half-open); a range near
  `maxInt(u64)` where `v += step` would overflow (must terminate, not wrap);
  `step == 0` handling.
- `difftest64`/`validate64`: build the same `(min, max, step)` on both sides and
  assert `assertAgreement` (+ serialization) vs `roaring64_bitmap_from_range`.

## Acceptance

- `fromRange(min, max, step)` implemented with the half-open + stepped semantics,
  `step == 1` fast path, overflow-safe stepping.
- Oracled against `roaring64_bitmap_from_range`; green; no 32-bit regression.
