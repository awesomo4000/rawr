<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 10-08: `Roaring64Bitmap` jaccardIndex

Phase-2 parity chunk of [64-bit Roaring](10-roaring64.md). The Jaccard similarity
coefficient `|A ∩ B| / |A ∪ B|`.

## Feature

| rawr 64-bit (new) | CRoaring | Semantics |
|---|---|---|
| `jaccardIndex(other) f64` | `roaring64_bitmap_jaccard_index` | `\|A ∩ B\| / \|A ∪ B\|` as `f64` |

`*const`, allocation-free. Mirrors 32-bit `RoaringBitmap.jaccardIndex`.

## Implementation

Pure delegation to the existing cardinality walks — **no result bitmap**:
`inter = andCardinality(other)`, `uni = orCardinality(other)`, return
`@as(f64, @floatFromInt(inter)) / @as(f64, @floatFromInt(uni))`. Compute both in
one `twoWayCardinality` pass if convenient, but two calls is fine (each is a
cheap bucket walk).

**Empty edge:** define the both-empty case to match CRoaring
(`jaccard_index` of two empty bitmaps). Check what the 32-bit rawr
`jaccardIndex` returns for empty∩empty and mirror it; the oracle is
`roaring64_bitmap_jaccard_index` on two empty bitmaps — assert rawr agrees
(likely `0.0` or `NaN`; whatever CRoaring yields, match and document).

## Wrapper decl

```c
double roaring64_bitmap_jaccard_index(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
```

## Tests / oracle

- Inline: disjoint (→ 0), identical (→ 1), partial overlap (hand-computed),
  empty vs non-empty, empty vs empty.
- `difftest64`: add `jaccardIndex` to the cardinality-ops agreement (float
  compare with a small epsilon, matching the 32-bit `expectEqualFloat` posture).

## Acceptance

- `jaccardIndex` implemented via `and`/`or` cardinality delegation, no allocation.
- Empty-case behavior matches CRoaring and is documented.
- Float agreement vs `roaring64_bitmap_jaccard_index` in `difftest64`; green.
