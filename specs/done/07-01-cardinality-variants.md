<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 07-01: Cardinality variants + jaccard + strict subset

First piece of the [CRoaring parity effort](07-parity-inventory.md) (pick #1).
Five small, clean gaps that reduce to machinery rawr already has — a good warm-up
that also unlocks `jaccardIndex`.

## Features

| rawr (new) | CRoaring | Semantics |
|---|---|---|
| `orCardinality(other)` | `roaring_bitmap_or_cardinality` | `\|A ∪ B\|` without materializing the union |
| `xorCardinality(other)` | `roaring_bitmap_xor_cardinality` | `\|A △ B\|` |
| `differenceCardinality(other)` | `roaring_bitmap_andnot_cardinality` | `\|A \ B\|` (rawr uses "difference" for andnot, cf. `bitwiseDifference`) |
| `jaccardIndex(other)` | `roaring_bitmap_jaccard_index` | `\|A ∩ B\| / \|A ∪ B\|` as `f64` |
| `isStrictSubsetOf(other)` | `roaring_bitmap_is_strict_subset` | `A ⊊ B` |

All are `*const Self`, allocation-free, no `c`.

## Task 0 — Wrapper decls

Add to `vendor/croaring_wrapper.h` (none of these are present yet):

```c
uint64_t roaring_bitmap_or_cardinality(const roaring_bitmap_t*, const roaring_bitmap_t*);
uint64_t roaring_bitmap_xor_cardinality(const roaring_bitmap_t*, const roaring_bitmap_t*);
uint64_t roaring_bitmap_andnot_cardinality(const roaring_bitmap_t*, const roaring_bitmap_t*);
double   roaring_bitmap_jaccard_index(const roaring_bitmap_t*, const roaring_bitmap_t*);
bool     roaring_bitmap_is_strict_subset(const roaring_bitmap_t*, const roaring_bitmap_t*);
```

Confirm `zig build validate` / `bench-compare` still build (translate-c picks them
up automatically).

## Task 1 — Implementation

### `orCardinality` / `xorCardinality` / `differenceCardinality`

Mirror `andCardinality`'s key merge-join (`bitmap.zig`) — walk both `keys`
arrays in lockstep, accumulating per the three branches. Let
`ix = containerIntersectionCardinality(a, b)` for a matched key, and `|a|`/`|b|`
be the per-container cardinalities:

| branch | `orCardinality` | `xorCardinality` | `differenceCardinality` (A\B) |
|---|---|---|---|
| key in **both** | `\|a\| + \|b\| - ix` | `\|a\| + \|b\| - 2·ix` | `\|a\| - ix` |
| key in **A only** | `\|a\|` | `\|a\|` | `\|a\|` |
| key in **B only** | `\|b\|` | `\|b\|` | `0` (skip) |

This single-pass per-container form keeps everything `*const` and needs no
whole-bitmap cardinality cache. (Correctness reference / sanity cross-check, not
the recommended impl: `orCard = |A|+|B|-andCard`, `xorCard = |A|+|B|-2·andCard`,
`diffCard = |A|-andCard` — handy for a test assertion, but the per-container merge
above is the implementation.)

Use the existing const per-container cardinality accessor
(`Container.fromTagged(tp).getCardinality()`, as `serialize.zig` does); ensure it
doesn't require a mutable container.

### `jaccardIndex`

`@as(f64, andCard) / @as(f64, orCard)`. **Edge case:** when the union is empty
(both bitmaps empty) this is `0/0`. Define the result to **match CRoaring's**
`roaring_bitmap_jaccard_index` for that case — pin the exact value (0.0 vs NaN)
in the differential test rather than guessing; implement to agree with whatever
the oracle returns. Compute `andCard` and `orCard` once (don't double-walk).

### `isStrictSubsetOf`

`self.isSubsetOf(other) and !self.equals(other)`. Both callees are already
`*const` — no cardinality needed. (Equivalent to `isSubsetOf and |A| < |B|`, but
`!equals` avoids a separate cardinality computation.)

## Task 2 — Differential checks

In `diff_test.zig`, these are scalar/bool predicates — compare results directly
(like the existing `andCardinality`/`intersects` predicate checks), no
`assertAgree`. Cover:

1. The **9-pair container matrix** (sparse/dense/runs combinations) plus the
   empty-operand and full-chunk edges, **both `run_optimize` states**, asserting
   each new function equals its CRoaring counterpart.
2. The **randomized loop** — add all five to the per-iteration predicate
   comparisons over `randomMixed`.
3. **jaccard empty/empty edge** explicitly, to pin the div-by-zero behavior
   against the oracle.
4. **Non-commutativity:** test `differenceCardinality` and `isStrictSubsetOf` in
   **both** orders (A,B) and (B,A) — both are asymmetric.

`f64` compare for jaccard: assert exact bit-equality if both sides compute it the
same way, else a tight epsilon — but first try exact, since both are
`andCard/orCard` over identical integer inputs and should match exactly.

## Acceptance criteria

1. All five methods exist on `RoaringBitmap`, `*const`, allocation-free, no `c`.
2. Wrapper decls added; `validate`/`bench-compare` still build.
3. Each function matches CRoaring across the 9-pair matrix + edges, both
   run-optimized and not, and over the randomized loop, in `diff_test.zig`.
4. `differenceCardinality` and `isStrictSubsetOf` tested in both operand orders;
   jaccard empty-union edge pinned to the oracle.
5. `zig build test`, `zig build validate`, `zig build difftest` all pass.

## Notes

- Update the [parity inventory](07-parity-inventory.md): mark these five ✅ when
  done.
- No chunking — single small pass.
