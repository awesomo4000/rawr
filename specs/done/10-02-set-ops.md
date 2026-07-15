<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 10-02: `Roaring64Bitmap` set operations

Second piece of [64-bit Roaring](10-roaring64.md). The full set-op suite, built
as a **merge-walk over the sorted bucket sequences** that delegates per shared
key to the existing, validated 32-bit ops. This chunk also lands the
**empty-sub-bitmap prune** discipline (the spec's core invariant) since `and`,
`andnot`, and the in-place ops are where buckets first go empty.

## Features

| rawr 64-bit (new) | CRoaring | Semantics |
|---|---|---|
| `bitwiseAnd` / `bitwiseOr` / `bitwiseXor` / `bitwiseDifference` | `roaring64_bitmap_and`/`or`/`xor`/`andnot` | new bitmap |
| `bitwiseAndInPlace` / `…OrInPlace` / `…XorInPlace` / `…DifferenceInPlace` | `*_inplace` | mutate self |
| `andCardinality` / `orCardinality` / `xorCardinality` / `differenceCardinality` | `*_cardinality` | `u64`, no result bitmap |
| `intersects` | `roaring64_bitmap_intersect` | bool |
| `isSubsetOf` / `isStrictSubsetOf` | `is_subset` / `is_strict_subset` | bool |
| `equals` | `roaring64_bitmap_equals` | bool |

Allocator threading matches the 32-bit API: the result-producing ops take an
`allocator` (e.g. `bitwiseOr(self, allocator, other) !Self`); in-place ops mutate
`self` using `self.allocator`; cardinality/bool ops are `*const`, alloc-free.

## Task 0 — Wrapper decls

Append to `vendor/croaring_wrapper.h`:

```c
roaring64_bitmap_t *roaring64_bitmap_and(const roaring64_bitmap_t*, const roaring64_bitmap_t*);
roaring64_bitmap_t *roaring64_bitmap_or(const roaring64_bitmap_t*, const roaring64_bitmap_t*);
roaring64_bitmap_t *roaring64_bitmap_xor(const roaring64_bitmap_t*, const roaring64_bitmap_t*);
roaring64_bitmap_t *roaring64_bitmap_andnot(const roaring64_bitmap_t*, const roaring64_bitmap_t*);
void roaring64_bitmap_and_inplace(roaring64_bitmap_t*, const roaring64_bitmap_t*);
void roaring64_bitmap_or_inplace(roaring64_bitmap_t*, const roaring64_bitmap_t*);
void roaring64_bitmap_xor_inplace(roaring64_bitmap_t*, const roaring64_bitmap_t*);
void roaring64_bitmap_andnot_inplace(roaring64_bitmap_t*, const roaring64_bitmap_t*);
uint64_t roaring64_bitmap_and_cardinality(const roaring64_bitmap_t*, const roaring64_bitmap_t*);
uint64_t roaring64_bitmap_or_cardinality(const roaring64_bitmap_t*, const roaring64_bitmap_t*);
uint64_t roaring64_bitmap_xor_cardinality(const roaring64_bitmap_t*, const roaring64_bitmap_t*);
uint64_t roaring64_bitmap_andnot_cardinality(const roaring64_bitmap_t*, const roaring64_bitmap_t*);
bool roaring64_bitmap_intersect(const roaring64_bitmap_t*, const roaring64_bitmap_t*);
bool roaring64_bitmap_is_subset(const roaring64_bitmap_t*, const roaring64_bitmap_t*);
bool roaring64_bitmap_is_strict_subset(const roaring64_bitmap_t*, const roaring64_bitmap_t*);
```

## Task 1 — The merge-walk

A shared driver walks two sorted `buckets` sequences by `hi`. For each step one
of three cases holds — **left-only**, **right-only**, **both** — and the op
decides what each case contributes:

| op | left-only | right-only | both (shared key) |
|---|---|---|---|
| OR | copy left sub-bitmap | copy right sub-bitmap | `bm_l.bitwiseOr(bm_r)` |
| XOR | copy left | copy right | `bm_l.bitwiseXor(bm_r)` |
| AND | — (skip) | — (skip) | `bm_l.bitwiseAnd(bm_r)` |
| ANDNOT | copy left | — (skip) | `bm_l.bitwiseDifference(bm_r)` |

**Prune rule:** for AND / ANDNOT / XOR, a "both" result can come back empty —
**do not emit a bucket for it.** Only OR with two non-empty inputs is
guaranteed non-empty. The copy cases are always non-empty (inputs are pruned by
invariant).

This is the literal one-level-up lift of rawr's existing 32-bit cross-container
merge (`src/container_ops.zig` / the bitmap-level walks). Keep the output
buckets sorted by construction (merge order is ascending).

## Task 2 — Out-of-place ops

`bitwiseOr/And/Xor/Difference(self, allocator, other) !Self` — run the driver
into a fresh `Roaring64Bitmap`, computing cardinality as you go (or leave the
cache `null` = unknown, per the `?u64` cache in 10-00). Sub-bitmap clones for copy
cases use the result's allocator.

Prefer **computing cardinality as you go** here (you already visit every result
bucket). `cardinality()` is `*const` and does not repopulate the cache, so leaving
it `null` makes the next `cardinality()` call O(buckets) — and set-op results are
exactly the values callers tend to immediately measure. Setting the cache during
construction is nearly free and avoids that. Same guidance for the in-place ops.

## Task 3 — In-place ops

`bitwiseOrInPlace/AndInPlace/XorInPlace/DifferenceInPlace(self, other) !void` —
mutate `self`'s bucket slice in place where practical, but a clean first cut may
build the result and swap. **Either way the prune invariant holds afterward.**
Invalidate the cardinality cache. (Match the 32-bit in-place ops' behavior —
`src/bitmap.zig` `bitwiseOrInPlace` et al. — for the shared-key delegation.)

**Self-aliasing (`self == other`):** rawr should define `A ⊕ A = ∅` and
`A − A = ∅` for the in-place ops (they prune to a truly empty bitmap). But
CRoaring's `roaring64_bitmap_xor_inplace` / `andnot_inplace` **forbid identical
pointers** — the differential test must never hand CRoaring `(r, r)`. Test the
self-aliased case **rawr-only** (assert the result is empty), or clone the oracle
first (`copy` then op) when an aliased comparison is wanted. Same caveat noted in
10-05's property pass (`A ⊕ A`, `A − A`).

## Task 4 — Cardinality + predicate ops

- `andCardinality/orCardinality/xorCardinality/differenceCardinality(self, other)
  u64` — run the driver in a counting mode (delegate to the 32-bit
  `*Cardinality` per shared key; full sub-cardinality for copy cases), **no
  result bitmap allocated.**
- `intersects(self, other) bool` — first shared key whose sub-bitmaps
  `intersects` → `true`; early-exit.
- `isSubsetOf(self, other) bool` — every `self` bucket has a matching `other`
  bucket and `self.bm.isSubsetOf(other.bm)`. A `self` key absent from `other`
  → `false`.
- `isStrictSubsetOf` — `isSubsetOf and cardinality < other.cardinality`.
- `equals(self, other) bool` — same bucket keys in the same order and each
  sub-bitmap `equals`.

## Acceptance

- All ops in the feature table implemented, delegating to the 32-bit core.
- Inline tests covering, with values spread across shared and disjoint high-keys:
  the 4 set ops (out-of-place + in-place), the 4 cardinality variants,
  intersects, subset/strict-subset, equals.
- **Prune coverage:** an AND or ANDNOT whose only shared key cancels to empty
  produces a bitmap with `size == 0` / `isEmpty` (no zombie buckets); assert via
  `equals` against a freshly-built empty bitmap and via `validate`-style bucket
  walk.
- `difftest64` extended: the 9-pair operation matrix (mirroring the 32-bit
  `diff_test.zig` shape) asserts every set op + cardinality + predicate agrees
  with CRoaring `roaring64` over the generator.
- `zig build test64 difftest64` green; no 32-bit regression.
