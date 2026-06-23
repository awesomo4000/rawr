# Spec 07-05: N-way unions (`orMany` / `xorMany`)

Fifth piece of the [CRoaring parity effort](07-parity-inventory.md). Combine an
arbitrary list of bitmaps in one call. This is the last **perf-headline** parity
feature — n-way throughput is a CRoaring strength — but the deepest optimization
(lazy union + repair) is deliberately split into **`07-06`**. This chunk lands the
API, a correct **k-way merge** implementation, differential coverage, and a bench
that documents where we stand before lazy.

## Features

| rawr (new) | CRoaring | Semantics |
|---|---|---|
| `orMany(allocator, bitmaps)` | `roaring_bitmap_or_many` | union of all input bitmaps → new bitmap |
| `xorMany(allocator, bitmaps)` | `roaring_bitmap_xor_many` | symmetric difference of all (value present in an odd number of inputs) |

```zig
pub fn orMany(allocator: std.mem.Allocator, bitmaps: []const *const Self) !Self
pub fn xorMany(allocator: std.mem.Allocator, bitmaps: []const *const Self) !Self
```

Both produce a fresh bitmap (allocating). Add `orManyOwned` / `xorManyOwned`
mirroring the `*Owned` surface. **Out of scope here (→ `07-06`):** the
`*_many_heap` variants and the lazy/`repair_after_lazy` machinery — they're the
perf layer, built on top of this chunk's k-way structure.

## Edge cases

- empty list (`bitmaps.len == 0`) → empty bitmap.
- single element → a clone of it (don't alias the input).
- inputs are `*const` and must not be mutated.

## Task 0 — Wrapper decls

```c
roaring_bitmap_t* roaring_bitmap_or_many(size_t number, const roaring_bitmap_t** rs);
roaring_bitmap_t* roaring_bitmap_xor_many(size_t number, const roaring_bitmap_t** rs);
```

## Task 1 — `orMany` via k-way merge

The naive approach (clone `bitmaps[0]`, fold `bitwiseOrInPlace` over the rest) is
correct but re-normalizes and re-walks the whole accumulator N times. Implement
the **k-way merge** instead — build each output container exactly once:

- Keep a cursor per input bitmap into its `keys`/`containers` arrays.
- Repeatedly find the **minimum key** across the live cursor heads. For `07-05` a
  linear scan over the N heads is fine (a binary heap is the `or_many_heap`
  optimization — defer to `07-06`); note the O(N·distinct_keys) cost in a comment.
- For that key, **union all same-key containers** from the inputs that have it
  into one result container (fold `containerUnion` / `containerUnionInPlace` over
  them), append it to the result, and advance those cursors.

This reuses the existing container union ops; the new code is the k-way key
merge. The per-key fold is exactly where `07-06`'s lazy path will later avoid
intermediate normalization — keep the per-key union isolated in a small helper so
lazy can swap it.

## Task 2 — `xorMany`

Same k-way merge skeleton, but **xor-accumulate** the same-key containers: a key
present in only one input contributes that container as-is; a key in several
inputs folds `containerXor` across them (associative/commutative, so order within
a key doesn't matter). Drop a container that folds to empty (cardinality 0) — same
ghost-container discipline as `bitwiseXor`.

## Task 3 — Differential checks (`diff_test.zig`)

n-way results may pick a different *valid* container representation than CRoaring's
`or_many`/`xor_many` (same situation as flip/removeRange), so compare by
**semantic equality** (`assertSameValues`), not byte-identical `assertAgree`. (Try
byte-identity first if you like; if it diverges, semantic is the contract.)

- Build **N generated bitmaps** for several `N` (e.g. 0, 1, 2, 3, 8, ~32) using
  the mixed generator across profiles, both `run_optimize` states. `orMany` /
  `xorMany` them in rawr; build the same N oracles and call the CRoaring
  counterpart; assert `assertSameValues`.
- Mix the inputs so the same key is array in one bitmap, bitset in another, run in
  a third (the heterogeneous-per-key case the k-way fold must handle).
- **Cross-check vs the 2-way op (pure rawr):** `orMany([a,b])` equals
  `a.bitwiseOr(b)`; `xorMany([a,b])` equals `a.bitwiseXor(b)`. And
  `orMany([a,b,c])` equals `a.bitwiseOr(b).bitwiseOr(c)` — cheap, catches fold
  bugs without the oracle.
- Edge cases: empty list, single element (result is an independent clone — mutate
  it and confirm the input is unchanged), inputs containing empties.

## Task 4 — Benchmark vs CRoaring

Extend `bench_croaring.zig`: `orMany` over a list of many bitmaps (e.g. 16–64
mixed-profile bitmaps) vs `roaring_bitmap_or_many`. Record the ratio.

**Expectation/framing:** the k-way merge with eager per-key normalization will
likely still trail CRoaring's `or_many` (which is lazy internally). That gap is
**expected and is exactly what `07-06` closes** — don't treat a >1.0× here as a
failure; record it as the baseline the lazy work will improve. (Keep the bench so
`07-06` can show the before/after.)

## Acceptance criteria

1. `orMany` / `xorMany` (+ `*Owned`) exist with the signatures and edge-case
   behavior above; inputs never mutated; single-element result is an independent
   clone.
2. Implemented as a k-way merge (each output container built once), with the
   per-key union isolated in a helper for `07-06` to swap.
3. Match CRoaring `or_many` / `xor_many` by `assertSameValues` across varied `N`,
   profiles, heterogeneous-per-key containers, and both run-optimized states;
   plus the pure-rawr 2-way cross-checks.
4. Benched vs CRoaring with the ratio recorded (gap acceptable; `07-06` closes it).
5. No leaks (intermediate containers freed); `zig build test`, `validate`,
   `difftest` pass.

## Notes

- Deliberately omits `or_many_heap`/`xor_many_heap` and lazy/repair → `07-06`.
- The per-key union helper is the seam lazy will use (accumulate into the widest
  form, normalize once at the end) — keep it clean.
- Mark `or_many`/`xor_many` ✅ in the [inventory](07-parity-inventory.md) when
  done (leave the `_heap`/lazy rows for `07-06`).
