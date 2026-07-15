<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 07-04: Range operations

Fourth piece of the [CRoaring parity effort](07-parity-inventory.md). Four related
range queries/mutations that share per-container range logic. `removeRange` is the
counterpart to the existing `addRange`; `rangeCardinality` is the one
perf-sensitive member (masked popcount) and gets a bench.

## Features

| rawr (new) | CRoaring | Semantics |
|---|---|---|
| `removeRange(lo, hi)` | `roaring_bitmap_remove_range_closed` | remove every value in `[lo, hi]`; returns count removed (`u64`) |
| `rangeCardinality(lo, hi)` | `roaring_bitmap_range_cardinality_closed` | count of set values in `[lo, hi]` (`u64`) |
| `containsRange(lo, hi)` | `roaring_bitmap_contains_range_closed` | is **every** value in `[lo, hi]` present (`bool`) |
| `intersectsRange(lo, hi)` | `roaring_bitmap_intersect_with_range` | does the bitmap contain **any** value in `[lo, hi]` (`bool`) |

**Range convention:** inclusive both ends, matching `addRange`. `removeRange` and
`rangeCardinality` and `containsRange` map to CRoaring `*_closed` (inclusive)
oracles. **`intersect_with_range` has no `_closed` variant** — it takes
`(x, y)` with `y` **exclusive**, so call it with `(lo, @as(u64, hi) + 1)`.

`removeRange` is `*Self` (mutating, errorable — may demote/drop containers,
allocates); the other three are `*const Self`, allocation-free.

## Task 0 — Wrapper decls

```c
void     roaring_bitmap_remove_range_closed(roaring_bitmap_t*, uint32_t lo, uint32_t hi);
uint64_t roaring_bitmap_range_cardinality_closed(const roaring_bitmap_t*, uint32_t lo, uint32_t hi);
bool     roaring_bitmap_contains_range_closed(const roaring_bitmap_t*, uint32_t lo, uint32_t hi);
bool     roaring_bitmap_intersect_with_range(const roaring_bitmap_t*, uint64_t x, uint64_t y); // y exclusive
```

## Task 1 — `rangeCardinality` (perf-sensitive)

```zig
pub fn rangeCardinality(self: *const Self, lo: u32, hi: u32) u64
```
`lo > hi` → `0`. Add a per-container helper
**`containerRangeCardinality(c, a, b) u32`** (count in low-16 `[a, b]`), and walk
containers whose key is in `[high16(lo), high16(hi)]`:
- **fully-covered interior chunk** (sub-range is the whole `[0, 65535]`): add
  `getCardinality()` — O(1)-ish, don't popcount the whole bitset.
- **boundary chunk** (partial sub-range `[a, b]`): add
  `containerRangeCardinality(c, a, b)`, implemented as
  `containerRank(c, b) - (if a == 0) 0 else containerRank(c, a-1)` — reusing the
  `containerRank` primitive from `07-02`.
- chunks in the key range with no container contribute 0.

The boundary masked-popcount on a **bitset** is the hot path → benchmark it
including partial-within-bitset ranges (Task 6).

## Task 2 — `removeRange`

```zig
pub fn removeRange(self: *Self, lo: u32, hi: u32) !u64
```
`lo > hi` → no-op, return `0`. Recommended impl is the **difference-with-range
identity** (same trick `07-03` flip used with XOR): removing `[lo, hi]` is
`self \ range[lo, hi]`, which reuses the tested `bitwiseDifferenceInPlace` and its
container drop/demote handling:

```zig
const before = self.cardinality();
var mask = try Self.init(self.allocator);
defer mask.deinit();
_ = try mask.addRange(lo, hi);
try self.bitwiseDifferenceInPlace(&mask);
return before - self.cardinality();
```

(Return value is a rawr convenience mirroring `addRange`'s count; CRoaring
`remove_range` returns void, so it's verified via the cardinality delta in tests,
not the oracle directly.) The temporary `mask` must be freed on all paths.

Note for testing: because this is identity-based, its result is compared to the
oracle by **value, not bytes** (see Task 5.2) — same as flip.

**Optional perf follow-up (`Task 2b`, gated on bench):** like flip's Task 1b, the
mask is one range container per touched chunk. A direct per-container clear-range
(interior chunks dropped, boundary chunks cleared + possibly demoted bitset→array)
avoids the allocation. Not required to land; correctness first.

## Task 3 — `containsRange`

```zig
pub fn containsRange(self: *const Self, lo: u32, hi: u32) bool
```
`lo > hi` → vacuously **true** (empty range; confirm against the oracle). Otherwise
**early-exit** false on the first gap, using a direct per-container helper
**`containerContainsRange(c, a, b) bool`** (not `rangeCardinality`): for each chunk
key in `[high16(lo), high16(hi)]`, the container must exist and fully cover its
sub-range (interior chunks → hold all 65536 values; boundary chunks → hold all of
`[a, b]`). A missing container in the range → immediately false. (Correctness
cross-check for tests only: `containsRange == (rangeCardinality(lo,hi) ==
@as(u64, hi) - lo + 1)` — but the implementation is the early-exit helper, not a
full count.)

## Task 4 — `intersectsRange`

```zig
pub fn intersectsRange(self: *const Self, lo: u32, hi: u32) bool
```
`lo > hi` → **false**. **Early-exit** true on the first present value, using a
direct per-container helper **`containerIntersectsRange(c, a, b) bool`** (not
`rangeCardinality`): for each chunk key in `[high16(lo), high16(hi)]` with a
container, test whether it holds any value in its sub-range (array: binary-search
for a value in `[a,b]`; bitset: any nonzero masked word; run: any run overlapping
`[a,b]`). Early-exit is the whole point — a hit in the first chunk is O(1).
(Cross-check for tests only: `intersectsRange == (rangeCardinality(lo,hi) > 0)`.)

**Oracle call:** `roaring_bitmap_intersect_with_range` has no `_closed` variant
and takes an exclusive upper bound — call it as
`c.roaring_bitmap_intersect_with_range(oracle, @as(u64, lo), @as(u64, hi) + 1)`.
Widen to `u64` **before** the `+1` so `hi == maxInt(u32)` doesn't overflow.

## Task 5 — Differential checks (`diff_test.zig`)

Over the mixed generator, both `run_optimize` states. `removeRange` produces a
bitmap → `assertAgree`; the other three are scalar/bool → compare directly.

1. **rangeCardinality / containsRange / intersectsRange:** over generated bitmaps,
   test a spread of ranges: within one chunk, spanning several chunks (including
   currently-empty chunks), a full chunk, ranges touching `0` / `65535` /
   `65536`, `lo == hi`, and `lo > hi` (empty). Compare each to its oracle
   (`intersect_with_range` called as `(@as(u64, lo), @as(u64, hi) + 1)` — see
   Task 4; the others use their `*_closed` oracles directly).
2. **removeRange:** use **semantic equality, not byte-identical `assertAgree`** —
   reuse the `assertSameValues` helper (the same one `flip` uses, `diff_test.zig`).
   The difference-with-range identity may choose a different *valid* container
   representation than CRoaring's direct `remove_range_closed` (exactly the flip
   situation: same set, different bytes), so portable-byte equality can spuriously
   fail. Compare a clone passed through `roaring_bitmap_remove_range_closed` by
   value, and **also** assert the returned count equals the cardinality delta.
   (If `Task 2b` direct per-container removal is ever implemented, byte-identity
   could be revisited, but semantic equality is the contract for this spec.)
3. **internal cross-checks (pure rawr):** `containsRange(lo,hi) ==
   (rangeCardinality == hi-lo+1)` and `intersectsRange(lo,hi) ==
   (rangeCardinality > 0)` over the probe ranges — cheap, catches off-by-ones.
4. Add a random range to each of these in the **randomized loop** (reuse the
   loop's existing flip-range generation style, including an occasional `lo > hi`).

## Task 6 — Benchmark `rangeCardinality` vs CRoaring

Extend `bench_croaring.zig`: many `rangeCardinality(lo, hi)` calls over random
sub-ranges on a dense bitmap vs `roaring_bitmap_range_cardinality_closed`. Include
**ranges that fall partially within a single bitset container** (not only
full/interior chunks), since the masked bitset popcount on the boundary is the hot
path this is meant to measure. Record the ratio. (`removeRange` may get a bench too if the difference-identity proves
slow on wide ranges — that's the Task 2b signal — but `rangeCardinality` is the
required one.)

## Acceptance criteria

1. `removeRange`, `rangeCardinality`, `containsRange`, `intersectsRange` exist
   with the inclusive-range convention and the signatures above.
2. All four match CRoaring across the range spread (within/cross/full-chunk,
   boundaries, `lo==hi`, `lo>hi`), both run-optimized and not — `removeRange` via
   **`assertSameValues` (semantic, not byte)** + count-delta, the others by direct
   compare (with `intersect_with_range` called as `(@as(u64,lo), @as(u64,hi)+1)`).
3. The pure-rawr cross-checks (Task 5.3) pass.
4. `containsRange` and `intersectsRange` early-exit (don't full-count).
5. `rangeCardinality` benched vs CRoaring with a recorded ratio.
6. No leaks (removeRange's mask freed on all paths); `zig build test`,
   `validate`, `difftest` pass.

## Notes

- Reuses `07-02`'s `containerRank` (for `rangeCardinality`) and the
  difference/`addRange` machinery (for `removeRange`) — little genuinely new code.
- Mark `remove_range(_closed)`, `range_cardinality(_closed)`,
  `contains_range(_closed)`, `intersect_with_range` ✅ in the
  [inventory](07-parity-inventory.md) when done.
- No chunking; `Task 2b` is an optional perf follow-up, not a sub-chunk.
