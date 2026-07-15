<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 07-02: rank / select / get_index

Second piece of the [CRoaring parity effort](07-parity-inventory.md). The marquee
missing capability — positional queries. rank and select share machinery (a
per-container rank and a per-container select primitive); get_index falls out of
rank. **Perf-sensitive** (Tier-1 / High): rank/select are hot-loop ops and
CRoaring ships `rank_many` specifically for throughput, so this spec includes a
bench task against CRoaring.

## Features

| rawr (new) | CRoaring | Semantics |
|---|---|---|
| `rank(value)` | `roaring_bitmap_rank` | count of set elements `≤ value` (`u64`) |
| `select(k)` | `roaring_bitmap_select` | the `k`-th smallest element, 0-based; `null` if `k ≥ cardinality` |
| `getIndex(value)` | `roaring_bitmap_get_index` | 0-based position of `value` if present, else `null` |
| `rankMany(values, out)` | `roaring_bitmap_rank_many` | rank of each value in a **sorted** input slice, written to `out` |

Zig conventions: `select` and `getIndex` return optionals (CRoaring uses a
bool+out-param and a `-1` sentinel respectively). All `*const Self`,
allocation-free (except `rankMany` writes caller-provided `out`).

## Task 0 — Wrapper decls

Add to `vendor/croaring_wrapper.h`:

```c
uint64_t roaring_bitmap_rank(const roaring_bitmap_t*, uint32_t x);
void     roaring_bitmap_rank_many(const roaring_bitmap_t*, const uint32_t* begin, const uint32_t* end, uint64_t* ans);
bool     roaring_bitmap_select(const roaring_bitmap_t*, uint32_t rank, uint32_t* element);
int64_t  roaring_bitmap_get_index(const roaring_bitmap_t*, uint32_t x);
```

## Task 1 — Per-container primitives

Add to `container_ops.zig` (or alongside the container types), const, no alloc:

- **`containerRank(c, low: u16) u32`** — count of elements `≤ low` within the
  container:
  - array: binary search for the first value `> low`; that index is the count.
  - bitset: `popcount(words[0..word_idx])` + `popcount(words[word_idx] & mask≤bit)`
    where `word_idx = low >> 6`. **Bit-63 caveat:** the "bits ≤ bit" mask is
    `(1 << (bit+1)) - 1`, which shifts by 64 (UB) when `bit == 63`. Special-case
    it — when `bit == 63` the mask is all-ones (`~@as(u64,0)`), or build the mask
    in `u128`/via `@shlWithOverflow`, or count the whole word. Don't `1 << 64`.
  - run: sum `(length+1)` for runs fully below `low`; for the run containing
    `low`, add `low - start + 1`.
- **`containerSelect(c, k: u32) ?u16`** — the `k`-th smallest low-16 (0-based)
  within the container, or `null` if `k ≥ container cardinality`:
  - array: `values[k]`.
  - bitset: walk words, subtract popcounts until the word holding the `k`-th bit,
    then find that bit (e.g. iterated `@ctz`/clear-lowest, or a select-in-word).
  - run: walk runs subtracting `(length+1)` until the run containing `k`, then
    `start + remainder`.

## Task 2 — `rank` and `getIndex`

```zig
pub fn rank(self: *const Self, value: u32) u64
```
Walk containers in key order. Let `hi = high16(value)`, `lo = low16(value)`:
- for each container with `key < hi`: add its full `getCardinality()`.
- for the container with `key == hi` (if present): add `containerRank(c, lo)`.
- stop at the first `key > hi`.

```zig
pub fn getIndex(self: *const Self, value: u32) ?u64
```
If `value` is not present, `null`. Otherwise its 0-based index = `rank(value) - 1`
(rank counts `≤ value`, inclusive of `value` when present). Implement by reusing
the rank walk and checking membership in the `key == hi` container in the same
pass (avoid a separate `contains` walk).

## Task 3 — `select`

```zig
pub fn select(self: *const Self, k: u64) ?u32
```
Walk containers accumulating `getCardinality()`. When the running total would
exceed `k`, the target is in the current container at offset
`k - prior_total`: return `combine(key, containerSelect(c, offset).?)`. If the
walk ends with `k ≥ total cardinality`, return `null`.

## Task 4 — `rankMany`

```zig
pub fn rankMany(self: *const Self, values: []const u32, out: []u64) void
```
**Debug-assert both** `out.len == values.len` **and** that `values` is sorted
ascending (document both as preconditions). Baseline: a single forward walk over
containers shared across all queries (cursor advances monotonically), so it's
O(containers + values) rather than `values.len ×` a full rank each.

**Decision (cursor-shared preferred, not blocking):** the cursor-shared walk is
the intended form and is only modestly more code than the loop, so aim for it in
the first pass. If it proves fiddly, landing the simple cut
`for (values, out) |v, *o| o.* = rank(v);` is acceptable — then the Task 6
benchmark ratio is recorded as **known debt** (a follow-up to optimize), not a
failed implementation. Don't block the chunk on the optimized walk.

## Task 5 — Differential checks (`diff_test.zig`)

These are scalar results — compare directly (like the `andCardinality`
predicates), over the mixed generator, both `run_optimize` states:

1. **rank:** over generated bitmaps (all profiles + the 9-pair-style container
   variety), probe a spread of values: `minimum-1`, `minimum`, `maximum`,
   `maximum+1`, several interior present + absent values, and chunk boundaries
   (`0`, `65535`, `65536`). Assert `rank` equals `roaring_bitmap_rank`.
2. **select:** probe `k` in `0 .. cardinality+2` (including `0`,
   `cardinality-1`, and out-of-range `≥ cardinality` → both `null`/false).
   Assert agreement (and that rawr `null` ⇔ CRoaring returns false). **Cast
   caveat:** `roaring_bitmap_select` takes `uint32_t rank` — only cast rawr's
   `u64 k` to `u32` for the oracle call when `k <= maxInt(u32)`; skip/guard the
   oracle compare otherwise (rawr should still return `null` for such `k`).
3. **getIndex:** for present values assert `index` matches `get_index`; for absent
   values assert `null` ⇔ `-1`.
4. **rankMany:** **oracle = repeated `roaring_bitmap_rank`**, not
   `roaring_bitmap_rank_many`. CRoaring's `rank_many`
   (`vendor/roaring.c:~17154`) does **not** fill remaining `ans` entries after it
   advances past the last container, so for empty bitmaps or probes `> maximum`
   the tail of `ans` is left unwritten — an unreliable oracle. Compare rawr
   `rankMany(values, out)` element-wise to `roaring_bitmap_rank(v)` for each
   `v` (the full documented behavior). Optionally also spot-check against
   `rank_many` for probes `≤ maximum` only, but `rank` is the authoritative
   oracle here.
5. **rank/select round-trip:** for `k in 0..cardinality`, assert
   `rank(select(k).?) == k+1` and `select(rank(v)-1) == v` for present `v`
   (internal consistency, pure rawr — cheap and catches off-by-ones).
6. Add `rank`, `select`, `getIndex` to the **randomized loop** comparisons. These
   need probe **values** (not just the two operand bitmaps), so add a small
   helper that derives a deterministic probe set per generated bitmap — e.g.
   `minimum`/`maximum` ± 1, a few values sampled from the bitmap (via `select` at
   spread ranks) plus a few known-absent ones, and chunk boundaries. Seed it off
   the loop's existing seed/iteration so failures stay reproducible.

## Task 6 — Benchmark vs CRoaring (perf-sensitive)

Extend `bench_croaring.zig` (mirrors the existing add/and/or/diff comparisons):

- **rank** on a dense bitmap: many `rank(x)` over random `x`.
- **select** on a dense bitmap: many `select(k)` over random `k` in range
  (the iteration-style hot path).
- **rankMany** vs `roaring_bitmap_rank_many` on a large sorted probe set (this is
  the batched throughput case CRoaring optimizes for). **Bench probe constraint:**
  keep all probe values `≤ maximum` (or pre-init / ignore the CRoaring `ans`
  tail), because `rank_many` leaves entries past the last container unwritten —
  otherwise the bench reads uninitialized memory. (Correctness is still validated
  against repeated `rank` per Task 5.4; this constraint is only for the
  `rank_many`-vs-`rankMany` *timing* comparison.)

Record the rawr/CRoaring ratio as we do for the set ops; flag a regression if
rawr is materially slower. (Good moment to also add the `andCardinality` /
`orCardinality` dense bench that `07-01` deferred, since you're touching this
file — optional.)

## Acceptance criteria

1. `rank`, `select`, `getIndex`, `rankMany` exist on `RoaringBitmap`, `*const`,
   with the optional/sorted-precondition contracts above.
2. Per-container `containerRank` / `containerSelect` primitives for all three
   container types.
3. All four match CRoaring across the mixed generator + edges (boundaries,
   min/max±1, out-of-range `k`, present/absent values), both run-optimized and
   not, plus the rank/select round-trip consistency check. `rankMany` is
   validated against repeated `roaring_bitmap_rank` (not `rank_many` — see Task
   5.4).
4. `bench_croaring.zig` has rank / select / rankMany comparisons with recorded
   ratios.
5. `zig build test`, `zig build validate`, `zig build difftest` pass; bench builds.

## Notes

- Mark `rank`/`select`/`get_index`/`rank_many` ✅ in the
  [inventory](07-parity-inventory.md) when done.
- No chunking expected; if `rankMany`'s cursor-shared walk balloons, it can be a
  fast-follow rather than a sub-chunk.
