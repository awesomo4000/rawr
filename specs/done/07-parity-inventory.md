# Spec 07: CRoaring parity inventory (planning reference)

**This is a living reference / menu, not an implementation spec.** It maps every
CRoaring `roaring_bitmap_*` function to rawr's current surface, with an effort
estimate and a "maps cleanly / needs design" flag, so we can cherry-pick features
into focused implementation specs (08+) deliberately. It does **not** move to
`done/` the usual way — update it as features land.

Source: `vendor/roaring.h` (full CRoaring API) vs `src/bitmap.zig` /
`src/frozen.zig` (rawr public surface), as of spec 06.

Legend — **Status:** ✅ have · ◑ partial · ❌ missing · ⛔ skip-by-design.
**Effort:** S (a few hrs) · M (a day) · L (multi-day / needs design).
**Test:** how it plugs into the differential harness.

---

## Already at parity (✅)

`add` (+checked → `add` returns bool), `remove` (+checked), `contains`,
`add_range`/`add_range_closed` (rawr `addRange`, inclusive), `and`/`or`/`xor`/
`andnot` (+`_inplace`), `and_cardinality`, `or_cardinality`,
`xor_cardinality`, `andnot_cardinality`, `jaccard_index`, `intersect`,
`is_subset`, `is_strict_subset`, `equals`, `get_cardinality`, `is_empty`,
`minimum`, `maximum`, `run_optimize`, `copy` (`clone`), `internal_validate`
(`validate`), portable serialize/deserialize (+`_safe`,
+`portable_deserialize_frozen` → `FrozenBitmap`), `create`/`init` (`init`),
`of_ptr` (≈ `fromSorted`/`fromSlice`).

---

## Tier 1 — core gaps, highest value

| CRoaring | Status | Effort | Maps cleanly? | Notes |
|---|---|---|---|---|
| `rank`, `rank_many` | ✅ | M | needs design | rawr `rank` / `rankMany` (cursor-shared). Per-container rank primitive. |
| `select` | ✅ | M | needs design | rawr `select(k)` → optional; per-container select primitive. |
| `get_index` | ✅ | S | clean | rawr `getIndex` → optional. |
| `flip`, `flip_closed`, `flip_inplace(_closed)` | ✅ | M | clean | rawr `flip`/`flipInplace`/`flipOwned` via XOR-with-range identity (inclusive). |
| `remove_range`, `remove_range_closed` | ✅ | M | clean-ish | rawr `removeRange` via difference-with-range identity. |
| `or_cardinality`, `xor_cardinality`, `andnot_cardinality` | ✅ | S | clean | rawr `orCardinality` / `xorCardinality` / `differenceCardinality`. |

**Test:** rank/select/get_index/range-cardinality are scalar results → new
differential assertion shape (compare scalar over the mixed generator). flip /
remove_range produce bitmaps → drop straight into `assertAgree` + the 9-pair
matrix.

---

## Tier 2 — moderate, workload-dependent

| CRoaring | Status | Effort | Maps cleanly? | Notes |
|---|---|---|---|---|
| `or_many`, `xor_many` | ✅ | M | clean | rawr `orMany`/`xorMany` (k-way + lazy, forces bitset accumulation). orMany ~1.14× (parity); xorMany ~0.54× — rawr takes the dense bitset fast path that CRoaring's `xor_many` (no bitset-conversion option, unlike `or_many`) skips. Win is xor-specific + dense-input-specific, not allocator/structure. |
| `or_many_heap` | ✅ | M | clean | rawr `orManyHeap` — **thin alias of `orMany`** (no distinct algorithm). CRoaring's `or_many_heap` is ~2× slower than its `or_many` on uniform-size inputs, so the alias suffices. |
| `xor_many_heap` | ⛔ | — | — | **not exported in this vendored CRoaring** (TODO comment only). No oracle; rawr-only alias of `xorMany` at most. |
| `lazy_or(_inplace)`, `lazy_xor(_inplace)`, `repair_after_lazy` | ✅ | L | needs design | rawr `lazyOr`/`lazyXor`/`repairAfterLazy` (+ in-place). Lazy bitset accumulation, single repair. |
| `range_cardinality`, `range_cardinality_closed` | ✅ | S | clean | rawr `rangeCardinality`; vectorized windowed popcount (beats CRoaring on large single-chunk ranges). |
| `contains_range`, `contains_range_closed` | ✅ | S | clean | rawr `containsRange` (early-exit). |
| `to_uint32_array` | ✅ | S | clean | rawr `toArrayAlloc`/`toArray` (bulk per-container fill). |
| `range_uint32_array` | ❌ | S | clean | values in a range to array. |
| `add_many` | ✅ | S | clean | rawr `addMany` (cursor reuse + array append fast-path). Sequential ~1.08×. |
| `remove_many` | ✅ | S | clean | rawr `removeMany`. |
| `add_bulk` | ◑ | M | needs design | uses a reusable "bulk context" cursor for sorted adds. `fromSorted` covers fresh; this is amortized incremental. |
| `contains_bulk` | ◑ | S | clean | contains with a cursor hint; `contains` covers it, this is the cursor optimization. |
| `jaccard_index` | ✅ | S | clean | rawr `jaccardIndex`. |
| `is_strict_subset` | ✅ | S | clean | rawr `isStrictSubsetOf`. |
| `intersect_with_range` | ✅ | S | clean | rawr `intersectsRange` (early-exit). |

---

## Tier 3 — nice-to-have / debug / maintenance

| CRoaring | Status | Effort | Notes |
|---|---|---|---|
| `clear` | ❌ | S | empty but keep allocation. |
| `shrink_to_fit` | ❌ | S | trim over-allocation. |
| `remove_run_compression` | ❌ | S | inverse of `run_optimize`. |
| `add_offset` | ❌ | M | shifted copy (add a constant to every value). |
| `overwrite` | ❌ | S | copy src into existing dst, reusing allocation. |
| `to_bitset` | ❌ | M | materialize a flat `[]u64` bitset. Niche. |
| `statistics` | ❌ | S | per-container-type counts/sizes. Useful for tests + tuning. |
| `printf`, `printf_describe` | ❌ | S | debug printing. |
| `create_with_capacity`, `init_with_capacity`, `init_cleared` | ◑ | S | rawr has `init` + `ensureCapacity`; a capacity ctor is cosmetic. |
| `from`, `from_range`, `of` | ◑ | S | convenience constructors; `from_range` = init+addRange, `of` = `fromSlice`. |

---

## Deliberately skip (⛔) — confirm these stay out

| CRoaring | Why skip |
|---|---|
| `get/set_copy_on_write` | rawr's one-alloc-per-container model has no COW. Architectural divergence. |
| native `serialize`/`deserialize`/`size_in_bytes` (non-portable) | rawr is portable-format only by design; portable is the interop format. |
| `frozen_serialize`/`frozen_size_in_bytes`/`frozen_view` (native frozen) | rawr's `FrozenBitmap` is a zero-copy view over the **portable** format — a different, deliberate design. |
| `portable_deserialize_size` | length/size pre-check; low value given `deserialize` already validates. Could add later if a streaming caller needs it. |
| `free`, `init`, low-level C lifecycle | N/A in the Zig allocator model. |

---

## Pieces (impl specs `07-NN`)

Each parity pick is a chunk under this umbrella: `07-NN`, self-contained with its
own wrapper/impl/differential-test and pass/fail. Tick them off here as they land.

1. **`07-01` — `or/xor/andnot` cardinality + `jaccard` + `is_strict_subset`**
   *(done)* — all S-effort, all "clean," unlocks jaccard. Warm-up.
2. **`07-02` — rank / select / get_index** *(done)* — marquee capability; shared
   machinery. Includes `rankMany` + bench vs CRoaring.
3. **`07-03` — flip (+inplace, +closed)** *(done)* — XOR-with-range identity.
4. **`07-04` — range operations** *(done)* (`remove_range` + `range_cardinality`
   + `contains_range` + `intersect_with_range`) — related per-container range
   logic. rangeCardinality uses a vectorized windowed popcount.
5. **`07-05` — n-way unions** (`or_many`/`xor_many`) *(done)* — k-way merge.
6. **`07-06` — lazy + repair** *(done)* — lazy fold + single repair; orMany
   85.78×→1.25×, xorMany 40.25×→0.55×. **`07-06b`** *(done)* — added `orManyHeap`
   parity alias; the proposed reusable-workspace fix **regressed** orMany (~1.38×)
   and was dropped, so the residual orMany gap (~1.14×) is left at parity. Final:
   orMany ~1.14×, orManyHeap ~0.52× (vs CRoaring's slower `or_many_heap`), xorMany
   ~0.54×.
7. **`07-07` — bulk + extract** *(done)* (`add_many`/`remove_many`/
   `to_uint32_array`) — cursor reuse + array append fast-path (sequential addMany
   1.37×→1.08×), bulk per-container extract.
8. **`07-08` — merge-join refactor** *(done)* — consolidated the two-way merge for
   the 4 allocating + 4 cardinality ops into comptime-generic helpers
   (`twoWayAllocatingMerge`/`twoWayCardinality`); AND scratch path preserved
   (comptime-routed), in-place four left as-is. Bench: set-op/cardinality rows
   unchanged (no regression). **This closes the parity umbrella.**
9. Tier 3 as opportunistic one-offs.

(Numbering is a guide, not a contract — reorder as priorities shift.)

### Known follow-ups (post-umbrella, optional)

- **flip / removeRange wide-range perf** — *closed.*
  [`07-09`](07-09-bitset-run-wordwise.md) made bitset×run ops word-wise
  (1.54 ms→0.13 ms); [`07-10`](07-10-inplace-xor-difference.md) made XOR/difference
  in-place (eliminating the redundant per-container copy), landing both at
  ~CRoaring (0.00 ms, timer floor). `07-03` Task 1b fully obsoleted.
- **Tier-3 conveniences** — `clear` (most likely wanted), `add_offset`
  (domain-specific), then the rest as needed.
- **`08-fuzzing`** (parked in `todo/`) — the next major effort.
- `rankMany` — *closed* ([`07-11`](07-11-rankmany-batch.md)): batch single-sweep
  per container, 1.50×→1.03× (parity).
- **Perf work effectively complete.** Remaining >1.0× rows are timer-floor
  (sub-20 µs ops: `orMany` ~10 µs/1.23×, flip/removeRange both 0.00 ms) or
  run-to-run noise (`contains`, `serialize`, `add`-sequential). rawr is
  at-or-faster than CRoaring on everything else.
- `lazyOr+repair (sparse)` ~1.24× is a **real ~3 ms apples-to-apples gap** (rawr's
  public 2-way `lazyOr`+repair vs CRoaring's), *not* noise — but low-value: the
  scenario is pathological (forcing bitset conversion on sparse 2-way; plain
  `bitwiseOr` is 0.96×, rawr faster) and the public `lazyOr` 2-way path is rarely
  called directly (orMany uses the internal k-way fold, not `lazyOr`). Residual is
  the per-chunk bitset alloc/zero/free traffic; profile before touching.
- *Optional completionist item:* `orMany` ~2 µs residual — would need a profile
  pass first (the workspace guess regressed it), but it's chasing microseconds.

Each impl spec follows the established discipline: extend `croaring_wrapper.h`
with the oracle decl, implement, add a differential check (scalar-compare or
`assertAgree`) over the mixed generator, run-optimized and not.
