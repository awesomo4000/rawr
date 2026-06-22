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
`andnot` (+`_inplace`), `and_cardinality`, `intersect`, `is_subset`, `equals`,
`get_cardinality`, `is_empty`, `minimum`, `maximum`, `run_optimize`, `copy`
(`clone`), `internal_validate` (`validate`), portable serialize/deserialize
(+`_safe`, +`portable_deserialize_frozen` → `FrozenBitmap`), `create`/`init`
(`init`), `of_ptr` (≈ `fromSorted`/`fromSlice`).

---

## Tier 1 — core gaps, highest value

| CRoaring | Status | Effort | Maps cleanly? | Notes |
|---|---|---|---|---|
| `rank`, `rank_many` | ❌ | M | needs design | "count of elements ≤ x". Per-container rank (array: binsearch; bitset: popcount of words below; run: sum). Needs a per-container `rank` primitive. |
| `select` | ❌ | M | needs design | "k-th smallest element". Inverse of rank; walk containers accumulating cardinality, then per-container select. |
| `get_index` | ❌ | S | clean | index of a value in sorted order (= rank-1 if present). Falls out of the rank work. |
| `flip`, `flip_closed`, `flip_inplace(_closed)` | ❌ | M | needs design | Complement within a range. rawr has **no flip at all**. Per-container flip + may create/destroy containers across the range. Common op. |
| `remove_range`, `remove_range_closed` | ❌ | M | clean-ish | Counterpart to `addRange`; rawr can add a range but not remove one. Per-container clear-range + drop emptied containers. |
| `or_cardinality`, `xor_cardinality`, `andnot_cardinality` | ❌ | S | clean | rawr has only `andCardinality`. Same shape — compute cardinality without materializing. Reuse the `containerIntersectionCardinality` pattern in `container_ops.zig`. |

**Test:** rank/select/get_index/range-cardinality are scalar results → new
differential assertion shape (compare scalar over the mixed generator). flip /
remove_range produce bitmaps → drop straight into `assertAgree` + the 9-pair
matrix.

---

## Tier 2 — moderate, workload-dependent

| CRoaring | Status | Effort | Maps cleanly? | Notes |
|---|---|---|---|---|
| `or_many`, `or_many_heap`, `xor_many`, `xor_many_heap` | ❌ | M | clean | n-way unions. Naive = fold pairwise; heap/lazy variants are the optimization. |
| `lazy_or(_inplace)`, `lazy_xor(_inplace)`, `repair_after_lazy` | ❌ | L | needs design | Skip cardinality/normalization maintenance during bulk, repair once at end. The handoff's "lazy n-way unions." Real perf win for `or_many`; meaningful internal change (lazy container state). |
| `range_cardinality`, `range_cardinality_closed` | ❌ | S | clean | count of set bits in [a,b). Per-container masked popcount. |
| `contains_range`, `contains_range_closed` | ❌ | S | clean | is the whole range present. Per-container all-set check. |
| `to_uint32_array` | ❌ | S | clean | bulk dump to `[]u32`. Iterator covers it functionally; this is the allocate-and-fill convenience + speed. |
| `range_uint32_array` | ❌ | S | clean | values in a range to array. |
| `add_many` | ❌ | S | clean | bulk add into existing bitmap (rawr has `fromSlice` for fresh only). |
| `remove_many` | ❌ | S | clean | bulk remove. |
| `add_bulk` | ◑ | M | needs design | uses a reusable "bulk context" cursor for sorted adds. `fromSorted` covers fresh; this is amortized incremental. |
| `contains_bulk` | ◑ | S | clean | contains with a cursor hint; `contains` covers it, this is the cursor optimization. |
| `jaccard_index` | ❌ | S | clean | `|A∩B| / |A∪B|`. Trivial once or/and cardinality exist. |
| `is_strict_subset` | ❌ | S | clean | `is_subset && !equals`. |
| `intersect_with_range` | ❌ | S | clean | does the bitmap intersect [a,b). |

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
   *(written)* — all S-effort, all "clean," unlocks jaccard. Warm-up.
2. **`07-02` — rank / select / get_index** — the marquee missing capability;
   shared machinery.
3. **`07-03` — flip (+inplace, +closed)** — high-use, self-contained.
4. **`07-04` — range operations** (`remove_range` + `range_cardinality` +
   `contains_range` + `intersect_with_range`) — related per-container range logic.
5. **`07-05` — n-way unions** (`or_many`/`xor_many`), then **`07-06` lazy +
   repair** as a follow-on (the lazy machinery is the L-effort design piece —
   keep it separate; may itself sub-split `07-06a/b`).
6. **`07-07` — bulk + extract** (`add_many`/`remove_many`/`to_uint32_array`).
7. Tier 3 as opportunistic one-offs (`07-08+`).

(Numbering is a guide, not a contract — reorder as priorities shift.)

Each impl spec follows the established discipline: extend `croaring_wrapper.h`
with the oracle decl, implement, add a differential check (scalar-compare or
`assertAgree`) over the mixed generator, run-optimized and not.
