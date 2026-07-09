# Spec 10: 64-bit Roaring bitmaps (`Roaring64Bitmap`)

A new public type that extends the value domain from `u32` to `u64`. It is an
**additive layer** over the existing, validated 32-bit `RoaringBitmap` — no new
container types, no SIMD work, no changes to the 32-bit core.

## Model

The on-disk Roaring format is 32-bit. 64-bit is a layer on top: split each `u64`
into a **high 32 bits (the key)** and **low 32 bits (the value)**, and keep an
ordered map from each key to a 32-bit `RoaringBitmap` holding the low halves.

```
key = @truncate(v >> 32)     // u32
lo  = @truncate(v)           // u32
```

**Backing structure: sorted slice of `{ hi: u32, bm: RoaringBitmap }`**, kept in
ascending `hi` order — the same model rawr already uses one level down (sorted
containers keyed by high-16). This is preferred over a hashmap because it gives
ordered `iterator` / `minimum` / `maximum` / `rank` / `select` for free,
serializes naturally, and lets the set ops be a literal lift of rawr's existing
sorted cross-container merge walk.

`Roaring64Bitmap` stores its own `allocator` and an optional cached cardinality,
mirroring `RoaringBitmap` (`src/bitmap.zig`).

### Core invariant — prune empty sub-bitmaps

A sub-bitmap that becomes empty (after `and`, `andnot`, `remove`, `removeRange`,
in-place set ops) **must be removed** from the slice, and its `RoaringBitmap`
deinit'd. Otherwise cardinality, iteration, equality, and serialization all
drift. Every operation that can empty a sub-bitmap is responsible for the prune.
This is the single most error-prone part of the layer and gets explicit test
coverage.

## What each operation becomes

Nearly everything is one-line delegation to the existing 32-bit method on the
sub-bitmap for `key`:

- **`add` / `contains` / `remove`** — find-or-create / look-up the sub-bitmap for
  `key`, delegate with `lo`.
- **Set ops (`and`/`or`/`xor`/`andnot`, + in-place, + `*Cardinality`)** —
  merge-walk the two key sequences. Shared key → delegate to the 32-bit op; key
  unique to one side → copy (union/xor/andnot) or skip (intersection). Prune any
  empty result sub-bitmap.
- **`cardinality`** — sum of sub-bitmap cardinalities (cached).
- **`minimum` / `maximum` / `iterator`** — first/last sub-bitmap; iterator is the
  ordered concat of sub-iterators with the `key` re-attached to the high 32 bits.

Two operations have **genuinely new logic** (not pure delegation):

- **`addRange(u64, u64)` / `removeRange`** — a range can span many keys. Partial
  first/last keys delegate to the sub-bitmap's `addRange`/`removeRange`;
  fully-covered interior keys get materialized full (`addRange(0, 0xFFFFFFFF)`).
  Inclusive-both-ends semantics carry over from the 32-bit API. For the CRoaring
  oracle, bind the **closed** variants (`roaring64_bitmap_add_range_closed` /
  `remove_range_closed`), which take inclusive bounds directly — no `end + 1`
  adjustment, and no overflow at `maxInt(u64)`. (The half-open `add_range` exists
  too but is not what we oracle against; see 10-03.)
- **`rank(u64)` / `select(u64)` / `getIndex`** — prefix-sum of sub-bitmap
  cardinalities over the key sequence, plus one sub-bitmap `rank`/`select`/
  `getIndex`. Worth caching per-key cardinalities.

## Cardinality edge

A fully-saturated 64-bit bitmap holds 2⁶⁴ elements, which overflows `u64`.
CRoaring has the same limitation. **Document it; do not engineer around it.**

## Overflow policy for materializing APIs (decided)

Distinct from the `u64` cardinality edge above: any API that **materializes**
memory sized by the set can need more than `usize` can address — a 64-bit
bitmap's element count or serialized byte length may exceed `maxInt(usize)`
(always possible on 32-bit-`usize` targets; the cardinality itself can even
exceed `maxInt(u64)`). Affected APIs: `toArrayAlloc`, `serialize`,
`serializedSizeInBytes` (and any future owned/frozen variants).

**Policy:** compute these sizes with **checked arithmetic** (`std.math.mul`/`add`
or `@addWithOverflow`), and **return an error** (`error.Overflow`) from the
allocating/writing API when the required size exceeds `maxInt(usize)` — never
truncate or wrap. Consequences per API:
- `toArrayAlloc` — already `!`; add `error.Overflow` when `cardinality *
  @sizeOf(u64)` (or the count itself) exceeds `usize`. `toArray(out)` (caller
  buffer) writes `min(cardinality, out.len)` values and **returns the number
  written** — it never overflows because it's bounded by `out.len` (a `usize`).
  The caller checks the return against `out.len` to detect a partial fill; when
  cardinality exceeds `usize` a full extraction is simply impossible and the
  bounded write is the honest behavior.
- `serialize` / `serializeToWriter` — already `!`; propagate `error.Overflow`.
- `serializedSizeInBytes` — make it **`!usize`** for the 64-bit type (it can
  legitimately overflow, unlike the 32-bit one whose max encoded size fits
  `usize`). This is a deliberate signature divergence from the 32-bit method;
  document it. (Alternatively keep it `usize` and document "valid only when the
  encoded size fits `usize`" — but `!usize` is the honest signature and is
  preferred.)

## Serialization — interop scope (decided)

"Portable 64-bit" is **not** universally interoperable: CRoaring's
`roaring64_bitmap_portable_serialize`, Java `Roaring64NavigableMap`, and Java
`Roaring64Bitmap` disagree on layout.

**Decision:** target **CRoaring's roaring64 portable format only.** Validate by
round-trip through `roaring64_bitmap_portable_{serialize,deserialize}`. State in
the docs exactly that — **no Java-interop claim we can't test.** Same rigor as the
32-bit validate.

## Test / bench harness layout (decided)

64-bit gets its **own** harness steps, stood up in chunk 10-00. Rationale is
iteration isolation, not runtime: validate/round-trip/difftest are the *fast*
loops (generated-case round-trips, seconds); bench/perf are the slow ones.

| step | when | speed | notes |
|---|---|---|---|
| `test64` | 10-00 | fast | focused subset during bring-up; folds into `test` automatically once `roaring64.zig` is imported from `roaring.zig` |
| `validate64` | 10-00 (stub) → real from 10-01 | fast | needs `roaring64_*` decls added to `vendor/croaring_wrapper.h` (already compiled in the amalgam — no amalgam change) |
| `difftest64` | 10-00 (stub) → real from 10-01 | fast | parallel to `difftest`, separate program |
| `bench64` / `bench-compare64` | **deferred, post-parity** | slow | correctness-first; no point benchmarking before parity |

`test64` is a bring-up convenience, not a permanent fork: once `roaring64.zig` is
imported from `roaring.zig`, its inline `test {}` blocks run under the default
`test` step too. We do not maintain two divergent test worlds.

## Differential bar is cheap and reachable

The vendored `vendor/roaring.c` amalgam already compiles the full
`roaring64_bitmap_*` API (declared in `vendor/roaring.h`). Our minimal
`vendor/croaring_wrapper.h` only exposes the 32-bit subset, so `validate64` /
`difftest64` just need the relevant `roaring64_*` declarations added to the
wrapper header — **no change to the vendored amalgam.**

## Naming

- Type: **`Roaring64Bitmap`** (matches CRoaring naming), in new file
  `src/roaring64.zig`, re-exported from `src/roaring.zig`.
- Methods mirror the 32-bit names exactly (`add`, `contains`, `bitwiseOr`,
  `rank`, `select`, `serialize`, …) so the API reads the same at both widths.

## Scope — v1 (done) and Phase 2 (full CRoaring parity)

**v1 (chunks 10-00 … 10-05, + 10-06 test-support): DONE.** Core type + lifecycle;
full set-op suite + cardinality variants + subset/intersects; rank/select/getIndex;
addRange/removeRange; CRoaring portable-64 serialize/deserialize; property +
differential tests; consolidated 64-bit test support. Everything implemented is
validated against CRoaring `roaring64`.

**Phase 2 — CRoaring parity completion (chunks 10-07 …).** Decision: finish the
full `roaring64_bitmap_*` surface, one feature per sub-spec, *then* do the
internals prettification. Measured against the vendored `roaring64_bitmap_*` API,
the remaining gaps are below. **Lazy ops are NOT a gap** — CRoaring's `roaring64`
has no `lazy_or`/`lazy_xor`/`repair` at all, so rawr's `Roaring64Bitmap` needs
none. Likewise `rankMany` is a rawr-only 32-bit batch helper with no `roaring64`
oracle, so it's excluded from parity.

| # | feature | CRoaring | effort |
|---|---|---|---|
| 10-07 | **flip** (`flip`, `flipInPlace`; inclusive, cross-key) | `flip*`/`flip_closed*` | M — the substantial one |
| 10-08 | **jaccardIndex** | `jaccard_index` | S |
| 10-09 | **runOptimize + shrinkToFit** (representation compaction) | `run_optimize`, `shrink_to_fit` | S |
| 10-10 | **clear** (reset, keep capacity) | `clear` | XS |
| 10-11 | **intersectsRange** | `intersect_with_range` | S |
| 10-12 | **fromRange** (range constructor) | `from_range` | S |
| 10-13 | **array constructors** (`fromSortedSlice`/`fromSlice`) | `of_ptr`, `from_array` | S |
| 10-14 | **32↔64 conversion** (build from / extract to `RoaringBitmap`) | `move_from_roaring32` | M |
| 10-15 | **bulk ops with context** (locality-cursor add/remove/contains) | `add_bulk`/`remove_bulk`/`contains_bulk` | M |
| 10-16 | **validate** (structural invariants) | `internal_validate` | M |
| 10-17 | **statistics** | `statistics` | S |
| 10-18 | **frozen-64** (zero-copy read-only view) | `frozen_serialize`/`_size_in_bytes`/`_view` | L — mirrors `FrozenBitmap` |

**Deliberately excluded** (not parity, or rawr-internal only): lazy ops (absent in
CRoaring `roaring64`), `rankMany` batch (no oracle), `OwnedBitmap`/FBA 64-bit
variants (rawr convenience, add later if wanted). Benchmarks (`bench64`) and the
internals refactor (key-span helper, merge-skeleton dedup) come **after** parity —
prettification last, per the phase decision.

> Grouping note for the chunk breakdown: same-feature CRoaring functions are folded
> into one sub-spec (all `flip*` variants → 10-07; `run_optimize`+`shrink_to_fit`
> compaction → 10-09; `of_ptr`+`from_array` → 10-13; the three `*_bulk` → 10-15).
> Open to splitting or trimming niche ones (statistics, bulk-with-context) on
> review.

## Chunk plan

**v1 (done):**

- **10-00** — harness scaffold: **empty** `Roaring64Bitmap` type
  (`init`/`deinit`/`isEmpty`), the three build steps (`test64`/`validate64`/
  `difftest64`) with trivial green bodies, and the lifecycle `roaring64_*`
  wrapper decls. Small first handoff that de-risks the rig before any behavior
  lands.
- **10-01** — core per-value ops on top of the scaffold: sorted-bucket infra,
  `clone`, `add`/`addMany`/`contains`/`remove`/`cardinality`/`minimum`/
  `maximum`/`toArray`/`iterator`; fills the validate64/difftest64 stubs with real
  agreement checks.
- **10-02** — set ops (`bitwiseAnd`/`Or`/`Xor`/`Difference` + in-place) via the
  merge-walk delegation, `*Cardinality` variants, `intersects`/`isSubsetOf`/
  `isStrictSubsetOf`/`equals`. Empty-sub-bitmap pruning lands here.
- **10-03** — positional + range: `rank`/`select`/`getIndex`,
  `addRange`/`removeRange`, `rangeCardinality`/`containsRange`.
- **10-04** — serialization: CRoaring portable-64 `serialize`/`serializedSizeInBytes`/
  `deserialize`/`deserializeSafe` + the `validate64` round-trip path.
- **10-05** — property tests + randomized differential loop against
  `roaring64_*`, mirroring `src/property_tests.zig` / `src/diff_test.zig`.
- **10-06** — test-support consolidation (`roaring64_test_support.zig` +
  `roaring64_oracle.zig`); no behavior change.

**Phase 2 — parity completion (10-07 … 10-18):** one feature per sub-spec, per the
table in "Scope" above. Each sub-spec stands alone: CRoaring mapping + semantics +
delegation approach + wrapper decls + `difftest64`/`validate64` agreement + tests.
Sub-specs are drafted after Morty signs off on this breakdown.

## Acceptance (umbrella)

**v1:** chunks 10-00 … 10-06 landed; `zig build test test64 validate64 difftest64`
green; docs state the CRoaring-only serialization interop scope. No 32-bit
regression. ✓

**Phase 2:** every gap in the parity table implemented and validated against its
CRoaring `roaring64` oracle; the only remaining `roaring64_bitmap_*` functions
without a rawr equivalent are the deliberately-excluded ones (lazy, rankMany, owned
variants). Then internals refactor + `bench64` follow.
