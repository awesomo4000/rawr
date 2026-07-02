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
  Inclusive-both-ends semantics carry over from the 32-bit API (CRoaring's
  `roaring64_bitmap_add_range` is exclusive on max → pass `end + 1`).
- **`rank(u64)` / `select(u64)` / `getIndex`** — prefix-sum of sub-bitmap
  cardinalities over the key sequence, plus one sub-bitmap `rank`/`select`/
  `getIndex`. Worth caching per-key cardinalities.

## Cardinality edge

A fully-saturated 64-bit bitmap holds 2⁶⁴ elements, which overflows `u64`.
CRoaring has the same limitation. **Document it; do not engineer around it.**

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

## Scope — v1 vs deferred

**v1 (this spec, chunks 10-01 … 10-05):** core type + lifecycle; full set-op
suite + cardinality variants + subset/intersects; rank/select/getIndex;
addRange/removeRange; CRoaring portable-64 serialize/deserialize; property +
differential tests.

**Deferred to a later parity pass (not in 10):** `flip` (cross-key range flip is
fiddly), `lazyOr`/`lazyXor`/`repairAfterLazy`, `rankMany` batch, `jaccardIndex`,
`runOptimize` exposure (sub-bitmaps can still be run-optimized internally),
`OwnedBitmap`/FBA variants, frozen 64-bit. Benchmarks (`bench64`).

## Chunk plan

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

## Acceptance (umbrella)

All five chunks land; `zig build test test64 validate64 difftest64` is green;
docs state the CRoaring-only serialization interop scope; the deferred list above
is recorded for a future parity pass. No regression to the 32-bit suite.
