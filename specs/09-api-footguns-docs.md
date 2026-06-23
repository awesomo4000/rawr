# Spec 09: API footguns + documentation completeness

First concrete piece of the API/ergonomics work (see
[`api-design-notes.md`](api-design-notes.md)). The API is in good shape — this is
**small code + thorough docs**, no restructuring and no behavior change to any
existing operation. Goal: fix the few real footguns and bring `API.md` +
`README.md` to complete and footgun-aware.

## Task 1 — `OwnedBitmap` completeness (additive)

`OwnedBitmap` only exposes `deinit`/`contains`/`cardinality`/`iterator`/
`serialize`, so a deserialized or set-op result can't be queried with
`minimum`/`maximum`/`equals`/`isSubsetOf`/`rank`/`select`/`andCardinality`/etc. It
already wraps a `bitmap: RoaringBitmap` field, so the fix is one documented
accessor:

```zig
/// Borrow the underlying bitmap for read-only queries
/// (minimum, equals, rank, andCardinality, ...).
pub fn asBitmap(self: *const OwnedBitmap) *const RoaringBitmap {
    return &self.bitmap;
}
```

Then `owned.asBitmap().minimum()` etc. work. Keep OwnedBitmap's existing
convenience methods. Document the accessor as the way to reach the full read-only
surface from an `OwnedBitmap`.

## Task 2 — `const` cardinality (fix an inconsistency)

`cardinality()` takes `*Self` (it maintains the cache), while every other query
(`contains`/`isEmpty`/`minimum`/`maximum`) is `*const`. So you can't get the count
of a `*const` bitmap — and `owned.asBitmap().cardinality()` (Task 1) wouldn't even
compile. Make `cardinality` `*const Self` on both `RoaringBitmap` and
`OwnedBitmap`.

Wrinkle to handle: the incremental cache (`cached_cardinality`) and run
containers' lazy cardinality both want to write on recompute. Options (implementer's
call):
- return the cache when valid (`>= 0`); on the recompute path, sum a **non-caching**
  per-container cardinality (don't write `cached_cardinality` or the run caches) —
  correct and const, at the cost of no memoization after a bulk op; or
- keep memoization via `@constCast` on the cache field (document the
  not-thread-safe-during-concurrent-reads caveat).

Prefer the non-caching const recompute unless the post-bulk-op repeated-read
pattern shows up as a real cost; incremental `add`/`remove` keep the cache valid in
the common case regardless.

## Task 3 — Demarcate public vs internal exports (low-effort)

`roaring.zig` presents container internals (`TaggedPtr`, `container_ops`,
`optimize`, the container types) and `test_gen` as `pub`, with no signal that only
three types are the stable API. They **can't be removed** — `bench`/`diff_test`/
`validate` consume `rawr.test_gen`/`rawr.TaggedPtr`/`rawr.BitsetContainer`/
`rawr.optimize` and need them from the same module as `RoaringBitmap` (type
identity). So:
- In `roaring.zig`, split the exports into a clearly-commented **"Public API"**
  block (`RoaringBitmap`, `OwnedBitmap`, `FrozenBitmap`, `ValidateError` / error
  sets) and an **"Internal — not part of the stable API, exposed for tooling; may
  change"** block (everything else).
- State in `API.md` that the public surface is the three bitmap types (+ errors).

Optional, **out of scope here**: a true split (public `rawr` root + a separate
internal-tools root so external consumers can't see internals at all). Note it as
a future option; don't do the build refactor in this spec.

## Task 4 — `API.md` completeness + footgun callouts

`API.md` predates the parity work (`07-*`) and is missing many methods. Add them
all, with Quick-Reference entries:
- **Positional:** `rank`, `select`, `getIndex`, `rankMany`.
- **Ranges (all inclusive):** `flip`, `removeRange`, `rangeCardinality`,
  `containsRange`, `intersectsRange`.
- **Cardinality variants:** `orCardinality`, `xorCardinality`,
  `differenceCardinality`; `jaccardIndex`; `isStrictSubsetOf`.
- **N-way:** `orMany`, `xorMany`, `orManyHeap` (+ `*Owned`).
- **Lazy:** `lazyOr`/`lazyXor`/`lazyOrInPlace`/`lazyXorInPlace`/`repairAfterLazy`.
- **Bulk / extract:** `addMany`, `removeMany`, `toArray`, `toArrayAlloc`.
- **Safety:** `validate`, `deserializeSafe`.
- **OwnedBitmap:** `asBitmap` (Task 1).

Footgun callouts to add (prominent, with short examples):
1. **Ranges are inclusive** — *all* range ops (`addRange`/`removeRange`/`flip`/
   `rangeCardinality`/`containsRange`/`intersectsRange`). `addRange(0, 100)` adds
   101 values. This is the most likely off-by-one; call it out clearly (it differs
   from the common half-open convention).
2. **`deserialize` vs `deserializeSafe`** — `deserialize` is bounds-safe but not
   semantically validated; **untrusted input → `deserializeSafe`** (or `deserialize`
   then `validate()`); trusted input → `deserialize`. This is currently absent.
3. **Iterator invalidation** — mutating a bitmap while iterating invalidates the
   iterator; finish iterating (or snapshot) first.
4. **Lazy state** — `lazyOr`/`lazyXor` leave the result invalid until
   `repairAfterLazy`; for a 2-way union/xor use eager `bitwiseOr`/`bitwiseXor`
   (lazy is for n-way bulk).
5. **`fromSorted` ⇄ `fromSlice`** — tighten the cross-link: `fromSorted` is UB on
   unsorted/dup input (release: silent corruption); use `fromSlice` when input
   isn't guaranteed sorted+unique.
6. **`OwnedBitmap` querying** — use `asBitmap()` for the full read-only surface.

## Task 5 — `README.md` update

Keep README a focused quickstart, but make it accurate and non-stale:
- Mention the feature breadth that now exists (positional queries, range ops,
  n-way unions, bulk add/extract) so README doesn't read as if rawr only does
  add/contains/set-ops — point to `API.md` for the full list.
- Add the **`deserializeSafe` for untrusted input** note (the README's deserialize
  example currently uses plain `deserialize`).
- Keep/clarify the inclusive-range note (already says "inclusive").
- Mention the three bitmap types and when to use each (or link API.md's table).

## Acceptance criteria

1. `OwnedBitmap.asBitmap` exists and is documented; the full read-only surface is
   reachable from an `OwnedBitmap`.
2. `cardinality` is `*const` on `RoaringBitmap` and `OwnedBitmap`; results
   unchanged; no caching regression in the incremental case.
3. `roaring.zig` demarcates public vs internal exports; all internal tooling
   (`bench`/`diff_test`/`validate`) still builds (exports retained).
4. `API.md` documents every public method including all `07-*` additions, with the
   six footgun callouts and updated Quick Reference.
5. `README.md` is accurate and complete (feature breadth, `deserializeSafe`,
   bitmap-type guidance).
6. `zig build test`/`validate`/`difftest` pass; **no behavior change** to any
   existing operation (changes are additive + docs).

## Notes

- No restructuring; the only code is `asBitmap`, the `const` cardinality, and the
  export-comment demarcation.
- True hiding of internals from external consumers is a deferred option (module
  split), not this spec.
- Broader ergonomics ideas (lazy-state type safety, naming, error-union story)
  remain in [`api-design-notes.md`](api-design-notes.md) for a later pass.
