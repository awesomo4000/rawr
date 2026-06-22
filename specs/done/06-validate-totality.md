# Spec 06: Make `validate()` a total function (no panics)

## Goal

`validate()` (added in spec 05) must be **total**: for *any* `RoaringBitmap`
value it returns either `void` (well-formed) or a `ValidateError` — it must never
panic. A validator whose contract is "report bad state" defeats itself if a bad
(or even just empty) bitmap crashes it.

This closes one **real bug** reachable on valid input, plus one defense-in-depth
gap surfaced by comparing against CRoaring's `roaring_bitmap_internal_validate`.

## Background

`validate()` currently starts with:

```zig
for (self.keys[1..self.size], 1..) |key, i| { ... }
```

and walks `self.containers[0..self.size]`. Two problems:

1. **Empty bitmap panics (real bug, valid input).** `init()` produces
   `size == 0`. `self.keys[1..self.size]` is then `keys[1..0]` — start `1` > end
   `0`, an illegal slice that panics in safe builds. This is reachable through
   normal use: `deserialize` returns an empty bitmap for `size == 0`
   (`serialize.zig:214`), so `deserializeSafe` on a legitimately-empty serialized
   bitmap panics. The spec-05 accept-side tests only round-trip populated
   `test_gen` profiles, so it was not caught.

2. **`size` vs allocation not guarded (defense-in-depth).** `validate()` slices
   `keys`/`containers` by `self.size` without checking `size` against the actual
   slice lengths. A corrupted in-memory bitmap with `size > keys.len` panics
   instead of returning an error. CRoaring's `internal_validate` checks
   `size <= allocation_size` before walking; this is the rawr analogue.

What is **already handled** (no work needed): `validateRunContainer` already
guards `rc.n_runs > rc.capacity` before slicing `rc.runs[0..rc.n_runs]`, and
`runs.len == capacity`, so the run-level slice can't OOB. The other CRoaring
checks (negative sizes, null pointers, invalid typecodes, refcount/COW
bookkeeping) don't map to rawr — Zig's types and the tagged-union representation
rule them out (`.reserved` is `unreachable`).

## Task 1 — Fix the empty-bitmap panic (the real bug)

Make `validate()` handle `size == 0` without slicing `keys[1..0]`. Cleanest:
early-return for the empty bitmap, which is trivially well-formed.

```zig
pub fn validate(self: *const Self) ValidateError!void {
    if (self.size == 0) return; // empty bitmap is valid
    // ... existing keys[1..size] walk and container walk ...
}
```

(Equivalent alternatives — guarding the loop, or iterating `keys[0..size]` with an
index-based previous-key compare — are fine; the requirement is no `[1..0]`
underflow at `size == 0`.)

## Task 2 — Top-level size/allocation guard (`BitmapSizeRange`)

Add a new error and check it **first**, before any `keys`/`containers` slicing:

```zig
pub const ValidateError = error{
    BitmapSizeRange,   // new: size exceeds allocated keys/containers
    UnsortedKeys, DuplicateKeys, EmptyContainer, UnsortedArray,
    ArrayCardinalityRange, BitsetCardinalityMismatch, BitsetCardinalityRange,
    RunOrdering, RunCardinalityMismatch,
};
```

Guard (after the `size == 0` early return from Task 1):

```zig
if (self.size > self.keys.len or self.size > self.containers.len)
    return ValidateError.BitmapSizeRange;
```

Check against the **slice lengths** (`keys.len` / `containers.len`) — those are
what gets sliced — not just the `capacity` field, which could itself be the
desynced value. Optionally also assert the invariant
`capacity == keys.len == containers.len` (treat a violation as `BitmapSizeRange`),
but the slice-length check is the one that prevents the panic.

**Framing:** this is robustness/parity, **not** a serialized-input fix. A
`size > allocation` state cannot come from `deserialize` (it builds `size` via
`ensureCapacity` and fills exactly that many); it only arises from in-memory
corruption or a rawr bug. The value is making `validate()` total and matching
`internal_validate`.

## Task 3 — Run-count error classification (optional, naming only)

`validateRunContainer` already returns `RunOrdering` for `n_runs > capacity`.
That's a structural-range violation, not an ordering one, so the name is
misleading. Optionally reclassify that single check to `BitmapSizeRange` for
consistency with Task 2 (both mean "a count exceeds its allocation"). This is a
naming choice with no behavior change beyond the returned error tag; the existing
"validate rejects adjacent runs" test stays `RunOrdering`. If reclassified, the
Task 4 run-count test expects `BitmapSizeRange`; if not, it expects `RunOrdering`.
Pick one and make the test match.

## Task 4 — Parity tests

Add deterministic tests (pure rawr, in `bitmap_tests.zig`):

1. **Empty bitmap validates ok** — `var bm = try RoaringBitmap.init(allocator);
   try bm.validate();` returns without error/panic. Also round-trip an empty
   bitmap through `serialize` → `deserialize` → `validate()` (and
   `deserializeSafe`) and assert success. This is the regression test for the
   Task 1 bug; confirm it panics against the pre-fix tree.
2. **`size > allocation` → `BitmapSizeRange`** — construct/poke a bitmap whose
   `size` exceeds `keys.len`/`containers.len` and assert the error (rather than a
   panic). Defense-in-depth; like spec 05's size-cap test, treat it as a
   hardening test, not a "prior tree accepted it" reproducer.
3. **Empty container** — a container with zero cardinality (array card 0 / run
   `n_runs == 0`) → `EmptyContainer`.
4. **Array cardinality range** — an array container with cardinality outside
   `[1, MAX_CARDINALITY]` → `ArrayCardinalityRange`.
5. **Bitset cardinality range** — a bitset container with cardinality
   `<= MAX_CARDINALITY` → `BitsetCardinalityRange`.
6. **Run `n_runs > capacity`** → the error chosen in Task 3
   (`BitmapSizeRange` or `RunOrdering`).

These tighten parity with `internal_validate`'s container-level coverage, which
spec 05's tests left partial.

## Acceptance criteria

1. `validate()` on an empty bitmap (`size == 0`) returns `void`, no panic;
   regression test added and shown to panic against the pre-fix tree.
2. Empty bitmaps round-trip through `deserialize` / `deserializeSafe` +
   `validate()` without panic.
3. `validate()` returns `BitmapSizeRange` (never panics) when `size` exceeds the
   allocated `keys`/`containers` length.
4. The Task 4 container-level parity tests pass (empty container, array card
   range, bitset card range, run count range).
5. All existing spec-05 `validate()` tests still pass; valid bitmaps still
   validate clean.
6. `zig build test`, `zig build validate`, `zig build difftest` all pass.

## Out of scope

- "Repairing" malformed bitmaps — `validate()` reports, never mutates.
- Defending against arbitrary in-memory corruption beyond making `validate()`
  total (e.g. a container tagged pointer pointing at freed memory is unreachable
  from safe rawr usage and out of scope).
- Differential rejection agreement with `internal_validate` — accept-side parity
  only, as established in spec 05.

## Sequencing

Test-first: write the empty-bitmap regression test (Task 4.1) and watch it panic,
then Task 1. Then Task 2 + its test, the container-level parity tests, and the
Task 3 naming decision. Single pass; trivial scope; no chunking.
