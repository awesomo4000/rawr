<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 05: `deserialize` hardening + semantic `validate()`

## Goal

Two related pieces for the **allocating** deserialization path
(`src/serialize.zig`), following CRoaring's split of *bounds safety* from
*semantic validity*:

1. **Cheap hardening** folded unconditionally into the base `deserialize` — a
   `size` cap and overflow-safe arithmetic. Near-zero cost; closes the Finding-2
   fragility from the baseline security review.
2. **An opt-in `validate()`** that checks the structural/semantic invariants of a
   bitmap (key ordering, container sortedness, cardinality consistency). This is
   the analogue of CRoaring's `roaring_bitmap_internal_validate`, kept off the hot
   path so trusted callers don't pay for it.

## Background / threat model

rawr's allocating `deserialize` (`serialize.zig:177` / `deserializeFromReader`)
reads through `std.Io.Reader.fixed`, which errors on any short read. It is
therefore **already bounds-safe** against truncation — it is at the level of
CRoaring's `portable_deserialize_safe`, not its unsafe `portable_deserialize`.
The remaining gaps are:

- **Finding 2 (low):** `size` is read as a full attacker-controlled `u32`
  (`serialize.zig:209`) with no cap, and `allocator.alloc(u16, size * 2)`
  (`:227`) computes `size * 2` in `u32`, which overflows for `size >= 2^31`. It is
  currently gated behind the multi-GB allocation failing first, so not a clean
  exploit — but it is fragile (a future arena/over-commit allocator could change
  that). Task 1 removes the fragility for free.
- **Semantic validity:** a malformed-but-in-bounds buffer deserializes into a
  structurally-wrong bitmap (unsorted keys, unsorted/duplicate array values, a
  bitset whose stored cardinality disagrees with its bit population, overlapping
  runs). The result returns **wrong answers** from `contains` / set ops /
  `cardinality` — an integrity issue, **not** memory unsafety. Task 2 lets a
  caller reject such input.

Priority is **lower than spec 04** (that was real OOB; this is wrong-answers).
Task 1 is nearly free and should land regardless. Task 2 is opt-in and only
matters when untrusted serialized bytes reach the *allocating* path (the
zero-copy frozen path is handled by spec 04).

## Task 1 — Cheap hardening in base `deserialize` (unconditional)

In `deserializeFromReader` (`serialize.zig:188`):

1. **Cap `size`.** The maximum legal container count is 65536 (one per high-16
   key). In **both** cookie branches, immediately after `size` is known, reject
   `size > 65536` with `error.InvalidFormat`:
   - run cookie (`:201`): `size = ((cookie >> 16) & 0xFFFF) + 1` is already
     `<= 65536`; no change needed, but assert/document it.
   - no-run cookie (`:209`): `size = reader.takeInt(u32, ...)` — add the check
     here, **before** any allocation or `size`-derived arithmetic.
2. **Widen arithmetic to `usize`.** Compute `size * 2` (the `desc_buf` length,
   `:227`) and the offset-skip `size * 4` (`:240`) in `usize`, not `u32`. With the
   cap in place these can't overflow anyway, but the widening is belt-and-suspenders
   and matches how `frozen.zig` already does it.

3. **Preserve run-container header cardinality.** Today deserialize reads each
   container's header cardinality into `cardinalities[i]` but then builds run
   containers with `rc.cardinality = -1` (`serialize.zig:261`), discarding it —
   so `validate()` would have nothing to compare a recomputed run cardinality
   against. Change deserialize to store the header cardinality into
   `rc.cardinality` (i.e. `@intCast(cardinalities[i])`), exactly as it already
   does for bitset containers (`:269`). This makes deserialize uniform across
   container types (header cardinality is trusted on the fast path for *all*
   types), and gives `validate()` a stored value to check the recomputed
   `sum(length+1)` against. **This is the one change that unblocks the run
   cardinality check below — without it, run-cardinality corruption is not
   observable.**

Points 1–2 are behavior-preserving for valid input (valid `size` is always
`<= 65536`); they only add rejection of out-of-range `size`. Point 3 is also
behavior-preserving for valid input (the stored value equals the recomputed one)
and additionally makes `cardinality()` on a freshly-deserialized run container
O(1) instead of lazily recomputed.

## Task 2 — `validate()` semantic check

Add a read-only method that verifies a bitmap's structural invariants. Pure
rawr, no allocation, no `c`.

```zig
pub const ValidateError = error{
    UnsortedKeys,
    DuplicateKeys,
    EmptyContainer,
    UnsortedArray,        // array values not strictly ascending
    ArrayCardinalityRange,// array container with cardinality outside [1, MAX_CARDINALITY]
    BitsetCardinalityMismatch, // stored cardinality != actual popcount
    BitsetCardinalityRange,    // bitset cardinality not > MAX_CARDINALITY
    RunOrdering,          // runs not strictly ordered / overlapping / adjacent
    RunCardinalityMismatch,    // sum(run length + 1) != stored cardinality
};

/// Verify structural invariants. Returns the first violation found, or void if
/// the bitmap is well-formed. Read-only; does not mutate or repair.
pub fn validate(self: *const Self) ValidateError!void
```

Checks (mirroring CRoaring `internal_validate`, scoped to rawr's representation):

1. **Keys** (`self.keys[0..size]`): strictly ascending — each `keys[i] <
   keys[i+1]`. A non-increasing step is `UnsortedKeys`; an equal step is
   `DuplicateKeys`.
2. **Per container**, by type:
   - **array**: `cardinality` in `[1, ArrayContainer.MAX_CARDINALITY]`
     (else `ArrayCardinalityRange` / `EmptyContainer`); `values[0..cardinality]`
     strictly ascending (`UnsortedArray` — catches both unsorted and duplicate).
   - **bitset**: `cardinality > ArrayContainer.MAX_CARDINALITY` (else
     `BitsetCardinalityRange` — a small set should be an array); the **actual**
     popcount of `words` equals the stored `cardinality`
     (`BitsetCardinalityMismatch`).
   - **run**: `n_runs >= 1` (`EmptyContainer`); runs strictly ordered and
     **non-overlapping, non-adjacent** — for consecutive runs, the next run's
     `start` must be `> prev.start + prev.length + 1` (a gap of at least one;
     adjacent/overlapping runs should have been merged) → `RunOrdering`; each run
     within `u16` range; `sum(length + 1)` over all runs equals
     `rc.cardinality` → `RunCardinalityMismatch`.
- **`.reserved`**: cannot arise from deserialize (the tagged-pointer type is
  always array/bitset/run on that path) and cannot be constructed by user code.
  Match the rest of the codebase and treat it as `unreachable` in the switch — do
  **not** add a `ValidateError` variant for it.

**Run cardinality — firm decision:** `validate`
stays strictly non-mutating (`self: *const Self`). It reads the **field**
`rc.cardinality` directly (not `getCardinality()`, which may recompute/mutate) and
compares it to the freshly recomputed `sum(length + 1)`. This is observable only
because **Task 1 point 3** makes deserialize store the header cardinality into
`rc.cardinality` instead of `-1`. (If some other code path ever leaves a run
container at `rc.cardinality == -1`, treat that as "nothing to check" for the
mismatch test — the ordering/range checks still apply — rather than recomputing
and persisting.) The same pattern applies to bitset: compare the **field**
`bc.cardinality` to the recomputed popcount, no mutation.

Rationale notes for the implementer:
- The array-vs-bitset cardinality-range checks encode rawr's own representation
  invariant (`deserialize` routes `card > MAX_CARDINALITY` → bitset, else array).
  A deserialized bitmap always satisfies them by construction; the checks matter
  when `validate` is run on a bitmap built some other way or from corrupted
  bytes where the type bit and the cardinality disagree.
- Keep it allocation-free and non-mutating: all checks are scans over existing
  container memory, reading cardinality fields directly.

### Performance

`validate()` is **O(serialized size)** — a single linear pass, no nested loops.
The dominant term is the bitset popcount (1024-word `@popCount` per bitset
container; worst case an all-bitset bitmap ≈ one full scan of the data). It runs
at multiple GB/s (sequential, hardware popcount), so validating roughly **doubles
deserialize time** in the bitset-heavy worst case and less otherwise. There is no
superlinear behavior. This linear-but-real extra pass is the reason `validate()`
is opt-in and kept off the trusted deserialize fast path.

## Task 3 — `deserializeSafe` convenience wrapper

Included (not optional): if untrusted allocating input is the use case, the API
story should be complete so callers don't have to remember `deserialize` +
`validate` + cleanup themselves. A one-call wrapper:

```zig
/// Deserialize, then validate. On validation failure, frees the partially-built
/// bitmap and returns the error (no leak). For trusted input use `deserialize`.
pub fn deserializeSafe(allocator, data) (DeserializeError || ValidateError)!Self {
    var bm = try deserialize(allocator, data);
    errdefer bm.deinit();
    try bm.validate();
    return bm;
}
```

The only subtlety is the `errdefer bm.deinit()` so a validation failure doesn't
leak the bitmap. Add the `OwnedBitmap` equivalent (`deserializeSafeOwned`) to
match the existing `*Owned` API surface.

## Task 4 — Tests (test-first for the reject cases, like spec 04)

**Reject side (deterministic, write red first):** for each `validate` violation,
build a valid serialized bitmap, corrupt the one relevant field, `deserialize`
(bounds-ok), then assert `validate()` returns the specific error:

1. unsorted keys / duplicate keys (corrupt descriptive-header key bytes)
2. unsorted or duplicate array values (corrupt array data bytes)
3. bitset cardinality mismatch (flip bitset data bytes so popcount ≠ stored card,
   leaving the header cardinality untouched)
4. overlapping/adjacent runs (corrupt run pairs)
5. run cardinality mismatch

Build the bitset fixture as a **real bitset** (scattered adds / `fromSorted` of
>4096 non-contiguous values / `test_gen` `dense`), not `addRange`, since
contiguous ranges store as runs (same caveat as spec 04). Build run fixtures via
`addRange` + `runOptimize`.

**`size > 65536` cap (Task 1):** assert `expectError(error.InvalidFormat,
deserialize(bad))`. Treat this as a **hardening test, not a "the old tree
accepted it" reproducer** — a crafted oversized-`size` buffer may, on the pre-fix
tree, fail earlier via EOF or a failed giant allocation rather than by being
accepted, so don't require demonstrating prior acceptance (unlike the semantic
reproducers above). Just assert the cap rejects it cleanly.

**Accept side:** every valid bitmap passes. Round-trip a `test_gen` bitmap for
each profile (`sparse`/`dense`/`full`/`runs`/`single`/`boundary`),
run-optimized and not, through `serialize` → `deserialize` → `validate()` and
assert no error. These are pure rawr, no CRoaring needed.

**Test placement:** `validate()` semantic tests (reject reproducers + accept
side) go in `bitmap_tests.zig` since `validate()` is a `RoaringBitmap` method;
the `size > 65536` deserialize-time cap test goes with the other serialize tests
in `serialize.zig`.

**Differential (optional, nice-to-have):** expose
`roaring_bitmap_internal_validate` in `vendor/croaring_wrapper.h` and, in
`diff_test.zig` or `validate_croaring.zig`, assert that for the valid accept-side
corpus **both** rawr `validate()` and CRoaring `internal_validate` accept. Exact
agreement on *rejection* is not required (the two implementations may flag
different first-violations), so only the accept side is differentially anchored.

## Acceptance criteria

1. `deserialize` rejects `size > 65536` with `error.InvalidFormat`; `size * 2`
   and `size * 4` are computed in `usize`. Valid round-trips unchanged.
2. `deserialize` stores run-container header cardinality into `rc.cardinality`
   (Task 1 point 3), matching the bitset path; valid round-trips unchanged.
3. `validate()` exists, is allocation-free, non-mutating (`*const Self`), and
   non-`c`, and returns the listed errors for each malformed shape; `.reserved`
   is `unreachable`, not an error variant.
4. Each **semantic** Task 4 reject reproducer was shown to fail against the
   pre-`validate` tree (the malformed bitmap deserializes "successfully") and now
   returns the specific `ValidateError`. The `size > 65536` case is a hardening
   test (clean rejection), not a prior-acceptance reproducer.
5. Every valid `test_gen` bitmap (all profiles, run-optimized and not) passes
   `validate()` (accept side).
6. `deserializeSafe` (and `deserializeSafeOwned`) free the bitmap on validation
   failure with no leak — confirm under the leak-checking test allocator.
7. `zig build test`, `zig build validate`, `zig build difftest` all pass.

## Out of scope

- Any further work on the frozen path (done in spec 04).
- "Repairing" a malformed bitmap (sorting keys, recomputing cardinalities).
  `validate()` reports; it does not fix.
- Streaming/`anytype`-reader-specific bounds concerns beyond the `size` cap — the
  reader contract already errors on EOF.

## Sequencing

Task 1 first (trivial, lands regardless). Then Task 4's reject reproducers (red)
→ Task 2 `validate()` (green). Accept-side tests and the optional Task 3 wrapper /
differential last. Single pass; no chunking expected.
