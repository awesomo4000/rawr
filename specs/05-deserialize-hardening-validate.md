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

This is a behavior-preserving change for all valid input (valid `size` is always
`<= 65536`); it only adds rejection of out-of-range `size`.

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
     within `u16` range; `sum(length + 1)` over all runs equals the run
     container's cardinality (`RunCardinalityMismatch`). Note rawr stores run
     cardinality lazily (`cardinality == -1` until computed) — compute it for the
     check without persisting if you want to keep `validate` non-mutating, or
     allow it to populate the cache (document which).

Rationale notes for the implementer:
- The array-vs-bitset cardinality-range checks encode rawr's own representation
  invariant (`deserialize` routes `card > MAX_CARDINALITY` → bitset, else array).
  A deserialized bitmap always satisfies them by construction; the checks matter
  when `validate` is run on a bitmap built some other way or from corrupted
  bytes where the type bit and the cardinality disagree.
- Keep it allocation-free: all checks are scans over existing container memory.

## Task 3 (optional) — `deserializeSafe` convenience wrapper

A one-call wrapper for the untrusted-allocating-input case:

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
leak the bitmap. Add the `OwnedBitmap` equivalent (`deserializeSafeOwned`) only if
the owned API already mirrors the others.

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
6. `size > 65536` rejected by **Task 1** at `deserialize` time
   (`expectError(error.InvalidFormat, deserialize(...))`)

Build the bitset fixture as a **real bitset** (scattered adds / `fromSorted` of
>4096 non-contiguous values / `test_gen` `dense`), not `addRange`, since
contiguous ranges store as runs (same caveat as spec 04). Build run fixtures via
`addRange` + `runOptimize`.

**Accept side:** every valid bitmap passes. Round-trip a `test_gen` bitmap for
each profile (`sparse`/`dense`/`full`/`runs`/`single`/`boundary`),
run-optimized and not, through `serialize` → `deserialize` → `validate()` and
assert no error. These are pure rawr — put them in `bitmap_tests.zig` /
`serialize.zig` tests, no CRoaring needed.

**Differential (optional, nice-to-have):** expose
`roaring_bitmap_internal_validate` in `vendor/croaring_wrapper.h` and, in
`diff_test.zig` or `validate_croaring.zig`, assert that for the valid accept-side
corpus **both** rawr `validate()` and CRoaring `internal_validate` accept. Exact
agreement on *rejection* is not required (the two implementations may flag
different first-violations), so only the accept side is differentially anchored.

## Acceptance criteria

1. `deserialize` rejects `size > 65536` with `error.InvalidFormat`; `size * 2`
   and `size * 4` are computed in `usize`. Valid round-trips unchanged.
2. `validate()` exists, is allocation-free and non-`c`, and returns the listed
   errors for each malformed shape.
3. Each Task 4 reject reproducer was shown to fail against the pre-`validate`
   tree (i.e. the malformed bitmap deserializes "successfully") and now returns
   the specific `ValidateError`.
4. Every valid `test_gen` bitmap (all profiles, run-optimized and not) passes
   `validate()` (accept side).
5. `deserializeSafe` (if implemented) frees the bitmap on validation failure
   (no leak — confirm under the leak-checking test allocator).
6. `zig build test`, `zig build validate`, `zig build difftest` all pass.

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
