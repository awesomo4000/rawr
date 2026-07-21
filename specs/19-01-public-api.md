<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 19-01: Public `bitwiseOrInPlaceConsume` (gated on 19-00 GO)

Second chunk of [consuming (move) in-place OR](19-consuming-move-union.md). Promotes the
`19-00` prototype to a public method with the full contract, differential coverage, and
exhaustive failure injection.

## Gate — do not start until both hold

1. **`19-00` cleared its gates:** the exact-allocation result held (drop == `2 × moved`,
   zero consuming clones), and the fixpoint union-only bench showed **≥ 15% median**
   improvement over cloning `bitwiseOrInPlace` at the realistic overlap.
2. **Real-driver overlap data exists** — a supplied trace/distribution or driver
   instrumentation establishing the actual unmatched-chunk-key frequency, and that
   frequency is where the 15% gate is met. A synthetic sweep alone does not authorize this
   chunk.

If either fails, this chunk is **not written** — `19-00`'s prototype stays benchmark-only
and no public API ships (the toplevel NO-GO).

## Deliverable

Public **`bitwiseOrInPlaceConsume(self: *Self, other: *Self) !void`**, promoting the
`19-00` algorithm unchanged, with the documented contract:

- Same allocator required (`ptr` + `vtable`) → `error.AllocatorMismatch` before any
  mutation; caller falls back to cloning `bitwiseOrInPlace`.
- Distinct bitmaps required → `error.AliasedOperands` before any mutation.
- Success leaves `other` a valid empty bitmap (`size = 0`, `cached_cardinality = 0`,
  top-level capacity retained).
- Basic-guarantee error semantics per the commit protocol: `other` unchanged on any error,
  `self` valid (possibly with completed matched merges), no leak, no double free.
- Doc comment states all four preconditions/post-states and the exact error guarantee.

## Tests

- **Differential:** result **set-equal** to `bitwiseOrInPlace` and logically to the
  CRoaring oracle, across container-type mixes and the chunk-key overlap range.
- **Exhaustive allocation-failure injection:** sweep every fallible site across the reserve
  and matched-merge steps; after **each** injected failure assert:
  - `other` is unchanged;
  - both bitmaps pass `validate()`;
  - **`cardinality()` on both returns the correct value** (called explicitly — `validate()`
    does not catch a stale bitmap-level cardinality cache);
  - both `deinit` leak-free under a leak-checking GPA.
- **Precondition tests:** `error.AllocatorMismatch` using the **same vtable but different
  allocator state** (e.g. two separate arenas), and `error.AliasedOperands` on `self ==
  other` — both returned **before** any mutation.
- **Post-state reuse:** after a successful consume, `other` is added to, re-validated, used
  in a further operation, and deinited — all clean.

## Acceptance

- Public method matches the contract; all tests above green under `ReleaseSafe` and
  `ReleaseFast`.
- **Additive:** `bitwiseOr` / `bitwiseOrInPlace` semantics and numbers unchanged; existing
  in-place OR stays within **≤ 2% median** of current numbers.
- Full build green; no regression elsewhere in `bench_croaring`.

## Deferred

XOR (`bitwiseXorInPlaceConsume`) is **out of scope** here — revisit only after this ships
and a concrete driver wants it. Difference remains excluded (nothing to move).
