<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 07-08: Consolidate the two-way key-merge-join (refactor)

Closing piece of the [parity umbrella](07-parity-inventory.md). **Not a feature —
a pure, behavior-preserving refactor.** Now that every merge variant exists
(parity is complete), collapse the two-way container-list merge that's been
copy-pasted across the set ops into one comptime-parameterized helper.

## The duplication

The same two-way merge walk — step both sorted `keys` arrays; for each key handle
**A-only / B-only / both** — is repeated across:

- **Allocating** (build a result): `bitwiseOr`, `bitwiseAnd`, `bitwiseXor`,
  `bitwiseDifference`.
- **Cardinality** (accumulate a `u64`): `andCardinality`, `orCardinality`,
  `xorCardinality`, `differenceCardinality`.
- **In-place** (mutate `self`): `bitwiseOrInPlace`, `bitwiseAndInPlace`,
  `bitwiseXorInPlace`, `bitwiseDifferenceInPlace`.

Per-op policy is the only thing that varies:

| op | A-only | B-only | both |
|----|--------|--------|------|
| union (or) | keep A | keep B | `containerUnion` |
| intersection (and) | skip | skip | `containerIntersection` (drop empty) |
| xor | keep A | keep B | `containerXor` (drop empty) |
| difference (andnot) | keep A | skip | `containerDifference` (drop empty) |

## Goal / scope

- **Primary:** unify the **4 allocating** ops under one comptime-op merge, and the
  **4 cardinality** variants under one comptime-op merge. These are the clearest,
  highest-duplication wins (the cardinality four are near-identical, per `07-01`).
- **Secondary — default to leaving the in-place four alone.** Only unify them if
  the abstraction comes out *obviously* simpler. Their ownership bookkeeping (the
  `owned[]` tracking + `errdefer` in OR/XOR differs from AND/DIFF) is different
  enough that forcing them into the first pass is more risk than payoff. Expect to
  skip them.
- **Out of scope (do not touch):** the k-way `manyMerge`/`foldManyKey`; the 2-way
  lazy `lazyMergeTwo` (another tempting-but-different nearby abstraction — leave
  it); the 9-pair `containerX` dispatch in `container_ops.zig` (inherent, clear);
  the `cached_cardinality` handling (consistent, correct).

## Approach

A **comptime-generic** merge so it inlines to exactly what's there today — zero
runtime cost. Parameterize by a comptime op enum (and/or per-branch policy), and
have the per-branch behavior be comptime-known so the compiler specializes each
instantiation. The allocating and cardinality consumers differ in their "sink"
(append-a-container vs accumulate-a-scalar), so model that as a comptime
sink/visitor or two thin wrappers over a shared key-walk — whichever reads
cleaner.

**Readability is the actual goal** — if a comptime abstraction ends up *harder*
to read than the duplication, it's not worth it. In particular:
- `bitwiseDifference` is the only *allocating* op still on the asymmetric walk
  (`while i < size`, advance `j`) instead of the symmetric `while i<size and
  j<size` + tail-drain. **Convert it to the common A-only/B-only/both walk** —
  it's probably clearer and lets it share the helper. (`differenceCardinality` is
  already in the symmetric shape, so the cardinality family needs no such change.)
- Don't over-generalize: covering the 4 allocating + 4 cardinality cleanly is the
  win; the in-place four can stay as-is if unifying them hurts clarity.

## Hard constraints (this is the hottest code in the repo)

1. **Behavior-preserving** — identical results. The existing `diff_test`
   coverage (every set op + cardinality variant against CRoaring, 9-pair matrix,
   randomized loop) is the correctness gate; it must stay green with **no test
   changes**.
2. **No perf regression** — the `bitwiseAnd` perf ghost lived in exactly this
   code. Run the full `bench-compare` before and after; every set-op and
   cardinality row must be within noise of its pre-refactor ratio. A comptime
   merge *should* be free; **prove it with the bench**, don't assume.
3. **Allocation profile unchanged** — same allocations as before (no new temporary
   per merge); leak-checked.
4. **Preserve `bitwiseAnd`'s scratch-buffer / optimization path exactly.** AND has
   op-specific tuning tied to the sparse-AND perf work; it's easy to flatten into
   a generic allocating merge and silently lose. Keep its fast path intact (or
   special-case AND out of the generic if needed) — the `bench-compare` AND rows
   are the check.
5. **Leak-safe container production** — in the shared allocating helper, a
   newly-created result container must be freed (`errdefer`/explicit) if the
   subsequent `appendContainer` fails, so an alloc failure mid-merge doesn't leak.
   This is achievable without changing the successful-path allocation profile.

## Method

Refactor **incrementally**, one family at a time, running `difftest` +
`bench-compare` after each step, so any regression is attributable to the step
that caused it:
1. Cardinality four → shared merge. (Lowest risk: no allocation, pure accumulate;
   `differenceCardinality` already symmetric.)
2. Allocating four → shared merge — **preserving AND's scratch/fast path** and
   converting `bitwiseDifference` to the symmetric walk.
3. In-place four → **expected to stay as-is**; only touch if the pattern comes out
   obviously cleaner.

## Acceptance criteria

1. The two-way merge walk exists **once** (per consumer shape) instead of copied
   across the set ops; the per-op policy is the only specialization.
2. `zig build test`, `zig build validate`, `zig build difftest` pass **unchanged**
   (no test edits needed — behavior identical).
3. `bench-compare` shows every set-op / cardinality row within noise of the
   pre-refactor numbers (record before/after in the commit).
4. No new allocations or leaks; allocation profile matches pre-refactor.
5. Net line reduction in `bitmap.zig`'s set-op section, and the result reads at
   least as clearly as the duplicated version (the point of the exercise).

## Notes

- This closes the `07` parity umbrella. After it, optional follow-ups are the
  Tier-3 conveniences (`clear`/`add_offset` if wanted) and the parked
  `08-fuzzing`.
- Next planned effort: the **API-design / ergonomics pass** (separate from this) —
  a consolidated set-op core makes that cleaner to build on.
