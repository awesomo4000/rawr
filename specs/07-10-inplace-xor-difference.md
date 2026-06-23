# Spec 07-10: In-place container XOR / difference (finish the in-place pattern)

Post-umbrella perf fix. **Behavior-preserving** — no API/result change. Closes the
flip / removeRange wide-range residual from `07-09`, and the broader structural
gap: in-place XOR and difference re-allocate every matched container instead of
mutating in place.

## Root cause (confirmed by trace)

`07-09` made the word-level helpers (`setRange`/`clearRange`/`toggleRange`) and
wired `setRange` into **union**-in-place — but XOR-in-place and difference-in-place
were left routing through the **allocating** container ops:

- `bitwiseOrInPlace` → `containerUnionInPlace`, with an `is_same` check; bitset×run
  hits `bitsetUnionRunInPlace` (mutates words via `setRange`, returns the same
  pointer → no copy, no free). **Genuinely in-place.**
- `bitwiseXorInPlace` → **`containerXor` (allocating)** — always allocs a fresh
  container + frees the old. No `containerXorInPlace`.
- `bitwiseDifferenceInPlace` → **`containerDifference` (allocating)** — same.

So `flip` (= `clone` + `bitwiseXorInPlace(mask)`) copies each flipped chunk
*twice* (once in clone, once in `bitsetXorRun`), and `removeRange`
(= `bitwiseDifferenceInPlace(mask)`) does likewise. CRoaring copies once and
negates in place. This is just the unfinished half of `07-09` — union got the
in-place treatment; xor/diff didn't.

## Task 1 — `containerXorInPlace`

Mirror `containerUnionInPlace` (`container_ops.zig`). Mutate `a` in place where the
type allows, returning the **same** container pointer; only allocate a new
container when a type change forces it (caller detects via `is_same` and frees the
old — see Task 3).

In-place cases (the valuable ones — `a` is a bitset, which is exactly flip's
cloned-dense case):
- **bitset × run** → `bitsetXorRunInPlace`: `toggleRange(run.start, run.end())`
  per run on `a`'s words (the `toggleRange` from `07-09`). Same pointer.
- **bitset × bitset** → in-place word XOR on `a`'s words (the existing
  `simdBitsetOp(.xor, ...)` already XORs into `dst` in place).
- **bitset × array** → toggle each array bit in `a`'s words.

Fallbacks (allocating, fine — not the hot path): `array × *` and `run × *` may
grow/convert; route them to the existing allocating `containerXor`.

**Demotion / empty:** after an in-place bitset mutation, recompute cardinality
(or leave `-1` lazy, as rawr bitsets allow). If the result is now
`≤ MAX_CARDINALITY`, demote to an array (allocates → new pointer, `is_same` false).
If it's empty, that's handled by the caller's drop logic (Task 3) — make sure the
container is freed exactly once in that case.

## Task 2 — `containerDifferenceInPlace`

Same shape, for `a \ b`:
- **bitset × run** → `bitsetDifferenceRunInPlace`: `clearRange(run.start,
  run.end())` per run on `a`'s words (the `clearRange` from `07-09`).
- **bitset × bitset** → in-place `andnot` on `a`'s words (`simdBitsetOp(.andnot,
  ...)` already does this in place).
- **bitset × array** → clear each array bit in `a`'s words.
- `array × *` / `run × *` → allocating `containerDifference` fallback.
- Demotion/empty as in Task 1.

## Task 3 — Wire into the bitmap in-place ops

In `bitwiseXorInPlace` and `bitwiseDifferenceInPlace`, replace the
`containerXor`/`containerDifference` call in the both-key branch with the new
in-place variant, using the **exact `is_same` pattern** `bitwiseOrInPlace` already
has:

```zig
const result = try ops.containerXorInPlace(self.allocator, old_container, other_container);
const result_tp = result.toTagged();
const is_same = (@as(u64, @bitCast(result_tp)) == @as(u64, @bitCast(self.containers[i])));
if (!is_same) { old_container.deinit(self.allocator); owned[k] = true; }
else { owned[k] = false; }
// ...then the existing non-empty-result check / drop
```

**Ownership subtlety (call out):** XOR/difference can produce an **empty** result
that the both-branch drops. When the result was produced *in place* (`is_same`),
the container to free on drop is the same pointer as `self.containers[i]` — free it
**exactly once** (don't both drop-free it and let the `errdefer`/owned-tracking
free it again). Get this right or it's a double-free; the existing OR path doesn't
hit it because union never empties a container.

## Verification

- **Behavior-preserving — no test changes.** `diff_test` already covers
  `bitwiseXorInPlace`/`bitwiseDifferenceInPlace` (in-place == allocating
  cross-check across the 9-pair matrix), plus `flip` and `removeRange`. It must
  stay green unchanged — that's the gate. Pay attention to the in-place==allocating
  assertions and the empty-result/ghost-container cases.
- **Bench:** re-run `flip wide range` and `removeRange wide` — both should drop to
  ~parity with CRoaring (clone-copy + in-place toggle, like CRoaring's copy +
  negate). Record before/after. Confirm the other in-place set-op rows and the
  9-pair matrix are unregressed (leak-checked).

## Acceptance criteria

1. `containerXorInPlace` / `containerDifferenceInPlace` exist, mutating `a` in
   place for the bitset-as-`a` cases (run via `toggleRange`/`clearRange`, bitset
   via `simdBitsetOp`, array via per-bit), allocating fallback otherwise; demotion
   and empty handled.
2. `bitwiseXorInPlace` / `bitwiseDifferenceInPlace` use them with the `is_same`
   pattern; empty-result drop frees the in-place container exactly once.
3. `zig build test`/`validate`/`difftest` pass **unchanged** (behavior preserved).
4. `flip wide range` and `removeRange wide` benched to ~parity; before/after
   recorded; no regressions on other set-op rows.
5. No leaks / double-frees.

## Notes

- Completes `07-09`'s pattern symmetrically (union/xor/diff all in-place now).
- **Fully obsoletes `07-03` Task 1b** — flip needs no direct-negation rewrite once
  XOR-in-place is genuinely in-place.
- Broader than flip/removeRange: speeds *all* in-place XOR and difference.
