<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 07-09: Word-wise bitset×run container ops (perf)

Post-umbrella perf fix. **Behavior-preserving** — no API change, no result change.
Fixes the `flip wide range (dense)` benchmark (rawr 1.54 ms vs CRoaring ~0) and a
matching latent `removeRange` wide-range slowness.

## Root cause

The bitset×run container ops toggle **bit-by-bit** over run ranges instead of
operating on whole words. `bitsetUnionRunInPlace` already shows the right pattern
(uses `BitsetContainer.setRange` for word-level fills); the others never got it:

- `bitsetXorRun` — `while v<=run.end : v+=1` with `contains`+`add`/`remove` per
  value → **the flip 1.54 ms** (flip = clone + addRange mask + `bitwiseXorInPlace`,
  and the mask's full chunks are run containers, so it hits this path).
- `bitsetDifferenceRun` — bit-by-bit `remove` → **`removeRange` wide-range** (same
  identity: `removeRange` = `bitwiseDifferenceInPlace` of an addRange mask).
- `bitsetIntersectRun` — bit-by-bit.
- `bitsetUnionRun` (allocating) — bit-by-bit, even though its in-place sibling is
  word-wise.

For a full-chunk run that's 65,536 iterations where ~1,024 word ops would do —
the same O(bits)-vs-O(words) shape as the rangeCardinality popcount fix.

Diagnosis confirmed by reading the code; the flip bench was confirmed fair (both
sides flip `[100000, 650000]` on the same dense bitmap, both consume the result).

## Task 1 — Word-level range helpers on `BitsetContainer`

`setRange(start, end)` already exists (word-level set). Add its siblings:

```zig
pub fn clearRange(self: *Self, start: u16, end: u16) void   // AND-NOT the range words
pub fn toggleRange(self: *Self, start: u16, end: u16) void  // XOR the range words
```

Each is O(words): whole-word op for the interior `[first_word+1 .. last_word]`,
masked boundary words for the partial first/last word. Reuse the boundary-mask
logic from `bitsetRangeCardinality`/`bitRangeMask` — **mind the bit-63 mask trap**
(don't shift by 64). Match `setRange`'s cardinality convention (it leaves
`cardinality = -1`; do the same so callers recompute once).

## Task 2 — Rewrite the run paths word-wise

- **`bitsetUnionRun`** (allocating): `@memcpy(result.words, bc.words)` then
  `setRange(run.start, run.end())` per run — matching `bitsetUnionRunInPlace`.
- **`bitsetDifferenceRun`**: `@memcpy` then `clearRange` per run.
- **`bitsetXorRun`**: `@memcpy` then `toggleRange` per run.
- **`bitsetIntersectRun`**: keep only bits inside runs — zero the result, then for
  each run OR in `bc.words` masked to `[start,end]` (word-wise copy-within-range).
  (Equivalently: copy `bc`, then `clearRange` the complement of the runs — pick
  whichever is cleaner.)

Each then recomputes cardinality once (`computeCardinality`, one popcount pass) —
keep the existing trailing recompute and the existing
`bitsetToArrayOrRun`/array-demotion tail so result typing is unchanged.

## Verification

- **Behavior-preserving — no test changes.** `diff_test` already exercises all
  four ops against CRoaring (run operands across the 9-pair matrix, plus the
  `flip` and `removeRange` cases). It must stay green unchanged; that's the
  correctness gate.
- **Bench:** re-run `flip wide range (dense)` (expect it to drop from ~1.54 ms
  toward ~CRoaring), and **add a `removeRange wide range` bench row** (it shares
  the same fix). Record before/after.
- Leak-checked (the allocating paths still `@memcpy` + build one result; no new
  temporaries).

## Acceptance criteria

1. `clearRange`/`toggleRange` exist on `BitsetContainer`, O(words), correct at the
   bit-63 boundary.
2. `bitsetXorRun`/`bitsetDifferenceRun`/`bitsetIntersectRun`/`bitsetUnionRun` are
   word-wise (no per-value loop over run ranges).
3. `zig build test`/`validate`/`difftest` pass **unchanged** (behavior preserved).
4. `flip wide range` bench improved to ~CRoaring; a `removeRange wide range` row
   added and likewise improved; before/after recorded.
5. No leaks.

## Notes

- **Obsoletes `07-03` Task 1b** (direct per-container flip negation) — the flip
  XOR-mask identity is fast enough once the primitive is word-wise; no rewrite of
  flip itself needed.
- Broader than flip: speeds any op mixing bitset + run containers (common once
  run-optimized data meets dense data).
