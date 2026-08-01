<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 33: Fused N-way bitset accumulation for `orMany` (E2)

Campaign: [31-structural-parity-campaign.md](31-structural-parity-campaign.md) (Wave 1). Close
**orMany (mixed) 1.248x** — a **compute/bandwidth** gap, not a top-level allocation gap.

**Parity is a hard requirement** — closes at ≤ 1.10x; a partial is adopted by owner judgement
(spec-30 policy) and stays open.

## The gap is accumulation, not allocation

Prior attribution: **~14.18 of 14.71 µs is mixed-container accumulation**, not top-level allocation.
The corpus maps **multiple bitset inputs to one output key**; today rawr streams the destination
**once per input bitset** (K passes over the 1024-word destination for K inputs → K loads+stores of
the destination). A **word-major N-way OR** loads each input word, reduces in registers, and stores
the destination **once** — cutting destination memory traffic ~K-fold.

## Establish the bitset share first (gating)

The word-major kernel **only helps the bitset share** of accumulation. Before building it:

- **Split accumulation time by source type (array / bitset / run).** If the bitset share is small,
  the kernel cannot close the gap — record and stop.
- **Assert bitset multiplicity per output key** — word-major only helps keys receiving **multiple**
  bitsets; a one-bitset-per-key corpus gains nothing. Report the multiplicity distribution.

## Cells

1. **Baseline** — zero destination + input-major accumulation (current).
2. **First-bitset seed** — clone the first bitset instead of zero-then-OR it.
3. **Word-major N-way OR** per key — one destination store per word across all bitset inputs.
4. **Seed + word-major** — first-bitset seeding combined with the word-major loop.
5. **Bitset-only ceiling cell** — the maximum the kernel could recover (bitset inputs only), measured
   **before** the full mixed-corpus cell, to bound the achievable gain.

## Pointer-collection discipline

**Pin how the per-key bitset pointers are gathered** for word-major traversal — a **fresh allocation
per output key** to hold the pointer list could **erase the gain**. State the mechanism (e.g. a
reused scratch pointer buffer sized once) and **include collection overhead inside the full
mixed-corpus cell's timing** — it may **not** be silently excluded.

## Correctness

- **Input immutability** — inputs are read, never mutated; assert source bitmaps unchanged.
- **Unknown-cardinality handling** — the word-major result's cardinality is recomputed/marked
  consistently with the baseline (lazy `-1` where the baseline would be).
- **Representation-identical output** — set-identity + CRoaring differential (and byte-identity via
  `serialize` where defined) across the mixed corpus, including keys with 1 vs many bitsets and
  mixed array/bitset/run inputs.
- **OR-specific — do not touch `xorMany`** (already well ahead).

## Measurement / gates

- **Both hosts, SMP, canonical protocol** (3 warmup / 21 timed, five process medians + full range),
  vs one CRoaring reference per host. E2 owns its own bench module (no shared-file edits).
- **Board gate + spec-28 layout exception** on any production adoption; **Zen 4 policy** (spec 30);
  **one architecture-neutral shape**.
- **Rebaseline note:** E2 accesses container representations E1 may change — if E1 adopts first
  (Wave 2), **re-measure E2 after rebasing** onto the accepted E1 state before its board gate.

## Acceptance

- **Phase 1 GO:** bitset share and per-key multiplicity reported; bitset-only ceiling establishes the
  achievable gain; the four accumulation cells timed both hosts; collection overhead counted;
  input-immutability + differential green. If the ceiling cannot reach ≤ 1.10x, record and stop
  (no production change).
- **Phase 2 (if the ceiling justifies it):** the word-major shape closes orMany to **≤ 1.10x M4 SMP**
  (or a beneficial partial adopted by owner judgement, row stays open), Zen 4 within noise,
  representation-identical output, board gate held.
- `zig build test`; `zig build difftest`; `ReleaseSafe` / `ReleaseFast` green; canonical
  `run-compare-bench.sh` both hosts on adoption; `docs/parity-measurement.md` updated.

## Proposed chunk plan (confirm at review)

- **`33-00`** — attribution (source-type split, per-key multiplicity) + the five cells including the
  bitset-only ceiling, both hosts; collection overhead counted; no production change. Decides
  whether the ceiling justifies proceeding.
- **`33-01`** — production word-major kernel (conditional on `33-00`): identity, input-immutability,
  board gate, ship the winning shape.

## Estimate

M for `33-00` (attribution + five cells + ceiling, two hosts). S–M for `33-01` (kernel + correctness
+ board gate).
