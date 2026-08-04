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

## Canonical corpus (authoritative generator + expected fingerprint)

The canonical row is `or-many` (`batch_count = 128`). The authoritative generator is
**`initRawrManyBitmaps` / `addManyPatternRawr`** in `bench_croaring.zig` — the diagnostic **calls the
same generator** (or asserts an identical fingerprint), never a re-implementation. Expected
fingerprint:

- **`N_MANY_BITMAPS = 32`** input bitmaps; **6 output keys** (chunks 0–5, `base = chunk << 16`).
- Per bitmap `i`, per chunk, the pattern is `(i + chunk) % 4`: **0** → 128 scattered adds (array);
  **1** → 5000 adds (bitset, > 4096); **2** → `addRange(start, start+12000)` (run); **3** → 4 values
  (tiny array). Bitmaps with `i % 3 == 0` (11 of 32) get `runOptimize()`.
- **Per output key: for fixed chunk, `i` ∈ 0..31 gives 8 of each pattern** before `runOptimize`. The
  post-`runOptimize` per-key type counts are the multiplicity the word-major kernel exploits.

`33-00` **generates and pins the exact post-`runOptimize` per-key type counts** (array / bitset / run
sources per key) — not an approximate "~8 bitsets" — and the diagnostic **asserts that pinned
fingerprint** (32 inputs, 6 keys, exact per-key source composition) so it **cannot drift** from the
shipping row's workload.

## Establish the bitset share first (gating)

The word-major kernel **only helps the bitset share** of accumulation. Before building it:

- **Split accumulation time by source type (array / bitset / run).** If the bitset share is small,
  the kernel cannot close the gap — record and stop.
- **Assert bitset multiplicity per output key** — word-major only helps keys receiving **multiple**
  bitsets; a one-bitset-per-key corpus gains nothing. Report the multiplicity distribution over the
  fingerprinted corpus.

## Cells

1. **Baseline** — zero destination + input-major accumulation (current).
2. **First-bitset seed** — clone the first bitset instead of zero-then-OR it.
3. **Word-major N-way OR** per key — one destination store per word across all bitset inputs.
4. **Seed + word-major** — first-bitset seeding combined with the word-major loop.
5. **Bitset-only ceiling cell** — the maximum the kernel could recover (bitset inputs only), measured
   **before** the full mixed-corpus cell, to bound the achievable gain.

## `foldManyKey` — pin the algorithm

Define the exact per-key fold used by the mixed accumulation. **Distinguish total sources from bitset
sources** — the existing **total-source-count == 1** clone fast path is preserved; bitset
multiplicity is classified **only when multiple sources share the key**:

- **cursor advancement** — how the per-key merge advances across the sorted inputs and which source
  types feed the word-major path;
- **total sources == 1 for a key** → **existing clone fast path unchanged** (copy the single source
  into the result; word-major not invoked), regardless of its type;
- **multiple sources, 0 bitsets** → array/run-only path unchanged (word-major not invoked);
- **multiple sources, 1 bitset** → seed path: **copy/clone the bitset into independently owned
  storage** (inputs are borrowed and immutable — **never adopt source words**), then OR the array/run
  remainder;
- **multiple sources, ≥2 bitsets** → word-major N-way OR across the bitset sources into an
  independently owned accumulator, then fold the array/run sources;
- **fallback behavior** — when the key's composition does not qualify, fall back to the baseline
  accumulation with no regression.

## Scratch ownership (pin)

Pin the per-key bitset-pointer buffer used for word-major traversal — a **fresh allocation per output
key could erase the gain**:

- **allocation count** (ideally **one** reused buffer for the whole `orMany`, not per key),
- **capacity** — sized to **`bitmaps.len`** (the arbitrary input count is the only sound upper bound
  on sources-per-key; **not** the canonical corpus's maximum, since `orMany` accepts arbitrary
  inputs). If a tighter bound is wanted, define an explicit **production prepass** that computes the
  actual max sources-per-key before sizing — but the default upper bound is `bitmaps.len`.
- **lifetime and reuse** (reset, not reallocated, between keys),
- **cleanup** (freed once at the end),
- **OOM behavior** (buffer allocation failure → clean error, no partial result leaked).

**Collection overhead is counted inside the full mixed-corpus cell's timing** — it may **not** be
silently excluded.

## Ceiling → full-ratio projection (pin)

A fast **synthetic bitset-only ceiling does not by itself prove the end-to-end `orMany` row reaches
≤ 1.10x** — the row also pays array/run folding, top-level assembly, and repair. Two-step:

- **Projection = stop-gate (early):** the bitset-share fraction from attribution × the ceiling's
  per-share improvement, applied to the measured full-row time, gives the **projected mixed ratio**.
  If that projection **cannot** reach the gate, stop — do not build the kernel.
- **Direct end-to-end measurement = GO evidence (decisive):** if the projection clears, `33-00`
  builds a **benchmark-only implementation of the winning shape and times it on the complete
  canonical `orMany` row** — including **pointer collection, Array/Run folding, top-level assembly,
  and repair**. This **directly measured full-row ratio**, not the Amdahl projection, is the GO
  evidence, since the candidate can be measured directly rather than extrapolated.

## Correctness

- **Input immutability** — inputs are read, never mutated; assert every source bitmap unchanged
  (serialization identical before/after).
- **Cardinality contract** — the **accumulator is unknown (`-1`) before repair**, but **public
  `orMany` returns a repaired result with known cardinalities**. Assert the returned bitmap's
  per-container cardinalities are known and equal to the baseline's; the transient unknown state is
  internal only.
- **Testable output invariants** — same container kind / cardinality / values as baseline, identical
  portable bytes where `serialize` is valid, + CRoaring differential, across the mixed corpus.
- **Edge cases** — **empty input list, single input, duplicate input pointers** (same bitmap listed
  twice), **zero / one / many bitsets per key**, **mixed array/bitset/run per key**, and the
  **4096 array→bitset conversion boundary** (a key whose OR crosses it).
- **OR-specific — do not touch `xorMany`** (already well ahead).

## Failure injection

- Focused **OOM tests** on the **new scratch buffer** and on a **partially constructed result**
  (mid-key or mid-assembly allocation failure) — valid-or-cleanly-errored, **inputs untouched**, no
  leak.

## Measurement / gates

- **Both hosts, SMP, canonical protocol** (3 warmup / 21 timed, five process medians + full range),
  vs one CRoaring reference per host. E2 owns its own bench module (no shared-file edits).
- **Board gate + spec-28 layout exception** on any production adoption; **Zen 4 policy** (spec 30);
  **one architecture-neutral shape**.
- **Rebaseline note:** E2 accesses container representations E1 may change — if E1 adopts first
  (Wave 2), **re-measure E2 after rebasing** onto the accepted E1 state before its board gate.

## Acceptance

- **Phase 1 GO:** corpus fingerprint (exact post-`runOptimize` per-key type counts) asserted; bitset
  share and per-key multiplicity reported; bitset-only ceiling **projected** as the early stop-gate;
  the four accumulation cells timed both hosts; and — decisively — a **benchmark-only end-to-end
  implementation of the winning shape timed directly on the full canonical `orMany` row** (pointer
  collection + folding + assembly + repair), both hosts; collection overhead counted;
  input-immutability + differential green. If the projection cannot reach ≤ 1.10x, stop before
  building the kernel; **the direct full-row measurement is the GO evidence**, not the projection.
- **Phase 2 (if the ceiling justifies it):** the word-major shape closes orMany to **≤ 1.10x M4 SMP**
  (or a beneficial partial adopted by owner judgement, row stays open), Zen 4 within noise,
  the testable output invariants (same kind / cardinality / values + portable bytes where serialize
  valid), board gate held.
- `zig build test`; `zig build difftest`; `ReleaseSafe` / `ReleaseFast` green; canonical
  `run-compare-bench.sh` both hosts on adoption; `docs/parity-measurement.md` updated.

## Proposed chunk plan (confirm at review)

- **`33-00`** — attribution (source-type split, pinned post-`runOptimize` per-key type counts) + the
  five cells including the bitset-only ceiling **and a benchmark-only end-to-end implementation of
  the winning shape timed on the full canonical `orMany` row**, both hosts; collection overhead
  counted; no production change. The direct full-row number (not the projection) decides whether to
  proceed.
- **`33-01`** — production word-major kernel (conditional on `33-00`): identity, input-immutability,
  board gate, ship the winning shape.

## Estimate

M for `33-00` (attribution + five cells + ceiling, two hosts). S–M for `33-01` (kernel + correctness
+ board gate).
