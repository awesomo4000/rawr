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

## Canonical corpus (fingerprint before timing)

The canonical row is `or-many`: **"32 deterministic mixed array, bitset, and run-heavy bitmaps"**,
`batch_count = 128` (`bench_parity_worker.zig`). The diagnostic must **fingerprint this exact
corpus** — count of inputs (32), per-input container-type inventory, the set of output keys, and the
resulting per-key source composition — and **assert the fingerprint** so the new diagnostic **cannot
accidentally construct a different workload** than the shipping row measures.

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

Define the exact per-key fold used by the mixed accumulation for **zero, one, and multiple bitsets
mixed with Array/Run sources**:

- **cursor advancement** — how the per-key merge advances across the sorted inputs and which source
  types feed the word-major path;
- **0 bitsets for a key** → array/run-only path unchanged (word-major not invoked);
- **1 bitset for a key** → seed path (clone-or-adopt the single bitset, then OR the array/run
  remainder) — no multi-input word loop;
- **≥2 bitsets** → word-major N-way OR across the bitset sources, then fold the array/run sources;
- **fallback behavior** — when the key's composition does not qualify (e.g. all-array), fall back to
  the baseline accumulation with no regression.

## Scratch ownership (pin)

Pin the per-key bitset-pointer buffer used for word-major traversal — a **fresh allocation per output
key could erase the gain**:

- **allocation count** (ideally **one** reused buffer for the whole `orMany`, not per key),
- **capacity** (sized once to the max bitsets-per-key over the corpus),
- **lifetime and reuse** (reset, not reallocated, between keys),
- **cleanup** (freed once at the end),
- **OOM behavior** (buffer allocation failure → clean error, no partial result leaked).

**Collection overhead is counted inside the full mixed-corpus cell's timing** — it may **not** be
silently excluded.

## Ceiling → full-ratio projection (pin)

A fast **synthetic bitset-only ceiling does not by itself prove the end-to-end `orMany` row reaches
≤ 1.10x** — the row also pays array/run folding, top-level assembly, and repair. The spec must define
**how the ceiling projects into the complete mixed ratio**: the bitset-share fraction from
attribution × the ceiling's per-share improvement, applied to the measured full-row time, gives the
**projected mixed ratio**. Phase 1 proceeds to production only if that **projected full-row ratio**
(not the synthetic ceiling alone) can reach the gate.

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

- **Phase 1 GO:** corpus fingerprint asserted; bitset share and per-key multiplicity reported;
  bitset-only ceiling **projected into the full mixed ratio** (not the synthetic ceiling alone); the
  four accumulation cells timed both hosts; collection overhead counted; input-immutability +
  differential green. If the **projected full-row ratio** cannot reach ≤ 1.10x, record and stop (no
  production change).
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
