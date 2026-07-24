<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 21: Skewed `andCardinality` — the last real parity gap

**Diagnosis first, then a conditional fix.** After spec 20a corrected the parity board,
skewed `andCardinality` is the **one remaining real gap** — rawr is at or ahead of CRoaring
everywhere else. It is a genuine algorithm/kernel gap (the op does **not allocate** in its
timed path, so it never carried the broad-harness allocator-history penalty).

> **Outcome (2026-07-23) — CLOSED; parity reached across the board.** `21-00` attributed the
> gap to the galloping-count kernel (dispatch at parity, generalizes across ratios/overlaps);
> `21-01`'s fused cached-cursor kernel (commit `5dae1e1`, count-only — write kernels,
> dispatch, and thresholds unchanged) closed it. M4 forced-gallop **75.561 → 43.426
> ns/container = CRoaring's 43.426**; full API both 0.014 ms. Natively validated on x86-64 /
> Zen 4 (rawr 41.958 vs CR 42.755 ns/container — slightly ahead). Empty/singleton/reversed-arg
> tests, `bench_aa`, CRoaring differential, and `ReleaseSafe`/`ReleaseFast` all green on both
> hosts. **With this, rawr is at or ahead of CRoaring across the entire parity board.** Full
> data: [`docs/parity-measurement.md`](../../docs/parity-measurement.md).

**Size honesty up front:** this is a ~6 µs, ratio-only target — isolated **~1.46x**
(rawr-SMP **0.019 [0.018, 0.019] ms** vs CRoaring **0.013 [0.012, 0.014] ms**, ranges
non-overlapping so the ratio is real). We scope it to *finish parity cleanly*, not because
it is a hot path. Effort is proportionate: a cheap diagnosis, a fix only if reachable and
generalizing.

## What the op is, and why threshold selection is ruled out

`andCardinality(a, b)` counts `|a ∩ b|` without materializing the result. The measured
"skewed" case is two **array** containers, **32 × 4096** — a **128:1** ratio, all-hit. It
dispatches to the count-only galloping kernel **`intersectCardGallop`**
(`src/array_kernels.zig`, dispatch ~`:190`), selected by the per-arch skew thresholds from
spec 11 (scalar 64, x86 12, **NEON 40** — NEON is the M4 host).

**A threshold re-tune cannot explain the measured point.** At 128:1, rawr gallops (NEON
threshold ≥ 40) *and* CRoaring gallops (its dispatch, `vendor/roaring.c:6971`, gallops at
ratio > 64). Both are already in their galloping-cardinality kernels, so **threshold
selection is ruled out**. What that does *not* establish is where the 1.46x actually lives:
it could be the galloping kernels, or the surrounding top-level key traversal, container
dispatch, or call overhead. **Phase 1 will determine how much comes from the galloping
kernels versus surrounding traversal and dispatch** — exactly what the two-layer measurement
below separates. Threshold tuning could only help *nearby* boundary cases, never the
32 × 4096 result, so it is not the lever here.

## Phase 1 — Diagnosis (fresh-process, kernel comparison first)

Per the 20a discipline: isolated fresh-process focused executable, five runs, median +
range; never off a broad-harness number.

### Separate full-API cost from direct-kernel cost

The existing benchmark is **not one container pair** — 200 containers per bitmap, 180
matching keys — so its time is bitmap key traversal + dispatch **plus 180 kernel calls**.
Measure both layers:

- **Full API:** `RoaringBitmap.andCardinality` vs public CRoaring `roaring_bitmap_and_cardinality`
  — the five-process median+range result (the number of record).
- **Direct kernel:** rawr's normal dispatch, rawr **forced gallop**, rawr **forced
  SIMD/merge**, and CRoaring's **direct array-cardinality function** — batched over enough
  calls to clear clock-resolution noise, reported as **ns/container**.

This says whether the ~1.46x is in the kernel, the per-key dispatch/traversal, or both.

### Sweep corpus (define precisely — hit distribution matters as much as ratio)

- Exact `(small, large)` cardinality pairs, **all ≤ 4096** so both stay arrays; include the
  original **32 × 4096 all-hit** case.
- Three overlap distributions per pair: **all-hit**, **disjoint**, and **deterministic
  mixed / random-overlap**.
- **Boundary cases** immediately below / at / above each threshold — concrete pairs, all
  ≤ 4096 so both stay arrays:
  - rawr NEON 40: `64×2496` (39:1), `64×2560` (40:1), `64×2624` (41:1);
  - CRoaring 64: `32×2016` (63:1), `32×2048` (64:1), `32×2080` (65:1);
  - original: `32×4096` (128:1).
- **Confirm container representations** (both arrays, expected cardinalities) before any
  timing is accepted.

### Attribute (measured, not assumed)

With both layers and the sweep in hand, name where rawr loses at the original case — the
galloping search/count loop in `intersectCardGallop` vs CRoaring's equivalent — and whether
the gap holds across overlap distributions and ratios or is specific to 32 × 4096 all-hit.
Inspect the generated code for the gallop loop where useful. Report absolute medians +
ranges and ns/container with a **named residual**, not forced-100%.

Phase 1 stands alone: "is 1.46x general, and where does it come from" is the deliverable
even if no fix follows.

## Phase 2 — Fix (conditional on Phase 1)

Only if Phase 1 finds a reachable, generalizing improvement.

- **Primary lever follows Phase 1's attribution:** if the galloping kernel dominates the
  1.46x, a tighter galloping-count kernel; if surrounding traversal/dispatch dominates,
  target that instead. (Threshold selection is already ruled out for the measured point.)
- **Not the lever for 32 × 4096:** a count-only crossover threshold re-tune only helps
  boundary cases, never the original result. If touched, keep it **separate from the
  write-kernel threshold** if they differ.
- **Threshold-change test scope:** if a change touches a **shared kernel or the x86/scalar
  threshold**, it **requires x86 testing**. If only the **measured NEON threshold** changes,
  explicitly **scope it to NEON**.
- **Correctness:** `andCardinality` result unchanged (`== |a ∩ b|`); the array-kernel
  differential coverage (`bench_aa` cross-checks gallop / merge / SIMD kernels for identical
  results) stays green; the write-kernel dispatch is untouched unless the diagnosis
  justifies re-tuning it too.
- **No regression** at other skew ratios or on the balanced-array kernels (spec 11).
  Allocator-independent (count-only path) — pure kernel/threshold work.

## Measurement / GO

- Isolated fresh-process focused executable; `ReleaseFast`, native CPU, spec-16 M4 host; five
  process runs, median + range. Report SMP (default) explicitly; libc only if it matters.
- **Phase 1 GO:** the ~1.46x is characterized across the sweep and both layers, attributed to
  kernel vs dispatch.
- **Phase 2 GO (if attempted):** the isolated skewed `andCardinality` ratio moves toward
  parity at the original case without regression elsewhere, differential green. Given the
  ~6 µs absolute, a partial improvement or a documented "intrinsic at this size" is an
  acceptable terminal outcome — do not over-invest.

## Validation commands

- `zig build test`
- `zig build bench-aa -Dcpu=native && ./zig-out/bin/bench_aa` — build **and run** the
  array-kernel differential + timing (the build step alone does not execute it)
- `zig build difftest` — differential validation (rawr result `== |a ∩ b|` `==` CRoaring
  oracle)
- the new focused five-process runner for the isolated skewed-`andCardinality` result, once
  it is named/added

## NO-GO

- The ~1.46x is specific to 32 × 4096 all-hit and does not generalize → record it, parity is
  effectively complete, stop.
- The gap is real but intrinsic to a count-only galloping kernel at ~6 µs with no clean
  improvement → document as an explained residual and stop.

## Estimate

S for Phase 1. Phase 2 is S–M and only if the diagnosis points to a reachable, generalizing
kernel improvement (a threshold re-tune is small; a new/tighter kernel is larger).
