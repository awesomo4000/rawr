<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 21: Skewed `andCardinality` — the last real parity gap

**Diagnosis first, then a conditional fix.** After spec 20a corrected the parity board,
skewed `andCardinality` is the **one remaining real gap** — rawr is at or ahead of CRoaring
everywhere else. It is a genuine algorithm/kernel gap (the op does **not allocate** in its
timed path, so it never carried the broad-harness allocator-history penalty).

**Size honesty up front:** this is a ~6 µs, ratio-only target — isolated **1.46x**
(rawr-SMP 0.020 [0.018, 0.021] ms vs CRoaring 0.014 [0.012, 0.014] ms, ranges
non-overlapping so the ratio is real). We scope it to *finish parity cleanly*, not because
it is a hot path. Effort is proportionate: a cheap diagnosis, and a fix only if it is
reachable and generalizes.

## What the op is

`andCardinality(a, b)` counts `|a ∩ b|` without materializing the result. The "skewed" case
is two **array** containers of very different cardinalities, which dispatches to the
count-only galloping kernel **`intersectCardGallop`** (`src/array_kernels.zig`), selected by
the per-arch skew crossover thresholds established in spec 11 (scalar 64, x86 12, **NEON
40** — NEON is the M4 host). CRoaring has its own skewed-array cardinality path.

## Phase 1 — Diagnosis (fresh-process, no preselected cause)

Following the 20a discipline: confirm and attribute in an **isolated fresh-process focused
executable**, five runs, median + range — never off a broad-harness number.

1. **Does 1.46x generalize, or is it one corpus point?** Sweep the skew ratio (small-vs-large
   array cardinality, e.g. 1:8, 1:64, 1:512, 1:4096) and the small-side size. Report
   rawr-SMP vs CRoaring across the sweep. A gap only at the bench's single skew point is a
   very different result from a gap across the range.
2. **If it generalizes, attribute where rawr loses.** Candidate territory — measured, not
   assumed:
   - the **`intersectCardGallop` kernel** itself (galloping search + count) being looser than
     CRoaring's skewed-cardinality kernel;
   - the **count-only crossover threshold** — the NEON skew threshold (40) was tuned for the
     *write* kernel (which pays compaction); the count-only variant has no compaction, so its
     SIMD-vs-gallop crossover may sit elsewhere, meaning rawr may be on the wrong side of the
     dispatch for some skews;
   - the **SIMD count path** vs galloping at the measured ratios.
   Inspect the generated code for the galloping search / count loop where useful; report
   absolute medians + ranges with a named residual, not forced-100%.

Phase 1 stands alone: even if no fix follows, "the 1.46x is / isn't general, and here's
where it comes from" is the deliverable.

## Phase 2 — Fix (conditional on Phase 1)

Only if Phase 1 finds a reachable, generalizing improvement — e.g. a tighter count-only
kernel or a re-tuned count-only crossover threshold (kept per-arch, and kept separate from
the write-kernel threshold if they differ). Constraints:

- **Correctness:** `andCardinality` result unchanged (`== |a ∩ b|`); the existing array-kernel
  differential coverage (`bench_aa` byte/■count cross-check across gallop / merge / SIMD) stays
  green, and the write-kernel dispatch is untouched unless the diagnosis explicitly justifies
  re-tuning it too.
- **No regression** at other skew ratios or on the balanced-array kernels (spec 11 numbers).
- Allocator-independent (count-only path), so this is pure kernel/threshold work.

## Measurement / GO

- Isolated fresh-process focused executable; `ReleaseFast`, native CPU, spec-16 M4 host; five
  process runs, median + range. Report SMP (default) explicitly; libc only if it matters.
- **Phase 1 GO:** the 1.46x is characterized across the skew sweep and attributed.
- **Phase 2 GO (if attempted):** the isolated skewed `andCardinality` ratio moves toward
  parity across the sweep, no regression elsewhere, differential green. Given the ~6 µs
  absolute, a partial improvement or a documented "intrinsic at this size" is an acceptable
  terminal outcome — do not over-invest.

## NO-GO

- The 1.46x is a single-skew-point artifact that does not generalize → record it, parity is
  effectively complete, stop.
- The gap is real but intrinsic to a count-only galloping kernel at ~6 µs with no clean
  improvement → document as an explained residual and stop.

## Estimate

S for Phase 1. Phase 2 is S–M and only if the diagnosis points to a reachable, generalizing
fix (a threshold re-tune is small; a new kernel is larger).
