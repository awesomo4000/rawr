<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 43-02: Default adoption and Gate 2

Toplevel: [43-address-ordered-lazy-construction.md](43-address-ordered-lazy-construction.md).
Gated on: [43-01](43-01-diagnostic-production-path.md) clearing **Gate 1**.

This chunk changes the default. It is where the canonical row either closes or does not.

## 1. Switch the default

`ConstructionMode` defaults to `.batched_sorted` for `op == .bor`. No public API is added — the mode
remains reachable only through the internal export.

## 2. Retain the old baseline as a diagnostic row — by REPURPOSING, not adding

**Row count stays at 42.** `43-01` raised the guards in `bench_parity_worker.zig:778` and
`run-compare-bench.sh:72` from 40 to 42. This chunk **repurposes
`lazy-or-construction-batched-sorted` into `lazy-or-construction-baseline`** rather than adding a row:
once the default is sorted, the board row *is* the sorted arm, so a separate sorted diagnostic row would
be redundant — measuring the same code twice under two names. No guard change in this chunk.

**Do this in the same change as the switch.** Once the default is sorted, `lazy-or-construction` measures
the *candidate*, so the pre-adoption baseline must survive under a separate diagnostic row name or the
negative control loses its reference the moment the default flips.

| Row | After adoption |
| --- | --- |
| `lazy-or-construction` | **sorted default** — the board row, gates |
| `lazy-or-construction-baseline` | pre-adoption interleaved path, for the §4 negative control |
| `lazy-or-construction-batched` | batched, unsorted — kept for attribution |

## 3. Gate 2 is a fresh measurement, not an inference

A Gate 1 pass does **not** predict the canonical row's post-adoption value. Adoption changes what code the
binary contains, and spec 28 established that this alone moves untouched rows — including CRoaring's —
with instruction-identical disassembly. So Gate 2 requires a **full rerun**, not a re-reading of Gate 1.

- **Canonical harness only** (`run-compare-bench.sh`, fresh process per cell, 3 warmup / 21 timed, ≥5
  process medians + full ranges). Never a focused harness: spec 35's read 1.155x where canonical read
  1.727x, purely from SMP preconditioning earlier in the same process.
- **Both hosts**, all three canonical tuples — rawr/SMP, rawr/libc, CRoaring/libc.
- **Whole board**, for spec-28 layout noise. Sub-~1.2x M4 ratios sit at the measurement floor.

## 4. Negative control — three conditions, not "the gap returns"

**Corrected: an earlier draft said disabling sorting must make "the gap return", which contradicts the
three-arm model.** If batching independently helps — which arm 2 vs arm 1 exists to measure — then
disabling *sorting alone* will not restore the full original gap, and a real ordering win would be
falsely invalidated.

The control is three **distinct** conditions — each a different comparison against a different reference.
*(An earlier draft's conditions 2 and 3 were the same comparison stated twice; condition 2 now pins a
**magnitude against Gate 1**, which is what makes it independent.)*

1. **Anchor — `lazy-or-construction-baseline` reproduces its own pre-adoption measurement** within the
   noise policy (≤5% on median, ranges considered). Establishes that the retained path and the harness
   are unchanged, so conditions 2 and 3 are being measured against a stable reference.
2. **Magnitude — `.batched_unsorted` reproduces Gate 1's arm-2 measurement** within the same tolerance.
   This is the claim that the *batching* contribution is what Gate 1 said it was, and that the
   ordering increment attributed in Gate 1 is still the increment being observed.
3. **Direction — the sorted default beats `.batched_unsorted` with non-overlapping ranges.** The
   ordering effect survives adoption and did not evaporate into layout.

All three must hold. Condition 1 validates the reference, condition 2 validates the decomposition
against Gate 1, condition 3 validates the effect itself. Overlapping ranges → rerun; still overlapping →
inconclusive → **NO-GO and rollback**.

## Acceptance

- Default is `.batched_sorted` for `op == .bor`; no public API added; `lazyXor` still byte-identical to
  baseline.
- Pre-adoption baseline retained as `lazy-or-construction-baseline`, **repurposed from
  `lazy-or-construction-batched-sorted`**, in the same change as the switch. **Row count still 42; no
  guard change** in either `bench_parity_worker.zig` or `run-compare-bench.sh`.
- **GATE 2:**
  - canonical `lazy-or-construction` **≤1.10x on M4**, measured fresh post-adoption;
  - combined `lazyOr+repair` does not regress;
  - **no other board row moves beyond the 5% layout tolerance**;
  - Zen 4 not regressed;
  - **libc not regressed — a libc regression is a STOP**, not a fallback to opt-in within this spec.
  - **Decision rules, not impressions:** ≥5 fresh-process medians with full ranges per cell; **≤5% on
    median** is the no-regression threshold; **rerun any ambiguous overlap**; an unresolved difference
    after rerun is **inconclusive → NO-GO and rollback**, never a marginal pass.
- **Negative control passes — all three distinct §4 conditions:** (1) baseline reproduces its own
  pre-adoption measurement within ≤5%; (2) `.batched_unsorted` reproduces **Gate 1's arm-2 measurement**
  within ≤5%; (3) sorted default beats `.batched_unsorted` with non-overlapping ranges. Record the
  output.
- All four suites green — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
  `check-package`.
- **ROLLBACK on Gate 2 failure — mandatory.** If canonical parity, libc, Zen 4, or any whole-board gate
  fails, **restore `.baseline` as the default** and record a NO-GO. The candidate does not stay shipped
  merely because the measurement happens after the implementation; that ordering is an artifact of how
  the work is sequenced, not a reason to ship a failed default. The diagnostic rows and the internal
  dispatch may remain for future work.
- **Outcome recorded on the umbrella (spec 31)** either way. If the row closes, say so with the measured
  ratio and the date. If it does not, record the measured result and the reason, and do **not** report it
  as closed or as "at parity when enabled" — that phrasing belongs to opt-in outcomes like spec 39-01, and
  this spec has no opt-in path.

## Estimate

**S/M** — the change itself is small; the measurement and the whole-board rerun are the work.
