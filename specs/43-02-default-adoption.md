<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 43-02: Default adoption and Gate 2

Toplevel: [43-address-ordered-lazy-construction.md](43-address-ordered-lazy-construction.md).
Gated on: [43-01](43-01-diagnostic-production-path.md) clearing **Gate 1**.

This chunk changes the default. It is where the canonical row either closes or does not.

## 1. Switch the default

`ConstructionMode` defaults to `.batched_sorted` for `op == .bor`. No public API is added — the mode
remains reachable only through the internal export.

## 2. Retain the old baseline as a diagnostic row

**Do this in the same change as the switch.** Once the default is sorted, `lazy-or-construction` measures
the *candidate*, so the pre-adoption baseline must survive under a separate diagnostic row name or the
negative control loses its reference the moment the default flips.

| Row | After adoption |
| --- | --- |
| `lazy-or-construction` | **sorted default** — the board row, gates |
| retained baseline row | pre-adoption interleaved path, for the §4 negative control |
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

## 4. Negative control on the mechanism

With the default adopted, run the production path with sorting disabled (via the retained baseline row and
the `.batched_unsorted` arm) and **confirm the gap returns**.

A win that survives disabling the lever was never the lever — it would mean the improvement came from
batching, from layout, or from the measurement, and the spec's causal claim would be false even though the
number improved.

## Acceptance

- Default is `.batched_sorted` for `op == .bor`; no public API added; `lazyXor` still byte-identical to
  baseline.
- Pre-adoption baseline retained as a named diagnostic row, in the same change as the switch.
- **GATE 2:**
  - canonical `lazy-or-construction` **≤1.10x on M4**, measured fresh post-adoption;
  - combined `lazyOr+repair` does not regress;
  - **no other board row moves beyond the 5% layout tolerance**;
  - Zen 4 not regressed;
  - **libc not regressed — a libc regression is a STOP**, not a fallback to opt-in within this spec.
- **Negative control passes:** disabling sorting returns the gap. Record the output.
- All four suites green — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
  `check-package`.
- **Outcome recorded on the umbrella (spec 31)** either way. If the row closes, say so with the measured
  ratio and the date. If it does not, record the measured result and the reason, and do **not** report it
  as closed or as "at parity when enabled" — that phrasing belongs to opt-in outcomes like spec 39-01, and
  this spec has no opt-in path.

## Estimate

**S/M** — the change itself is small; the measurement and the whole-board rerun are the work.
