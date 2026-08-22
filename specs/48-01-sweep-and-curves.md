<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 48-01: The cardinality sweep — Q1 to Q4

Toplevel: [48-tiny-bitmap-cost-measurement.md](48-tiny-bitmap-cost-measurement.md).
Gated on: [48-00](48-00-harness-and-fixtures.md) complete and green.

Answers **Q1, Q2, Q3, Q4**. Q5 and the recommendation are `48-02` — and the recommendation needs both,
so this chunk **states no verdict**.

## 1. What to run

`Mtiny` across the sweep **0, 1, 2, 4, 6, 8, 12, 16, 20, 32, 64, 128**, for **all three shapes**
(`localized`, `spread`, `one-per-container`).

Cells: **rawr/SMP, rawr/libc, CRoaring/libc**, and the **heap-owned plain-list reference** under the
matching allocators. Both hosts, **`ReleaseFast`, native CPU**.

Fresh process per cell, warmup then timed iterations, **≥5 process medians with full ranges**. Batching
in **whole pool cycles** (102,400 = 100 × 1,024).

**Timing boundaries** (§9): serialized-buffer allocation and `serializedSizeInBytes` **inside**; value
generation, uniqueness, sorting, validation and checksum **outside**.

## 2. What to report

**Q1 — per-bitmap cost**, per shape: wall time, allocation/free counts, byte figures, serialized bytes.

**Q2 — full ratio curves**, per shape:

- **`rawr / plain-list-reference`**, computed **from means, never as the mean of per-fixture ratios**;
- **time curves per allocator, not collapsed** — `rawr/SMP ÷ ref/SMP` and `rawr/libc ÷ ref/libc`;
- crossovers derived only against the pre-registered **sustained** rule: smallest **nonzero** cardinality
  where that point *and every subsequent measured point* stay ≤ 2.0, else "none in range". Report
  `crossover_time_smp`, `crossover_time_libc`, `crossover_bytes`.
- **`crossover_bytes` is portable serialized bytes vs the §5a byte reference.** Memory ratios are
  reported separately and get **no crossover figure** — a peak-memory curve is not a size curve.

**Q3 — gap to the plain-list references**, plus **rawr's and CRoaring's actual portable serialized sizes
reported separately**. State the gap as a number. **Do not call it a maximum achievable win.**

**Q4 — allocation activity**: counts, size histogram, checkpoint deltas, and allocator sensitivity shown
by **comparing SMP and libc cells** — never by subtracting within one. Any trace replay is
**directional and non-additive** and may not be subtracted from a measured total.

**Lifecycle peak bytes: both mean and maximum.**

## 3. Interpretation rules

- The CRoaring control answers **"is rawr unusually expensive relative to the reference"** — nothing more.
  **The phrase "Roaring-inherent" may not be used**: allocation layout is an implementation choice on
  both sides, and §5a is a hypothetical plain-list encoding, not the portable format.
- The **create→build checkpoint delta** is what separates *lazy top-level allocation* from
  *container/header cost*. Report it explicitly per shape — it is the design-relevant split.
- Shape matters as much as cardinality; **never report a sweep number without its shape**.

## Acceptance

- Full sweep run, three shapes, all cell types, both hosts, protocol per §1.
- Q1–Q4 answered with numbers, per shape where required.
- Ratio curves computed from means; time curves per allocator; crossovers per the sustained rule.
- Portable serialized sizes for rawr and CRoaring reported alongside the references.
- Create→build delta reported per shape.
- Lifecycle peak reported as both mean and maximum.
- No board row moves; all four suites plus `check-32`, `check-docs`, `check-package` green.
- **No recommendation and no design conclusion** — Q5 is required for that, and it is `48-02`.

## Estimate

**S/M** — the harness exists; this is running it and reporting carefully.
