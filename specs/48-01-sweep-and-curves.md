<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 48-01: The cardinality sweep — Q1 to Q4

Toplevel: [48-tiny-bitmap-cost-measurement.md](48-tiny-bitmap-cost-measurement.md).
Gated on: [48-00](48-00-harness-and-fixtures.md) complete and green.

Answers **Q1, Q2, Q3, Q4**. Q5 and the decision inputs are `48-02` — and the decision needs Q3 **and**
Q5, so this chunk **states no verdict**.

## 1. What to run

`Mtiny` across the sweep **0, 1, 2, 4, 6, 8, 12, 16, 20, 32, 64, 128**, for **all three shapes**
(`localized`, `spread`, `one-per-container`).

**Five timing tuples**, because the reference must run under both allocators to match rawr's two:
**rawr/SMP, rawr/libc, CRoaring/libc, reference/SMP, reference/libc**. Both hosts, **`ReleaseFast`,
native CPU**.

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

- Full sweep run, three shapes, **all five tuples**, both hosts, protocol per §1.
- Q1–Q4 answered with numbers, per shape where required.
- Ratio curves computed from means; time curves per allocator; crossovers per the sustained rule.
- Portable serialized sizes for rawr and CRoaring reported alongside the references.
- Create→build delta reported per shape.
- Lifecycle peak reported as both mean and maximum.
- No board row moves; all four suites plus `check-32`, `check-docs`, `check-package` green.
- **No verdict and no design conclusion** — Q5 is required for that, and it is `48-02`. Reporting the Q3
  gap here is the deliverable; interpreting it is not.

## Outcome — Q1–Q4 answered, no verdict drawn (correct)

Reports: `misc/tiny-bench-20260823-084027-summary.txt` (M4),
`misc/tiny-bench-20260823-091104-summary.txt` (Zen 4).

**Protocol verified in the artifacts, not taken on report:** five tuples present in every table
(`rawr/smp`, `ref/smp`, `rawr/libc`, `ref/libc`, `CR/libc`); five processes per tuple; header states
*"Ratios divide those means; they are not means of per-fixture ratios"*; *"No verdict: Q5 is deferred to
spec 48-02"*; both hosts.

**Sustained crossover rule applied correctly** — spot-checked on localized bytes: card 1 = 2.25x (>2), and
card 2 = 1.67x with every subsequent point ≤ 2.0, giving `crossover_bytes=2`. Not the first point under
threshold.

### The finding the summary understated: rawr ≈ CRoaring

| shape | `rawr/CR` at card 0 | at card 6 | at card 128 |
|---|---:|---:|---:|
| localized | 1.66x | 1.11x | 1.14x |
| spread | 1.54x | 1.37x | 1.11x |
| one-per-container | 1.57x | 1.39x | 1.22x |

**rawr sits at ~1.1–1.4x of CRoaring across the whole tiny range**, tightening above cardinality 8. So the
5–77x gap to a plain list is **not a rawr defect** — it is what Roaring costs *as both implementations
build it*. That is the question the CRoaring control exists to answer, and the answer is "rawr is not
unusually expensive."

**Per the spec's own rule, this still may not be called "Roaring-inherent."** Both implementations making
similar choices is not proof the choices are forced.

### One place rawr IS measurably behind

**Allocations per lifecycle:** rawr **13–14**, CRoaring **9**, plain-list reference **3** (card 1–8,
localized). rawr does ~50% more allocations than CRoaring while costing only ~10–40% more time — so the
extra allocations are not proportionally expensive, but they are real and they are rawr-specific. CRoaring
also uses `resize` (1–3 per lifecycle) where rawr uses none: different growth strategies.

**This is the most actionable number in the chunk** and belongs in `48-02`'s design-candidate reasoning.

### Shape dominates cardinality — the three-shape decision paid for itself

At **cardinality 128**, rawr/SMP vs plain list: localized **12.00x**, spread **74.42x**, one-per-container
**77.33x**. Same cardinality, ~6x spread in the answer. A shape-free sweep would have averaged these into
a number describing nothing.

Byte behaviour splits the same way: `crossover_bytes` = **2** (localized), **128** (spread), **none**
(one-per-container). For localized, rawr's serialized size is *smaller* than the plain list from
cardinality 8 upward — u16 values in one container beat u32 in a list.

**rawr and CRoaring serialized bytes are identical at every point** in the localized table, which is a
useful independent signal that the portable encoding is doing what it should.

### Anomaly to explain, not to average away

**`spread`, SMP, cardinality 16 → 20: time DROPS 8.5%** (761.21 → 696.28 ns) while **libc rises 24.3%**
(1256.16 → 1561.77 ns) over the same step. One-per-container is monotonic on both.

An allocator-specific non-monotonicity is exactly the size-class behaviour this campaign has documented
before. **`48-02` must state monotonicity per §3, so this point needs an explanation rather than being
smoothed** — and it is a reason the spec forbade reducing the 1–12 range to a single headline figure.

### Cross-check against independent measurement

`create` storage is a flat **40 bytes** at every cardinality, matching the 40 bytes measured
independently in the spec-48 scratch probe before this harness existed.

## Estimate

**S/M** — the harness exists; this is running it and reporting carefully.
