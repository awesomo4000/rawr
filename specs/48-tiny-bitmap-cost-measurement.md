<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 48: Tiny-bitmap cost — measurement before design

**Measurement only. No design decisions, no production changes.** The point is to produce the numbers
that the small-representation design would branch on, *before* committing to a shape.

## 1. Why measure first

External survey evidence says the tiny case is the most common real-world shape and the most
under-benchmarked: ClickHouse measured **90–94% of distinct tokens at cardinality ≤6, median 1–2**, and
built two non-Roaring encodings rather than pay Roaring's overhead. `groupBitmap` keeps a plain set below
33 elements. Lucene switches away from Roaring below ~0.05% density and above 1%. Delta inlines small
deletion vectors. Spark declined a build optimization because ~3 containers had nothing to win.

**Scratch measurement on rawr** (counting allocator, values `i*7`):

| cardinality | allocations | live bytes | serialized |
|---:|---:|---:|---:|
| 0 | 2 | 40 | 8 |
| 1 | 4 | 72 | 18 |
| 6 | 5 | 80 | 28 |
| 12 | 6 | 96 | 40 |

An empty bitmap costs two allocations; a one-element bitmap costs four and 72 bytes for 4 bytes of data.
**That scratch run is indicative, not authoritative** — it is one host, Debug, no timing, no distribution.
This spec produces the real numbers.

**The design branches on results we do not have:**

- If the cost is dominated by the two *top-level array* allocations, **lazy allocation alone** may be
  enough, and inline storage is unnecessary complexity.
- If the container header + payload allocations dominate, **inline small-set storage** is required.
- If CRoaring shows the same profile, this is **Roaring-inherent** and the honest answer may be "document
  it, tell callers to layer their own encoding" — which is CRoaring's own position.
- If the achievable floor is close to current cost, **there is no win to chase**.

Committing to a design before knowing which of those holds would be guessing.

## 2. Questions the measurement must answer

- **Q1 — What does the tiny path cost end to end?** Per bitmap: wall time, allocation count, peak and
  live bytes, serialized bytes.
- **Q2 — Where is the crossover?** At what cardinality does fixed overhead stop dominating? Sweep
  0,1,2,4,6,8,12,16,20,32,64,128.
- **Q3 — What is the floor?** What would the same sequence cost with an ideal minimal representation
  (a plain sorted `[]u32` plus a length)? **This bounds the available win** and is the number that decides
  whether any design is worth building.
- **Q4 — Allocation versus everything else.** Split the per-bitmap cost into allocator time and the rest.
  This is what distinguishes "lazy allocation suffices" from "inline storage required".
- **Q5 — Does it matter in aggregate?** Under a realistic Zipf mix (median 2, p99 ~5000), what fraction
  of total time and total bytes lands in the tiny tail? Measuring uniformly-tiny bitmaps alone would
  overstate the case.

**Q3 and Q5 are the load-bearing ones.** Q1/Q2/Q4 describe the current state; Q3 says whether an
improvement exists and Q5 says whether it is worth having.

## 3. Benchmark shape — sequences, not operations

Per the survey's own methodology point: compare **as a sequence**, because per-op measurement cannot see
allocation strategy or fixed overhead. Our existing 40-row board is per-op and structurally blind here.

**`Mtiny`** — the full lifecycle, repeated ×100k:

```
create → [add(v)]×card → serialize → deserialize → cardinality → free
```

**`Mtiny-zipf`** — same sequence, cardinalities drawn from a Zipf tail (median 2, p99 ~5000), reporting
the tiny-tail share of the total (Q5).

Report **ns/bitmap and bytes/bitmap**, not aggregate throughput — the per-bitmap fixed cost *is* the
subject.

## 4. Controls

- **CRoaring reference, same sequence.** This is the control that decides attribution: if CRoaring shows
  the same allocation profile, the cost is Roaring-inherent and the conclusion changes from "fix it" to
  "document it". Without this control the result is uninterpretable.
- **Floor reference (Q3):** the same sequence against a plain sorted `[]u32`. Not a proposal — a bound.
- **Existing board unaffected:** this chunk adds measurement only; no board row may move.

## 5. Protocol

Standard campaign discipline, because the failure modes are known:

- **Canonical harness style**: fresh process per cell, warmup then timed iterations, **≥5 process medians
  with full ranges**. Spec 35's warmed-context artifact read 1.155x where canonical read 1.727x.
- **Both hosts**, SMP and libc, allocator stated per cell.
- **Allocation counting must not contaminate timing** — a counting allocator wrapper may not substitute
  for the real allocator in a timed cell, nor run before one in the same process (spec 45-02 §4). Collect
  allocation and byte figures in a **separate untimed run**.
- **Release mode.** The scratch numbers above are Debug and are not comparable.

## 6. Explicitly out of scope

- **Any design or production change.** No inline storage, no lazy allocation, no thresholds. This spec
  produces numbers, not code paths.
- Changing container types or serialization.
- The expression-pipeline idea, which addresses a different archetype.

## 7. Acceptance

- `Mtiny` and `Mtiny-zipf` implemented as **sequence** benchmarks, both hosts, protocol per §5.
- **Q1–Q5 each answered explicitly with numbers**, not prose.
- CRoaring control run for the same sequences; **the Roaring-inherent versus rawr-specific split stated
  outright**.
- Floor reference measured; **the available win stated as a number**, with the honest verdict if it is
  small.
- Allocation/byte figures collected out of band from timing.
- No board row moves; all four suites plus `check-32`, `check-docs`, `check-package` green.
- **Recommendation recorded — including "no design warranted" if that is what the numbers say.**

## 8. Estimate

**S/M** — the benchmark is small; the two-host protocol and the controls are the work.
