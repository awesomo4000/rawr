<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 48: Tiny-bitmap cost — measurement before design

**Measurement only. No design decisions, no production changes.** This spec produces the numbers a
small-representation design would branch on, *before* any shape is committed to.

## 1. Why measure first

Survey evidence says the tiny case is the most common real-world shape and the most under-benchmarked:
ClickHouse measured **90–94% of distinct tokens at cardinality ≤6, median 1–2**, and built two
non-Roaring encodings rather than pay Roaring's overhead. `groupBitmap` keeps a plain set below 33
elements. Lucene switches away below ~0.05% density and above 1%. Delta inlines small deletion vectors.
Spark declined a build optimization at ~3 containers.

**Scratch measurement on rawr** (counting allocator, values `i*7`, one host, **Debug**, untimed):

| cardinality | allocations | live-at-peak bytes | serialized |
|---:|---:|---:|---:|
| 0 | 2 | 40 | 8 |
| 1 | 4 | 72 | 18 |
| 6 | 5 | 80 | 28 |
| 12 | 6 | 96 | 40 |

**Indicative only — not a result.** One host, Debug, no timing, one bitmap shape, no distribution. Do not
quote it. This spec produces the authoritative numbers.

**The design branches on answers we do not have:**

- If the two **top-level array** allocations dominate → lazy allocation may suffice and inline storage is
  unnecessary complexity.
- If **container header + payload** allocations dominate → inline storage is required.
- If the **achievable floor is close to current cost** → no design is warranted.

## 2. Bitmap shape is a parameter, not a constant

**Cardinality alone does not determine cost.** The scratch run used `i*7`, which keeps every value inside
one container. Six values spread across a large universe may need **six containers** — a completely
different allocation profile at identical cardinality.

Pin three shapes, and report **Q2 crossover separately for each**:

| Shape | Definition |
| --- | --- |
| **localized** | all values within a single high-key container |
| **spread** | sorted unique values over a fixed realistic universe (models real row IDs) |
| **one-per-container** | one value per distinct high key — **negative control**, the library's own documented bad case |

## 3. Questions the measurement must answer

- **Q1 — End-to-end cost per bitmap**, per shape: wall time, allocation/free counts, byte figures (§6),
  serialized bytes.
- **Q2 — Crossover**, *per shape*: at what cardinality does fixed overhead stop dominating? Sweep
  0,1,2,4,6,8,12,16,20,32,64,128.
- **Q3 — The floor** (§5). Bounds the available win. **This is the number that decides whether any design
  is worth building.**
- **Q4 — Allocation activity and allocator sensitivity** (§4).
- **Q5 — Aggregate significance** under a realistic mixed corpus (§7).

**Q3 and Q5 are load-bearing.** Q1/Q2/Q4 describe the current state; Q3 says whether an improvement
exists, Q5 says whether it is worth having.

## 4. Q4 — allocation activity, NOT a time split

*(Corrected: an earlier draft asked to split cost into "allocator time versus everything else." That is
not cleanly measurable — allocation timing is context-sensitive, and subtracting an allocator
microbenchmark from a total is exactly the class of error this campaign has repeatedly paid for.)*

Report instead:

- **allocation and free counts** per lifecycle;
- **requested, live, and peak bytes** (§6 checkpoints);
- **allocation-size histogram** — which size classes are hit, and how often;
- **deltas at lifecycle checkpoints**;
- **separate SMP and libc timing cells** — allocator sensitivity shown by *comparison across cells*, never
  by subtraction within one.

Any isolated allocation-trace replay is **directional and non-additive** and must be labelled as such. It
may not be subtracted from a measured total.

## 5. Q3 — the floor, defined exactly

*(An earlier draft said "a plain sorted `[]u32` plus a length", which leaves capacity, serialization, and
ownership unspecified — three things that dominate at this scale.)*

Report **two** floors; they answer different questions:

**(a) Byte-format floor** — serialized size only, no execution:
- wire format: `u32 count` followed by `count` little-endian `u32` values;
- this is the honest lower bound on bytes and is **format-inherent**.

**(b) Executable heap-owned floor** — the same lifecycle as `Mtiny`:
- capacity **known and allocated exactly** (no growth, no slack);
- serialize: emit the §5(a) wire format;
- deserialize: **copies** into an owned allocation (matching rawr's owning semantics — a borrowed variant
  would be a different comparison and must not be conflated);
- cardinality: the stored length, O(1);
- **all** allocations counted, including the values array and the serialized buffer.

## 6. Byte accounting — checkpoints, not one number

*(An earlier draft reported "live bytes" for a lifecycle ending in `free`, where the final answer is
necessarily zero.)*

Record byte and allocation figures at each checkpoint:

1. after `create`
2. after build (all `add`s)
3. after `serialize`
4. after `deserialize`
5. after complete teardown — **must be zero**

The create→build delta is precisely what distinguishes *lazy top-level allocation* from
*container/header cost*, which is one of this spec's primary design questions (§1).

## 7. Q5 — mixed corpus, reproducible, with a stated share method

### 7.1 Corpus — pinned like a fixture

Following spec 40's cross-width fixture practice:

- exact generator and **pinned seed**;
- cardinality cap and **corpus count**;
- reported quantiles;
- **checked-in corpus hash**, asserted at generation, so drift fails loudly rather than silently changing
  the answer.

### 7.2 "Tiny" is a set of bands, not a word

Report per band: **0**, **1–2**, **3–6**, **7–12**, **13–32**, **33–128**, **129+**.

### 7.3 Share method — stated, because the obvious approaches are wrong

Per-item timers dominate the tiny cases; separately timing each band changes allocator history and
therefore the answer.

- **one mixed-corpus total cell** — the ground truth for total time;
- **independent batched cells per band** — per-band cost, each in its own fresh process;
- **a weighted projected share**, combining the two and **explicitly labelled a projection**, never
  reported as measured;
- **an exact byte and allocation share** from the untimed accounting pass — this one *is* measured, and is
  the reliable half of Q5.

## 8. Controls and what they do and do not establish

**CRoaring reference, same sequences.** It answers: **is rawr unusually expensive relative to the
reference implementation?**

**It does NOT establish that any cost is "Roaring-inherent."** *(An earlier draft claimed exactly that.)*
Allocation layout, container ownership, and execution time are **implementation choices**. CRoaring
matching rawr means both made similar choices — not that improvement is impossible. **The only
format-inherent quantity here is the portable serialized floor** (§5a), because that one is fixed by the
interop contract.

**Existing board unaffected:** measurement only; no board row may move.

## 9. Execution matrix and protocol

- **Hosts:** M4 and Zen 4. **`ReleaseFast`, native CPU.** *(The §1 scratch numbers are Debug and are not
  comparable.)*
- **Cells:** rawr/SMP, rawr/libc, CRoaring/libc, and the floor under the matching allocators.
- **Fresh process per cell**, warmup then timed iterations, **≥5 process medians with full ranges** —
  spec 35's warmed-context artifact read 1.155x where canonical read 1.727x.
- **Timing boundaries, pinned:** serialized-buffer allocation **and** `serializedSizeInBytes` are
  **inside** the timed region — they are part of the sequence a caller pays for. **Validation and a
  consumed checksum are outside** it.
- **Accounting is out of band:** a counting allocator may not substitute for the real allocator in a timed
  cell, nor run before one in the same process (spec 45-02 §4).

## 10. Out of scope

- **Any design or production change** — no inline storage, no lazy allocation, no thresholds.
- Container-type or serialization changes.
- The expression-pipeline idea (different archetype).

## 11. Acceptance

- `Mtiny` and `Mtiny-mixed` implemented as **sequence** benchmarks across all three §2 shapes, on both
  hosts, per §9.
- **Q1–Q5 each answered with numbers**, per shape where §2 requires it.
- **Both floors** (§5a, §5b) measured; **the available win stated as a number**, with an explicit verdict
  if it is small.
- CRoaring control run; conclusion phrased as **"unusually expensive vs the reference" or not** — the
  phrase "Roaring-inherent" may be used **only** of the §5a serialized floor.
- Byte accounting reported at all five §6 checkpoints, teardown proven zero.
- Corpus pinned per §7.1 with its hash; bands per §7.2; share reported per §7.3 with the projection
  labelled.
- No board row moves; all four suites plus `check-32`, `check-docs`, `check-package` green.
- **Recommendation recorded — including "no design warranted" if that is what the numbers say.**

## 12. Estimate

**M** — the benchmark itself is small, but CRoaring allocation accounting, the three shapes, and the
mixed-corpus attribution are careful harness work.
