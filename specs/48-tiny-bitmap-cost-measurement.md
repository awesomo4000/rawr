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
- If **container header + payload** allocations dominate → inline storage becomes a **candidate**.
  *(A measurement can rule designs out; it cannot prove one is required.)*
- If the **gap to the plain-list reference is small** → no design is warranted.

## 2. Bitmap shape is a parameter, not a constant

**Cardinality alone does not determine cost.** The scratch run used `i*7`, which keeps every value inside
one container. Six values spread across a large universe may need **six containers** — a completely
different allocation profile at identical cardinality.

Pin three shapes, and report **Q2 crossover separately for each**:

| Shape | Definition — **pre-registered, not chosen at implementation time** |
| --- | --- |
| **localized** | `base + i*7`, `base = 3 << 16` — every value inside one container |
| **spread** | sorted unique values drawn over a **10,000,000-row universe** (models row IDs in one data file; Delta/Iceberg DVs cover a bounded universe, often <10M), `std.Random.DefaultPrng`, **seed `0x48_5350_2026`** |
| **one-per-container** | `i * 65536` — one value per distinct high key. **Negative control**: the library's own documented bad case |

**All three shapes, at every cardinality in the Q2 sweep, get a checked-in fixture hash** (spec 40
practice) asserted at generation — not just the mixed corpus. Silent corpus drift would change the answer
without changing the reported inputs.

## 2.1 `Mtiny` — the lifecycle, defined

*(Restored: the revision referenced "the same lifecycle as `Mtiny`" after the definition had been lost in
an edit, and the original `×100k` batching contract went with it.)*

**Object lifetimes are part of the definition** — checkpoints 3 and 4 (§6) are uninterpretable without
them:

```text
create
→ add sorted unique values                         (shape per §2)
→ serialized-size query
→ allocate serialization buffer
→ serialize
→ owning deserialize, WHILE source bitmap and bytes remain live
→ cardinality (on the deserialized bitmap)
→ free deserialized bitmap, then bytes, then source
```

The source and the byte buffer stay live across the deserialize precisely so checkpoint 4 shows peak
concurrent residency, which is what a caller actually pays.

**Outside the timed region:** value generation, uniqueness enforcement, and sorting. Those are corpus
preparation, not the sequence under test — timing them would measure the harness.

**Batching:** **100,000 bitmaps per timed cell**, calibrated so a cell runs long enough to time reliably;
if 100k is too short at a given cardinality, raise the repeat count and **report it** rather than
shortening the sequence.

## 3. Questions the measurement must answer

- **Q1 — End-to-end cost per bitmap**, per shape: wall time, allocation/free counts, byte figures (§6),
  serialized bytes.
- **Q2 — Cost curve, per shape.** Sweep 0,1,2,4,6,8,12,16,20,32,64,128 and report the **complete
  rawr/floor ratio curve** — **separately for time and for bytes**, which need not cross at the same
  cardinality. *(An earlier draft asked when "fixed overhead stops dominating", which is not operational
  and could be decided post hoc.)*

  The curves are the primary output. A single crossover figure may be derived from them **only** against
  the pre-registered threshold **ratio ≤ 2.0** — reported as `crossover_time` and `crossover_bytes`, or
  "none in range".

  **`crossover_bytes` means portable serialized bytes vs the §5a plain-list byte reference**, and nothing
  else. There are now several byte metrics (serialized, post-build live, lifecycle peak); **memory ratios
  are reported separately and get no crossover figure at all** — a peak-memory curve does not have the
  same meaning as a size curve and should not be collapsed into one number.

  **Time ratios are reported per allocator, not collapsed:** `rawr/SMP ÷ reference/SMP` and
  `rawr/libc ÷ reference/libc` as separate curves. Picking one as canonical would hide precisely the
  allocator sensitivity §4 exists to expose.
- **Q3 — Gap to the plain-list references** (§5). **The number that decides whether any design is worth
  building** — though it bounds a simple alternative, not every possible one.
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

## 5. Q3 — the plain-list references, defined exactly

*(An earlier draft said "a plain sorted `[]u32` plus a length", which leaves capacity, serialization, and
ownership unspecified — three things that dominate at this scale.)*

Report **two references**; they answer different questions.

**Neither is a true floor** *(an earlier draft called them that)*: delta/VarInt or inline-in-header
encodings can be smaller than the byte reference, and an inline representation can avoid the heap-owned
reference's allocation entirely. Report **"gap to the plain-list reference"**, never "the available
maximum win" — these bound a *simple* alternative, not every possible representation.

**(a) Plain-list byte reference** — serialized size only, no execution:
- wire format: `u32 count` followed by `count` little-endian `u32` values;
- a lower bound on bytes for a naive encoding.

  **This is NOT format-inherent and NOT the interop contract.** *(An earlier draft called it
  "format-inherent", which is wrong: it is a hypothetical plain-list encoding, not the Roaring portable
  format.)* **Report rawr's and CRoaring's actual portable serialized sizes separately** — those are the
  real format numbers, and the plain-list figure only says what a non-Roaring encoding could achieve.

**(b) Heap-owned plain-list reference** — the same lifecycle as `Mtiny`:
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

Following spec 40's cross-width fixture practice. **Pre-registered here, not chosen during
implementation** — Q5 is load-bearing, so its inputs must be fixed before any number is seen:

| Parameter | Value |
| --- | --- |
| generator | Zipf over cardinality, `std.Random.DefaultPrng` |
| **seed** | `0x48_5A_49_50_2026` |
| exponent `s` | **1.48** — verified: median **2**, p99 **4,935** over `1..100000` |
| corpus count | **100,000** bitmaps |
| cardinality cap | **100,000** |
| shape per bitmap | **spread** (§2) |
| sampling | **inverse-CDF**: precompute `cum[k] = Σ_{i≤k} i^-s` in `f64` for `k = 1..cap`; draw `u = random.float(f64)`; take the smallest `k` with `cum[k] ≥ u * cum[cap]` by binary search |
| corpus hash | **checked in**, asserted at generation |

**Realized quantiles must be reported and asserted**: median in `[1,2]`, p99 in `[1000, 20000]`.

`s = 1.48` was **computed against this band, not guessed** — an earlier draft pinned `s = 1.15`, which
yields median **21** and p99 **71,688**, missing the target so badly that the "re-pin if it misses"
fallback would have fired on the first run. That would have made the escape hatch the actual mechanism,
defeating the point of pre-registration. **A quantile-assertion failure now means the sampler is wrong,
not that the exponent needs tuning.**

**The mixed corpus has no `0` band** — Zipf support starts at cardinality 1. Cardinality 0 is covered by
`Mtiny` (§2.1 sweep), not here.

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
matching rawr means both made similar choices — not that improvement is impossible. **And "Roaring-inherent" is not a conclusion this measurement can reach at all** — §5a is a hypothetical
plain-list encoding, not the portable format. The portable sizes rawr and CRoaring actually emit are
reported as their own numbers (§5).

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
- **CRoaring allocation accounting, pinned:** count via CRoaring's **memory hooks**, and include the
  **caller-owned serialization buffer** — both must appear in the untimed accounting pass, or the
  comparison silently favours CRoaring by omitting allocations rawr is charged for.
- **Validation, outside timing** — beyond a consumed checksum: the deserialized bitmap's **cardinality and
  full value set must match the input**, and **rawr↔CRoaring cross-deserialization** must round-trip. A
  measurement of a sequence that produced wrong results is not a measurement.

## 10. Scope and out of scope

**Scope: `RoaringBitmap` / `u32` only.** `Roaring64Bitmap` is **deferred, not dismissed** — the survey
flags tiny 64-bit bitmaps explicitly (Delta and Iceberg deletion vectors are 64-bit and frequently tiny),
but adding it doubles the matrix and 64-bit has no benchmark harness at all yet. `10-21-bench64` is the
natural home; this spec's findings should inform it.

Out of scope:

- **Any design or production change** — no inline storage, no lazy allocation, no thresholds.
- Container-type or serialization changes.
- The expression-pipeline idea (different archetype).

## 11. Acceptance

- **`Mtiny` across all three §2 shapes; `Mtiny-mixed` on `spread` only** — both as **sequence**
  benchmarks, both hosts, per §9. *(An earlier draft's acceptance demanded all three shapes for both,
  contradicting §7.1's pinning of the mixed corpus to `spread`.)*
- **Q1–Q5 each answered with numbers**, per shape where §2 requires it.
- **Both plain-list references** (§5a, §5b) measured, plus **rawr's and CRoaring's actual portable
  serialized sizes reported separately**; **the gap to the plain-list reference stated as a number**, with
  an explicit verdict if it is small. Do not describe it as a maximum achievable win.
- CRoaring control run; conclusion phrased as **"unusually expensive vs the reference" or not**. **The
  phrase "Roaring-inherent" may not be used at all** — §5a is a hypothetical plain-list encoding, not the
  portable format, and allocation layout is an implementation choice on both sides.
- **`Mtiny` lifecycle implemented exactly as §2.1**, including object lifetimes across the deserialize,
  and the batching contract reported.
- **All shapes and the mixed corpus fixture-hashed** per §2 and §7.1; realized quantiles reported and
  asserted.
- **Q2 reported as full ratio curves** (time and bytes, per shape), with `crossover_time` /
  `crossover_bytes` derived only against the pre-registered ratio ≤ 2.0.
- CRoaring allocations counted via memory hooks including the caller-owned buffer; validation per §9
  (cardinality, full value set, rawr↔CRoaring cross-deserialization) passing outside timing.
- Byte accounting reported at all five §6 checkpoints, teardown proven zero.
- Corpus pinned per §7.1 with its hash; bands per §7.2; share reported per §7.3 with the projection
  labelled.
- No board row moves; all four suites plus `check-32`, `check-docs`, `check-package` green.
- **Recommendation recorded — including "no design warranted" if that is what the numbers say.**

## 12. Estimate

**M** — the benchmark itself is small, but CRoaring allocation accounting, the three shapes, and the
mixed-corpus attribution are careful harness work.
