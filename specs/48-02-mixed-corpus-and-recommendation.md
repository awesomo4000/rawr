<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 48-02: Mixed corpus, aggregate share, and the recommendation

Toplevel: [48-tiny-bitmap-cost-measurement.md](48-tiny-bitmap-cost-measurement.md).
Gated on: [48-01](48-01-sweep-and-curves.md) complete.

Answers **Q5** and produces the **recommendation** — the only chunk that reaches a conclusion, because
the conclusion requires Q3 and Q5 together.

## 1. `Mtiny-mixed`

The pinned Zipf corpus (§7.1), **`spread` shape only** — `Mtiny` covers all three shapes, `Mtiny-mixed`
does not.

Bands: **0, 1–2, 3–6, 7–12, 13–32, 33–128, 129+**. The **`0` band is empty** — Zipf support starts at 1;
cardinality 0 is covered by `48-01`'s sweep. Report it as count 0, share 0, timing `N/A`.

## 2. Share method — three parts, each labelled for what it is

1. **One mixed-corpus total cell** — the measured ground truth for total time.
2. **Independent batched cells per band**, each in a fresh process. **Each band cell replays the actual
   mixed-corpus bitmaps in that band**, cycled in **whole cycles** over that band's membership — *not*
   freshly generated bitmaps of similar cardinality. Otherwise the projection multiplies counts from one
   distribution by costs from another.
3. **Weighted projection**, explicitly labelled a projection:

   ```
   projected_band_time = band_count × mean_time_per_bitmap(band)
   projected_share(b)  = projected_band_time(b) / Σ_bands projected_band_time
   ```

   Plus the **projection residual**: `Σ projected_band_time − measured mixed-corpus total`, reported as
   **signed time and signed percentage**. A share quoted without its residual hides how much the band
   decomposition misses.

4. **Byte and allocation share** from the untimed accounting pass. **This part is measured, not
   projected**, and is the reliable half of Q5.

## 3. The recommendation

**Both conditions are required for "no design warranted":**

- the **gap to the plain-list reference is small** (Q3, from `48-01`), **and**
- the **tiny tail is a small share of the mixed corpus** (Q5).

A large per-bitmap gap over a negligible share of real work does not justify a design; neither does a
small gap over a dominant share. Either alone is insufficient.

**If a design is warranted, name which one the evidence points at** — using `48-01`'s create→build
checkpoint delta:

- top-level array allocations dominate → **lazy allocation** is the candidate;
- container header + payload dominate → **inline small-set storage** becomes a candidate.

**"Candidate", not "required".** A measurement can rule designs out; it cannot prove one necessary.

**And record the honest negative if that is what the numbers say.** The archetype-F story is compelling
enough to create real pull toward finding a problem worth solving; "no design warranted" is a valid and
valuable outcome.

## 4. Scope reminder

**`RoaringBitmap` / `u32` only.** `Roaring64Bitmap` is **deferred, not dismissed** — the survey flags tiny
64-bit bitmaps (Delta and Iceberg deletion vectors are 64-bit and frequently tiny). `10-21-bench64` is
its natural home, and this chunk's findings should inform it. Say so in the write-up rather than leaving
the omission unexplained.

## Acceptance

- `Mtiny-mixed` run on both hosts per the toplevel protocol; corpus hash asserted; realized quantiles
  reported.
- Bands reported per §1, including the empty `0` band.
- All four §2 parts reported, each labelled measured or projected, **with the signed projection residual**.
- Byte/allocation share reported as measured.
- **Recommendation recorded**, satisfying §3's two-condition rule, with the candidate design named if one
  is warranted and "candidate" wording preserved.
- Scope note per §4 included.
- No board row moves; no production change; all four suites plus `check-32`, `check-docs`,
  `check-package` green.

## Estimate

**S/M** — one corpus, one benchmark, careful attribution and write-up.
