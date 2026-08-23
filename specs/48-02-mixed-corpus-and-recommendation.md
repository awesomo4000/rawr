<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 48-02: Mixed corpus, aggregate share, and the decision inputs

Toplevel: [48-tiny-bitmap-cost-measurement.md](48-tiny-bitmap-cost-measurement.md).
Gated on: [48-01](48-01-sweep-and-curves.md) complete.

Answers **Q5** and assembles the **decision inputs**. It is the only chunk that can, because the decision
needs Q3 and Q5 together — but it **does not itself decide**: thresholds are the owner's (§3).

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

## 2.1 Which cells get the full decomposition

*("One total cell" was ambiguous against the execution matrix — and the matrix is **five tuples**, not
four: the plain-list reference must run under **both** allocators to match rawr's two.)*

| Output | Cells |
| --- | --- |
| **mixed-corpus total** (measured) | **all five tuples**: rawr/SMP, rawr/libc, CRoaring/libc, **reference/SMP, reference/libc** — both hosts |
| **band decomposition + projection + residual** | **rawr/SMP and rawr/libc only** |
| **byte / allocation share** (measured, untimed) | rawr/SMP and rawr/libc |

The projection exists to **attribute rawr's own cost** and to show whether the allocator changes the
tail's share — not to compare implementations band by band. CRoaring and the reference contribute totals
for comparison; decomposing them would multiply the cell count for no question anyone asked.

## 3. The decision — inputs pre-registered, thresholds are the OWNER'S

*(An earlier draft required an automatic recommendation gated on "small gap" and "small share" with no
metric or threshold defined — which would let the verdict be chosen after seeing the numbers, the exact
failure this spec exists to prevent.)*

**No threshold is pre-registered, and no automatic recommendation is required.** We do not yet know
enough to set one honestly, and inventing a number would be false precision dressed as rigour.

**Instead, the decision inputs are pre-registered so they cannot be selected post hoc.** Report exactly
these, then the owner decides:

| Input | Pinned definition |
| --- | --- |
| **Q3 gap** | `spread` shape, **rawr/SMP**, **all six sweep points in 1–12 reported individually** — `1, 2, 4, 6, 8, 12` — each as time ratio **and** byte ratio vs the plain-list references. **No aggregate.** |
| **tiny tail** | bands **1–2, 3–6, 7–12** (i.e. ≤12 — matching the cutoff above which ClickHouse does use Roaring) |
| **Q5 shares** | **three separate numbers, never combined into an index**: time (projected, with residual), bytes (measured), allocations (measured) |

**Why six numbers and not one:** the range holds six measured points, and a mean, a maximum, and an
endpoint can each support a *different* owner decision. Choosing the reduction after seeing the curve
would reintroduce exactly the post-hoc freedom §3 exists to remove. **Report all six; also state whether
the ratio is monotonic across them**, since a non-monotonic curve is itself decision-relevant.
*(An earlier draft said "headline gap" without a reduction rule.)*

Cardinalities 1–12 and the ≤12 tail are chosen because that is where the survey's evidence actually sits
(ClickHouse: 90–94% of tokens at ≤6, its own Roaring cutoff at >12) — not because they flatter any
outcome.

**The owner applies judgement to those numbers.** A large per-bitmap gap over a negligible share does not
justify a design, and neither does a small gap over a dominant share — but where those lines fall is a
call this measurement cannot make for them.

**If a design is warranted, name which one the evidence points at** — using `48-01`'s create→build
checkpoint delta:

- top-level array allocations dominate → **lazy allocation** is the candidate;
- container header + payload dominate → **inline small-set storage** becomes a candidate.

**"Candidate", not "required".** A measurement can rule designs out; it cannot prove one necessary.

**And present the honest negative if that is what the numbers show.** The archetype-F story is compelling
enough to create real pull toward finding a problem worth solving; "the numbers do not support a design"
is a valid and valuable outcome, and must be reported as plainly as a positive one.

## 4. Scope reminder

**`RoaringBitmap` / `u32` only.** `Roaring64Bitmap` is **deferred, not dismissed** — the survey flags tiny
64-bit bitmaps (Delta and Iceberg deletion vectors are 64-bit and frequently tiny). `10-21-bench64` is
its natural home, and this chunk's findings should inform it. Say so in the write-up rather than leaving
the omission unexplained.

## Acceptance

- `Mtiny-mixed` run on both hosts per the toplevel protocol; corpus hash asserted; realized quantiles
  reported.
- Bands reported per §1, including the empty `0` band.
- All four §2 parts reported, each labelled measured or projected, **with the signed projection
  residual**; cell scope per **§2.1**.
- Byte/allocation share reported as measured.
- **All §3 decision inputs reported with their pinned definitions** — the **six individual Q3 ratios**
  (plus monotonicity), tiny-tail bands, and the three separate Q5 shares. **No aggregate, no threshold
  invented, no automatic verdict.**
- **Candidate design named** per §3's create→build evidence if the numbers point at one, with "candidate"
  wording preserved. The accept/reject decision is explicitly **the owner's**.
- Scope note per §4 included.
- No board row moves; no production change; all four suites plus `check-32`, `check-docs`,
  `check-package` green.

## Outcome — decision inputs assembled, no verdict taken (correct)

Reports: `misc/tiny-mixed-bench-20260823-100002-summary.txt` (M4),
`misc/tiny-mixed-bench-20260823-100806-summary.txt` (Zen 4).

**Protocol verified in the artifacts:** five tuples in the measured totals; band decomposition for
**rawr/smp and rawr/libc only** per §2.1; `0` band present as count 0 / share 0 / `N/A`; residual given as
signed ms **and** percentage; corpus hash and realized quantiles printed (median 2, p99 4961); header
states *"No automatic verdict or threshold is applied; the owner decides from Q3 and Q5."*

### Q5 — the tail is most of the objects and almost none of the work

| | value |
|---|---:|
| ≤12 share of **bitmaps** | **77.537%** |
| ≤12 share of projected **time** — M4 SMP / Zen 4 SMP / M4 libc | **4.591% / 5.125% / 7.315%** |
| ≤12 share of **requested bytes** | **5.806%** |
| ≤12 share of **allocation calls** | **17.075%** |
| `129+` — share of bitmaps → share of time | **7.284% → 87.330%** |

**Projection residuals are small**, so the shares are trustworthy: M4 SMP **−3.527%**, M4 libc
**+1.114%**, Zen 4 **−1.482%**.

**This is why Q5 was made load-bearing.** A uniformly-tiny benchmark would have shown 5–31x gaps and read
as urgent. Under a realistic Zipf mix the tail is ~5% of time, and **7.284% of bitmaps in the `129+` band
account for 87% of it**.

**The one number that is not small: allocation calls at 17.075%** — roughly 3.7x the tail's time share.
The tail is allocation-dense relative to its cost. Single-threaded measurement will not show whatever that
implies under allocator contention.

### Q3 — six ratios, and they are monotonically INCREASING

`spread`, rawr/SMP vs plain list, cardinalities 1, 2, 4, 6, 8, 12:

- **M4:** 5.80x, 7.53x, 10.52x, 15.24x, 18.99x, 31.64x
- **Zen 4:** 6.31x, 8.14x, 11.49x, 16.23x, 19.73x, 25.42x

**Monotonically increasing on both hosts** — the required monotonicity statement, and the interpretively
important one: **the gap widens with cardinality across the tiny range.** So this is *not* a fixed
per-bitmap tax that amortizes. For `spread`, each added value adds a container, so it is a **per-container
tax that never amortizes within this range**. That distinction shapes what any design would have to do:
lazy top-level allocation cannot touch it, because the cost is not in the top level.

### Whole-corpus totals — worth reading directly

| tuple | M4 total | vs |
|---|---:|---|
| rawr/smp | 260.836 ms | — |
| croaring/libc | 271.943 ms | **rawr/SMP is faster than CRoaring/libc** |
| rawr/libc | 353.519 ms | **1.30x** croaring/libc, same allocator |
| reference/smp | 29.971 ms | rawr/smp is **8.70x** the plain list |
| reference/libc | 16.513 ms | — |

**Caveat on the reference:** `reference/libc` (16.5 ms) is **1.8x faster than `reference/smp`** (30.0 ms).
The plain list prefers libc while rawr prefers SMP. Same-allocator pairing is the right comparison and the
spec pinned it — but a reader should know the plain-list gap would look **worse** if the reference were
run under its own best allocator.

### Candidate evidence, as reported

Cardinality 1 favours **lazy top-level allocation**; from cardinality 2 upward container/header storage
dominates, pointing at **inline small-set storage**. Consistent with `create` being a flat 40 bytes while
create→build grows with container count. **Candidates, not requirements** — and §3's thresholds remain the
owner's.

**No decision recorded in this chunk, correctly.**

## Estimate

**S/M** — one corpus, one benchmark, careful attribution and write-up.
