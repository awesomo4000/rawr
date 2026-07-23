<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 21-00: Skewed `andCardinality` — diagnosis (two-layer + sweep)

First chunk of [skewed `andCardinality`](21-skewed-andcardinality.md). **Diagnosis only,
benchmark-only.** Deliverable: the ~1.46x characterized across the sweep and **attributed to
galloping kernel vs surrounding traversal/dispatch** — the input that decides whether `21-01`
(a fix) is written and around what. Baseline of record: rawr-SMP **0.019 [0.018, 0.019] ms**
vs CRoaring **0.013 [0.012, 0.014] ms** (isolated). Threshold selection is already ruled out
for the original 32 × 4096 point (both sides gallop); this chunk does **not** pre-attribute
the rest.

## Two-layer measurement (fresh process, five runs, median + range)

### Full API (the number of record)

`RoaringBitmap.andCardinality` vs public CRoaring `roaring_bitmap_and_cardinality`, isolated
fresh-process focused executable — this is the ~1.46x, and it includes top-level key
traversal, per-container dispatch, and the kernel calls (the bench is 200 containers/bitmap,
180 matching keys).

### Direct kernel (ns/container, batched to clear clock noise)

Peel the layers apart with matched rows:

| row | rawr | CRoaring |
|---|---|---|
| **true kernel-to-kernel** (forced gallop, no dispatch) | forced `intersectCardGallop` | **`intersect_skewed_uint16_cardinality` called directly** |
| **dispatch-inclusive** (normal per-container dispatch) | rawr normal dispatch | **`array_container_intersection_cardinality` wrapper** |
| **cross-check** | forced SIMD/merge | — |

Batch each direct row over enough calls to clear clock resolution; report **ns/container**.
The true-kernel row isolates the galloping-count kernels head-to-head; the dispatch-inclusive
row shows what per-container dispatch adds on each side. Together with the full-API number
they say **how much of the 1.46x is kernel vs dispatch/traversal.**

## Sweep corpus (define exactly; confirm representations before timing)

All pairs have both sides **≤ 4096** so they stay arrays; **confirm representations** (both
arrays, expected cardinalities) before any timing. Run the **three overlap distributions** —
**all-hit**, **disjoint**, **deterministic mixed/random-overlap** — on the **direct-kernel**
cases.

**Generalization matrix** — does the kernel gap hold across ratios and small-side sizes, or
is it specific to 32 × 4096 all-hit?

- **ratio progression** (fixed small side): `32×256` (8:1), `32×1024` (32:1), `32×2048`
  (64:1), `32×4096` (128:1);
- **fixed 128:1 ratio, varying small side**: `8×1024`, `16×2048`, `32×4096`;
- **extreme small sides**: `1×4096` (4096:1), `8×4096` (512:1).

**Threshold boundary pairs** — dispatch selection near each crossover:

- rawr NEON 40: `64×2496` (39:1), `64×2560` (40:1), `64×2624` (41:1);
- CRoaring 64: `32×2016` (63:1), `32×2048` (64:1), `32×2080` (65:1).

**Scope split:** the **direct-kernel** ns/container rows run across the whole matrix × the
three overlap distributions (cheap, batched). The **full-API five-process** measurement only
needs the **original 32 × 4096** case, plus any points where the direct results reveal
something worth confirming at the API layer.

## Acceptance

- Full-API isolated ratio reproduced near the baseline; direct-kernel rows reported as
  **ns/container** with the true-kernel and dispatch-inclusive rows matched per the table.
- The ~1.46x **attributed**: how much is the galloping kernel vs traversal/dispatch, and
  whether it holds across overlap distributions and the sweep or is specific to
  32 × 4096 all-hit — absolute medians + ranges with a **named residual**, not forced-100%.
- Correctness: every measured op validated (`== |a ∩ b|` `==` CRoaring oracle) before timing
  is accepted.
- Environment: `ReleaseFast`, native CPU, spec-16 M4 host; five process runs; env header
  recorded.
- **Benchmark-only:** no public library / vendored-source change; findings + commands
  committed to `docs/parity-measurement.md` (or a sibling), not ignored `misc/`.
- Validation: `zig build test`; `zig build bench-aa -Dcpu=native && ./zig-out/bin/bench_aa`;
  `zig build difftest`; the focused five-process runner.

## Result to record (feeds `21-01`)

The kernel-vs-dispatch attribution and the sweep generality — this decides whether `21-01` is
written and whether its lever is a tighter kernel, a dispatch/traversal change, or (for
boundary cases only) a threshold re-tune. An intrinsic-at-~6 µs / non-generalizing result is a
valid terminal outcome recorded in the doc.
