<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 28-00: Serialize — factorial diagnosis

First chunk of [serialize direct buffer](28-serialize-direct-buffer.md). **Diagnosis only,
benchmark-only.** Attributes the serialize gap (M4 SMP **1.14x**, Zen 4 SMP **1.08x**) across the
`construction × output` factorial so `28-01` fixes the component that actually costs — and so we
learn whether "drop temp arrays" helps or replays spec 27's M4-SMP regression here.

## The four cells (true factorial: temp/direct construction × Writer/direct-index output)

1. **temp tables + Writer** — current;
2. **direct construction + Writer** — table entries streamed through the Writer, no temp tables
   (inherently more `Writer` calls — intrinsic to the cell);
3. **temp tables + direct index** — temp tables copied in via direct indexed stores, no Writer;
4. **direct construction + direct index** — in-place entries and direct indexing (shipping shape).

The 2×2 isolates the **construction** lever (temp allocation — the spec-27 M4-SMP risk) from the
**output** lever (Writer abstraction — allocator-independent) and their interaction.

## Method

- **Four rawr-SMP cells** (fresh process each) on **M4 and Zen 4**, against **one CRoaring
  reference per host**. Median + full range; untimed allocation/byte counters; no nested timers.
- **rawr-libc is a conditional control, not routine:** the target is the SMP path and the gap is
  M4 SMP; the serialize libc tuple isn't even whole-op libc (only the output buffer; temp tables
  stay on `bm.allocator` = SMP — `serialize.zig:194`, row wiring `bench_croaring.zig:730`). Run
  **one M4 rawr-libc control only if** an SMP cell's cost is ambiguous between allocation and
  other work. Not in the acceptance gate or routine reporting.
- **Allocation counts + bytes reported per cell** — output buffer vs temp tables **separately**;
  this reads the construction lever directly, no libc timing needed.
- **Reproducible artifacts:** `zig build bench-serialize-diag` →
  `./zig-out/bin/bench_serialize_diag`, five-process runner `scripts/run-bench-serialize-diag.sh`,
  timestamped summaries under `misc/` (named in `docs/parity-measurement.md`).
- **Cells share production serialization-layout helpers** where possible, or **each cell's output
  is proven byte-identical to unchanged `serializeToWriter()`** — so the winning benchmark cell
  cannot drift from what `28-01` ships.
- Interactions treated as interactions, not additive percentages.

## Acceptance / checklist

- [ ] Four **rawr-SMP** cells × (M4, Zen 4), fresh process each, median + range; **one CRoaring
      reference per host**; optional M4 rawr-libc control only if a cell needs allocator attribution
- [ ] Alloc counts + bytes per cell (output buffer vs temp tables separately)
- [ ] `bench-serialize-diag` build + runner + timestamped `misc/` summaries exist
- [ ] Cells share production layout helpers or are byte-identical to `serializeToWriter()`
- [ ] The M4 1.14x attributed to construction lever / output lever / interaction, per host
- [ ] Benchmark-only; results in `docs/parity-measurement.md`; `zig build test`, `difftest`,
      `ReleaseSafe`, `ReleaseFast` green

## Result to record (decides `28-01`)

Which lever(s) carry the gap, per host — and specifically whether removing the temp tables helps
or **regresses M4 SMP** (the spec-27 check). `28-01` implements only the winning cell's levers.
