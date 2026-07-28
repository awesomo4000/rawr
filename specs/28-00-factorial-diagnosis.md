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

- Untimed allocation/byte counters; each cell a **fresh process**; **rawr-SMP and rawr-libc**
  plus CRoaring; **M4 and Zen 4**; no nested timers.
- **Allocator provenance reported per variant:** the bench bitmap is SMP-built and the rawr-libc
  tuple routes only the **output buffer** through libc — `desc_buf`/`offset_buf` use
  `bm.allocator` (SMP) (`serialize.zig:194`; row wiring `bench_croaring.zig:730`). Report, per
  variant: output-buffer allocator, temp-table allocator, and allocation counts + bytes for each,
  so no cell is misread as fully allocator-matched.
- Interactions treated as interactions, not additive percentages.

## Acceptance / checklist

- [ ] Four cells × (rawr-SMP, rawr-libc, CRoaring) × (M4, Zen 4), fresh process per cell,
      median + range
- [ ] Allocator provenance + alloc counts/bytes reported per variant
- [ ] The M4 1.14x attributed to construction lever / output lever / interaction, per host
- [ ] Every cell's output validated byte-identical to unchanged `serializeToWriter()` (in-repo
      oracle) so cells differ only in *how* they write, not *what*
- [ ] Benchmark-only; results in `docs/parity-measurement.md`; `zig build test`, `difftest`,
      `ReleaseSafe`, `ReleaseFast` green

## Result to record (decides `28-01`)

Which lever(s) carry the gap, per host — and specifically whether removing the temp tables helps
or **regresses M4 SMP** (the spec-27 check). `28-01` implements only the winning cell's levers.
