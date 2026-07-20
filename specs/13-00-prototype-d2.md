<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 13-00: Single-alloc prototype + D2 decision

First chunk of [single-allocation container layout](13-single-alloc-containers.md).
**Measurement, not implementation.** It answers two questions with numbers before
any production layout is written:

- **Q1 (whole-spec go/no-go):** does one allocation per array container (vs today's
  two) actually reduce allocation count and yield a net time win on allocation-heavy
  workloads — enough to justify the L-effort ABI change?
- **Q2 (D2):** stored-slice vs derived-accessor — is the pointer-chase win of derived
  accessors big enough to justify the repo-wide `.values`/`.runs` migration, or do we
  keep stored slices?

Output: both answers recorded back into the umbrella (D2 marked resolved with
numbers; go/no-go stated).

**Retention (be explicit):**
- The **counting allocator** is a **permanent** reusable test util (13-06 reuses it).
- **`bench-proto` and the prototype `ArrayContainer` variants stay committed** — they
  are the reproducible record of the D2/Q1 measurement (a future reader re-runs
  `zig build bench-proto` to reproduce the decision). "Throwaway" only means they are
  **not production code and not the 13-03 implementation** — 13-03 is written fresh
  against the real `ArrayContainer`, informed by the winning layout; it does not lift
  the prototype code.

## Isolation

The prototype must **not** touch production containers or the bitmap. Build it as a
standalone module + bench (e.g. `src/proto_single_alloc.zig` / a `bench-proto` step),
so `test`/`validate`/`difftest` are unaffected and there's no behavior change to the
library from this chunk.

## What to build

Four `ArrayContainer` variants exercised under one harness. **The alignment change
(current 32-byte → proposed 16-byte, D5) is bundled into the proposed layout, so a
matching-alignment control is needed to attribute the effect:**

1. **Baseline** — the *current* two-alloc `ArrayContainer` exactly as shipped
   (`allocator.create` header + `alignedAlloc` values, **32-byte** value alignment).
   The real-world "before."
2. **Control: two-alloc, 16-byte** — same two-alloc shape as baseline but values at
   **16-byte** alignment. Isolates the alignment change: `(2) − (1)` = pure alignment
   effect.
3. **Single-alloc, stored-slice (16-byte)** — one aligned block; `Header {
   cardinality: u16, capacity: u16, values: []Elem }` co-located, `values` points into
   the block. `dataOffset() = alignForward(@sizeOf(Header), 16)`; block =
   `dataOffset() + cap*@sizeOf(u16)`. **Refresh `values` ptr *and* length after every
   capacity change** (in-place resize *and* move), per the umbrella D2 rule.
4. **Single-alloc, derived-accessor (16-byte)** — one aligned block; `Header {
   cardinality: u16, capacity: u16 }`, `fn values(self) [*]u16 = @ptrCast(self) +
   dataOffset()`, never stored.

Attribution: `(3 or 4) − (2)` = pure single-alloc effect; `(3 or 4) − (1)` = the
**complete proposed change** (what ships) — that difference is the Q1 headline.

Each variant needs just enough to run the workloads: `init`, `initCapacity` (reserve
up front), `add`, membership, iterate, cardinality, `clone`, `deinit`. Block
alignment ≥ 4 (trivially satisfied at 16). Use the shared `dataOffset()`/`blockSize()`
math from the umbrella.

**Growth mechanism differs by design — do not force one path on all variants:**
- **Baseline (1) and control (2)** keep the **current shipped growth**: alloc-new +
  copy + free-old, **no `allocator.resize`** (the two-alloc `ArrayContainer` never
  tries resize). Control is baseline's growth path, just at 16-byte alignment.
- **Single-alloc (3, 4)** grow via **`resize`-then-move** (try in-place `resize`
  first, alloc-new+copy+free-old on failure), per the umbrella D3/growth rule.

This asymmetry is *part of the proposed change* (trying resize is new), so the
build-growth workload measures shipped-growth vs proposed-growth honestly — but it's
exactly why build-growth is reported separately from the reserved-capacity headline.

## Counting-allocator harness (keep this — reused in 13-06)

A real, reusable `std.mem.Allocator` wrapper over the **authoritative backing
allocator `std.heap.smp_allocator`** (same one for every variant). It tallies,
distinctly:

- **`alloc` calls** and **`free` calls** (the headline count — a two-alloc container
  is 2 `alloc`, single-alloc is 1).
- **`resize` calls**, split into **in-place successes vs failures**. **An in-place
  `resize` is NOT a new allocation** — count it separately, never as an `alloc`. (A
  failed resize that falls back to alloc-new+free *does* increment `alloc`+`free`.)
- **`remap` calls** — Zig 0.16's `Allocator` vtable has **four** ops (`alloc`,
  `resize`, `free`, `remap`); the wrapper **must implement all four** even though this
  prototype's containers don't call `remap`. Forward `remap` to the backing and count
  it like `resize`: in-place success is **not** a new alloc; a relocation counts as one
  move (not alloc+free). Assert `remap` is unused in the prototype (count stays 0) so a
  stray call is caught.
- **cumulative bytes** requested, **live bytes** (outstanding), and **peak live
  bytes**.

**`resetStats` semantics (matters for clone/deinit, which run on already-live
containers):** reset zeros the **per-region call counters** (`alloc`/`free`/`resize`/
`remap` counts and cumulative bytes) but **must not touch the live-bytes gauge** —
those allocations are still outstanding and clone/deinit legitimately act on them.
`free` calls during the deinit region are counted in that region even though the
matching `alloc` happened earlier. **peak** is tracked per region (reset to the current
live-bytes value at region start).

Report all of these per workload. Make it a proper reusable test util, not throwaway
(13-06 reuses it).

## Benchmark environment (pinned)

Not just printed — **pinned**, and stated in the recorded result:

- **Optimize mode:** `ReleaseFast`.
- **CPU target:** `-Dcpu=native` (the prototype has no comptime-gated SIMD, but pin it
  so the block/copy codegen is the host's).
- **Allocator:** `std.heap.smp_allocator` (the counting wrapper's backing, above).
- **Machine(s):** the authoritative host is named in the result (spec 14's env header
  stamps zig/mode/os/arch/cpu automatically). If run on more than one machine, each
  table names its host; the go/no-go is judged on the authoritative one.

## Corpus (pinned — identical input for every variant)

The corpus is generated **once, before any timing**, and **all four variants receive
byte-identical input** (same values, same insertion order) so the only difference
measured is the layout:

- **N = 10,000** containers; **fixed seed** (a named constant, logged).
- **Shape proportions (fixed):** 50% small (cardinality 1–64), 35% medium (256–1024),
  15% near-threshold (≈4000). Cardinality within each band drawn from the seeded PRNG.
- **Values:** distinct u16 per container from the seeded PRNG.
- **Insertion order per container: pre-sorted.** Insert in ascending order so
  `add` is append-mostly — random-order insertion adds shift/search cost that would
  swamp the layout signal. (If insertion-order sensitivity matters, that's a separate
  measurement, not this one.)
- Materialize the value arrays up front; the timed builds consume them.

## Workloads (median of ≥9; boundaries defined)

**Benchmark protocol (applies to every workload):**
- Corpus generation, clone *sources*, and result buffers are allocated **outside** the
  timed/counted region.
- **Reset the allocator counters** at the start of each measured region; snapshot at
  the end.
- **`deinit` is timed and counted separately** from build (its own line), not folded
  into build.
- **Consume outputs** with `std.mem.doNotOptimizeAway` (membership booleans,
  iteration/cardinality sums, cloned pointers) so nothing is optimized out.
- **Run variants in rotated order** across trials (not variant-1-all-trials then
  variant-2), to spread thermal/cache drift evenly; report the median per variant.

Workloads:

- **build (reserved capacity)** — `initCapacity` to the final cardinality, then
  append. Expected: baseline/control ≈ **2N** allocs, single-alloc ≈ **N** (this is
  the clean 2→1 headline; it holds *because* capacity is reserved).
- **build (growth)** — `init` small, `add` one at a time, growing via resize/move.
  Alloc counts here are **> N / > 2N** and depend on the growth schedule (each grow is
  a resize or an alloc+free); report the actual counts and resize-success rate rather
  than a 2N→N expectation. This is the realistic-churn case; keep it separate from the
  reserved case so the headline isn't conflated with growth cost.
- **clone** — clone all N (sources built outside timing).
- **deinit** — free all N (its own line).
- **read: membership** — **16 probes per container** (fixed), **50% hit / 50% miss**,
  all drawn from the seeded PRNG so every variant probes the identical values. Reported
  **separately**.
- **read: iterate** — full iteration over all N. Reported **separately**.
- **read: cardinality** — cardinality sum. Reported separately, and treated as
  *secondary* for the D2 comparison — it barely touches `values`/`runs`, so it dilutes
  the stored-slice-vs-derived signal; membership and iterate are the D2-relevant reads.

Also record the **real accessor-migration count** for D2: grep `.values`/`.runs`,
then hand-filter to the actual `ArrayContainer`/`RunContainer` field accesses (the
~207/~160 grep is a rough upper bound).

## Decision criteria

**Noise floor `ε` — per workload, as a percentage.** One absolute value can't span
operations of different durations, so establish **`ε_w` per workload `w`** as a
percent: do **K = 5 complete benchmark reruns** (the whole variant×workload matrix,
fresh process each time). For each workload and variant, relative spread is
`(maximum median - minimum median) / median of the five medians`. The workload noise
floor `ε_w` is the maximum relative spread among its four variants (report every
`ε_w`). All "win/regression" judgments below are **relative** (percent) against that
workload's own `ε_w`.

- **Q1 go/no-go (whole spec):** GO requires **all** of:
  1. reserved-capacity build allocation count drops to ≈ N (from ≈ 2N) — deterministic,
     just confirm;
  2. the **complete proposed change** ((3 or 4) − baseline) is a time win of **≥ 2·ε_w**
     on **both `build-reserved` and `clone`** (both, not either — those are the
     allocation-heavy workloads the whole spec is justified by); **and**
  3. **no workload** regresses by more than its `ε_w` (no beyond-noise regression
     anywhere).
  If single-alloc lands within ±`ε_w` of baseline on build-reserved and clone
  (allocation savings don't translate to time), that's **no-go / park the spec** —
  record it and stop.
- **Q2 (D2):** pick **derived-accessor only if** it beats stored-slice by **≥ 2·ε_w on
  *both* membership *and* iterate** (both must win — a single-read win is ambiguous),
  repeatably across the 5 reruns, *and* the team judges it worth the measured migration
  count. Otherwise **default to stored-slice** (keeps every call site, no repo-wide
  refactor). Record the chosen option, the per-read numbers, and the `ε_w` values.

## Deliverable

- A results table: **4 variants** {baseline (32B, 2-alloc), control (16B, 2-alloc),
  single-alloc stored-slice, single-alloc derived} × **workloads** {build-reserved,
  build-growth, clone, deinit, membership, iterate, cardinality} for **alloc/free/resize
  counts, cumulative/live/peak bytes, and time** — plus the per-workload noise floors `ε_w` and the env
  header.
- The real accessor-migration count.
- **Edit the umbrella:** mark D2 **resolved** (chosen option + one-line numeric
  justification against the relevant `ε_w`) and record the Q1 go/no-go outcome. If go, 13-01+ proceed;
  if no-go, the spec parks with the evidence attached.

## Acceptance

- Four variants + reusable counting-allocator + pinned corpus + workloads build and run
  under the isolated `bench-proto` step; no production container/bitmap change; `zig
  build test` unaffected.
- All four variants measured on byte-identical input; the benchmark protocol
  (boundaries, counter resets, `doNotOptimizeAway`, rotated order, per-workload `ε_w` from 5 reruns) is
  followed; results table recorded.
- **D2 decided against `ε_w` with numbers and written back into the umbrella; Q1 go/no-go
  stated.**
- The counting allocator and `bench-proto` land **committed** (reusable util +
  reproducible measurement).

## Results (07/20/2026)

Authoritative host: `AARHODES-M-P120`, Apple M4 (`aarch64-macos`), Zig 0.16.0,
`ReleaseFast`, `-Dcpu=native`, counting wrapper over `std.heap.smp_allocator`.
Each number below is the median of the five fresh-process medians; each process used
one warmup and nine timed trials. All `remap` counts were zero as required.

| Variant | Workload | ms | alloc | free | resize ok/fail | requested B | live B | peak B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline-32 | build-reserved | 15.082 | 20000 | 0 | 0/0 | 18935408 | 18935408 | 18935408 |
| baseline-32 | build-growth | 17.063 | 77261 | 57261 | 0/0 | 37550816 | 18935408 | 18939504 |
| baseline-32 | clone | 0.986 | 20000 | 0 | 0/0 | 18935408 | 37870816 | 37870816 |
| baseline-32 | deinit | 0.164 | 0 | 20000 | 0/0 | 0 | 0 | 18935408 |
| baseline-32 | membership | 2.342 | 0 | 0 | 0/0 | 0 | 18935408 | 18935408 |
| baseline-32 | iterate | 2.323 | 0 | 0 | 0/0 | 0 | 18935408 | 18935408 |
| baseline-32 | cardinality | 0.006 | 0 | 0 | 0/0 | 0 | 18935408 | 18935408 |
| control-16 | build-reserved | 15.075 | 20000 | 0 | 0/0 | 18935408 | 18935408 | 18935408 |
| control-16 | build-growth | 17.524 | 77261 | 57261 | 0/0 | 37550816 | 18935408 | 18939504 |
| control-16 | clone | 0.942 | 20000 | 0 | 0/0 | 18935408 | 37870816 | 37870816 |
| control-16 | deinit | 0.170 | 0 | 20000 | 0/0 | 0 | 0 | 18935408 |
| control-16 | membership | 2.352 | 0 | 0 | 0/0 | 0 | 18935408 | 18935408 |
| control-16 | iterate | 2.330 | 0 | 0 | 0/0 | 0 | 18935408 | 18935408 |
| control-16 | cardinality | 0.006 | 0 | 0 | 0/0 | 0 | 18935408 | 18935408 |
| single-stored | build-reserved | 15.518 | 10000 | 0 | 0/0 | 19015408 | 19015408 | 19015408 |
| single-stored | build-growth | 19.468 | 48129 | 38129 | 19132/38129 | 77457376 | 19015408 | 19019536 |
| single-stored | clone | 1.545 | 10000 | 0 | 0/0 | 19015408 | 38030816 | 38030816 |
| single-stored | deinit | 0.096 | 0 | 10000 | 0/0 | 0 | 0 | 19015408 |
| single-stored | membership | 3.104 | 0 | 0 | 0/0 | 0 | 19015408 | 19015408 |
| single-stored | iterate | 2.478 | 0 | 0 | 0/0 | 0 | 19015408 | 19015408 |
| single-stored | cardinality | 0.020 | 0 | 0 | 0/0 | 0 | 19015408 | 19015408 |
| single-derived | build-reserved | 15.540 | 10000 | 0 | 0/0 | 18855408 | 18855408 | 18855408 |
| single-derived | build-growth | 19.258 | 57543 | 47543 | 9718/47543 | 76223008 | 18855408 | 18859520 |
| single-derived | clone | 1.233 | 10000 | 0 | 0/0 | 18855408 | 37710816 | 37710816 |
| single-derived | deinit | 0.091 | 0 | 10000 | 0/0 | 0 | 0 | 18855408 |
| single-derived | membership | 3.201 | 0 | 0 | 0/0 | 0 | 18855408 | 18855408 |
| single-derived | iterate | 2.507 | 0 | 0 | 0/0 | 0 | 18855408 | 18855408 |
| single-derived | cardinality | 0.019 | 0 | 0 | 0/0 | 0 | 18855408 | 18855408 |

Per-workload noise floors: build-reserved 3.30%, build-growth 4.59%, clone
31.74%, deinit 46.88%, membership 9.86%, iterate 6.90%, cardinality 25.00%.

**Q1: NO-GO / PARK.** Reserved construction did halve allocation calls from 20,000
to 10,000, but the chosen stored-slice layout was 2.89% slower on build-reserved and
56.69% slower on clone, rather than winning by `2*epsilon_w`. It also regressed
build-growth by 14.09% and membership by 32.54%, both beyond their noise floors.
The allocation reduction does not justify the moving internal ABI on this evidence.

**Q2: stored slices.** Derived accessors were 3.12% slower than stored slices for
membership and 1.17% slower for iteration; they missed the respective 19.73% and
13.80% (`2*epsilon_w`) win thresholds. The hand-filtered migration audit found **147
real `.values` references and 132 real `.runs` references** in container-owned code,
so the default stored-slice choice also avoids 279 unnecessary call-site changes.
