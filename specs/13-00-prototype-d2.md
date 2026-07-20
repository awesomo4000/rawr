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
up front), `add` (grow via `resize`-then-move), membership, iterate, cardinality,
`clone`, `deinit`. Block alignment ≥ 4 (trivially satisfied at 16). Use the shared
`dataOffset()`/`blockSize()` math from the umbrella.

## Counting-allocator harness (keep this — reused in 13-06)

A real, reusable `std.mem.Allocator` wrapper over a **fixed backing allocator**
(`std.heap.page_allocator` or a GPA — state which; use the same one for every
variant). It tallies, distinctly:

- **`alloc` calls** and **`free` calls** (the headline count — a two-alloc container
  is 2 `alloc`, single-alloc is 1).
- **`resize` calls**, split into **in-place successes vs failures**. **An in-place
  `resize` is NOT a new allocation** — count it separately, never as an `alloc`. (A
  failed resize that falls back to alloc-new+free *does* increment `alloc`+`free`.)
- **cumulative bytes** requested, **live bytes** (outstanding), and **peak live
  bytes**.

Report all of these per workload. Make it a proper reusable test util, not throwaway
(13-06 reuses it).

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
- **read: membership** — probe hits + misses across all N. Reported **separately**.
- **read: iterate** — full iteration over all N. Reported **separately**.
- **read: cardinality** — cardinality sum. Reported separately, and treated as
  *secondary* for the D2 comparison — it barely touches `values`/`runs`, so it dilutes
  the stored-slice-vs-derived signal; membership and iterate are the D2-relevant reads.

Also record the **real accessor-migration count** for D2: grep `.values`/`.runs`,
then hand-filter to the actual `ArrayContainer`/`RunContainer` field accesses (the
~207/~160 grep is a rough upper bound).

## Decision criteria

First establish the **noise floor**: run the *baseline against itself* (or one variant
twice) and take the max run-to-run median spread as `ε` (report it). Every comparison
below is judged against `ε`, so "faster/slower" has a concrete meaning.

- **Q1 go/no-go (whole spec):** GO requires **both**:
  1. reserved-capacity build allocation count drops to ≈ N (from ≈ 2N) — deterministic,
     just confirm; **and**
  2. the **complete proposed change** ((3 or 4) − baseline) is a **time win of ≥ 2×ε**
     on at least the build and clone workloads, and **no worse than −ε** (i.e. not a
     regression beyond noise) on any workload.
  If the single-alloc time is within ±ε of baseline everywhere (allocation savings
  don't translate to time), that's **no-go / park the spec** — record it and stop.
- **Q2 (D2):** pick **derived-accessor only if** its advantage over stored-slice on the
  **membership + iterate** reads is **≥ 2×ε** (repeatable across the trials, not a
  single run) *and* the team judges it worth the measured migration count. Otherwise
  **default to stored-slice** (keeps every call site, no repo-wide refactor). Record
  the chosen option, the numbers, and `ε`.

## Deliverable

- A results table: **4 variants** {baseline (32B, 2-alloc), control (16B, 2-alloc),
  single-alloc stored-slice, single-alloc derived} × **workloads** {build-reserved,
  build-growth, clone, deinit, membership, iterate, cardinality} for **alloc/free/resize
  counts, cumulative/live/peak bytes, and time** — plus the noise floor `ε` and the env
  header.
- The real accessor-migration count.
- **Edit the umbrella:** mark D2 **resolved** (chosen option + one-line numeric
  justification against `ε`) and record the Q1 go/no-go outcome. If go, 13-01+ proceed;
  if no-go, the spec parks with the evidence attached.

## Acceptance

- Four variants + reusable counting-allocator + pinned corpus + workloads build and run
  under the isolated `bench-proto` step; no production container/bitmap change; `zig
  build test` unaffected.
- All four variants measured on byte-identical input; the benchmark protocol
  (boundaries, counter resets, `doNotOptimizeAway`, rotated order, `ε` floor) is
  followed; results table recorded.
- **D2 decided against `ε` with numbers and written back into the umbrella; Q1 go/no-go
  stated.**
- The counting allocator and `bench-proto` land **committed** (reusable util +
  reproducible measurement).
