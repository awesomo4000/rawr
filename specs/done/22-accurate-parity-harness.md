<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 22: Accurate CRoaring parity benchmark harness

Rebuild the parity benchmark so **every published row is measured correctly**, and audit
them all — not patch individual suspicious ones. `run-compare-bench.sh` becomes the
canonical accurate runner producing the familiar full comparison table; the all-in-one
`bench_croaring` is retained only as a quick screening dashboard.

> **Outcome (2026-07-24, commit `2950477`) — DONE; the harness is canonical.** All five
> chunks landed. `run-compare-bench.sh` is the canonical accurate runner: each of the 38 rows
> measured per-`(row, impl, allocator)` in ≥5 fresh processes, 3w/21t median + full range,
> validated outside timing, allocator-labeled (SMP/libc side-by-side + arena `reference`
> reuse), tiny ops calibrated to `ns/op` (incl. the per-implementation cardinality counts that
> killed the ~19 s over-batching artifact), identical corpus/semantics/target with
> `croaring-avx512` recorded, cross-validated on M4 and Zen 4 under macOS Bash + Windows Git
> Bash. `bench_croaring` is demoted to a screening dashboard. Full board:
> [`docs/parity-measurement.md`](../../docs/parity-measurement.md).
>
> **The board is now trustworthy — and it corrects the record.** rawr leads or ties most ops,
> but **real default-SMP gaps remain**: iterate (**1.52x M4 / 1.88x Zen 4** — the largest),
> select, and sequential addMany are the strongest general next targets. Several other gaps
> are **architecture-specific** (dense AND/OR, flip, lazy-OR: rawr *ahead* on Zen 4, behind on
> M4) — codegen, not algorithm. So the spec-21 "parity reached across the board" was premature;
> the accurate harness surfaced these, which is exactly what it was built to do.

## Why

The parity effort (specs 16–21) repeatedly found the broad `bench_croaring` numbers were
**process-context artifacts, not real gaps**. Spec 20a proved it with a controlled matrix:
the harness's **rawr-SMP** rows carried *process-global `smp_allocator` history* (earlier
allocation-heavy groups) plus *warmup-protocol sensitivity* (2/9 → 3/21 moved rawr-SMP 27%),
while CRoaring (libc) escaped it — making the ratios **structurally asymmetric**. Two of the
three "gaps" (sparse AND, sparse OR) were phantoms; rawr is actually faster isolated. We could
only trust the real board by building **focused fresh-process** executables
(`bench_parity_isolated`, the and-cardinality diagnosis).

So the correct number for *any* row already required the isolated method. The lesson is not
"fix the three rows" — it is that an in-process, group-sequenced harness cannot be trusted for
parity, and **every row deserves the isolated treatment**. This spec operationalizes the
`docs/parity-measurement.md` recommendation into the standard runner.

## Target design (the eight requirements)

1. **Keep the existing benchmark workloads** — same corpora and operations. This changes *how*
   we measure, not *what*.
2. **Fresh process per measured variant** — one process per **`(row, implementation,
   allocator)` tuple** (rawr-SMP, rawr-libc, CRoaring each isolated). "Per operation or group"
   is not enough: `std.heap.smp_allocator` is a global singleton with no reset, so co-running
   variants in one process lets them bleed allocator state into each other.
3. **Aggregate ≥ 5 process-level medians** — median of per-process medians, reported with the
   **full min/max range** (matching the trusted `run-bench-parity-isolated.sh` style — range,
   not IQR).
4. **Validate without contaminating timing state** — either in a **separate validation
   process**, or **time first, then validate untimed** before emitting the row. Do **not**
   validate before the timed region: an allocating validation recreates allocator-history
   bias. Never validate in the hot path.
5. **Matched-allocator comparison, with per-row rules (in the manifest).** Report rawr-**SMP**
   (default) and rawr-**libc** side by side against CRoaring-libc for allocating ops; a single
   rawr number for non-allocating ops. **Which allocator controls which work is per-row:** set
   ops on prebuilt inputs vary only the *result* allocator (inputs SMP-built); **construction
   ops** (`add`, `addMany`, `deserialize`) allocate *throughout*, so the whole op runs under
   the variant's allocator; `serialize`, `toArrayAlloc`, `flip`, `removeRange` have distinct
   allocation boundaries the manifest states explicitly.
6. **Batch tiny operations mechanically** → `ns/op`, not `0.00 ms`. Use a fixed batch count
   per implementation, identical by default, with a manifest-recorded override when measured
   algorithmic asymmetry makes a shared count impractical. Each timed sample must exceed the
   **stated floor (≥ 1 ms)** and results are normalized to ns/op. **Stateful ops** (`flip`,
   `removeRange`, in-place) recreate/reset their state consistently between repetitions.
   **Tiny pure queries** (`cardinality`, `contains`)
   need more than `doNotOptimizeAway` — it prevents *removal* but not *hoisting* an invariant
   pure call, so require **equivalent non-inline / opaque call boundaries for rawr and CRoaring**
   and **runtime-varying inputs** where applicable. (This avoids repeating the earlier
   range-cardinality artifact where rawr inlined and the C reference did not.)
7. **Identical data, semantics, requested target/CPU, and timing boundaries** for rawr and
   CRoaring — same inputs, same `-Dcpu=native`, same warmup/timed protocol, same
   construct-vs-timed boundaries. Mind the known interop equivalences (`addRange` inclusive vs
   CRoaring exclusive → `end + 1`; `bitwiseDifference` ⇄ `andnot`). "Identical *features*" is
   too strong — CRoaring AVX-512 may be disabled — so **record each implementation's effective
   feature configuration** rather than assert they match: report **`croaring-avx512=on/off`**
   and rawr's `-Dcpu=native` features in the header.
8. **Retain `bench_croaring` only as a quick screening dashboard** — explicitly *not* the
   parity oracle.

## Row manifest (mandatory)

"Every published row" requires an explicit inventory — the audit (`22-01`) is driven by it and
it is the contract the harness executes. Each row records:

- **stable row ID** and display name;
- **corpus / seed**;
- **rawr operation** and **CRoaring operation** (the matched pair);
- **allocating / non-allocating** classification;
- **allocator variants** measured — rawr SMP, rawr libc, CRoaring, and **rawr arena** for rows
  that have an arena variant (`sparse AND arena`, `sparse OR arena`, `deserialize arena`).
  Arena is a **supplemental rawr variant**; CRoaring has no arena mode, so an arena row's
  CRoaring column **reuses the standard CRoaring baseline** (labeled as such), showing the
  arena's effect on rawr against the same reference;
- **setup and teardown timing boundaries** (what is inside vs outside the timed region);
- **validation oracle** (byte-identity vs logical-equality, against what);
- **batch count and reporting unit** (`ms` or `ns/op`).

**Outcome:** `run-compare-bench.sh` is the canonical accurate runner and prints the familiar
full comparison table, every row trustworthy for target selection.

## Proposed chunk plan (confirm at review)

- **`22-00` Harness architecture** — **define the manifest schema**, and implement the per-`(row,
  implementation, allocator)` fresh-process runner + ≥5-process median-with-range aggregation +
  validate-without-contaminating-timing framework + output-table format, proven on **two
  representative pilot rows** (one allocating, one tiny). The foundation the rest build on;
  generalizes `bench_parity_isolated` / `run-bench-parity-isolated.sh` into the standard runner.
- **`22-01` Workload audit** — **populate the complete manifest** for every published
  `bench_croaring` row using `22-00`'s schema, port each to the isolated harness, and confirm or
  correct its real number. The "audit every row" deliverable; produces the **complete functional
  table** — final trustworthiness is reached after `22-02`'s allocator completion and `22-03`'s
  tiny-op calibration.
- **`22-02` Allocator parity** — matched-allocator handling where **allocator scope follows the
  row manifest** (result-allocator-only for set ops, whole-op for construction ops, named
  boundaries for `serialize`/`toArrayAlloc`/`flip`/`removeRange`), plus the supplemental
  **rawr-arena** variants; single rawr number for non-allocating ops.
- **`22-03` Tiny-operation calibration** — batching + `ns/op` for the sub-clock ops that
  currently read `0.00 ms`; define batch sizes that clear clock resolution.
- **`22-04` Cross-machine validation** — run the canonical table on the M4 and the x86-64 /
  Zen 4 host, **same requested target/protocol with each implementation's effective feature
  configuration recorded**, so every row is validated on both architectures.

## Constraints

- **Benchmark-only** — no production/library change, no committed vendored-source change.
- **Do not silently change what is measured** — corpora and operation semantics stay as they
  are; only the measurement method changes. Any corpus that must change to enable batching or
  validation is called out explicitly.
- Record target / CPU / features **and `croaring-avx512=on/off`** in the header (spec 14 style).
- **Named committed artifact** — the canonical table is committed to a durable path (e.g.
  `docs/parity-measurement.md` and/or a named `misc/`-summary path that is *committed*, not
  ignored); the chunk names the exact path.
- **Portable runner** — `run-compare-bench.sh` must work under **macOS Bash** and **Windows Git
  Bash** (both hosts run the canonical table).

## Acceptance (toplevel; per-chunk gates in the chunks)

- `run-compare-bench.sh` produces the familiar full comparison table where **every row** is run
  as **≥5 independent fresh processes per `(row, implementation, allocator)` tuple**, reported as
  median + **full min/max range**, validated without contaminating timing, allocator-labeled
  (SMP/libc for allocating ops), and `ns/op` for tiny ops — with identical data / semantics /
  **requested target/CPU** / timing boundaries for rawr and CRoaring, and each implementation's
  effective feature configuration recorded (incl. `croaring-avx512=on/off`). Runs under macOS
  Bash and Windows Git Bash.
- **Every published row audited** against its manifest entry; no row remains that is a known
  process-context artifact. The prior sparse AND/OR isolated results are **sanity anchors, not a
  gate** — a correct harness is allowed to report a *changed* number; correctness is never
  gated on rawr staying ahead.
- `bench_croaring` retained but explicitly demoted to a screening dashboard.
- The canonical table is cross-validated on M4 and Zen 4, committed to the named artifact.

## Estimate

L. A harness rebuild plus a full row-by-row audit and cross-machine validation — five chunks,
sequenced with `22-00` first and the rest layered on the framework.
