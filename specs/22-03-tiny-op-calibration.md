<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 22-03: Tiny-operation calibration

Fourth chunk of [accurate parity harness](22-accurate-parity-harness.md). Replace the
sub-clock `0.00 ms` rows with mechanically-batched `ns/op`. **This is where the table reaches
final trustworthiness.**

## Gate

- `22-01` functional table (may proceed alongside `22-02`; they touch different manifest
  fields).

## Deliverables

- **Identify the sub-clock rows** — those currently reading `0.00 ms` (e.g. `addRange`,
  `cardinality`, `flip`, `removeRange`, and any others the manifest flags), and any whose
  single-shot time is at clock resolution.
- **Batch mechanically** → `ns/op`: use a fixed batch count per implementation and normalize
  to ns/op. Use an identical count by default. A manifest variant may override it when a measured
  algorithmic asymmetry would make the shared count impractical. The **≥ 1 ms floor applies per
  timed sample for each implementation on each host**; aggregate runtime over the whole process
  is *not* sufficient. Every effective batch count is recorded in the manifest.
- **Cardinality requires independently calibrated counts.** Rawr returns a cached bitmap-wide
  cardinality in O(1), while CRoaring's `roaring_bitmap_get_cardinality` walks the container array
  and sums container cardinalities on every call. A shared count high enough to measure Rawr made
  each CRoaring timed sample take about 19 seconds on the M4. Keep the corpus and public operation
  identical, calibrate each implementation above the timing floor, and compare normalized ns/op.
- **Stateful ops** (`flip`, `removeRange`, in-place) **recreate/reset their state consistently**
  between repetitions, on both sides, so each repetition measures the same work.
- **Genuinely tiny invariant queries** (e.g. `cardinality`): `doNotOptimizeAway` alone is
  insufficient — it stops removal but not hoisting of an invariant pure call. Require
  **equivalent non-inline / opaque call boundaries for rawr and CRoaring** and **runtime-varying
  inputs** where applicable, so neither side is silently hoisted. (This is exactly the
  range-cardinality inline-(rawr)-vs-out-of-line-(C) artifact to avoid.)
- **Do not batch already-substantial rows.** `contains (hit/miss)` already performs ~1M queries
  (~100 ms) — it is **not** a tiny row; preserve its existing workload rather than converting it
  to individual `ns/op` calls. The anti-hoisting rule applies only to genuinely tiny invariant
  rows.

## Acceptance

- **No `0.00 ms` rows remain** — every genuinely tiny op reports `ns/op` with a manifest-recorded
  batch count and **≥ 1 ms per timed sample on both hosts**. Counts are identical by default;
  justified per-implementation overrides preserve the same corpus and measured operation.
- Genuinely tiny invariant queries use symmetric non-inline/opaque call boundaries and
  runtime-varying inputs; stateful ops reset consistently between repetitions; `contains` (and
  other already-substantial rows) keep their existing workload.
- All calibrated rows still validated outside timing.
- With `22-02` complete, the full canonical table is now **trustworthy** for target selection.
- **Benchmark-only:** no production/library or vendored-source change; build green under
  `ReleaseSafe` and `ReleaseFast`.

## Validation

- `zig build test`; `zig build -Doptimize=ReleaseSafe`; `zig build -Doptimize=ReleaseFast`
- `scripts/run-compare-bench.sh` on **both** M4 and Zen 4 shows **no `0.00 ms` row**; every tiny
  row reports `ns/op` with ≥ 1 ms per timed sample on each host
- validate that tiny-query results still match their oracle (anti-hoisting did not change the
  computed value)

## Checklist

- [ ] Sub-clock rows identified and batched → `ns/op`; batch count recorded in the manifest
- [ ] Fixed batch count per implementation; identical by default, with justified manifest
      overrides; ≥ 1 ms **per timed sample on both** M4 and Zen 4
- [ ] Stateful ops (`flip`/`removeRange`/in-place) reset state consistently between repetitions
- [ ] Genuinely tiny invariant queries (`cardinality`): symmetric non-inline/opaque call
      boundaries + runtime-varying inputs
- [ ] `contains` and other already-substantial rows keep their existing workload (not batched)
- [ ] No `0.00 ms` rows remain; calibrated rows still validate against their oracle
- [ ] `zig build test`, ReleaseSafe, ReleaseFast all green; benchmark-only
