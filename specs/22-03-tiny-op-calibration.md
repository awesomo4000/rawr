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
- **Batch mechanically** → `ns/op`: a **fixed identical batch count** for rawr and CRoaring,
  normalized to ns/op. The **≥ 1 ms floor applies per timed sample on each host** — if a batch
  falls below 1 ms per timed sample on **either** M4 or Zen 4, increase the **shared** batch
  count and rerun **both**; aggregate runtime over the whole process is *not* sufficient. The
  batch count is recorded in the manifest.
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

- **No `0.00 ms` rows remain** — every genuinely tiny op reports `ns/op` with an identical batch
  count on both sides and **≥ 1 ms per timed sample on both hosts**.
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
- [ ] Fixed identical batch count both sides; ≥ 1 ms **per timed sample on both** M4 and Zen 4
      (raise shared count and rerun both if under)
- [ ] Stateful ops (`flip`/`removeRange`/in-place) reset state consistently between repetitions
- [ ] Genuinely tiny invariant queries (`cardinality`): symmetric non-inline/opaque call
      boundaries + runtime-varying inputs
- [ ] `contains` and other already-substantial rows keep their existing workload (not batched)
- [ ] No `0.00 ms` rows remain; calibrated rows still validate against their oracle
- [ ] `zig build test`, ReleaseSafe, ReleaseFast all green; benchmark-only
