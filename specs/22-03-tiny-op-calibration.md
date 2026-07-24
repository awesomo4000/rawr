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
  **total measured duration above a stated floor (≥ 1 ms)**, normalized to ns/op. The batch
  count is recorded in the manifest.
- **Stateful ops** (`flip`, `removeRange`, in-place) **recreate/reset their state consistently**
  between repetitions, on both sides, so each repetition measures the same work.
- **Tiny pure queries** (`cardinality`, `contains`): `doNotOptimizeAway` alone is insufficient —
  it stops removal but not hoisting of an invariant pure call. Require **equivalent non-inline /
  opaque call boundaries for rawr and CRoaring** and **runtime-varying inputs** where
  applicable, so neither side is silently hoisted. (This is exactly the range-cardinality
  inline-(rawr)-vs-out-of-line-(C) artifact to avoid.)

## Acceptance

- **No `0.00 ms` rows remain** — every tiny op reports `ns/op` with an identical batch count on
  both sides and total measured duration ≥ the floor.
- Tiny pure queries use symmetric non-inline/opaque call boundaries and runtime-varying inputs;
  stateful ops reset consistently between repetitions.
- All calibrated rows still validated outside timing.
- With `22-02` complete, the full canonical table is now **trustworthy** for target selection.
- **Benchmark-only:** no production/library or vendored-source change; build green under
  `ReleaseSafe` and `ReleaseFast`.
