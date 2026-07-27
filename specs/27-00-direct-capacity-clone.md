<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 27-00: Direct-capacity clone fix + measurement

First chunk of [clone optimization](27-clone-optimization.md). The Phase-1 quick win: remove
the measured init-then-grow waste from `clone`, prove it with focused tests, and re-measure
both hosts against the board gate.

## Change

- `clone` (`src/bitmap.zig:89`): replace `Self.init(allocator)` + `ensureTotalCapacity(self.size)`
  with **`initCapacity(allocator, self.size)`** — removing two allocations plus their frees per
  clone. `errdefer` discipline preserved.
- **Sibling audit (scoped):** confirm and record that `flipDirect` (already
  `initCapacity(result_capacity)`) has no clone-style waste. No other constructors touched.

## Tests (the empty-capacity regression set)

`initCapacity(self.size)` gives an empty clone **zero** top-level capacity where today it gets
`INITIAL_CAPACITY`:

- clone an **empty** bitmap; then **add to that clone** (growth from zero);
- **singleton** and **multi-container** clones (byte-identical to source);
- **allocation failure during partial container cloning** — leak-free under a leak-checking
  GPA, source untouched;
- the `26-00` range matrix and `difftest` stay green.

## Measurement / acceptance

- **Allocation count 20 → 18** on the `26a` wide-dense probe (`bench_range_attrib`).
- **Clone body improves on M4**; canonical `clone` and `clone + removeRange` rows improve;
  **teardown neutral within noise** (final arrays/containers are identical — no teardown
  improvement may be claimed).
- **Zen 4 no-regress:** clone body, teardown, and both canonical rows within ≤ 5% (rerun on
  range overlap).
- **Board gate:** full 39-row canonical tables on both hosts; no row > 5% worse vs the
  baseline of record (commit `75662a1`; `misc/range-attrib-20260727-182905-summary.txt` M4,
  `misc/range-attrib-20260727-183135-summary.txt` Zen 4).
- Results + updated rows recorded in `docs/parity-measurement.md`.
- `zig build test`; `zig build difftest`; `ReleaseSafe` / `ReleaseFast` green.

## Result to record (decides Phase 2)

The post-fix M4 canonical `clone` and `clone + removeRange` ratios. **≤ 1.10x → spec 27 closes
here** (`27-01` is not started). Otherwise `27-01` runs the feasibility analysis.

## Checklist

- [ ] `clone` uses `initCapacity(self.size)`; errdefer preserved
- [ ] `flipDirect` audit recorded (no waste; nothing else touched)
- [ ] Empty/add-after/singleton/multi/alloc-failure tests green; range matrix + difftest green
- [ ] 20 → 18 allocations confirmed on the probe
- [ ] M4 clone body + canonical rows improved; teardown neutral; Zen 4 within noise
- [ ] Full-board 39-row gate green both hosts vs `75662a1` baseline
- [ ] docs updated; test / difftest / ReleaseSafe / ReleaseFast green
