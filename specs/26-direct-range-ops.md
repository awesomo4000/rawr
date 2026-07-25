<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 26: Direct construction for flip and removeRange

Replace the clone-plus-mask composition behind `flip` / `flipInplace` / `removeRange` with
**direct per-container construction**, the way the reference implements these ops. This is the
largest remaining canonical-board number (removeRange **2.313x M4**) and — unlike the spec-25
SIMD theory — the cause here is **structural and verified in source**, not a timing hypothesis:

- `flip(lo, hi)` (`src/bitmap.zig:1139`): **deep-clones the entire bitmap**, then `flipInplace`
  (`:1642`) builds a **separate mask bitmap** via `addRange` and runs `bitwiseXorInPlace`
  against it. Every container is cloned, and a whole temporary bitmap is allocated, populated,
  and freed — to change only the containers the range touches.
- `removeRange(lo, hi)` (`:448`): computes **whole-bitmap `cardinality()`**, builds a **mask
  bitmap**, runs `bitwiseDifferenceInPlace`, then computes **whole-bitmap `cardinality()`
  again** for the return value.

The reference constructs directly: copy/keep unaffected containers untouched, mutate (or build)
only the containers whose chunk the range intersects, and compact the top level once. No mask
bitmap, no whole-bitmap clone, no whole-bitmap cardinality recompute.

This is allocation-**demand** reduction — the one lever that has repeatedly proven out (specs
18/19) — applied to the two ops still carrying the composition.

Canonical-board standing (rawr/CRoaring): removeRange **2.313x M4 / 1.08x Zen 4**; flip
**1.771x M4 / 0.56x Zen 4** (rawr *ahead* on Zen 4 — see the gate below).

## Design

For a range `[lo, hi]` (inclusive both ends — rawr's single range convention):

- **Partition the chunk keys** into: below/above the range (untouched), **edge chunks**
  (partially covered), and **interior chunks** (fully covered).
- **`removeRange`** (in-place): untouched containers stay as-is; edge containers get an
  in-container range removal (per type: array splice, bitset `clearRange` + demote check, run
  split/trim); interior containers are freed and their slots dropped; compact the top-level
  key/container arrays **once**. Return value = **sum of per-affected-container removed counts**
  (before − after per touched container) — no whole-bitmap cardinality pass. Update
  `cached_cardinality` consistently (subtract if it was valid, else stay `-1`).
- **`flipInplace`**: untouched containers stay; edge containers flip a sub-range in-container;
  interior chunks: an existing container is flipped whole (with type conversion / drop-if-empty
  as needed), a **missing** interior chunk becomes a full-range container (a 1-run
  `RunContainer`, as `addRangeToChunk` already builds). Compact once.
- **`flip` (by-value)**: construct the result directly — clone only untouched containers, build
  flipped containers for affected chunks. **No whole-bitmap clone, no mask bitmap.**
  `flipOwned` follows via the same path.
- Result-type discipline unchanged: containers produced by a flip/removal convert to the correct
  representation (array/bitset/run thresholds) and empty containers are dropped, matching what
  the current composition produces — **the resulting bitmap must be representation-identical to
  today's output**, not just set-equal, so canonical validation stays byte-stable.

## Constraints / gates

- **Zen 4 no-regress gate (hard).** flip is currently **0.56x — ahead — on Zen 4** with the
  existing composition; removeRange is 1.08x. The direct paths ship only if Zen 4 stays within
  noise of current (≤ 5% worse per row, rerun on range overlap) — an M4 win must not be bought
  with an x86 loss. If direct construction loses on Zen 4, that result is recorded and the
  design is revisited rather than shipped.
- **Error semantics (basic guarantee, matching existing in-place ops).** In-place variants may
  allocate (container conversions); on OOM the bitmap remains **valid** (passes `validate()`,
  `cardinality()` correct or cache invalidated), possibly partially modified, with no leak or
  double-free. By-value `flip` cleans up fully on error (`errdefer`), inputs untouched.
- **Semantics unchanged**: inclusive `[lo, hi]`; `removeRange` returns the removed count;
  `lo > hi` no-ops; public signatures unchanged.

## Correctness

- Differential vs the CRoaring oracle (`flip_closed` / `remove_range_closed`) across: range
  within one chunk, spanning chunk boundaries, whole-universe, empty bitmap, range over missing
  chunks, edges at chunk boundaries (`lo`/`hi` at multiples of 65536 ± 1), and all three
  container types on the edges (incl. conversions and drop-to-empty).
- **Exhaustive allocation-failure injection** over the fallible sites of the in-place paths:
  after each injected failure the bitmap passes `validate()`, `cardinality()` is correct, and
  deinit is leak-free under a leak-checking GPA.
- Existing tests, `difftest`, and the canonical `flip` / `removeRange` rows' validation stay
  green; `ReleaseSafe` / `ReleaseFast` both build.

## Measurement / acceptance

- Canonical parity harness, both hosts, five fresh processes, median + full range.
- **GO:** removeRange and flip reach **≤ 1.10x on M4** — or retain a statistically supported
  improvement with rationale — with the **Zen 4 no-regress gate** held and no other canonical
  row worsening > 5% vs the latest committed corrected baseline (rerun on range overlap).
- Allocation counts reported before/after (the mask-bitmap and whole-clone allocations should
  disappear outright); `docs/parity-measurement.md` updated.

## Estimate

M. Per-type in-container range removal / flip helpers (some exist: `clearRange`,
`toggleRange`, `setRange`), the chunk-partition walk, cardinality bookkeeping, and the
failure-injection coverage. The design is well-understood; the work is the per-type edge cases
and the error-path discipline.
