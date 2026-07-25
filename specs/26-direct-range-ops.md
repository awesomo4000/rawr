<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 26: Direct construction for flip and removeRange

Replace the clone-plus-mask composition behind `flip` / `flipInplace` / `removeRange` with
**direct per-container construction**, the way the reference implements these ops. This is the
largest remaining canonical-board number (removeRange **2.167x M4**) and — unlike the spec-25
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

Canonical-board standing (rawr/CRoaring, SMP): removeRange **2.167x M4 / 1.078x Zen 4**; flip
**1.767x M4 / 0.565x Zen 4** (rawr *ahead* on Zen 4 — see the gate below). **Baseline of
record for all gates:** the per-host canonical tables committed in
`docs/parity-measurement.md` at commit `190f6d4` (M4 and Zen 4 sections) — not the earlier
pre-select-fix summaries.

## Design

For a range `[lo, hi]` (inclusive both ends — rawr's single range convention):

- **Partition the chunk keys** into: below/above the range (untouched), **edge chunks**
  (partially covered), and **interior chunks** (fully covered). **A full chunk at either end of
  the range is interior**, not edge — the edge path is only for genuinely partial coverage
  (e.g. `lo` at an exact multiple of 65536 makes that first chunk interior).
- **Overflow-safe chunk iteration:** the walk over chunk keys must handle
  `[0, maxInt(u32)]` — never increment a `u16` key through 65535; iterate with a wider index
  (the existing top-level `usize` index over `keys[0..size]` is the natural shape).
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
- **Representation contract (pinned): direct-vs-legacy portable-byte equality.** The direct
  paths must produce results whose **portable serialization is byte-identical to the legacy
  rawr composition's output**, across the **full test matrix** (not just the one canonical
  corpus) — plus **CRoaring set parity** (logical equality) as the independent oracle. This is
  stronger than the existing `assertSameValues` flip/remove differential, deliberately: it pins
  rawr's container-selection behavior so the rewrite cannot silently change representations.
  **Legacy-implementation lifecycle:** the legacy composition **remains in-tree during
  development** as the byte-equality reference. If a **single direct implementation** wins both
  hosts, legacy is then removed and the contract is preserved by **pinned serialization
  fixtures** (or a test-only copy) — `26-00`'s equality harness is a development gate, not a
  permanent dual-implementation requirement. If **per-arch selection** ships, both
  implementations are retained and both stay tested (via the strategy override).
- **Sanctioned implementation shape:** a **stack-local one-run range view** fed to the existing
  `containerDifferenceInPlace` / `containerXorInPlace` kernels is an acceptable (likely the
  cleanest) way to eliminate the bitmap-level mask while preserving today's representation
  decisions — the per-container kernels already encode the conversion thresholds. Direct
  per-type range helpers are equally acceptable if they meet the byte-equality contract.

## Constraints / gates

- **Zen 4 no-regress gate (hard), with a sanctioned fallback.** flip is currently **0.56x —
  ahead — on Zen 4** with the existing composition; removeRange is 1.08x. Preference order,
  decided by measurement:
  1. **Single direct implementation** if it is neutral-or-better on both hosts (≤ 5% worse per
     row counts as noise, rerun on range overlap) — one code path is always preferred.
  2. **Comptime per-arch selection** if direct wins M4 but loses Zen 4 (zero runtime cost;
     precedent: the per-arch skew thresholds in `array_kernels.zig`). **Exact selector, per op:**
     `aarch64` → direct; `x86_64` → legacy only if direct measurably regresses there; **all
     other architectures → an explicit documented choice** (default direct, as the
     fewer-allocations path, unless stated otherwise). Provide an **internal comptime strategy
     override** (build option or comptime flag) so **tests exercise both implementations on any
     host** — without it, one branch is effectively untested off its home arch. Both arms must
     produce **byte-identical portable serializations** — only speed may differ by arch — and
     both arms keep full differential + failure-injection coverage. Accept the doubled
     maintenance surface only for the op(s) where the split is measured to pay.
  3. Direct loses on **both** hosts (unexpected) → record and stop; keep the composition.
  Either way, an M4 win is never bought with an x86 loss.
- **Error semantics — two sanctioned models (basic guarantee).** Reconciling the OOM contract
  with the sanctioned in-place kernels (whose bitset paths mutate first and may then fail while
  allocating a demoted array, `container_ops.zig:435`):
  - **Infallible mutation of an existing container may precede a fallible representation
    conversion.** If the conversion fails, the valid mutated container **remains installed**
    and the bitmap cache is invalidated — no temporary 8 KB bitset copy required.
  - **Transformations that require a separate replacement** (flip inserting missing chunks,
    conversions that build a new container, run growth/splitting) **build the replacement fully
    before swapping/freeing the original.**
  **Compute and reserve a safe upper bound** on top-level capacity before the first mutation
  (flip: current containers + missing covered chunk keys; removeRange: never needs additional
  top-level capacity), so commit-phase inserts cannot fail. On OOM the bitmap remains **valid**
  (passes `validate()`, `cardinality()` correct or cache invalidated), possibly partially
  modified, no leak or double-free. By-value `flip` cleans up fully on error (`errdefer`),
  inputs untouched.
- **Cardinality-cache accounting, per op:** `removeRange` subtracts the summed per-container
  removed count when the cache was valid (else stays `-1`); **on an allocation failure after
  partial mutation, the cache is either updated for the removals already committed or
  invalidated before returning — never left stale.** **`flip`/`flipInplace` invalidate to `-1`
  before the first committed mutation** — no delta accounting — and their failure path
  therefore always leaves the cache invalidated, never stale.
- **Semantics unchanged**: inclusive `[lo, hi]`; `removeRange` returns the removed count;
  `lo > hi` no-ops for the in-place variants and **by-value `flip(lo > hi)` still returns an
  independent clone** (today's behavior: clone then no-op); public signatures unchanged.

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

## Proposed chunk plan (confirm at review)

1. **`26-00`** — baseline representation tests (portable-byte legacy-vs-direct harness across
   the full matrix), allocation-count instrumentation, and the comptime **strategy test seam**.
2. **`26-01`** — direct `removeRange` with sanctioned OOM coverage.
3. **`26-02`** — direct `flipInplace` + by-value `flip` (and `flipOwned` via the same path).
4. **`26-03`** — cross-host performance gate, per-arch selection if the numbers require it, and
   `docs/parity-measurement.md` update.

## Estimate

M. Per-type in-container range removal / flip helpers (some exist: `clearRange`,
`toggleRange`, `setRange`), the chunk-partition walk, cardinality bookkeeping, and the
failure-injection coverage. The design is well-understood; the work is the per-type edge cases
and the error-path discipline.
