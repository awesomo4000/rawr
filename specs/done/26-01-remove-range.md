<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 26-01: Direct `removeRange` with sanctioned OOM coverage

Second chunk of [direct range ops](26-direct-range-ops.md). Replaces the mask-bitmap
composition in `removeRange` (`src/bitmap.zig:448`: whole-bitmap `cardinality()` + mask bitmap
+ `bitwiseDifferenceInPlace` + whole-bitmap `cardinality()` again) with direct per-container
construction, behind `26-00`'s strategy seam.

## Gate

- `26-00` complete (harness + seam + baselines).

## Design (per the parent contracts)

- **Chunk partition:** untouched / edge (partial) / interior (fully covered; a full chunk at
  either range end is interior). Overflow-safe iteration.
- Untouched containers are not visited beyond the key comparison. Edge containers get an
  in-container range removal — via the sanctioned **stack-local one-run range view** into
  `containerDifferenceInPlace`, or per-type helpers, whichever meets byte equality. Interior
  containers are freed and their slots dropped. **One** top-level compaction. removeRange
  never needs additional top-level capacity.
- **Return value** = sum of per-affected-container removed counts (before − after per touched
  container). No whole-bitmap cardinality pass.
- **OOM model (sanctioned, per parent):** infallible mutation may precede a fallible
  conversion — on conversion failure the valid mutated container remains installed; separate
  replacements are built before swap/free. On failure after partial mutation the cache is
  **updated for committed removals or invalidated — never stale**; bitmap stays `validate()`-
  green, no leak/double-free.
- **Cache:** subtract the removed sum when the cache was valid; else stays `-1`.

## Acceptance

- `26-00` harness: direct-vs-legacy **byte equality** across the full matrix + CRoaring parity,
  on both strategy settings.
- **Exhaustive allocation-failure injection** over the direct path's fallible sites: after each
  injected failure — `validate()` green, `cardinality()` correct (recomputed == cached-or-`-1`
  semantics), leak-free deinit under a leak-checking GPA.
- **Allocation counts**: the mask-bitmap allocations are gone; report legacy-vs-direct counts.
- Semantics unchanged (inclusive `[lo, hi]`, removed count returned, `lo > hi` no-op).
- `zig build test`, `zig build difftest`, `ReleaseSafe`, `ReleaseFast` green. Performance is
  **not** gated here — `26-03` owns the cross-host gate.

## Checklist

- [ ] Direct path behind the strategy seam; legacy retained
- [ ] Edge/interior partition with full-edge-chunks-as-interior; overflow-safe walk
- [ ] Removed count summed per affected container; no whole-bitmap cardinality pass
- [ ] Sanctioned OOM models followed; cache never stale on any path
- [ ] Byte-equality matrix + CRoaring parity green under both strategies
- [ ] Failure injection green (validate / cardinality / leak-free)
- [ ] Allocation counts reported vs baseline
- [ ] test / difftest / ReleaseSafe / ReleaseFast green
