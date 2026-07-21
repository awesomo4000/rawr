<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 19-00: Consuming-OR prototype + measurement (private, benchmark-only)

First chunk of [consuming (move) in-place OR](19-consuming-move-union.md). **No public
API.** Builds a private consuming-merge, recounts the real eager baseline, and produces the
GO/NO-GO numbers that decide whether `19-01` (the public method) is ever written. A NO-GO
leaves nothing behind but the benchmark.

## Deliverables

### Private consuming merge

A private (or benchmark-only) implementation of the consuming in-place OR following the
toplevel commit protocol exactly — this is the real algorithm, just not yet a public
method:

- Preconditions (no mutation): same allocator (`ptr` **and** `vtable`) else
  `error.AllocatorMismatch`; distinct pointers (`self != other`) else
  `error.AliasedOperands`.
- Reserve `self`'s index arrays to `self.size + other`-unmatched-chunk-key count (all
  fallible index allocation here).
- Set `self.cached_cardinality = -1` **before** the first matched merge.
- Merge matched chunk keys into `self`'s existing containers (existing in-place merge).
- Infallible commit: free `other`'s redundant matched containers, insert `other`'s
  unmatched tagged pointers by **backward merge** into the pre-reserved arrays (no growth,
  no allocation), then set `other.size = 0` and `other.cached_cardinality = 0`.

Correctness of this prototype is validated in-bench (below); the full differential +
failure-injection suite is `19-01`'s responsibility (it ships with the public API).

### Eager-baseline allocation recount

Measure the allocation count of the **current eager `bitwiseOrInPlace`** on the sweep and
fixpoint workloads. The spec-17 ~98k figure is forced-lazy A1, not this path — no savings
claim may reference it; this recount is the real baseline.

### Overlap sweep harness

Deterministic unmatched-chunk-key sweep at **0 / 25 / 50 / 75 / 100%** of `other`'s chunk
keys unmatched in `self`, with rounds, delta sizes, container types, and key distribution
pinned and documented. Reuse the spec-17 counting allocator; both variants (current
cloning `bitwiseOrInPlace` and the prototype) run on byte-identical inputs.

### Fixpoint-pattern bench

Repeated `R := R ∪ ΔR` over many rounds, `ΔR` freshly built then consumed each round, with
the three separated timing boundaries:

- union operation only (`ΔR` construction outside the timed region);
- full round lifecycle (including `ΔR` construction and cleanup);
- allocator counters reset immediately around the union operation.

## Acceptance

- **Allocation correctness (exact):** on the sweep, the current path's unmatched-right
  container clones equal **exactly `2 × moved_container_count`**; the prototype's are
  **exactly zero**; index-array growth and matched-merge allocations are reported
  **separately** and never folded into that count.
- **Prototype correctness in-bench:** the prototype's repaired result is **set-equal** to
  the current `bitwiseOrInPlace` on every sweep point and every fixpoint round; the emptied
  `other` reports `cardinality() == 0` and re-validates; a smoke run reuses and then
  deinits `other` leak-free under a leak-checking GPA. (Exhaustive failure injection is
  `19-01`.)
- **Numbers emitted** on the authoritative environment (`ReleaseFast`, native, M4 host),
  five independent process runs, median + range, for both variants across the sweep and the
  fixpoint bench, with the three timing boundaries and the allocation attribution.
- **No public API** and no change to `bitwiseOr` / `bitwiseOrInPlace`; full build green
  under `ReleaseSafe` and `ReleaseFast`.

## Result to record (the decision input)

- The eager-baseline allocation count (recounted), and the prototype's savings as a
  function of unmatched-chunk-key overlap.
- The fixpoint union-only median improvement at each sweep point.
- Whether the exact-allocation gate holds (drop == `2 × moved`, zero clones consuming).

This feeds the GO/NO-GO. **GO still additionally requires real-driver overlap data** (a
supplied trace or driver instrumentation) to know which sweep point is realistic —
`19-01` does not start until both this chunk clears its gates and that data exists.
