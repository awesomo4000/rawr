<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 24: Select parity — diagnosis-first

The one **persistent cross-host** default-SMP gap left after iterate was shown to be a
phantom: **`select (dense)`**, 1.675x (M4) / 1.20x (Zen 4) rawr/CRoaring. Present on both
architectures → likely a genuine implementation gap.

**Diagnosis first** (the 20a / 21 / 23 discipline). But note the honest asymmetry: unlike
iterate, the benchmark fairness issue here points **against** rawr, so this one probably is
real — verify, then likely fix.

## Fairness first — but it cuts the other way

`benchRawrSelectDense` loops `bm.select(query)` in pure Zig (`bench_croaring.zig:781`, inline-
able); `benchCRoaringSelectDense` loops `roaring_bitmap_select(bm, query, …)` — **~1M Zig→C
FFI calls** (`:1224`). Both iterate the same `select_queries` (1M random ranks < 500k).

That per-call FFI overhead **inflates CRoaring's** measured time, so it **understates** rawr's
gap. Removing it (measure CRoaring select through an in-C loop wrapper) will most likely make
rawr's ratio **worse than 1.675x**, not better — the opposite of iterate's pull-vs-push. So the
fairness check here is to get the *true* number before optimizing, not to look for a phantom.

## Phase 1 — Diagnosis (benchmark-only, canonical harness + focused split)

1. **True like-for-like select.** Measure CRoaring select via a **benchmark-only C wrapper**
   that runs the whole `select_queries` loop **in C** (one FFI call, local checksum) vs rawr's
   Zig loop. This removes the per-call FFI asymmetry and gives the real ratio, on both hosts.
2. **Where rawr's select cost goes.** `select(rank)` finds the value at sorted position `rank`:
   skip containers by cardinality until the one holding `rank`, then index within it. Split
   the **container-skip** cost (cumulative-cardinality walk across the top-level array —
   O(containers) per call, or is there an index?) from the **intra-container select** (array
   index / bitset rank-select / run walk). Compare against `roaring_bitmap_select`'s structure
   (does CRoaring keep a cheaper cumulative index or a faster intra-container path?).
3. **Corpus / container mix.** Characterize the dense `select` corpus (container types, the
   rank distribution of `select_queries`), so the dominant path is attributed, not assumed.
   Report absolute medians + ranges and **ns/query** with a named residual.

Symmetry: local/context-owned checksums, identical minimal work per query, state built inside
the timed scan; validate identical results untimed (rawr `select` == CRoaring `select` for
every query). Canonical spec-22 protocol: 3w/21t median, ≥5 fresh processes, one path per fresh
process, on M4 and Zen 4. Iteration/select does not allocate → the single **rawr non-allocating**
tuple vs CRoaring.

Phase 1 stands alone: the true like-for-like ratio + where rawr's select cost lives is the
deliverable.

## Phase 2 — Fix (conditional on Phase 1's true ratio)

Threshold (true like-for-like rawr vs CRoaring select):
- **≤ 1.10x on both hosts** → parity; no fix (and correct the row's per-call-FFI comparison).
- **> 1.10x on either host** → optimize rawr's select, lever following the attribution:
  - if the **container-skip** dominates: a cheaper cumulative-cardinality traversal / skip (or a
    cached prefix structure) — mind that any cached index must stay correct across mutation;
  - if the **intra-container select** dominates: tighten that path per container type.
- Correctness unchanged: `select` returns the same value for every rank; differential green,
  including array/bitset/run and empty/boundary ranks.

## Canonical-row note

Even if a real gap is confirmed, the `select` row's CRoaring side should move to the **in-C
loop wrapper** (removing the benchmarked per-call FFI), so the canonical number reflects
like-for-like select, not FFI overhead. If that changes the row's shape, update
`docs/parity-measurement.md`; it does not add a manifest row (still one `select` row).

## Acceptance

- **Phase 1 GO:** the true like-for-like select ratio (FFI removed) reported on both hosts, with
  rawr's cost split into container-skip vs intra-container and the container/rank mix recorded.
- **Phase 2 GO (if attempted):** true like-for-like select **≤ 1.10x on both M4 and Zen 4**, no
  canonical row regressing >5% vs the committed spec-22 baseline (rerun on range overlap),
  differential green.
- Benchmark-only for Phase 1; a Phase-2 kernel change is production, differential-covered.
- Validation: `zig build test`; `zig build difftest`; the canonical `run-compare-bench.sh` on
  both hosts; `ReleaseSafe` / `ReleaseFast` green.

## NO-GO

- The true like-for-like ratio is already ≤ 1.10x on both hosts (the 1.675x was mostly the
  per-call FFI) → correct the row, no kernel change. (Considered less likely here than for
  iterate, given the asymmetry direction.)

## Estimate

S for Phase 1 (C loop wrapper + attribution on the existing harness). Phase 2 is S–M — a
focused select-path change chosen by the diagnosis.
