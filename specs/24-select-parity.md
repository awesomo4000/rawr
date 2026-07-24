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
gap — removing it *raises* rawr/CRoaring, it does not lower it (the opposite of iterate's
pull-vs-push). But an in-C loop removes only the **Zig→C crossings**, not the **public
function-call** asymmetry: CRoaring's `roaring_bitmap_select` is a **non-inlined** call per
query, while rawr's `select` may **inline** into its loop. A fair comparison must separate those,
so the fairness step is a **call-boundary matrix**, not a single wrapper.

## Phase 1 — Diagnosis (benchmark-only, canonical harness + focused split)

1. **Call-boundary matrix (four paths).** Separate language-boundary, ordinary function-call,
   and implementation costs:
   - **rawr inline loop** — `select` **forced-inline** (`@call(.always_inline, …)` or equivalent),
     **confirmed by disassembly** to be incorporated into the loop (a bare `bm.select()` does not
     guarantee inlining);
   - **rawr via a `noinline` select wrapper** — same rawr kernel, forced non-inlined call;
   - **CRoaring via the current Zig loop** — per-query Zig→C FFI (the board's current path);
   - **CRoaring via an in-C loop** — one FFI call, loop in C.
   The **canonical-row comparison follows from this** — likely **rawr-noinline vs CRoaring-in-C**
   (both a non-inlined public call, neither paying a Zig→C tax), the honest public-API
   like-for-like. Report on both hosts.
   **Directional sanity checks** — the measured ranges must support these; if any fails
   materially, investigate codegen/harness shape rather than accept the ratio:
   - rawr-inline **no slower** than rawr-noinline;
   - CR-in-C **no slower** than the repeated Zig→C path (or their ranges overlap);
   - the existing board ratios (**1.675x M4, 1.20x Zen 4**) are a **lower-bound expectation per
     host** for rawr-noinline / CR-in-C — the fair comparison should not come out *better* for
     rawr than the FFI-inflated board number on **either** host.
   Confirm by **disassembly** that both selected canonical paths retain **one non-inlined public
   call per query**.
2. **Where rawr's select cost goes.** `select(rank)` finds the value at sorted position `rank`:
   skip containers by cardinality until the one holding `rank`, then index within it. Split
   the **container-skip** cost (cumulative-cardinality walk across the top-level array —
   O(containers) per call, or is there an index?) from the **intra-container select** (array
   index / bitset rank-select / run walk). Compare against `roaring_bitmap_select`'s structure
   (does CRoaring keep a cheaper cumulative index or a faster intra-container path?).
3. **Corpus / container mix.** Characterize the dense `select` corpus (container types, the
   rank distribution of `select_queries`), so the dominant path is attributed, not assumed.
   Report absolute medians + ranges and **ns/query** with a named residual.

Symmetry: only the **accumulator/checksum state is local to the timed loop**; the bitmap and the
`select_queries` are built **outside** timing. Identical minimal work per query; validate
identical results untimed (rawr `select` == CRoaring `select` for every query). Canonical spec-22
protocol: 3w/21t median, ≥5 fresh processes, one path per fresh process, on M4 and Zen 4. Select
does not allocate → the single **rawr non-allocating** tuple vs CRoaring.

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

Even if a real gap is confirmed, the `select` row should move to the **public-API like-for-like
comparison the matrix identifies** (likely rawr-noinline vs CRoaring-in-C), so the canonical
number reflects a fair public select call, not the Zig→C FFI or an inline-vs-noinline mismatch.
If that changes the row's shape, update `docs/parity-measurement.md`; it does not add a manifest
row (still one `select` row).

## Acceptance

- **Phase 1 GO:** the **matrix-selected public-API select ratio** reported on both hosts (four-path
  matrix + directional sanity checks passing), with rawr's cost split into container-skip vs
  intra-container and the container/rank mix recorded.
- **Phase 2 GO (if attempted):** matrix-selected public-API select **≤ 1.10x on both M4 and Zen 4**, no
  canonical row regressing >5% vs the **latest committed corrected parity baseline** (spec 23
  changed the iterate row) — rerun on range overlap — differential green.
- Benchmark-only for Phase 1; a Phase-2 kernel change is production, differential-covered.
- Validation: `zig build test`; `zig build difftest`; the canonical `run-compare-bench.sh` on
  both hosts; `ReleaseSafe` / `ReleaseFast` green.

## NO-GO

- The public-API like-for-like ratio turns out ≤ 1.10x on both hosts → correct the row to the
  chosen comparison, no kernel change. (Considered **less likely** here than for iterate: the FFI
  asymmetry *inflated* CRoaring, so removing it **raises** rawr/CRoaring — it does not lower it —
  and the inline-vs-noinline matrix isolates whatever remains.)

## Estimate

S for Phase 1 (C loop wrapper + attribution on the existing harness). Phase 2 is S–M — a
focused select-path change chosen by the diagnosis.
