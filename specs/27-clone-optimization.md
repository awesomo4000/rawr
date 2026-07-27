<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 27: Clone allocation/layout optimization

Close the component [spec 26a](done/26a-clone-attribution.md) named: rawr's **clone** costs
**254.4 ns vs CRoaring's 96.9 ns** on the M4 wide-dense probe (with **teardown 144.9 vs
48.3 ns**), while the mutation it usually precedes already beats CRoaring. Payload-copy volume
is ruled out (48 vs 56 copied bytes); the **leading measured hypothesis** is **allocation
count/layout and the frees they imply** — rawr makes 20 allocations / 440 requested bytes where
CRoaring makes 18 / 288 — and Phase 1 is the experiment that tests it. The
behavior is architecture-specific: Zen 4 rawr-SMP already wins substantially, so every change
carries a Zen 4 no-regress gate.

**Staged: a measured quick win first, deeper layout work only if the numbers still warrant.**

## Phase 1 — eliminate the measured init-then-grow waste

`clone` (`src/bitmap.zig:89`) calls `Self.init(allocator)` — which allocates the top-level
key/container arrays at `INITIAL_CAPACITY` — then immediately `ensureTotalCapacity(self.size)`,
reallocating both. **`initCapacity` already exists**: initialize directly at `self.size`,
removing two allocations plus their frees per clone. **Sibling audit (tightly scoped):**
`flipDirect` already computes `result_capacity` and calls `initCapacity` directly — **confirm
and record** that it has no clone-style waste; do **not** expand Phase 1 into unrelated
constructors (`fromSorted`, etc.).

Re-measure with the `26a` diagnostics (clone body, teardown, allocation counts) and the
canonical `clone` + `clone + removeRange` rows, both hosts.

## Phase 2 — conditional deeper layout work (only if M4 remains meaningfully behind)

Candidates from the attribution, in order:

- **Single-allocation run-container clones** — a cloned 1-run container is a struct + a tiny
  runs array; combining them halves its allocations and frees. **Two preconditions, both
  resolved before any prototype:**
  1. **Ownership/deinit invariant.** `RunContainer.deinit` frees `runs` and the struct
     separately; a clone-only combined allocation cannot be freed correctly without metadata,
     a new representation, or converting **all** run containers to the combined layout. The
     feasibility analysis must produce a concrete ownership design (or a NO-GO) — size-class
     analysis alone is insufficient.
  2. **Spec-13 class-boundary analysis, redone for this case:** spec 13's NO-GO was about
     *array* containers whose power-of-two payloads fit SMP classes exactly; run payloads are
     small and unaligned, so compute the SMP size classes for combined vs split run
     allocations across realistic run counts — if combining crosses classes, stop.
- **Combined top-level key/container storage** — one allocation for both arrays. Same
  class-boundary analysis required; touches every top-level growth path, so it is the more
  invasive option and needs its own review before implementation.

Each candidate is measured independently with the `26a` diagnostics before/after; a candidate
ships only on its own numbers.

## Constraints / gates

- **Do not touch the direct `removeRange` algorithm** — its mutation body beats CRoaring
  (49.8 vs 78.5 ns); it is exonerated and closed.
- **Correctness:** a clone remains **portable-byte identical to its source**; the `26-00`
  range matrix and the full differential/`difftest` suites stay green; error paths leak-free
  (`errdefer` discipline preserved through any constructor change). **Empty-clone regression
  cases required** — `initCapacity(self.size)` gives an empty clone **zero** capacity where
  today it gets `INITIAL_CAPACITY`: cloning an empty bitmap; **adding to that clone** (growth
  from zero); singleton and multi-container clones; allocation failure during partial
  container cloning (leak-free, source untouched).
- **Zen 4 no-regress (hard):** clone-body, teardown, and the canonical `clone` /
  `clone + removeRange` rows stay within noise (≤ 5%, rerun on range overlap) on Zen 4, where
  rawr already wins.
- **Board gate:** no canonical row worsens > 5% vs the **post-26a baseline of record** —
  commit `75662a1`, summaries `misc/range-attrib-20260727-182905-summary.txt` (M4) and
  `misc/range-attrib-20260727-183135-summary.txt` (Zen 4) plus the canonical tables recorded
  with them in `docs/parity-measurement.md` — both hosts, rerun on range overlap.

## Acceptance (GO)

- Phase 1: allocation count **20 → 18** on the probe; **clone body and the canonical rows
  improve** on M4; **teardown stays neutral within noise** — Phase 1 removes the temporary
  initial arrays from the clone *body*, but the returned clone's final arrays and containers
  are identical, so final `deinit` work is unchanged and no teardown improvement may be
  claimed. Both hosts re-measured; all gates green. If M4's canonical `clone` and
  `clone + removeRange` rows reach **≤ 1.10x**, stop — Phase 2 is not attempted.
- Phase 2 (per candidate, only if attempted): **both preconditions** (ownership/deinit design
  and the class-boundary analysis) are recorded **before** implementation; the candidate
  improves M4 beyond noise on the `26a` diagnostics and canonical rows without breaking any
  gate. A candidate whose analysis fails on either precondition is recorded and dropped, not
  built.
- A documented partial (M4 improved but above 1.10x, with the residual attributed) is an
  acceptable terminal outcome — clone is a secondary operation; do not over-invest.

## Estimate

Phase 1: **S** (swap to `initCapacity`, audit siblings, re-measure). Phase 2: **M** per
candidate, gated on Phase 1's result and the class analysis.
