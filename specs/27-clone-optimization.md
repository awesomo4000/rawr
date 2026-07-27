<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 27: Clone allocation/layout optimization

Close the component [spec 26a](done/26a-clone-attribution.md) named: rawr's **clone** costs
**254.4 ns vs CRoaring's 96.9 ns** on the M4 wide-dense probe (with **teardown 144.9 vs
48.3 ns**), while the mutation it usually precedes already beats CRoaring. Payload-copy volume
is ruled out (48 vs 56 copied bytes); the gap is **allocation count/layout and the frees they
imply** — rawr makes 20 allocations / 440 requested bytes where CRoaring makes 18 / 288. The
behavior is architecture-specific: Zen 4 rawr-SMP already wins substantially, so every change
carries a Zen 4 no-regress gate.

**Staged: a measured quick win first, deeper layout work only if the numbers still warrant.**

## Phase 1 — eliminate the measured init-then-grow waste

`clone` (`src/bitmap.zig:89`) calls `Self.init(allocator)` — which allocates the top-level
key/container arrays at `INITIAL_CAPACITY` — then immediately `ensureTotalCapacity(self.size)`,
reallocating both. **`initCapacity` already exists**: initialize directly at `self.size`,
removing two allocations plus their frees per clone. Audit for the same pattern in any other
by-value constructor paths touched by the range work (`flip` by-value builds a result the same
way it clones — check).

Re-measure with the `26a` diagnostics (clone body, teardown, allocation counts) and the
canonical `clone` + `clone + removeRange` rows, both hosts.

## Phase 2 — conditional deeper layout work (only if M4 remains meaningfully behind)

Candidates from the attribution, in order:

- **Single-allocation run-container clones** — a cloned 1-run container is a struct + a tiny
  runs array; combining them halves its allocations and frees. **Spec-13 caveat applies and
  must be checked, not assumed either way:** spec 13's NO-GO was about *array* containers whose
  power-of-two payloads fit SMP classes exactly; run-container payloads are small and not
  power-of-two-aligned, so the class-boundary analysis must be redone for this case before
  building anything (compute the SMP size classes for combined vs split run allocations across
  realistic run counts — if combining crosses classes, stop).
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
  (`errdefer` discipline preserved through any constructor change).
- **Zen 4 no-regress (hard):** clone-body, teardown, and the canonical `clone` /
  `clone + removeRange` rows stay within noise (≤ 5%, rerun on range overlap) on Zen 4, where
  rawr already wins.
- **Board gate:** no canonical row worsens > 5% vs the post-26a baseline, both hosts.

## Acceptance (GO)

- Phase 1: allocation count drops by the predicted 2 (+ their frees) on the probe; clone body
  and teardown improve on M4; both hosts re-measured; all gates green. If M4's canonical
  `clone` and `clone + removeRange` rows reach **≤ 1.10x**, stop — Phase 2 is not attempted.
- Phase 2 (per candidate, only if attempted): the class-boundary analysis is recorded
  **before** implementation; the candidate improves M4 beyond noise on the `26a` diagnostics
  and canonical rows without breaking any gate. A candidate whose class analysis fails is
  recorded and dropped, not built.
- A documented partial (M4 improved but above 1.10x, with the residual attributed) is an
  acceptable terminal outcome — clone is a secondary operation; do not over-invest.

## Estimate

Phase 1: **S** (swap to `initCapacity`, audit siblings, re-measure). Phase 2: **M** per
candidate, gated on Phase 1's result and the class analysis.
