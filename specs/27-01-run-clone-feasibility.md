<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 27-01: Run-container clone feasibility analysis (conditional)

Second chunk of [clone optimization](27-clone-optimization.md). **Analysis only — no
implementation.** Runs only if `27-00` leaves the M4 canonical `clone` /
`clone + removeRange` rows above **1.10x**.

## Gate

- `27-00` complete and its M4 rows still > 1.10x. (If ≤ 1.10x, this chunk is not started and
  spec 27 closes with `27-00`.)

## Deliverables — both preconditions, answered on paper before any prototype

1. **Ownership/deinit design for single-allocation run-container clones.**
   `RunContainer.deinit` frees `runs` and the struct separately; a clone-only combined
   allocation cannot be freed correctly as-is. Produce a concrete design — metadata
   discriminating the layouts, a new representation, or converting **all** run containers to
   the combined layout — with its full blast radius (every alloc/free/resize site touched), or
   a reasoned **NO-GO**. Size-class analysis alone is insufficient.
2. **SMP class-boundary analysis, redone for run containers** (the spec-13 lesson, not assumed
   either way): compute combined vs split allocation size classes across realistic run counts
   (1-run full-chunk clones up to multi-run containers). If combining crosses classes where
   splitting did not, that candidate is a NO-GO on spec-13 grounds.
3. **The same two questions for combined top-level key/container storage** (the more invasive
   candidate), at analysis depth only — including a statement of every top-level growth path
   it would touch.
4. **Recommendation:** implement (which candidate, with the design), or close spec 27 with the
   `27-00` result + documented residual. **Any implementation is its own future chunk** — it
   does not begin in this one.

## Acceptance

- **The applicable ownership/layout invariant plus class-boundary analysis** answered in
  writing per candidate: for run clones, the deinit-invariant design-or-NO-GO + class
  arithmetic; for top-level storage, its own allocation/growth/deinit ownership design +
  class arithmetic, at analysis depth.
- A clear recommendation recorded in the spec/`docs/parity-measurement.md`: implement X, or
  close with residual documented.
- No production or benchmark code changed by this chunk.

## Checklist

- [ ] Ownership/deinit design or NO-GO for combined run-container clones, blast radius stated
- [ ] SMP class arithmetic for combined vs split run allocations, realistic run counts
- [ ] Top-level combined-storage candidate assessed (growth paths enumerated)
- [ ] Recommendation recorded; no code changed
