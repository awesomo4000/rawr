<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 23: Iterate parity — diagnosis-first

Close the largest real default-SMP gap on the accurate parity board: **iterate**, 1.52x
(M4) / 1.88x (Zen 4) rawr/CRoaring. Persistent on **both** architectures, so it is a
genuine implementation gap, not an M4 codegen quirk — and high-leverage, since iteration
underlies `toArray`, serialization walks, and any consumer scanning the set.

**Diagnosis first, no preselected cause** (the 20a / 21 discipline: verify the comparison
is fair and attribute the cost before touching code). A fix is a conditional second phase.

## The first thing to check — are we comparing like for like?

The two benched paths use **different iteration models**:

- **rawr** — `bm.iterator()` then a pull `while (it.next()) |v|` loop (`src/bitmap.zig:2218`,
  `:2325`): one `next()` call per value, saving/restoring iterator state (container index,
  intra-container position) across every value.
- **CRoaring** — `roaring_iterate(bm, callback, …)` (`src/bench_croaring.zig:1164`): a **push**
  model where CRoaring drives a tight per-container inner loop and calls the callback per
  value, with no per-value state save/restore between values in a container.

CRoaring's **push** `roaring_iterate` is typically faster than its **pull**
`roaring_uint32_iterator`. So an unknown share of the 1.5–1.9x may be a **model mismatch in
the benchmark**, not a rawr kernel deficiency — the same shape as the sparse-AND/OR harness
artifacts. This must be resolved before attributing anything to rawr's iterator.

## Phase 1 — Diagnosis (on the canonical harness + a focused split)

1. **Iteration-model parity.** Measure both comparisons, per-value normalized:
   - rawr pull `next()` **vs CRoaring `roaring_uint32_iterator`** (pull ↔ pull, apples-to-apples);
   - rawr's bulk/callback path *if one exists* (else note its absence) **vs CRoaring
     `roaring_iterate`** (push ↔ push).
   Quantify how much of the board's 1.5–1.9x is the pull-vs-push model vs a real
   like-for-like gap. If a large share is model mismatch, the board row and/or rawr's API —
   not its kernel — is the thing to change.
2. **Container mix of the corpus.** The 1M-value iterate corpus — is it bitset-, array-, or
   run-dominated? That decides which inner loop dominates the number (bitset word-scan vs
   array walk vs run expansion).
3. **Where the time goes (kernel-level).** Within the dominant container type, split
   **per-value `next()` state overhead** (the pull-model tax) from the **container inner
   loop** (bitset `ctz` word-advance / array copy). Compare rawr's per-container iteration
   against CRoaring's equivalent. Inspect generated code for the hot loop where useful.
   Report absolute medians + ranges (and ns/value) with a **named residual**, not forced-100%.

Phase 1 stands alone: "how much is model vs kernel, and where the kernel cost is" is the
deliverable even if no fix follows.

## Phase 2 — Fix (conditional on Phase 1, lever follows the attribution)

- **If a large share is the pull-vs-push model:** add a **bulk / callback iteration path** to
  rawr (a `forEach`-style API mirroring `roaring_iterate`'s tight per-container loop). The
  idiomatic pull `iterator()` stays; the bulk path closes the benchmark *and* speeds real
  consumers (`toArray`, serialization). This is an **additive public API**, not a change to
  the existing iterator's semantics.
- **If rawr's pull iterator is genuinely slower than CRoaring's pull iterator:** tighten
  `next()` — reduce per-value state work, faster bitset word-advance, a per-container fast
  path — without changing iteration semantics.
- Threshold/dispatch and other container types only if the attribution implicates them.

## Constraints

- **Correctness:** iteration yields the same sorted value sequence; a differential check
  (rawr iteration == CRoaring order == `toArray`) stays green. Any new bulk API is validated
  against the pull iterator and CRoaring.
- Measured on the **canonical spec-22 harness**, default **rawr-SMP** vs CRoaring, five
  fresh-process runs median + range, on **M4 and Zen 4** (the gap must close on both, since
  it is present on both).
- **No regression** on `toArray` / serialization / other rows; existing `iterator()` semantics
  unchanged (a fix is additive or an internal tightening).

## Acceptance

- **Phase 1 GO:** the 1.5–1.9x is split into model-mismatch vs like-for-like kernel gap, with
  the container mix and the dominant cost named, on both hosts.
- **Phase 2 GO (if attempted):** the default rawr-SMP iterate ratio moves materially toward
  parity on **both** M4 and Zen 4, no regression elsewhere, differential green. If the honest
  finding is "the board compared pull-vs-push and like-for-like is already near parity," that
  is a valid terminal outcome — correct the row's comparison and record it rather than
  optimizing a phantom.

## NO-GO

- Phase 1 shows the gap is essentially a benchmark model mismatch and like-for-like iteration
  is at parity → fix the row's comparison (or add the bulk API purely for consumer ergonomics),
  do not chase a kernel that isn't slow.

## Estimate

S for Phase 1 (model-parity + attribution on the existing harness). Phase 2 is S–M: a bulk
`forEach` API is small and additive; a `next()` tightening is a focused kernel change — chosen
by the diagnosis.
