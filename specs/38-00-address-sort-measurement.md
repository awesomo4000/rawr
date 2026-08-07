<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 38-00: Address-sorted traversal — measurement

Toplevel: [38-address-sorted-traversal.md](38-address-sorted-traversal.md). Background:
[allocator-address-order-pathology.md](../docs/allocator-address-order-pathology.md).

**Diagnosis only — no production change.** Decide, per phase, whether address-sorted traversal is worth
shipping, and produce the numbers `38-01` needs. **`38-01` is separately blocked** on two owner
decisions (scope; control mechanism) — this chunk does not depend on either.

## Two phases, unequal confidence

- **Repair (sound target).** Operates on the ~16,364 bitset payloads the pathology was demonstrated on.
  Repair-alone is the largest lazy-OR phase (**8.616 ms**) at **1.069x**.
- **Teardown (lower confidence, and carries a state obligation).** The spec-26a teardown gap (144.9 vs
  48.3 ns) came from the **8-run-container clone corpus** — 8 pointers, below any plausible size gate,
  and not bitsets. It **cannot** motivate sorted teardown at scale.

## Sort key: PAYLOAD, not header

`TaggedPtr` addresses the **header**; the pathology concerns the **8 KB payload**, a *separate*
allocation with an independent order.

- **repair:** sort by **`bc.words`**.
- **teardown:** **type-specific payload key** — `bc.words` / `ac.values` / `rc.runs`. `deinit` frees
  payload then header, so two streams exist and only one can be ordered — **sort by payload**.
- **Required comparison:** **header-key vs payload-key sorting**, to confirm which stream matters.

## Corpora

- **Mass-bitset corpus** (~16k 8 KB payloads) — the diagnosis corpus for **both** phases.
- **Canonical 8-container clone corpus** — retained as a **control**, to confirm the size gate correctly
  excludes it.

## Measurement protocol

- Canonical: **3 warmup / 21 timed**, **five fresh-process medians + full ranges**, **one process per
  `(row, implementation, allocator)` tuple**, implementation-specific init only.
- Arms: **rawr/SMP** and **rawr/libc**; hosts: **M4** and **Zen 4/WSL2**.
- **Sort cost INSIDE the timed region.** Only the **sort-plus-traversal** number may gate a decision;
  the sort-outside-timing figure (which the retained probe also produces) overstates the win and must
  not be used for a verdict.
- **CRoaring references at matched boundaries** — CRoaring **teardown** and **repair**. Without them,
  results are **rawr-internal deltas only** and must be labelled as such; no gap arithmetic.
- **Range separation required** (spec-37 discipline): sorted vs unsorted five-process ranges must
  **separate** before claiming a win; overlap ⇒ inconclusive for that phase.
- **Expect the libc arm to gain little** (libc recovered 0.063–0.180 ms in the probe) — that asymmetry is
  the evidence bearing on the scope decision, so report it prominently.

## Repair specifics

- **Time the COMPLETE user-visible operation:** address-ordered cardinality pass **+ key-ordered
  conversion/compaction**. Report the cardinality sub-phase as **attribution only** — **a
  cardinality-only speedup cannot gate adoption if total `repairAfterLazy` is neutral or slower.**
- **Frees must stay in the key-ordered pass**, unchanged. That is what keeps repair
  allocator-state-neutral; verify it holds in the implementation.
- **Measure scratch BOTH ways — cold and reusable.** Cold = scratch alloc + sort + traversal + scratch
  free. Reusable = steady state. **Do not assume amortization**: the control mechanism `38-01` picks may
  not provide persistent scratch.

## Teardown specifics — two-stage, non-negotiable

`SmpAllocator.free` pushes onto a **per-size-class LIFO freelist**, so **free order determines the order
later allocations come back.** Sorted teardown is **state conditioning**; a process that exits right
after teardown cannot reveal the downstream effect.

**Stage 1 — immediate teardown**, three orders: **unsorted**, **ascending payload**, **descending
payload**.

**Stage 2 — refill and re-measure.** For each of the three orders, **refill the same allocator and
measure the next construction/traversal.** LIFO reuse implies **descending frees tend to produce
ascending allocations** (and vice versa), so descending is expected to be the interesting arm — and the
same inversion is a plausible mechanism for the measured words-only case (8 KB stride, descending
allocation order).

**Stage 3 (strongly preferred) — allocator-noise control.** Insert unrelated allocation/free traffic
**between** teardown and refill, and repeat stage 2. This tests whether any downstream effect survives a
realistically **shared** allocator rather than existing only in a rawr-only process.

**A teardown improvement reported without stage 2 does not count**, and one that disappears under
stage 3 must be recorded as **not dependable in a library**.

## Size-gate derivation

- **Per phase** (teardown and repair get separate thresholds).
- Sweep container counts; at each, require **separated ranges favouring sorted**.
- Threshold = **smallest count where sorted wins on BOTH hosts**, then take the **more conservative
  (larger)** of the two hosts' values.
- **Irreconcilable host disagreement is a finding, not a threshold** — report it; do not ship a value.

## Acceptance

- Both phases measured, both arms, both hosts, five fresh processes, sort cost inside timing, range
  separation applied.
- Payload-key vs header-key comparison reported.
- CRoaring teardown/repair references obtained, or results explicitly labelled rawr-internal deltas.
- Repair: complete-operation timing (cardinality as attribution only); **cold and reusable** scratch
  both reported; frees confirmed unmoved from the key-ordered pass.
- Teardown: stage 1 (three orders) **and** stage 2 (refill + next-cycle) reported; stage 3 noise control
  run or its absence justified; any stage-2-less teardown claim withheld.
- Per-phase size-gate thresholds derived by the pinned rule, or host disagreement reported.
- Each result labelled **absolute-throughput** or (only once the scope decision permits it) parity —
  **no parity language until the owner decides**.
- **No production change.** `zig build test`, `zig build difftest` green; `docs/parity-measurement.md`
  updated with cells and verdicts; `docs/allocator-address-order-pathology.md` cross-referenced.

## Estimate

M — the repair cells are straightforward; the teardown two-stage refill experiment (plus the noise
control) and the CRoaring matched-boundary references are the bulk of the work.
