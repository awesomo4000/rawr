<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 17: Transient-bitset arena for lazy union

**Prototype-first, decision-gated.** Give the lazy-union path an internal arena for the
short-lived bitset containers it spins up, so their allocation cost stops being
dominated by the general allocator. The deliverable is a measured go/no-go on both the
sparse and the n-way dense workloads, not a commitment to ship.

## Why

Spec 16 closed the algorithmic side of `lazyOr+repair (sparse)` and then proved the
residual is **allocator-bound, not loop-bound**: with all construction fixes in place
the combined ratio sat at ~1.22x under `smp_allocator`, but swapping only the lazy
output allocator to a different strategy took construction to ~0.99x and combined to
~1.07x. Forced lazy union builds thousands of 8 KB bitset payloads that are freed again
almost immediately (repair demotes most of them straight back to arrays), and the
general allocator's per-object cost on that pattern is the whole remaining gap. Full
context in `done/16-lazy-union-forced-bitset.md`.

The framing that makes this worth doing is **not** the sparse 2-way number — that
workload is off-design for lazy (lazy exists for n-way aggregation). It is that a
reusable *transient allocator for scratch containers* is a general lever: it helps the
n-way dense aggregation lazy is actually built for, it is the same "parent-owned
temporary storage" direction the spec 13 analysis pointed at, and it composes with
rawr's existing arena/`Owned` story rather than inventing a new concept.

## Why an arena (not a pool, not a malloc clone)

The lazy path builds **all** output bitsets first, then a single `repairAfterLazy`
sweep converts them. So during the pre-repair window every transient bitset is live at
once and they all die together at repair — **batch lifetime, not interleaved churn**. A
free-list pool (recycling one buffer across many objects) does not fit, because the
objects coexist. Batch lifetime is the textbook arena case: replace N allocator
round-trips with N bump-allocations + one slab acquire + one bulk free, which is
strictly less work than any general allocator does per object. Reimplementing a
general allocator is explicitly out of scope; the libc measurement in spec 16 was a
diagnostic proxy, not a target.

## Approach (prototype)

Back the lazy path's transient bitset storage with an arena whose child allocator is
**the operation's own allocator** (never a hardcoded global — rawr is allocator-
generic). Bump-allocate the scratch bitsets during construction; `repairAfterLazy`
produces the persistent results on the real allocator and the arena is released whole.

The persistent outputs must not live in the arena. Repair therefore, per transient
bitset:
- **demote** (cardinality ≤ `MAX_CARDINALITY`, the common sparse case): build the array
  container on the real allocator, leave the arena words untouched;
- **survive** (cardinality > `MAX_CARDINALITY`, the common dense case): copy the payload
  out to a real-allocator bitset;

then deinit the arena once, freeing all scratch payloads in a single step.

## Design decisions to settle (with review, before chunking)

- **D1 — Arena granularity.** Pool only the 8 KB words buffers (the expensive, allocator-
  slow part) while container structs and all persistent containers stay on the real
  allocator? Or arena the whole transient container? The words-only split keeps the
  copy-out and the "what's arena-backed" bookkeeping minimal; confirm by measurement.
- **D2 — Arena lifetime / ownership.** The arena must live from construction until the
  user calls `repairAfterLazy`, which is a separate call. Options: the lazy result
  carries the arena handle until repair; or, for the 2-way case only, fuse
  construct-then-repair so the arena never escapes. Decide whether carrying an arena on
  the result is acceptable given the lazy footgun contract (result already invalid until
  repaired), and how repair distinguishes arena-backed transients from real-allocator
  clones (non-overlapping keys are cloned from the real allocator and must be freed
  normally).
- **D3 — Survivor copy-out.** Confirm the dense path's copy-out cost is bounded and does
  not regress n-way dense. On sparse, survivors are ~0 and it is free.
- **D4 — Allocator genericity.** Child allocator = the caller's allocator. Define
  behavior when the caller already passed an arena/`Owned` allocator (wrapping is
  redundant — detect and skip, or accept the harmless double-arena?).
- **D5 — Arena vs pre-sized fixed buffer.** `ArenaAllocator` needs no size estimate and
  is simplest; a `FixedBufferAllocator` pre-sized to the overlap bound (`min(a.size,
  b.size)` × payload, already known from the merge) removes even the per-alloc arena
  check. Prototype with `ArenaAllocator`; escalate to a pre-sized buffer only if the
  split shows per-alloc overhead.
- **D6 — Which lazy sites.** `lazyMergeTwo` (2-way) is the measured case. Include the
  n-way fold (`foldManyKey` / the many-way accumulate) since it is lazy's real use case
  and creates the same transient bitsets; `lazyXor` (always bitset-accumulates) and the
  in-place `lazyOrInPlace` (delegates to `lazyOr`) inherit whatever the shared path does.

## Measurement

- **Both workloads**: the sparse 2-way corpus (from spec 16) **and** an n-way dense
  aggregation corpus (`orMany`-style), each with the construction/repair split from
  spec 16 and identical setup/teardown on both sides.
- Five independent process runs; report median and range/IQR per phase.
- Report allocation counts and peak transient bytes alongside times, so the arena's
  effect on allocator traffic is visible, not just wall-clock.

## Acceptance (GO)

- **Sparse**: `lazyOr+repair (sparse)` combined ratio meaningfully beats the current
  SMP baseline and approaches the isolated-allocator figure from spec 16 (~1.07x
  territory), with the construction phase at or near parity.
- **No regression on n-way dense** (`orMany` / `orManyHeap` / `xorMany`, dense
  aggregation) beyond noise — the survivor copy-out must not cost more than it saves
  there.
- **Correctness preserved**: forced/size-selected representation tests, lazy-or/xor and
  in-place differential cases, footgun and edge cases all green under both `ReleaseSafe`
  and `ReleaseFast`. No change to the deferred-cardinality contract or public API
  semantics; if the arena must be carried on the result, its lifetime is internal and
  invisible to callers.
- Full build green; no diagnostic allocator left in the tree.

## NO-GO / risks

- If the ownership model (D2) can only be made correct by carrying an arena on the lazy
  result in a way that leaks into the public contract, or if the dense copy-out (D3)
  regresses n-way, park it — the sparse workload alone does not justify contract
  complexity.
- The win is a property of the general allocator's transient-8 KB behavior; record that
  it may narrow under a different backing allocator, and that the arena's value is the
  reduced allocator traffic (visible in the allocation-count report), not a single
  platform's wall-clock.

## Estimate

M. The arena mechanism is small, but the deliverable is the ownership decision (D2), the
survivor copy-out, and the measured go/no-go across two workloads — not just wrapping an
allocator.
