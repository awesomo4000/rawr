<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 38: Address-sorted traversal for order-free phases

Campaign: [31-structural-parity-campaign.md](31-structural-parity-campaign.md). Background:
[allocator-address-order-pathology.md](../docs/allocator-address-order-pathology.md).

Apply the address-sort remedy to the phases whose visit order is **ours to choose**, behind a
**default-off** option. This is the *tactical* remedy — it compensates for allocator-returned order
rather than fixing it, and it is chosen because it is **robust to allocator sharing** (it depends only
on pointers we already hold, not on shared allocator state).

**Not in scope:** compaction/relayout, a private-allocator track (`*Owned` for lazy ops), any change
to `SmpAllocator`, and the open hardware-mechanism question.

## What can and cannot be sorted

The remedy is legal only where nothing downstream depends on visit order.

| phase | order-free? | eligible |
|---|---|---|
| **teardown / mass free** (`deinit`) | yes — frees may happen in any order | **yes** |
| **repair cardinality pass** (`repairAfterLazy`, per-bitset `computeCardinality`) | yes — each container is independent; the **compaction step must stay key-ordered** | **yes, split into address-ordered compute + key-ordered compact** |
| `serialize` / `serializeToWriter` | no — output byte order is the format | no |
| `iterate` / `toArray` | no — must emit ascending values | no |
| clone / set-op **writes** | no — result must be key-ordered | no |

**Measured motivation:** teardown is a known gap — spec 26a attributed clone's residual to `clone`
(254.4 vs 96.9 ns) **plus teardown (144.9 vs 48.3 ns)**. Repair-alone is the largest single lazy-OR
phase (**8.616 ms**) though already at **1.069x**, so improving it is an **absolute-throughput** win
(relevant to the datalog backend use case), **not** a parity-gap closure. State which goal each
measurement serves.

## API shape

**A raw public `sort()` has no coherent shape and is NOT proposed.** There is nothing persistent to
reorder: `keys`/`containers` must stay key-ordered, so the sorted order is inherently a *transient
traversal strategy*, not stored state. Reordering payload assignment permanently is **compaction** —
a different operation, out of scope here.

Therefore:

- **Internal:** an address-sorted traversal used by the eligible phases above.
- **Public control:** an **opt-in flag, default OFF**, set at bitmap creation. Users who know their
  allocator's behaviour turn it on; everyone else pays nothing.
- **Public surface, if one is wanted:** expose **sorted variants of the order-free operations** (e.g.
  a sorted-teardown entry point), not a raw pointer sort. Decide at review whether the flag alone is
  sufficient.

### Flag semantics to pin

- **Default OFF.** No behaviour change for existing users.
- **Propagation rule (must be stated):** operations returning a *new* bitmap (`lazyOr`, `bitwiseAnd`,
  `clone`, …) take a per-call `allocator`, so the flag and the allocator can disagree. Pin whether the
  result **inherits** the source's flag or **defaults off**, and document it.
- **The flag is about the allocator, not the data** — note this tension in the doc comment so users
  understand what they are asserting.

### Scratch buffer + failure semantics

- The sort needs a **permutation/pointer array** (~65–131 KB at 16k containers) — `containers` itself
  cannot be reordered. **Reuse a scratch buffer; do not allocate per call.**
- **Graceful degradation is mandatory:** sorting is purely an optimization, so if the scratch
  allocation fails, **fall back to unsorted traversal and continue** — never propagate an error and
  never fail the operation. This must be tested with allocation-failure injection.
- Note the irony to keep in mind: the scratch comes from the same allocator we are compensating for.

### Size gate

Below some container count the sort cannot pay (≈0.13 ms per 16k pointers is fatal overhead for a
small bitmap). **Determine the threshold by measurement in Phase 1** and apply it *in addition to* the
flag — an enabled flag on a tiny bitmap must still skip the sort.

## Phase 1 — measurement (before shipping anything)

- **Honest timing, non-negotiable:** the sort cost must be **inside the timed region**. The retained
  probe (`src/bench_smp_layout.zig`) distinguishes sort-outside-timing from sort-plus-traversal;
  **only the sort-plus-traversal number may gate a decision.** A recovery figure with the sort excluded
  overstates the win.
- **Per eligible phase** (teardown; repair cardinality pass), measure sorted vs unsorted:
  **rawr/SMP and rawr/libc**, **M4 and Zen 4/WSL2**, canonical protocol (3 warmup / 21 timed, five
  fresh-process medians + full ranges), one process per `(row, implementation, allocator)` tuple.
- **Expect the libc arm to show little or no gain** (libc recovered only 0.063–0.180 ms in the probe) —
  that asymmetry is the evidence the flag is allocator-dependent and must default off.
- **Range separation required** (spec-37 discipline): sorted vs unsorted five-process ranges must
  **separate** before claiming a win; overlap ⇒ inconclusive for that phase.
- **Determine the size-gate threshold**: sweep container counts to find where sorted stops paying.
- Report absolute times and whether each result is a **parity-gap closure** or an
  **absolute-throughput** gain.

## Phase 2 — ship what won (conditional, per phase)

Ship the sorted traversal only for phases whose Phase 1 result separated, behind the default-off flag
plus the measured size gate.

## Correctness

- **Order-invariance must be asserted, not assumed.** Teardown and per-container cardinality are
  order-invariant by construction; prove it holds: **byte-identical `serialize` output and identical
  cardinalities** with the flag on vs off, plus CRoaring differential, across container-type mixes.
- **Repair split correctness:** the address-ordered cardinality pass must leave the **key-ordered
  compaction** result identical to today's single-pass repair — same container kinds, same
  cardinalities, same order, same demote/survive decisions.
- **No leaks / double-frees in sorted teardown** — every container freed exactly once regardless of
  order; failure-injection green.
- `zig build test`, `zig build difftest`, `ReleaseSafe` and `ReleaseFast`.

## Gates

- **Board gate + spec-28 layout exception**, both hosts, on adoption.
- **Zen 4 policy (spec 30):** within-noise passes; a real regression needs an explicit owner exception.
- **Flag OFF must be indistinguishable from today** — verify the default path is unchanged (a
  scaffolding check in the spec-36/37 style: flag-off ranges overlap the pre-change baseline, medians
  within 5%).
- **One architecture-neutral shape.**

## Acceptance

- Eligible phases identified and justified; ineligible ones explicitly excluded.
- Phase 1: sorted vs unsorted, **sort cost inside timing**, both allocators, both hosts, five fresh
  processes, range separation applied; size-gate threshold measured; each gain labelled parity vs
  absolute.
- Phase 2: sorted traversal shipped only for phases that separated, **default OFF**, size-gated, flag
  propagation rule documented, scratch reused, **scratch-failure degrades to unsorted**.
- Correctness: order-invariance asserted (byte-identity + cardinalities + differential), repair-split
  identical to single-pass, no leak/double-free, failure injection green.
- Flag-OFF path verified indistinguishable from baseline; board gate held.
- `docs/parity-measurement.md` updated; `docs/allocator-address-order-pathology.md` cross-referenced.

## Chunk plan

- **`38-00`** — Phase 1 measurement (both phases, both allocators, both hosts, size-gate sweep). No
  production change.
- **`38-01`** — Phase 2: ship what won behind the default-off flag + size gate, with the correctness
  and failure-injection surface.

## Estimate

S–M for `38-00` (reuses the retained probe's discipline and the canonical worker). M for `38-01` — the
work is the flag plumbing, propagation rule, scratch reuse, graceful degradation, and the repair split,
not the sort itself.
