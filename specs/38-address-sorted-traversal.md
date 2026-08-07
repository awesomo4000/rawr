<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 38: Address-sorted traversal for order-free phases

Campaign: [31-structural-parity-campaign.md](31-structural-parity-campaign.md). Background:
[allocator-address-order-pathology.md](../docs/allocator-address-order-pathology.md).

Apply the address-sort remedy to the phases whose visit order is **ours to choose**, behind a
**to-be-decided control mechanism** (see the scope decision below — options include default-off and
default-on, so this spec uses **neutral wording** until the owner picks). This is the *tactical* remedy — it compensates for allocator-returned order rather than fixing it, and
it is chosen because it is **robust to allocator sharing** (it depends only on pointers we already hold,
not on shared allocator state).

**`38-00` (diagnosis) is ready. `38-01` (implementation) is NOT** — it is blocked on two unresolved
decisions recorded below: the **scope decision** (throughput feature vs parity lever) and the **public
control mechanism**.

**Not in scope:** compaction/relayout, a private-allocator track (`*Owned` for lazy ops), any change
to `SmpAllocator`, and the open hardware-mechanism question.

## BLOCKING SCOPE DECISION — throughput feature or parity lever? (owner call, gates `38-01`)

**A default-off flag cannot close a default-path parity gap.** The canonical board measures rawr/SMP on
the default path; if the flag is off there, **no board row moves**. This spec cannot claim both. Three
mutually exclusive positions — **`38-00` runs regardless; `38-01` cannot be written until one is
chosen:**

- **(A) Optional throughput feature (default OFF).** Honest and low-risk. **The board does not move and
  this is NOT a parity lever** — all gains are absolute-throughput gains for opted-in users (e.g. the
  datalog backend). Every parity claim must be struck.
- **(B) Parity lever (default ON).** Requires proving the sort is **non-harmful for every allocator**,
  including libc, which recovered only **0.063–0.180 ms** and would still pay the ~0.13 ms sort — i.e.
  plausibly **net-negative** for libc users. Needs an explicit no-regress gate on the libc arm.
- **(C) Default ON but auto-gated** by a cheap scatter/direction probe, so it is a no-op where it does
  not help. The only route to closing parity without harming libc — but it introduces a **heuristic
  tuned on our benchmark**, which is precisely the sole-allocator-client trap (see the pathology doc);
  a probe tuned here may mistune in production.

**Recommendation: (A) for now**, with (C) reconsidered only if `38-00` shows the libc arm is unharmed
and a scatter probe is reliable across both hosts. **Until this is decided, no parity language is
permitted in this spec.**

## What can and cannot be sorted

The remedy is legal only where nothing downstream depends on visit order.

**Two independent tests** — a phase qualifies only if the visit order is free (**correctness**) **and**
the reordering is allocator-state-neutral (**robustness**):

| phase | order-free? | state-neutral? | verdict |
|---|---|---|---|
| **repair cardinality pass** (per-bitset `computeCardinality`) | yes — containers independent; compaction stays key-ordered | **yes** — a read-only traversal; **frees are untouched** because the demote/free work stays in the key-ordered pass | **clean candidate** |
| **teardown / mass free** (`deinit`) | yes — frees may happen in any order | **NO** — see below | **candidate with a downstream obligation** |
| `serialize` / `serializeToWriter` | no — output byte order is the format | — | no |
| `iterate` / `toArray` | no — must emit ascending values | — | no |
| clone / set-op **writes** | no — result must be key-ordered | — | no |

### Sorted teardown is NOT allocator-state-neutral (blocking requirement)

`SmpAllocator.free` pushes onto a **per-size-class LIFO freelist**, so **the order we free in
determines the order later allocations come back.** Sorted teardown is therefore **state conditioning**,
not a local optimization: it may improve teardown while **poisoning the next allocation/traversal
cycle** — and **a fresh process that exits right after teardown can never reveal that.**

**Direction inverts, and this matters:** LIFO reuse means **freeing ascending tends to hand back
descending**, and **freeing descending tends to hand back ascending**. Note this is a plausible
mechanism for the measured words-only case (8 KB stride, *descending* allocation order). So the free
order that helps the *next* cycle may be **descending**, the opposite of "sort ascending before
freeing."

**`38-00` must therefore measure teardown as a two-stage experiment:**

1. **Immediate teardown** — unsorted, **ascending-payload**, and **descending-payload** free order.
2. **Then refill the same allocator and measure the next construction/traversal** for each of the three
   free orders — this is where the poisoning (or benefit) appears.
3. **Ideally insert an allocator-noise control between teardown and refill** (unrelated
   allocation/free traffic), to test whether any downstream effect survives a realistically shared
   allocator rather than only in a rawr-only process.

**A teardown improvement measured without stage 2 does not count.** Repair has no such obligation:
sorting the cardinality traversal does not reorder frees, and key-ordered compaction is unchanged.

**Confidence differs sharply between the two phases — say so:**

- **Repair is the sound target.** It operates on the **16,364 bitset payloads** that the pathology was
  actually demonstrated on. Repair-alone is the largest single lazy-OR phase (**8.616 ms**) though
  already at **1.069x**.
- **Teardown is LOWER CONFIDENCE, and its usual motivation does not transfer.** The spec-26a teardown
  gap (144.9 vs 48.3 ns) was measured on the **clone corpus of 8 run containers** — **8 pointers, far
  below any sensible size gate**, and run containers, not 8 KB bitsets. That number **cannot** motivate
  sorted teardown at scale. Therefore:
  - diagnose teardown on a **mass-bitset teardown corpus** (the ~16k 8 KB payload population), and
  - retain the **canonical 8-container clone corpus as a control** to confirm the size gate correctly
    excludes it.

## API shape

**A raw public `sort()` has no coherent shape and is NOT proposed.** There is nothing persistent to
reorder: `keys`/`containers` must stay key-ordered, so the sorted order is inherently a *transient
traversal strategy*, not stored state. Reordering payload assignment permanently is **compaction** —
a different operation, out of scope here.

**Internal:** an address-sorted traversal used by the eligible phases above. That part is settled.

### The PUBLIC control mechanism is NOT decided — and must be pinned before `38-01`

A **creation-time persistent flag** (this spec's first draft) **fits the property badly.** The flag
would describe *the allocator a bitmap is currently using*, but every operation that returns a new
bitmap (`lazyOr`, `bitwiseAnd`, `clone`, …) takes its **own per-call allocator**. So inheritance is
questionable from **either** direction: inheriting the source's flag may describe the wrong allocator,
and defaulting off silently drops the user's intent. There is no clean answer, which is the signal that
the flag is the wrong shape.

Candidates, to be decided at review — **`38-00` does not depend on this**:

| option | fit | cost |
|---|---|---|
| **Per-operation option** (e.g. an options struct on the ops that can use it) | **good** — matches the per-call allocator exactly | wider signature churn |
| **Explicit optimized variants** (e.g. a sorted-teardown entry point) | **good** — no hidden state, no propagation question | API surface growth; one variant per eligible op |
| Creation-time persistent flag | **poor** — describes an allocator that may not be the one used | cheap, but propagation is unanswerable |
| Raw public `sort()` | **rejected** — no coherent shape (nothing persistent to reorder) | — |

**Rule: `38-01` is not writable until this and the scope decision above are both resolved.**

### Sort KEY — payload address, not header address (pinned)

**`TaggedPtr` points at the container HEADER; the proven pathology is about the 8 KB PAYLOAD.** Header
and payload are **separate allocations** with independent orders, so sorting `TaggedPtr` sorts the wrong
addresses. Required keys:

- **repair (bitsets):** sort by **`bc.words`** (the 8 KB payload).
- **teardown (mixed types):** an explicit **type-specific payload key** — `bc.words` / `ac.values` /
  `rc.runs`. Note `deinit` frees **both** payload and header (payload first), so there are two address
  streams and only one can be sorted; **sort by payload** (the large one) and say so.
- **`38-00` must measure header-address sorting vs payload-address sorting** — that comparison confirms
  which stream actually matters and is cheap to run.

### Scratch buffer + failure semantics

- The sort needs a **permutation/pointer array** (~65–131 KB at 16k containers) — `containers` itself
  cannot be reordered.
- **"Reuse scratch" does NOT fit `deinit`, which runs once.** There is no steady state to amortize into,
  so teardown must be measured **cold**. Hiding the scratch allocation before the timer would make
  teardown results **dishonest**. Measure both, and label which applies to which phase:
  - **cold:** scratch allocation + sort + traversal + scratch free — **the honest teardown number**;
  - **reusable steady state:** where repeated invocation could make reuse legitimate (repair).
  **Measure repair BOTH ways — cold and reusable-scratch.** The eventual control mechanism may not
  provide persistent scratch at all, so `38-00` **must not assume amortization** before `38-01` chooses
  it. Report both numbers.
- **Graceful degradation is mandatory:** sorting is purely an optimization, so if the scratch
  allocation fails, **fall back to unsorted traversal and continue** — never propagate an error and
  never fail the operation. Test with allocation-failure injection.
- Note the irony to keep in mind: the scratch comes from the same allocator we are compensating for.

### Size gate — derivation rule (pinned)

Below some container count the sort cannot pay (≈0.13 ms per 16k pointers is fatal overhead for a small
bitmap). **Range separation identifies whether sorted wins at a given size; it does not by itself pick a
threshold.** Rule:

- **Per phase** (teardown and repair get their own thresholds — different work per container).
- Sweep container counts; for each, require **separated five-process ranges favouring sorted**.
- The threshold is the **smallest count at which sorted wins on BOTH hosts**, then take the
  **more conservative (larger) of the two hosts' thresholds** so one architecture-neutral value never
  enables the sort where it loses.
- If the two hosts disagree irreconcilably (sorted wins on one, loses on the other at every size), that
  is a **finding, not a threshold** — report it and do not ship a single value.
- Apply the gate *in addition to* whatever the opt-in mechanism turns out to be: enabled-but-tiny must
  still skip the sort.

## Phase 1 — measurement (before shipping anything)

- **Honest timing, non-negotiable:** the sort cost must be **inside the timed region**. The retained
  probe (`src/bench_smp_layout.zig`) distinguishes sort-outside-timing from sort-plus-traversal;
  **only the sort-plus-traversal number may gate a decision.** A recovery figure with the sort excluded
  overstates the win.
- **Per eligible phase** (teardown; repair), measure sorted vs unsorted: **rawr/SMP and rawr/libc**,
  **M4 and Zen 4/WSL2**, canonical protocol (3 warmup / 21 timed, five fresh-process medians + full
  ranges), one process per `(row, implementation, allocator)` tuple.
- **CRoaring reference required, at matched boundaries.** Sorted-vs-unsorted rawr alone cannot support
  any statement about gaps. Measure **CRoaring teardown and CRoaring repair** at the **same boundaries**
  as the rawr cells, so any gap arithmetic is computable rather than asserted. Without these the phase
  results are rawr-internal deltas only — label them that way if the references cannot be obtained.
- **Expect the libc arm to show little or no gain** (libc recovered only 0.063–0.180 ms in the probe) —
  that asymmetry is the evidence the flag is allocator-dependent and must default off.
- **Range separation required** (spec-37 discipline): sorted vs unsorted five-process ranges must
  **separate** before claiming a win; overlap ⇒ inconclusive for that phase.
- **Repair must time the COMPLETE user-visible operation**, not just the cardinality pass. The shipped
  operation is *address-ordered cardinality pass + key-ordered conversion/compaction*. **A
  cardinality-only speedup cannot gate adoption if total `repairAfterLazy` is neutral or slower** —
  report the cardinality sub-phase as attribution, and gate on the total.
- **Determine the size-gate threshold**: sweep container counts to find where sorted stops paying.
- Report absolute times and whether each result is a **parity-gap closure** or an
  **absolute-throughput** gain.

## Phase 2 — ship what won (conditional, per phase; BLOCKED until the two decisions are made)

Ship the sorted traversal only for phases whose Phase 1 result separated, behind the chosen opt-in
mechanism plus the measured per-phase size gate. **Cannot be written until the scope decision and the
public control mechanism are resolved.**

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
- Phase 1 (`38-00`): sorted vs unsorted with **sort cost inside timing**; **payload-key vs header-key
  sorting compared**; **CRoaring teardown and repair references at matched boundaries** (or the results
  explicitly labelled rawr-internal deltas); both allocators, both hosts, five fresh processes, range
  separation applied; **repair timed as the complete operation**, cardinality sub-phase as attribution
  only; **mass-bitset teardown corpus** plus the 8-container clone corpus as size-gate control;
  **cold** scratch accounting for teardown and **BOTH cold and reusable-scratch for repair**; per-phase
  size-gate thresholds derived by the pinned rule.
- **Teardown two-stage requirement:** immediate teardown measured **unsorted / ascending-payload /
  descending-payload**, then **the same allocator refilled and the next construction/traversal
  measured** for each order, ideally with an **allocator-noise control** between. **A teardown win
  without the downstream stage does not count.**
- Phase 2 (`38-01`): **blocked** until the scope decision and control mechanism are pinned. Then:
  sorted traversal shipped only for phases that separated, size-gated per phase, chosen opt-in
  mechanism documented, **scratch-failure degrades to unsorted**.
- Correctness: order-invariance asserted (byte-identity + cardinalities + differential), repair-split
  identical to single-pass, no leak/double-free, failure injection green.
- Flag-OFF path verified indistinguishable from baseline; board gate held.
- `docs/parity-measurement.md` updated; `docs/allocator-address-order-pathology.md` cross-referenced.

## Chunk plan

- **`38-00`** — Phase 1 measurement: both phases, both allocators, both hosts, CRoaring references,
  payload-vs-header key comparison, mass-bitset teardown corpus + clone control, honest scratch
  accounting (repair cold **and** reused), the **teardown two-stage refill experiment**, per-phase
  size-gate sweep. **No production change.** Ready to implement — see
  [38-00](38-00-address-sort-measurement.md).
- **`38-01`** — Phase 2 implementation. **NOT ready** — blocked on (i) the scope decision
  (throughput-only vs parity lever) and (ii) the public control mechanism (per-op option vs explicit
  variants). Write it after `38-00` reports and both are pinned.

## Estimate

S–M for `38-00` (reuses the retained probe's discipline and the canonical worker). M for `38-01` — the
work is the flag plumbing, propagation rule, scratch reuse, graceful degradation, and the repair split,
not the sort itself.
