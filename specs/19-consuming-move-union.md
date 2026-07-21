<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 19: Consuming (move) set operations — cut clone demand

Reduce allocation **demand** in the operations that build a result from two bitmaps, by
**moving** the containers of a consumed input instead of cloning them. Driven by the one
perf lever left standing after specs 16–18: not a faster allocator (spec 18 closed that),
but fewer allocations.

## Why

The chain of findings converges here:

- Spec 18 showed the whole-result allocator is **not** the lever — libc barely helps and
  regresses the container-heavy set ops; rawr's container model is already SMP-optimal.
- Spec 17 Phase A surfaced the real cost: a sparse 2-way union does **~98k allocations**,
  most of them **clones of unmatched containers**, not transient bitsets.
- The clone sites are concrete. `bitwiseOrInPlace` (`src/bitmap.zig:1130`) clones every
  unmatched container from `other` into the result (`:1173`, `:1214`); the in-place xor
  and difference paths do the same (`:1417`, `:1457`). Each clone is a fresh payload
  allocation (array/run bytes, or an 8 KB bitset) plus a copy.
- rawr's containers are **tagged pointers, one heap alloc each**, so an unmatched
  container can be transferred by **moving its tagged pointer** — no payload allocation,
  no copy — *when the source input is being discarded anyway.*

The concrete driver is the datalog fixpoint (`R := R ∪ ΔR`, `ΔR` discarded each round) and
the novelty step (`ΔR := new \ R`) — see the datalog-driver context. Those loops throw
away one operand every iteration, so cloning its unique containers is pure waste. A
consuming operation turns that clone traffic into pointer moves, allocator-independently.

This is scoped on its own terms (the engine's fixpoint churn), not as a comparison — the
reference has no move-union, so this is an ergonomic rawr adds for its own driver.

## Mechanism

A **consuming** variant of the in-place ops takes ownership of `other` and, for each key:

- **unmatched in `other`** (result/`self` lacks the key): **move** `other`'s tagged
  pointer into `self` — no clone, no free of the payload;
- **matched**: merge as today (in-place where the existing path already does);
- **unmatched in `self`**: unchanged (already no clone).

`other` is left in a **consumed** state: the moved slots no longer own their containers, so
`other`'s later `deinit` frees only the husk (top-level arrays + any un-moved containers).

## Design decisions (settle with review before chunking)

- **D1 — Surface.** Core is the fixpoint fit: an in-place consuming union
  (`self` grows, `other` consumed) — working name `orInPlaceConsuming(self, other: *Self)`
  pending the naming convention. Decide whether to also add consuming **xor** and
  **difference** (the datalog novelty/delta steps want them), and whether a *by-value*
  consuming union (`a ∪ b` consuming both) earns its place or the in-place form suffices.
- **D2 — Ownership contract (primary risk).** Define `other`'s exact post-state so its
  `deinit` frees the husk and never double-frees a moved container or leaks an un-moved
  one. Guard against use-after-consume and against consuming `self` (aliasing). This is the
  correctness crux, as with spec 17's ownership work.
- **D3 — Move vs merge partition.** Enumerate per container-pair which keys move (unmatched
  in `other`) vs merge (matched), and confirm run/array/bitset moves are all just tagged-
  pointer transfers under the existing model (no type-specific copy needed to move).
- **D4 — Index-array cost is not eliminated.** Moving containers removes the *payload*
  clones (the bulk of the 98k), but `self`'s key/container arrays still grow and may
  reallocate. Measurement must **attribute** the saved allocations: payload-clone share
  (removed) vs index growth (remains). State the realistic ceiling, not "zero allocs."
- **D5 — Genericity + failure atomicity.** Allocator-generic; on OOM mid-operation (e.g.
  growing `self`'s index while partway through moving `other`'s containers), define a state
  with no double-free and no leak — likely "moves are all-or-nothing per key, and a failure
  leaves both bitmaps valid," to be specified.
- **D6 — Naming.** Fit the existing convention (`*InPlace`, `*Owned`); `*Consuming` vs
  `*Into` vs an explicit `consume: bool`. This connects to the naming-consistency theme in
  `api-design-notes.md`.

## Measurement (gate the win, prototype-first where cheap)

- **Allocation count** on the sparse 2-way union: consuming vs the current in-place path,
  reported with the D4 attribution (how much of the ~98k is removed).
- **Timing**, two workloads:
  - the sparse 2-way union (single call);
  - a **fixpoint-pattern** bench — repeated `R := R ∪ ΔR` over many rounds with a freshly
    built, then consumed, `ΔR` each round — the realistic datalog driver, measuring
    cumulative allocations and time across rounds.
- Report on the spec-16/18 authoritative environment (`ReleaseFast`, native, M4 host),
  five independent process runs, median + range.

## Acceptance (GO)

- **Allocation demand drops materially** on the sparse union (payload clones of unmatched
  containers eliminated), and the **fixpoint-pattern** bench shows a cumulative time win
  beyond noise.
- **Correctness**: consuming result is **set-equal** to the current eager/in-place union
  (and to the CRoaring oracle, logically); differential tests cover consuming or/xor/diff
  by value and in place; the consumed husk deinits leak-free under a leak-checking GPA;
  use-after-consume and self-aliasing are guarded.
- **Non-consuming paths unchanged** — this is additive; existing `bitwiseOr` /
  `bitwiseOrInPlace` semantics and numbers are untouched.
- Full build green under `ReleaseSafe` and `ReleaseFast`; no regression elsewhere in
  `bench_croaring`.

## NO-GO

- If the D4 attribution shows index-array growth, not payload clones, dominates the ~98k
  on the target workloads — then moving containers saves little and the win is illusory;
  stop and record it.
- If clean consuming ownership (D2/D5) can't be given leak-free, double-free-free semantics
  without contorting the container model, park it.

## Estimate

M. The move itself is a tagged-pointer transfer, but the deliverable is the ownership
contract (D2/D5), the consuming variants across or/xor/diff, the differential coverage,
and the measured allocation/time win on the fixpoint pattern.
