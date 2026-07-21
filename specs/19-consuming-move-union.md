<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 19: Consuming (move) in-place OR — cut clone demand

**Prototype-first, decision-gated.** Add one operation — a consuming in-place union that
**moves** an unmatched container from the consumed operand instead of cloning it — and
gate shipping on whether the intended workload actually has enough move opportunity to
matter. The lever is allocation **demand** (spec 18 closed the faster-allocator track),
but the benefit is real only under a specific condition made explicit below.

## Why, and the one condition it hinges on

- Spec 18: the whole-result allocator is not the lever; rawr's container model is already
  SMP-optimal.
- Spec 17 Phase A: a sparse 2-way union does ~98k allocations, most of them **clones of
  unmatched containers**. Each container is **two allocations** (struct + payload), so
  ~98k allocations is roughly **~49k cloned containers** — and moving a tagged pointer
  transfers the whole ownership graph, saving **both** allocations per moved container.
- The clone site: `bitwiseOrInPlace` (`src/bitmap.zig:1130`) clones every unmatched
  container from `other` (`:1173`, `:1214`). Because rawr containers are tagged pointers
  (one heap alloc each, struct+payload), an unmatched container can be **moved** when
  `other` is discarded anyway.

**The condition (this is the gate, not a footnote).** A container is "unmatched" only when
its **high-16-bit chunk key** exists in one operand and not the other. If `R` and `ΔR`
hold *different values within the same 65,536-value chunk*, that is a **matched** container
and **cannot move** — it must merge. A fixpoint that quickly populates most chunk keys
leaves little to move. So the benefit depends entirely on **unmatched-chunk-key
frequency** in the real driver, not on unmatched values. **Measuring that frequency in the
intended datalog workload is the most important precondition** — without it this risks
being a sound optimization for a rare case.

The driver is the datalog fixpoint (`R := R ∪ ΔR`, `ΔR` discarded each round). Scoped on
its own terms, not as a comparison — the reference has no move-union.

## Scope — one op, deliberately narrow

- **In this spec: consuming in-place OR only.** Working name
  **`bitwiseOrInPlaceConsume(self, other: *Self)`** — `self` accumulates the union,
  `other` is consumed.
- **Difference is removed.** `bitwiseDifferenceInPlace` never clones right-only containers
  — it only mutates existing `self` containers — and the novelty step `new \ R` already
  consumes/mutates `new` while preserving `R`. There is nothing to move; no consuming
  difference.
- **XOR is deferred.** It has the same move opportunity, but only add it *after* union
  clears its gates and if a concrete driver wants it.
- **No by-value form.** The pointer-based in-place consuming op already takes ownership of
  both operands' roles (`self` = result, `other` = consumed); a by-value variant adds no
  capability.

## Public contract (settle before chunking)

`bitwiseOrInPlaceConsume(self: *Self, other: *Self) !void`:

- **Same allocator required.** Containers do **not** record their allocator; a moved
  container will later be freed through `self.allocator`. So a move is valid only when
  `self` and `other` share the **exact** allocator handle (both `ptr` **and** `vtable`).
  On mismatch, return a defined **`error.AllocatorMismatch` before any mutation** (caller
  falls back to the cloning `bitwiseOrInPlace`).
- **Distinct bitmaps required.** `self != other` (guard against self-aliasing).
- **Success post-state:** `other` is left a **valid, empty bitmap** — `size = 0`,
  cardinality 0, top-level capacity retained — safe to reuse, add to, validate, or
  `deinit`. No "consumed/husk" flag and no use-after-consume state to track.
- **Error post-state:** stated by the commit protocol below.

## Failure semantics — basic guarantee via a late commit

Existing in-place OR is not strongly atomic for matched containers, so this spec adopts
the **basic guarantee** with a commit protocol that makes it clean:

1. **Preconditions** (no mutation): allocator match, distinct pointers.
2. **Reserve** (fallible, up front): grow `self`'s key/container index arrays to
   `self.size + (count of `other`'s unmatched chunk keys)`. All index allocation happens
   here.
3. **Merge matched** (fallible, mutates `self`): merge each `other` matched container into
   `self`'s existing container using the current in-place merge.
4. **Commit** (infallible): free `other`'s now-redundant matched containers, **move**
   `other`'s unmatched tagged pointers into the pre-reserved slots, then set `other.size =
   0`. No allocation here — capacity was reserved in step 2.

Guarantee: **`other` is unchanged on any error** (moves and frees are deferred entirely to
the infallible commit), and **`self` remains valid** though it may already contain some
completed matched unions. Both bitmaps are always valid; neither leaks nor double-frees.
(A stronger "both inputs unchanged on error" guarantee would require non-mutating matched
merges — out of scope; note it as the alternative if a caller ever needs it.)

## Design decisions to settle with review

- **D1 — Move partition.** Confirm array/bitset/run unmatched containers are all just
  tagged-pointer transfers under the current model (no per-type copy to move).
- **D2 — Index-array cost is not removed.** Moving containers eliminates the *payload+struct*
  clones (the bulk), but `self`'s index arrays still grow. Measurement must **attribute**
  saved allocations: moved-container savings (2 allocs each) vs residual index growth.
  State the realistic ceiling, never "zero allocs."
- **D3 — Naming/convention.** `bitwiseOrInPlaceConsume` vs `...Consuming`; avoid a boolean
  option and avoid the `Owned` suffix (already means arena ownership). Fits the
  `api-design-notes.md` naming theme.

## Measurement — the chunk-key sweep is the core

- **Deterministic unmatched-chunk-key overlap sweep:** 0 / 25 / 50 / 75 / 100% of
  `other`'s chunk keys unmatched in `self`, with exact rounds, delta sizes, container
  types, and key distribution pinned and documented. This maps benefit directly to the
  condition that governs it.
- **Realistic fixpoint-pattern bench:** repeated `R := R ∪ ΔR` over many rounds with a
  freshly built, then consumed, `ΔR` each round — cumulative allocations and time.
- **Best: derive the delta/overlap distribution from the real datalog driver** so the
  measured unmatched-chunk-key frequency reflects the actual workload, not a synthetic
  guess.
- Allocation count (with the D2 attribution) and timing, on the spec-16/18 authoritative
  environment (`ReleaseFast`, native, M4 host), five independent process runs, median +
  range.

## Acceptance (GO)

- The overlap sweep shows allocation demand dropping materially **as unmatched-chunk-key
  frequency rises**, and the fixpoint-pattern bench shows a cumulative time win beyond
  noise at the frequencies the real driver actually exhibits.
- **Correctness:** result set-equal to the current `bitwiseOrInPlace` (and logically to the
  CRoaring oracle); differential coverage for the consuming op; `AllocatorMismatch` and
  self-aliasing guarded and tested; the emptied `other` is valid and leak-free (reuse it,
  then `deinit`, under a leak-checking GPA); error-path leaves both bitmaps valid.
- **Additive:** existing `bitwiseOr` / `bitwiseOrInPlace` semantics and numbers untouched.
- Full build green under `ReleaseSafe` and `ReleaseFast`.

## NO-GO

- If the real driver's **unmatched-chunk-key frequency is low** (deltas mostly touch chunks
  `R` already has), moves are rare and the win is illusory → record it and stop; this
  becomes a rare-case optimization not worth the API surface.
- If the D2 attribution shows residual index-array growth dominates the savings at the
  realistic frequency → same call.

## Estimate

M. The move is a tagged-pointer transfer, but the deliverable is the ownership/commit
contract, the allocator-match precondition, differential coverage, and — decisively — the
chunk-key-frequency measurement that says whether this pays off at all.
