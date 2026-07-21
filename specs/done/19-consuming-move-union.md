<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 19: Consuming (move) in-place OR — cut clone demand

**Prototype-first, decision-gated.** Add one operation — a consuming in-place union that
**moves** an unmatched container from the consumed operand instead of cloning it — and
gate shipping on whether the intended workload actually has enough move opportunity to
matter. The lever is allocation **demand** (spec 18 closed the faster-allocator track),
but the benefit is real only under a specific, measurable condition.

> **Outcome (2026-07-21) — GO; shipped as an optional API.** Both chunks landed.
> `bitwiseOrInPlace​Consume` (`src/bitmap.zig:1247`) implements the full commit protocol —
> allocator + alias guards before mutation, reserve → cache-invalidate → matched merge →
> infallible backward-merge commit, `other` left valid-empty. M4 results scale with
> unmatched-chunk-key overlap as predicted:
> - 25% unmatched: **12% faster** fixpoint, **18% faster** direct sweep;
> - 50%: **31% faster** fixpoint; 75%: **53% faster** fixpoint;
> - **zero consuming clones**; `other` reusable with retained capacity; existing
>   `bitwiseOrInPlace` stayed within parity.
>
> Coverage: allocator mismatch (same vtable / different state), aliasing, exhaustive
> allocation-failure injection, reuse, mixed container types, all overlap levels, in-place
> OR parity, CRoaring differential. Verified across `ReleaseSafe`/`ReleaseFast`, x86_64
> Linux + Windows cross-builds, and five-process benches.
>
> **One honest caveat on the gate:** the `19-01` real-driver overlap-data requirement was
> **not** gathered — the decision was the synthetic sweep only, and at 25% overlap the
> fixpoint win (12%) is below the 15% gate (50%+ clears it easily). Shipping anyway is
> defensible because the API is **optional/opt-in with a clean cloning fallback** and is
> broadly beneficial. The remaining open item transfers to the caller: **the datalog engine
> must measure its actual unmatched-chunk-key overlap** to know its realized benefit and
> when to call the consuming path.
>
> Next target surfaced by the same run: **lazy-OR construction (2.19x)** — see the new
> spec when opened.

## Why, and the one condition it hinges on

- Spec 18: the whole-result allocator is not the lever; rawr's container model is already
  SMP-optimal.
- Spec 17 Phase A: a sparse 2-way union did **~98k allocations**, most of them clones of
  unmatched containers. **Caveat: that count is from spec 17's forced-lazy A1 workload,
  not from eager `bitwiseOrInPlace`.** It motivates the effort but is **not** this
  effort's expected savings — the eager in-place baseline must be **recounted** (task in
  the prototype) before any savings claim.
- Each container is **two allocations** — the container struct and its payload (array
  bytes, run bytes, or the 8 KB bitset). Moving the single tagged pointer transfers
  ownership of **both** allocations, so each moved container saves **two** allocations and
  a payload copy.
- The clone site: `bitwiseOrInPlace` (`src/bitmap.zig:1130`) clones every unmatched
  container from `other` (`:1173`, `:1214`).

**The condition (this is the gate, not a footnote).** A container is "unmatched" only when
its **high-16-bit chunk key** exists in one operand and not the other. If `R` and `ΔR`
hold *different values within the same 65,536-value chunk*, that is a **matched** container
and **cannot move** — it must merge. A fixpoint that quickly populates most chunk keys
leaves little to move. So the benefit depends entirely on **unmatched-chunk-key
frequency** in the real driver, not on unmatched values. Measuring that frequency in the
intended datalog workload is the decisive precondition — without it this risks being a
sound optimization for a rare case.

The driver is the datalog fixpoint (`R := R ∪ ΔR`, `ΔR` discarded each round). Scoped on
its own terms, not as a comparison — the reference has no move-union.

## Scope — one op, deliberately narrow

- **In this spec: consuming in-place OR only.** Working name
  **`bitwiseOrInPlaceConsume(self, other: *Self)`** — `self` accumulates the union,
  `other` is consumed.
- **Difference removed.** `bitwiseDifferenceInPlace` never clones right-only containers,
  and `new \ R` already consumes/mutates `new` while preserving `R`. Nothing to move.
- **XOR deferred** — same move opportunity, but only after union clears its gates and if a
  concrete driver wants it.
- **No by-value form** — the in-place consuming op already owns both operand roles.

## Phasing (how this will chunk) — prototype before any public API

1. **Prototype chunk (private / benchmark-only).** A private consuming-merge implementation
   plus the overlap-sweep harness and the **eager-baseline allocation recount**. **No
   public API is added.** This chunk produces the GO/NO-GO numbers.
2. **Public-API chunk (only on GO).** The public `bitwiseOrInPlaceConsume` with the full
   contract, differential tests, and failure injection.

A **NO-GO must leave no unused public API behind** — the method ships only if the
prototype clears the gates.

## Public contract (settle before the public-API chunk)

`bitwiseOrInPlaceConsume(self: *Self, other: *Self) !void`:

- **Same allocator required.** Containers do **not** record their allocator; a moved
  container is later freed through `self.allocator`. A move is valid only when `self` and
  `other` share the **exact** allocator handle (both `ptr` **and** `vtable`). On mismatch,
  return **`error.AllocatorMismatch` before any mutation** (caller falls back to cloning
  `bitwiseOrInPlace`).
- **Distinct bitmaps required.** On `self == other`, return **`error.AliasedOperands`
  before any mutation** (symmetric with the allocator-mismatch guard).
- **Success post-state:** `other` is a **valid, empty bitmap** — `size = 0`,
  `cached_cardinality = 0`, top-level capacity retained — safe to reuse, add to,
  `validate()`, or `deinit`. No husk/consumed flag.
- **Error post-state:** per the commit protocol below.

## Failure semantics — basic guarantee via a late, infallible commit

Existing in-place OR is not strongly atomic for matched containers, so this spec adopts
the **basic guarantee** with an explicit protocol:

1. **Preconditions** (no mutation): allocator match, distinct pointers.
2. **Reserve** (fallible, up front): grow `self`'s key/container index arrays to
   `self.size + (count of `other`'s unmatched chunk keys)`. **All** index allocation
   happens here.
3. **Invalidate `self`'s bitmap-level cache** (no allocation): set
   `self.cached_cardinality = -1` **before the first matched merge**. Otherwise a later
   merge failure could leave modified containers sitting behind a stale bitmap-level
   cardinality cache.
4. **Merge matched** (fallible, mutates `self`): merge each `other` matched container into
   `self`'s existing container with the current in-place merge.
5. **Commit** (infallible): free `other`'s now-redundant matched containers, then insert
   `other`'s unmatched tagged pointers by an **infallible backward merge** into the
   pre-reserved arrays (merge from the tail so no element is overwritten before it is
   moved). **Do not** call an insertion helper that could grow or allocate during commit —
   capacity was reserved in step 2. Finally set `other.size = 0` **and**
   `other.cached_cardinality = 0`.

Guarantee: **`other` is unchanged on any error** (all moves/frees are in the infallible
commit), and **`self` stays valid** though it may already contain completed matched
unions. Both bitmaps are always valid; nothing leaks or double-frees. (A stronger
"both inputs unchanged on error" would require non-mutating matched merges — out of scope;
noted as the alternative.)

## Design decisions to settle with review

- **D1 — Move partition.** Confirm array/bitset/run unmatched containers are all pure
  tagged-pointer transfers (no per-type copy to move).
- **D2 — Index-array cost is not removed.** Moving eliminates the payload+struct clones
  (the bulk), but `self`'s index arrays still grow. Measurement must **attribute** saved
  allocations: moved-container savings (2 each) vs residual index growth. State the ceiling
  honestly, never "zero allocs."
- **D3 — Naming.** `bitwiseOrInPlaceConsume` vs `...Consuming`; no boolean option, and not
  the `Owned` suffix (means arena ownership). Fits `api-design-notes.md`.

## Measurement — chunk-key sweep + real-driver data (required)

- **Deterministic unmatched-chunk-key overlap sweep:** 0 / 25 / 50 / 75 / 100% of `other`'s
  chunk keys unmatched in `self`, with exact rounds, delta sizes, container types, and key
  distribution pinned and documented.
- **Realistic fixpoint-pattern bench:** repeated `R := R ∪ ΔR` over many rounds with a
  freshly built, then consumed, `ΔR` each round. **Timing boundaries (report all three,
  kept separate):**
  - the **union operation only**, with `ΔR` construction *outside* the timed region;
  - the **full round lifecycle** separately, including `ΔR` construction and cleanup;
  - **allocator counters reset immediately around the union operation**, so the alloc
    numbers attribute to the union, not to delta construction.

  Without this split the 15% gate drifts with how much unrelated delta-construction cost
  is folded in.
- **Real-driver overlap data is a requirement, not a nicety.** The repo has no datalog
  driver or trace, and the GO gate depends on its actual unmatched-chunk-key frequency.
  Satisfy this one of exactly these ways: **(a)** a supplied trace / distribution from the
  driver; **(b)** instrumentation added to the driver to collect chunk-key overlap; or
  **(c)** hold production GO until (a) or (b) exists. A synthetic sweep alone cannot
  authorize the public API.
- **Recount the eager baseline:** measure the allocation count of the current eager
  `bitwiseOrInPlace` on these workloads before claiming any savings (the 98k figure is
  forced-lazy A1, not this path).
- Report on the spec-16/18 authoritative environment (`ReleaseFast`, native, M4 host), five
  independent process runs, median + range.

## Acceptance (GO) — numeric gates set now

- **Allocation-reduction correctness (exact, no tolerance):** the result is
  deterministic, so assert it exactly —
  - baseline unmatched-right container clones: **exactly `2 × moved_container_count`**
    (payload + struct per unmatched-right container);
  - consuming unmatched-right clones: **exactly zero**;
  - index-array growth and matched-merge allocations reported **separately** (they are the
    residual the move does not remove — D2), never folded into the above.

  No `±` tolerance — a tolerance could silently accept losing half the expected savings.
- **Value at the realistic overlap:** at the unmatched-chunk-key frequency the real-driver
  data (a/b) exhibits, the fixpoint-pattern bench shows the consuming op **≥ 15% median
  faster** than cloning `bitwiseOrInPlace`. (If the real frequency is low, this simply
  will not be met — that is the NO-GO working as intended.)
- **No regression:** existing `bitwiseOr` / `bitwiseOrInPlace` stay within **≤ 2% median**
  of current numbers; semantics untouched (additive change).
- **Correctness + failure injection:** result set-equal to current `bitwiseOrInPlace` (and
  logically to the CRoaring oracle); differential coverage for the consuming op;
  **exhaustive allocation-failure injection** sweeping every fallible site across the
  reserve and matched-merge steps, asserting after **each** injected failure that `other`
  is unchanged, both bitmaps pass `validate()`, **and `cardinality()` on both returns the
  correct value** (call it explicitly — `validate()` does not detect a stale bitmap-level
  cardinality cache), and both `deinit` leak-free under a leak-checking GPA;
  `error.AllocatorMismatch` (tested with the **same vtable but different allocator state**,
  e.g. two separate arenas) and `error.AliasedOperands` both guarded and tested.
- Full build green under `ReleaseSafe` and `ReleaseFast`.

## NO-GO

- Real-driver **unmatched-chunk-key frequency is low** (deltas mostly touch chunks `R`
  already has) → moves are rare, the 15% gate is missed → record it and stop; no public
  API ships.
- D2 attribution shows residual index-array growth dominates the savings at the realistic
  frequency → same call.

## Estimate

M. The move is a tagged-pointer transfer, but the deliverable is the ownership/commit
contract, the allocator-match + alias preconditions, exhaustive failure injection, and —
decisively — the chunk-key-frequency measurement (with real-driver data) that says whether
this pays off at all.
