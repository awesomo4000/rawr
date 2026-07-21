<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 17: Transient-bitset arena for lazy union

**Experiments before ownership.** The lazy-union path spins up short-lived bitset
containers whose allocation cost — not the loop — is the remaining gap on
`lazyOr+repair (sparse)` (spec 16). An internal arena for those transient bitsets could
close it. But the production workflow (`lazyOr` returns, user calls `repairAfterLazy`
later) makes arena *ownership* the hard part, so this spec is staged: **prove the
performance ceiling with two non-shipping experiments first, and only design the
escaping arena-backed result if those clear both timing and memory gates.**

## Why

Spec 16 closed the algorithmic side of `lazyOr+repair (sparse)` and proved the residual
is **allocator-bound, not loop-bound**: with all construction fixes in place the
combined ratio sat at ~1.22x under `smp_allocator`, but swapping only the lazy output
allocator took construction to ~0.99x and combined to ~1.07x. Forced lazy union builds
thousands of 8 KB bitset payloads that are freed again almost immediately (repair
demotes most straight back to arrays); the general allocator's per-object cost on that
churn is the whole gap. Full context in `done/16-lazy-union-forced-bitset.md`.

The value is **not** the sparse 2-way number itself (off-design for lazy). It is that a
reusable *transient allocator for scratch containers* is a general lever for any
allocation-churn-heavy caller — the same "parent-owned temporary storage" direction the
spec 13 analysis pointed at, composing with rawr's existing arena/`Owned` story.

## The unifying constraint: only arena guaranteed-demote bitsets

The single decision that makes the rest tractable: **arena a transient bitset only when
its input-cardinality upper bound guarantees it will demote to an array at repair**
(`c_a + c_b ≤ ArrayContainer.MAX_CARDINALITY` for the 2-way case; the running
upper-bound sum for n-way). Consequences:

- **No survivor copy-out.** A guaranteed-demote bitset never persists as a bitset, so it
  is always freed in bulk with the arena; only the demoted *array* (built on the real
  allocator) escapes. The entire dense >4096 copy-out problem disappears.
- **Dense is untouched by construction.** Keys whose sum exceeds the threshold keep
  today's path exactly, so a genuinely dense union — which may survive as a bitset — is
  never routed through the arena. This directly answers the dense-regression risk:
  `manyMerge` already repairs internally and retains surviving dense bitsets, and the
  arena must not make it allocate transiently, allocate again persistently, copy 8 KB,
  and hold both.
- Forced-bitset **semantics are preserved**: a guaranteed-demote key still builds a
  bitset pre-repair (representation unchanged); only its *allocation source* changes.

This turns "transient-bitset arena" into "cheap allocation for the forced bitsets that
were always going to demote" — precisely the waste spec 16 measured.

## Phase A — non-shipping experiments (prove the ceiling)

Neither experiment changes `RoaringBitmap` or ships. Both must clear the Phase-A gates
below before any ownership design is attempted.

### A1 — Fused 2-way construct+repair (upper bound on sparse benefit)

A benchmark-only path that constructs and repairs a single 2-way forced lazy OR in one
scope, so the arena's lifetime never escapes. This measures the **maximum** sparse
benefit available from arena allocation, with zero ownership risk. It does not optimize
the real `lazyOr` → later `repairAfterLazy` workflow — it establishes whether the
ceiling justifies pursuing that workflow at all.

### A2 — Local-arena n-way prototype (inside `manyMerge`)

`manyMerge` already encloses construction and repair in one call, so an arena's lifetime
is naturally local there — no escaping result. Prototype transient storage for the
guaranteed-demote keys only, and measure against a real dense n-way corpus. Purpose: show
the arena does **not** regress the n-way case (its intended use), given the
guaranteed-demote gate.

### Phase-A gates (both required to proceed to Phase B)

- **Timing:** A1 sparse combined reaches ~1.07x territory (approaching the spec-16
  isolated-allocator figure); A2 dense shows **no beyond-noise regression**.
- **Memory:** peak live transient bytes do not exceed the current path's peak by more
  than a small margin (ceiling stated below), measured — not assumed.

## Phase B — production integration (only if Phase A clears its gates)

This is the hard part and is explicitly **not** designed until A justifies it.

- **D2 — Escaping ownership (primary blocker).** The public workflow needs the arena to
  outlive the `lazyOr` return and be used by a later `repairAfterLazy`. An **embedded**
  arena is unsafe: `ArenaAllocator`'s `Allocator` holds a pointer to the arena object,
  which moves when `RoaringBitmap` is returned by value. `OwnedBitmap` sidesteps this by
  never using its bitmap allocator after return (`src/bitmap.zig:2380`); `repairAfterLazy`
  would have to. Production therefore needs an explicit stable-ownership mechanism —
  candidates: a heap-boxed arena sidecar with a stable address, or a **distinct
  lazy-result type** that owns the arena and exposes only `repair()` → `RoaringBitmap`.
  The lazy-result type also cleanly fixes the existing lazy footgun (invalid until
  repaired). Decide this before any code.
- **Mixed-ownership representation.** `repairAfterLazy` and `deinit` free every bitset
  through `self.allocator` (`src/bitmap.zig:1537`). Arena-backed bitsets must be
  identifiable **without** losing that identity when cardinality is computed, so repair
  frees each container through the correct owner. Specify a representation, not "minimal
  bookkeeping."
- **Enumerated edge cases (all must have defined behavior):**
  - deinit of an unrepaired result (arena still owns transients);
  - repair failing partway through (some demoted arrays built, arena not yet freed);
  - repeated `lazyOrInPlace` before a repair;
  - clone or move of an unrepaired result;
  - construction failure with a mix of persistent (cloned) and transient (arena) containers.
- **Caller-allocator interaction — no detection.** `std.mem.Allocator` has no supported
  "is this an arena?" query; vtable-identity checks are brittle and are **out**.
  Double-arena is **not** harmless: an inner arena's frees become near-no-ops inside an
  outer arena, so transient bitsets persist until the outer `OwnedBitmap` dies — the
  existing ~512 MB inflate/demote hazard (`api-design-notes.md`). The design must make
  the transient arena's lifetime self-contained regardless of the caller's allocator,
  not conditionally skip based on what the caller passed.
- **Failing-allocator tests** are a precondition for production integration, given the
  mixed-ownership free paths.

## Measurement

- **Both workloads with a real corpus.** Sparse 2-way (spec 16 corpus) and a **concrete
  dense n-way** corpus — the current `orMany` case (≈6 chunks, ~0.01 ms) is far too small
  to decide this; define enough shared keys and repetitions for stable construction,
  repair/copy-out, allocation, and memory numbers.
- **Construction/repair split** (spec 16 method), five independent process runs, median
  and range/IQR per phase, identical setup/teardown on both sides.
- **Report allocator behavior, not just wall-clock.** `ArenaAllocator` is **not** one
  slab acquisition — it allocates geometrically growing nodes and attempts resizes
  (`std/heap/ArenaAllocator.zig`), and an 8 KB payload plus node/alignment overhead
  initially crosses SMP's 8 KB class. So report, per variant: child-allocator call count,
  `queryCapacity()`, requested bytes, effective SMP size-class bytes, and actual peak live
  memory.
- **Fixed-buffer sizing (if used) must be exact.** `min(a.size, b.size) × 8 KB` counts
  *potential* overlaps and can reserve ~2× the sparse need; a fixed-buffer variant must
  first count actual forced-bitset (guaranteed-demote) keys, or explicitly accept and gate
  the over-allocation against the peak-memory ceiling.

## Acceptance (hard gates)

- **Sparse:** `lazyOr+repair (sparse)` combined **≤ 1.10x**, construction at/near parity.
- **Dense:** n-way combined regression **within the measured noise band** (the
  guaranteed-demote gate should make this hold by construction; verify it does).
- **Peak memory:** transient peak **≤ 110%** of the current path's peak on both workloads.
- **Correctness:** representation tests (forced/size-selected, by-value/in-place),
  lazy-or/xor and in-place differential cases, footgun, edge cases, and
  **failing-allocator** tests all green under `ReleaseSafe` and `ReleaseFast`. No change
  to the deferred-cardinality contract; any ownership vehicle (sidecar/lazy-result type)
  keeps the arena lifetime internal.
- Full build green; no diagnostic allocator left in the tree.

## NO-GO

- Phase A misses its timing or memory gate → stop; do not build the ownership machinery.
- Phase B can only be made correct by leaking arena lifetime into the public contract in
  a way heavier than a distinct lazy-result type, or the enumerated edge cases can't be
  given clean semantics → park; the sparse workload alone does not justify it.

## Estimate

Phase A: S–M (two benchmark-scoped prototypes + the measurement harness). Phase B: M–L
and only attempted if A clears its gates — the cost is the ownership decision (D2), the
mixed-ownership free paths, and the failing-allocator hardening, not the arena itself.
