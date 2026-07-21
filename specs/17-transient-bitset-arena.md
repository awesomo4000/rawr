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

## Why an arena (not a pool, not a malloc clone)

The lazy path builds **all** output bitsets first, then a single `repairAfterLazy` sweep
converts them. So during the pre-repair window every transient bitset is live at once and
they all die together at repair — **batch lifetime, not interleaved churn**. A free-list
pool (recycling one buffer across many objects) does not fit, because the objects
coexist. Batch lifetime is the arena case: replace N allocator round-trips with N
bump-allocations plus a bulk free. Reimplementing a general allocator is out of scope;
the libc measurement in spec 16 was a diagnostic proxy, not a target.

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
  never routed through the arena. This answers the dense-regression risk: `manyMerge`
  already repairs internally and retains surviving dense bitsets, and the arena must not
  make it allocate transiently, allocate again persistently, copy 8 KB, and hold both.
- Forced-bitset **semantics are preserved**: a guaranteed-demote key still builds a
  bitset pre-repair (representation unchanged); only its *allocation source* changes.

### Eligibility prepass (must be cheap and exact)

Before allocating a key group's accumulator, compute the same-key input-cardinality sum
and route to the transient arena **only when the complete bound is known to be ≤ 4096**:

- Accumulate the sum in `u64` (or with an early-saturating cutoff that stops once the
  running sum exceeds 4096) — no overflow, no full second pass.
- Array/run/bitset inputs expose their cardinality cheaply. rawr has no distinct "lazy
  bitset" type, so the rule is on the stored value: **any bitset whose stored cardinality
  is `< 0` is ineligible** (unknown, however it arose — e.g. an in-place accumulator
  mid-fold), and that key uses the **normal allocator**. Do **not** trigger a fresh
  cardinality scan merely to qualify a key for the arena; the prepass must never add work
  that the transient path was meant to save.

## Phase A — non-shipping experiments (prove the ceiling)

Neither experiment ships or modifies the production `lazyOr` / `manyMerge` code paths:
each is a **separate prototype function or a compile-time benchmark-only path**. Both
must clear the Phase-A gates before any ownership design is attempted.

**Allocation granularity (decided for Phase A):** for a guaranteed-demote key, allocate
the **entire temporary `BitsetContainer`** (struct + words) from the transient arena.
Only the resulting demoted array container is built on the persistent allocator. This is
the cleanest experiment; words-only splitting is not needed to measure the ceiling.

**Reuse production kernels (control the variables):** the prototypes must call the same
accumulation (`setList`) and bitset→array demotion kernels the production path uses.
**Eligibility, allocator source, and lifetime are the only permitted behavioral
differences** — otherwise a kernel divergence, not the arena, could move the number.

**Two allocator variants per experiment (avoid a false no-go):** measure both a
`std.heap.ArenaAllocator` and an **exactly sized single-allocation
`FixedBufferAllocator`** (one child allocation of the counted guaranteed-demote size).
`ArenaAllocator`'s node geometry and atomic bump could miss the gate on their own even
though the transient-lifetime design is sound; the fixed-buffer variant isolates that, so
a miss is attributed to the right cause.

### A1 — Fused 2-way construct+repair (upper bound on sparse benefit)

A benchmark-only path that constructs and repairs a single 2-way forced lazy OR in one
scope, so the arena's lifetime never escapes. Measures the **maximum** sparse benefit
available from arena allocation with zero ownership risk. It does not optimize the real
`lazyOr` → later `repairAfterLazy` workflow — it establishes whether the ceiling
justifies pursuing that workflow at all. **Arena teardown (bulk free) is inside the timed
region**, so the free cost is in the comparison.

### A2 — Local-arena n-way prototype (benchmark-only)

A benchmark-only n-way prototype (a separate function, **not** an edit to `manyMerge`)
whose arena lifetime is fully local — `manyMerge` already encloses construction and
repair in one call, so this models that shape without touching production. Two workloads,
both required:

- **Sparse-heavy n-way** — many *shared sparse* keys whose summed bound is ≤ 4096, so the
  transient arena is actually exercised. This is the case that must show a win.
- **Dense n-way control** — genuinely dense shared keys that bypass the arena, held as a
  **no-regression control**.

### Phase-A gates (all required to proceed to Phase B)

- **Value parity, two levels:** each prototype's repaired output is **byte-identical to
  rawr's current path**; against the CRoaring oracle require **logical set/cardinality
  equality**, not byte-identity — portable serialization can legitimately differ when the
  two implementations pick different container representations.
- **Leak-free teardown:** arena plus all persistent allocations fully released under a
  leak-checking GPA; no leaked bytes, no double free.
- **Timing:** A1 sparse combined (including teardown) reaches ~1.07x territory; A2
  sparse-heavy shows a win and demonstrably enters the arena; A2 dense control shows **no
  beyond-noise regression**.
- **Memory:** peak child-allocator live bytes ≤ **110%** of the current path's peak (see
  measurement — this is physical/size-class bytes, not logical requested bytes).
- **Allocator measurements reported** for every variant (list below).

## Phase B — production integration (only if Phase A clears its gates)

The hard part; explicitly **not** designed until A justifies it.

- **D2 — Escaping ownership (primary blocker), with an honest API fork.** The public
  workflow needs the arena to outlive the `lazyOr` return and be used by a later
  `repairAfterLazy`. An **embedded** arena is unsafe: `ArenaAllocator`'s `Allocator`
  holds a pointer to the arena object, which moves when `RoaringBitmap` is returned by
  value. `OwnedBitmap` sidesteps this by never using its bitmap allocator after return
  (`src/bitmap.zig:2380`); `repairAfterLazy` would have to. The two viable vehicles are
  **not** equivalent in API impact, and the spec does not pretend otherwise:
  - a **heap-boxed arena sidecar** with a stable address — fully **internal**, no public
    API change; or
  - a **distinct lazy-result type** that owns the arena and exposes `repair()` →
    `RoaringBitmap` — an **additive public API change** (and it also retires the lazy
    footgun).

  The choice is deferred to Phase B, but it is a real fork: the sidecar preserves the
  current public surface; the lazy-result type extends it. What is invariant either way
  is the **deferred-cardinality contract and the semantics of a repaired result** — not
  the public surface.
- **Mixed-ownership representation.** `repairAfterLazy` and `deinit` free every bitset
  through `self.allocator` (`src/bitmap.zig:1537`). Arena-backed bitsets must be
  identifiable **without** losing that identity when cardinality is computed, so repair
  frees each container through the correct owner. Specify a representation, not "minimal
  bookkeeping."
- **Enumerated edge cases (all must have defined behavior):** deinit of an unrepaired
  result; repair failing partway (some demoted arrays built, arena not yet freed);
  repeated `lazyOrInPlace` before a repair; clone or move of an unrepaired result;
  construction failure with a mix of persistent (cloned) and transient (arena)
  containers.
- **Caller-allocator interaction — no detection, and honest guarantees.**
  `std.mem.Allocator` has no supported "is this an arena?" query; vtable-identity checks
  are brittle and are **out**. rawr can guarantee **correct ownership and cleanup calls
  for every caller allocator** — but *physical reclamation and the performance win are
  only guaranteed for reclaiming allocators*. If the caller's backing allocator is itself
  a non-reclaiming arena, the inner arena's bulk free is ignored by the outer allocator,
  so the transient bitsets are not physically reclaimed until the outer `OwnedBitmap`
  dies — the existing ~512 MB inflate/demote hazard (`api-design-notes.md`). State this
  limitation explicitly rather than claiming reclamation "regardless of caller
  allocator." The design must not *depend* on detecting the caller; correctness holds for
  all, reclamation holds for reclaiming allocators.
- **Failing-allocator tests** are a Phase-B precondition, given the mixed-ownership free
  paths.

## Measurement

- **Corpora.** Sparse 2-way (spec 16 corpus); **sparse-heavy n-way** (many shared
  ≤4096-bound keys — actually enters the arena); **dense n-way control**. The current
  `orMany` case (≈6 chunks, ~0.01 ms) is far too small — size the n-way corpora with
  enough shared keys and repetitions for stable construction, repair, allocation, and
  memory numbers.
- **Construction/repair split** (spec 16 method), five independent process runs, median
  and range/IQR per phase, identical setup/teardown on both sides, **teardown inside the
  combined timing**.
- **Report allocator behavior, not just wall-clock.** `ArenaAllocator` is **not** one
  slab acquisition — it allocates geometrically growing nodes and attempts resizes
  (`std/heap/ArenaAllocator.zig`), and an 8 KB payload plus node/alignment overhead
  initially crosses SMP's 8 KB class. Per variant report: child-allocator call count,
  requested bytes, **effective SMP size-class bytes**, and **actual peak live memory**.
  `queryCapacity()` and logical requested bytes are useful diagnostics but are **not**
  the peak-memory figure — the 110% gate applies to actual child-allocator live/peak
  (size-class) bytes.
- **Fixed-buffer sizing must be exact.** The `FixedBufferAllocator` variant is required
  (above), and its slab must be sized from a **count of actual guaranteed-demote keys**,
  not `min(a.size, b.size) × 8 KB` — that counts *potential* overlaps and can reserve ~2×
  the sparse need. Any deliberate over-allocation must be explicit and gated against the
  peak-memory ceiling.

## Acceptance

**Phase A (benchmark-only prototypes) — required to unlock Phase B:**

- Value parity: byte-identical to rawr's current path; logical set/cardinality equality
  vs the CRoaring oracle.
- Leak-free teardown under a leak-checking GPA.
- Both allocator variants measured (`ArenaAllocator` and exactly sized
  `FixedBufferAllocator`), reusing production accumulation/demotion kernels.
- Allocator measurements reported (calls, requested, SMP-class, peak).
- Timing: A1 sparse combined ≤ ~1.10x (approaching ~1.07x), A2 sparse-heavy wins and
  enters the arena, A2 dense control within noise.
- Memory: peak child-allocator live (size-class) bytes ≤ 110% of the current path on all
  workloads.

**Phase B (production integration) — only if Phase A passes:**

- Ownership vehicle chosen (sidecar vs lazy-result type) with the API impact stated;
  mixed-ownership representation and all enumerated edge cases given defined behavior.
- Sparse ≤ 1.10x preserved through the real escaping `lazyOr` → `repairAfterLazy`
  workflow; dense within noise; peak ≤ 110%.
- Correctness: representation tests (forced/size-selected, by-value/in-place),
  lazy-or/xor and in-place differential cases, footgun, edge cases, and
  **failing-allocator** tests green under `ReleaseSafe` and `ReleaseFast`.
- Full build green; no diagnostic allocator left in the tree.

## NO-GO

- Phase A misses its timing or memory gate → stop; do not build the ownership machinery.
- Phase B can only be made correct by leaking arena lifetime into the public contract in
  a way heavier than a distinct lazy-result type, or the enumerated edge cases can't be
  given clean semantics → park; the sparse workload alone does not justify it.

## Estimate

Phase A: S–M (two benchmark-scoped prototypes + the measurement harness), and it is
ready to chunk. Phase B: M–L, attempted only if A clears its gates — the cost is the
ownership decision (D2 fork), the mixed-ownership free paths, and the failing-allocator
hardening, not the arena itself.
