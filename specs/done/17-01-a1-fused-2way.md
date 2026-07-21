<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 17-01: A1 — fused 2-way construct+repair (sparse ceiling)

Second chunk of the [transient-bitset arena](17-transient-bitset-arena.md) Phase A.
Builds experiment **A1** on the `17-00` harness: the benchmark-only fused 2-way path
that measures the **maximum** sparse benefit of arena allocation with zero ownership
risk. Its whole job is to produce the ceiling number that decides whether Phase B is
worth pursuing.

## What it is

A benchmark-only function that constructs **and** repairs a single 2-way forced lazy OR
in one scope, so the transient arena's lifetime never escapes. It does **not** touch the
production `lazyOr` / `repairAfterLazy` workflow — it is the upper bound, not the
shippable path.

## Behavior

- **Eligibility prepass:** for each overlapping key, compute the input-cardinality sum
  (`c_a + c_b`) in `u64`, or with an early-saturating cutoff at 4096. A key is
  arena-eligible only when the bound is **known ≤ 4096**. Any bitset input with stored
  `cardinality < 0` makes the key ineligible; never run a fresh cardinality scan to
  qualify a key.
- **Allocation granularity:** for an eligible key, allocate the **entire temporary
  `BitsetContainer`** (struct + words) from the transient allocator; build the demoted
  array on the persistent allocator. Ineligible keys use the current path unchanged.
- **Reuse production kernels:** call the same `setList` accumulation and bitset→array
  demotion kernels the production path uses. **Eligibility, allocator source, and
  lifetime are the only behavioral differences.**
- **Mixed-ownership bookkeeping (benchmark-local):** eligible (transient) and ineligible
  (persistent) bitsets coexist before repair, so the prototype keeps a **benchmark-local
  flag or side table** marking which containers are arena-backed, and repair + error
  cleanup free each through the correct allocator. This bookkeeping is confined to the
  benchmark: **do not** change production container tags or `RoaringBitmap` layout for it.
- **Variants registered with the harness:**
  1. baseline (current path, no arena);
  2. `std.heap.ArenaAllocator` as the transient allocator;
  3. exactly sized single-allocation `FixedBufferAllocator`, slab sized from a **count of
     actual arena-eligible keys** (not `min(a.size,b.size)`).
- **The FBA pre-count is timed.** Counting eligible keys, computing the aligned slab
  footprint, and allocating the slab all happen **inside** the construction and combined
  timed regions — FBA gets no precomputed-setup advantage over the arena. Assert the
  exact slab never falls back or returns `OutOfMemory` (the count must be exact).
- Arena/buffer teardown (bulk free) is inside the harness's combined timed region.

## Acceptance

- **Value parity:** each variant's repaired output is **byte-identical to rawr's current
  path** on the sparse 2-way corpus, and **logically equal (set + cardinality)** to the
  CRoaring oracle.
- **Leak-free:** arena/buffer plus all persistent allocations fully released under a
  leak-checking GPA; no leaks, no double free (exercise the eligible and ineligible key
  mixes).
- **Timing (the ceiling), two denominators:** report construction / repair / combined
  (median + range) for all three rawr variants **and the timed CRoaring reference**, and
  give both numbers per transient variant:
  - **improvement** = transient ÷ rawr baseline (informational);
  - **gate** = transient ÷ **CRoaring reference**, which must be combined **≤ 1.10x**
    (approaching the spec-16 isolated-allocator ~1.07x) for at least one transient
    variant; report which.
  Gate measured in the authoritative environment (`ReleaseFast`, native, spec-16 M4 host).
- **Memory:** peak child-allocator live (size-class) bytes ≤ **110%** of baseline.
- **Attribution:** if `ArenaAllocator` misses but `FixedBufferAllocator` clears the gate,
  that is a *pass with a noted cause* (arena node geometry / atomic bump), not a no-go —
  record it explicitly so Phase B picks the right vehicle.
- Benchmark-only; full build green under `ReleaseSafe` and `ReleaseFast`.

## Result to record

The A1 ceiling: the best transient variant's combined **ratio vs the CRoaring reference**
(the gate) plus its **improvement vs rawr baseline**, and which allocator variant produced
it — this is half of the Phase-A go/no-go input.
