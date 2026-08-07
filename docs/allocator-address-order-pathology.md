<!-- SPDX-License-Identifier: MPL-2.0 -->

# Allocator address-order pathology

A generalizable write-up of a performance pathology found while closing rawr's lazy-OR construction
gap (campaign spec 31; diagnosis specs 35–37). It is **not** rawr-specific — it was reproduced with
**no rawr code at all** — so it is recorded here as a pattern to recognize and test for whenever an
allocator-sensitive gap appears.

## The pathology in one sentence

A general-purpose, multi-thread-oriented allocator can be **faster at allocating** while returning
addresses in an order that is **much more expensive to traverse**, so the cost shows up in whatever
code later touches those buffers — not in the allocator at all.

## Symptoms

- Swapping only the allocator moves a large share of the gap, with **no change to bytes, alignment,
  algorithm, or codegen**.
- The allocator's own `alloc`/`free` time is **comparable or better** than the faster alternative.
- Time appears to be inside a routine you have proven identical (same instructions, same size, same
  libc entry point) — e.g. a `memset`/`bzero` of a fixed size.
- **Page-fault counts are low and do not explain the gap.**
- Interleaving small allocations among the large ones makes it **worse**, even though total bytes are
  unchanged.

## The cheap diagnostic: the address-sort control

**Sort the pointers you already have by address, then re-time the traversal.** Nothing else changes —
same allocator, same buffers, same addresses, same byte volume, same routine. Only traversal *order*
differs.

- If sorting recovers most of the gap → **allocator-induced address order/locality.**
- If sorting recovers nothing → look elsewhere.

The sort is cheap enough not to distort the result: **measured 0.132 ms best / 0.157 ms mean** for
16,364 8-byte pointers (`std::sort`, shuffled 64 KB-stride addresses, Apple M4) — a small fraction of
the 1.7–2.7 ms recovered in our case. **But do not take that as licence to exclude it:** report the
honest sort-plus-traversal number (see Measured instance below).

Two refinements worth knowing:

- **Order-sensitivity asymmetry is the real signal.** Measure the sorted-vs-unsorted delta for *both*
  allocators. An allocator whose addresses are already near-sequential is only weakly order-sensitive
  (libc recovered **0.063–0.180 ms**); the pathological one is strongly order-sensitive (SMP recovered
  **1.681–2.737 ms**).
- **UNTESTED IDEA:** bucketing by slab base would be O(N) and might get near-sorted order more
  cheaply than a comparison sort. **This has not been measured** — it is a hypothesis, not a
  recommendation. At ~0.13 ms per 16k pointers there is little to optimize anyway, so prefer the plain
  sort until someone demonstrates otherwise.

## Critical reconciliation: TLB misses are NOT page faults

This pathology is easy to misdiagnose as a **first-touch / page-fault** problem, and a fault-counter
experiment will then *refute* the fault hypothesis while the layout effect remains completely real.
Both can be true at once:

- **Pages resident** ⇒ few or no page faults.
- **Translations not cached** ⇒ TLB misses and page-table walks per access.

In our case a dedicated experiment (spec 36) measured **40 operation faults** across a construction
touching ~134 MB, with **100% page reuse** proven and no gain from pre-conditioning residency — a
clean refutation of first touch. The address-order effect was nonetheless 1.681–2.737 ms. **A refuted
page-fault hypothesis does not clear the allocator.** Check order separately.

## Why the ordering arises — ALLOCATOR IMPLEMENTATION (not intent)

**No allocator deliberately disperses allocations to hurt you.** Zig 0.16's `std.heap.SmpAllocator`
contains no dispersal policy. The observed ordering is an **emergent consequence** of ordinary
scalability machinery:

- **64 KB slabs** as the backing unit,
- **LIFO freelists** (most-recently-freed handed back first),
- **size-class segregation**,
- **rotating thread-metadata slots** keyed by thread id.

Each is sensible for multi-threaded throughput; together they produce an allocation stream whose
order is unrelated to address order. **Measured, and the two cases differ — do not conflate them:**

- **Header-interleaved** (16 B headers requested between the 8 KB buffers): **median 64 KB stride.**
- **Words-only** (8 KB buffers only): **8 KB absolute stride but DESCENDING order** — consistent with
  a LIFO freelist handing back most-recently-freed first.

libc's stream was roughly **8 KB and ascending** (essentially sequential) in both cases.

**Note what this implies:** in the words-only case the problem is **direction, not stride magnitude** —
an 8 KB *descending* walk was materially slower than the same 8 KB walk ascending. Any mechanism story
must account for that, and it is a further reason the stride-magnitude/TLB-reach account below is
hypothesis rather than explanation.

### Hardware mechanism — HYPOTHESIS, NOT PROVEN

The following is a *plausible* account, **not** something this work demonstrated. Treat it as a
direction for a future experiment, not an explanation to cite:

- Measured **virtual span ≈ 211 MB** for ~134 MB of payload — i.e. the span is **not** the ~1 GB a
  uniform 64 KB stride across all buffers would imply, so the stride distribution is skewed rather
  than a uniform march. An earlier draft of this document asserted ~1 GB; that was wrong.
- On Darwin's **16 KB pages an 8 KB buffer occupies about half a page**, so two buffers can share a
  page — accesses are *not* necessarily one-page-per-access.
- Prefetcher and TLB/page-walk behaviour are the leading candidates for why monotonic order is
  cheaper, but **no fault counter, TLB counter, or prefetch measurement has confirmed them.**

What *is* established: **sorting the same buffers into ascending order recovers most of the cost**,
and the near-sequential allocator is far less order-sensitive. The mechanism behind that remains open.

## What the literature says

This is a known *category*, though the application-side sort remedy is closer to folklore than to
documented practice.

- **Allocator layout determines application spatial locality.** Feng & Berger (MSP'05) state it
  directly: applications exhibit temporal locality, but "their spatial locality is dictated by the
  memory allocator." Their `Vam` allocator improves cache- and page-level locality via page-sized
  chunks, aggressively returning free pages, **eliminating object headers**, and careful size-class
  selection.
- **Address-ordering is established allocator practice.** FreeBSD's PHKmalloc keeps free pages in a
  list **sorted by address** specifically for page-level locality; address-ordered free lists are
  standard for coalescing and locality.
- **Typical magnitudes are small — ours was not.** Vam reports **4–8% average** vs DLmalloc/PHKmalloc
  with adequate memory; its 2×/10× results require memory scarcity and paging. **Our effect is far
  outside that range with no paging at all**, which argues the ordering here is *pathological* rather
  than merely suboptimal. Treat an outsized effect as evidence of a **specific interaction worth
  isolating**, not as general "layout matters." (Do **not** describe it as a *uniform* slab stride —
  the measured stride distribution is skewed; see the hypothesis section.)
- **Multi-thread allocators penalizing single-threaded programs is current research.** "Old is Gold:
  Optimizing Single-threaded Applications with Exgen-Malloc" (2025) targets this configuration and
  supports **cache/TLB locality concerns** and **single-thread allocator overheads**. **It does NOT
  discuss hardware prefetching or slab-stride conflicts** — an earlier draft of this document
  attributed those to it in error. Cite it only for the locality and single-thread-cost claims.

References:
- <https://people.cs.umass.edu/~emery/pubs/p33-feng.pdf>
- <https://arxiv.org/pdf/2510.10219>
- <https://www.gingerbill.org/article/2021/11/30/memory-allocation-strategies-005/>

## Measured instance (Apple M4, 16,364 × 8 KB, no rawr/CRoaring code)

**Authoritative source: the retained probe `src/bench_smp_layout.zig`.** (An earlier temporary probe
produced slightly different figures; use the retained one.) Recovery from address-sorting:

| case | SMP recovery | libc recovery |
|---|---:|---:|
| words-only | **1.681 ms** | 0.063 ms |
| header-interleaved | **2.737 ms** | 0.180 ms |

**The order-sensitivity asymmetry is the finding** — SMP recovers 1.7–2.7 ms from reordering; libc
recovers 0.06–0.18 ms, i.e. its addresses were already close to sorted. Note libc is **not perfectly**
order-insensitive (0.180 ms on the interleaved case), so treat "insensitive" as relative.

**Timing discipline:** the retained probe distinguishes **sorting performed outside the timed region**
from **honest sort-plus-zero timing**. Any go/no-go decision must use the **sort-plus-zero** number —
a recovery figure measured with the sort excluded overstates the achievable win.

## Zig 0.16 allocator options (and why none is a drop-in fix)

| allocator | single-thread friendly? | usable here? |
|---|---|---|
| `SmpAllocator` | No — built for multi-threading; process-wide singleton, per-thread freelists, 64 KB slabs | current default; the pathology's source |
| `BrkAllocator` | **Yes, explicitly** (`@compileError` unless `builtin.single_threaded`) | **No — Linux/WASM only** (needs an sbrk-like primitive); unavailable on Darwin, and requires a single-threaded *build* |
| `ArenaAllocator` | n/a | tested and rejected (spec 17): bulk free cost exceeded individual frees on M4; lifetime failed the memory gate |
| `MemoryPool` | n/a | single-size pool — a natural *shape* for uniform 8 KB buffers, but **does NOT guarantee ascending addresses** (arena-backed with a **LIFO** preheated freelist; arenas may grow non-contiguously) and inherits arena lifetime/peak-memory characteristics |
| `FixedBufferAllocator` | n/a | no growth; already used for bounded scratch |
| `PageAllocator` | n/a | mmap per allocation — page granularity and syscall per call |
| `debug_allocator` | n/a | safety/debug tooling, not a performance option |

**There is no general-purpose, single-thread-friendly, address-ordered allocator in Zig 0.16 stdlib
that works on Darwin.** `BrkAllocator` is the only explicitly single-threaded one and it is
unavailable on the platform where the pathology was measured.

## Robust vs fragile remedies (library context)

A general-purpose allocator is typically a **process-wide singleton shared with every other data
structure in the program** — Zig's `SmpAllocator` says so explicitly ("it uses global state and only
one should be instantiated for the entire process"), with per-thread freelists keyed by thread id.
A library therefore **cannot own or rely on that allocator's state.** This splits candidate remedies
cleanly:

**Robust — depend only on things you own:**

- **Sort the pointers you already hold before a READ-ONLY traversal** (zeroing, scanning, computing).
  Works regardless of provenance, other clients, or threads, and **leaves no trace in the allocator.**
  This is the only unambiguously robust form of the sort remedy.
- **Allocate from something private** (arena / pool owned by the operation). Being private makes the
  order **independent of other clients**, which is the robust part. **It does NOT guarantee ascending
  addresses:** Zig's `MemoryPool` is arena-backed but its preheated freelist is **LIFO**, and an
  arena can grow through **non-contiguous backing allocations**. So a private allocator removes the
  *interference*, but the ordering it produces still has to be **measured, not assumed**.

**Fragile — depend on shared allocator state:**

- **Sorting FREES is not the same as sorting reads — it is state conditioning.** `SmpAllocator.free`
  pushes onto a **per-size-class LIFO freelist**, so **the order you free in determines the order later
  allocations come back.** A sorted teardown may improve teardown itself while **poisoning the next
  allocation/traversal cycle** — and a benchmark process that exits immediately after teardown **can
  never reveal that.** Any sorted-free result must be paired with a **refill-and-re-measure** of the
  following cycle.
- **Direction matters, and it inverts.** Because reuse is LIFO, **freeing ascending tends to hand back
  descending**, and **freeing descending tends to hand back ascending.** This is a plausible mechanism
  for the measured words-only case (8 KB stride, *descending* allocation order): an ascending free pass
  would produce exactly that. If the goal is a fast *next* cycle, the useful free order may be
  **descending**, not ascending — the opposite of what "sort before freeing" suggests.
- Conditioning the shared allocator so a *later* burst behaves better, in general. Other data
  structures and other threads perturb that state between operations, so the effect is not dependable
  in a library.

> **Benchmark warning.** Fragile remedies **measure well and deploy badly.** A single-purpose
> benchmark process is usually the *only* allocator client, so allocator-state conditioning looks
> excellent there and evaporates in a real program. This is the deployment-side twin of the
> measurement-side trap (allocator process history contaminating benchmark numbers): same cause,
> opposite direction. Before believing any state-conditioning result, ask whether the benchmark is
> the sole allocator client — and if it is, treat the result as unproven.

**Corollary — two independent tests, not one.** A phase is a candidate only if it passes **both**:

1. **Is the visit order free?** (correctness test) — independent per-object computation and frees pass;
   format-defined or sorted-key output order fails.
2. **Is it allocator-state-neutral?** (robustness test) — **read-only traversals pass; FREES DO NOT**,
   because they rewrite the freelist order.

So **independent per-object reads/computations are clean candidates**, while **teardown/mass-free is
order-free but NOT state-neutral** — it belongs in the fragile column and needs downstream measurement,
not the robust one. An earlier version of this document listed teardown as simply eligible; that was a
contradiction with the fragile-conditioning entry above.

## Open question

**Which hardware effect** makes the order expensive — hardware prefetching, TLB/page-walk locality,
cache conflict/replacement behaviour, or a combination. **All remain candidates.**

An earlier draft claimed reordering's success ruled out **cache set-aliasing**. That reasoning was
wrong: **cache conflict and replacement behaviour are themselves order-sensitive**, so a win from
reordering does not exclude them. Nothing is eliminated yet.

Testable prediction *if* TLB/page-walk dominates: reducing *span* (not merely ordering) should help,
larger pages should help, and **neither would move fault counts**. Distinguishing that from
conflict/replacement effects needs actual TLB and cache-miss counters, not inference.

## Checklist for next time

1. A/B the allocator on the canonical workload — does swapping it move the gap?
2. Time the allocator's own `alloc`/`free` separately — if it is *faster*, stop blaming call overhead.
3. Run the **address-sort control** on the *same* buffers; compute the sorted-vs-unsorted delta for
   **both** allocators and compare their order-*sensitivity*.
4. Measure the allocation stride distribution (median stride, total span, distinct pages).
5. Do **not** treat a refuted page-fault hypothesis as clearing the allocator — TLB ≠ faults.
6. Check whether interleaving small allocations among large ones worsens the returned order.
7. Remember the sort remedy only helps traversals you are free to reorder; order dictated by data
   (e.g. key-ordered merges) cannot be sorted away, which favours fixing order at the source.
8. **Apply the two-test rule before sorting anything: order-free AND allocator-state-neutral.** Sorting
   a read traversal leaves no trace; **sorting frees rewrites a LIFO freelist and changes the next
   cycle's allocation order** — measure the following cycle before believing a sorted-free win.
