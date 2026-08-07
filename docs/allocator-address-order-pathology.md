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
16,364 8-byte pointers (`std::sort`, shuffled 64 KB-stride addresses, Apple M4) — around **8–9%** of
the ~1.66 ms this recovered in our case.

Two refinements worth knowing:

- **Order-sensitivity asymmetry is the real signal.** Measure the sorted-vs-unsorted delta for *both*
  allocators. An allocator whose addresses are already near-sequential will be **order-insensitive**
  (ours moved 0.011–0.073 ms); the pathological one will be **order-sensitive** (1.663–2.759 ms).
- **A full sort is usually unnecessary** for diagnosis or remedy — bucketing by slab base is O(N) and
  gets near-sorted order. (At 0.13 ms there is nothing to optimize, so prefer the simple sort while
  diagnosing.)

## Critical reconciliation: TLB misses are NOT page faults

This pathology is easy to misdiagnose as a **first-touch / page-fault** problem, and a fault-counter
experiment will then *refute* the fault hypothesis while the layout effect remains completely real.
Both can be true at once:

- **Pages resident** ⇒ few or no page faults.
- **Translations not cached** ⇒ TLB misses and page-table walks per access.

In our case a dedicated experiment (spec 36) measured **40 operation faults** across a construction
touching ~134 MB, with **100% page reuse** proven and no gain from pre-conditioning residency — a
clean refutation of first touch. The address-order effect was nonetheless ~1.66 ms. **A refuted
page-fault hypothesis does not clear the allocator.** Check order separately.

## Why allocators do this

Multi-thread-oriented allocators deliberately disperse allocations to avoid contention: per-thread or
per-CPU freelists, size-class segregation, and slab-based backing. Each is good for scalability and
bad for a single-threaded caller's spatial locality.

Concretely, Zig 0.16's `std.heap.SmpAllocator` uses **64 KB slabs with per-size-class freelists** and
per-thread metadata keyed by thread id. For a sequential single-threaded caller requesting many 8 KB
buffers, the returned stream had a **median 64 KB stride**, versus **8 KB** (i.e. essentially
sequential) from libc `malloc`.

**Why stride matters more than span:** a 64 KB stride over 16,364 buffers walks ~**1 GB** of address
space while only ~134 MB holds data. Every access lands on a different page (16 KB pages on Darwin),
far exceeding TLB reach, and defeats stride prefetchers that operate within limited page-local
windows. Sorting does not shrink the span — it makes the walk **monotonic**, which is what
prefetchers and page-walk caches exploit. That also explains why the near-sequential allocator was
order-insensitive: its addresses were already sorted.

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
  with adequate memory; its 2×/10× results require memory scarcity and paging. **A ~60% effect with
  no paging is far outside that range**, which argues a uniform slab stride is *pathological* rather
  than merely suboptimal. Treat an outsized effect as evidence of a specific stride interaction, not
  of general "layout matters."
- **Multi-thread allocators penalizing single-threaded programs is current research.** "Old is Gold:
  Optimizing Single-threaded Applications with Exgen-Malloc" (2025) targets this configuration and
  names per-thread region dispersion, TLB pressure, prefetcher-hostile access, and **size-class/slab
  striding conflicting with hardware prefetchers**. *(Framing only — the results section was not
  retrieved; read directly before citing figures.)*

References:
- <https://people.cs.umass.edu/~emery/pubs/p33-feng.pdf>
- <https://arxiv.org/pdf/2510.10219>
- <https://www.gingerbill.org/article/2021/11/30/memory-allocation-strategies-005/>

## Measured instance (Apple M4, 16,364 × 8 KB, no rawr/CRoaring code)

| operation | SMP | libc |
|---|---:|---|
| allocate blocks | 0.207 ms | 0.232 ms |
| allocate headers + blocks | **0.132 ms** | 0.305 ms |
| zero in allocation order | **4.482 ms** | 2.753 ms |
| zero header-interleaved, allocation order | **5.686 ms** | 2.721 ms |
| zero same blocks, **sorted by address** | **2.819 ms** | 2.680 ms |
| sort header-interleaved, then zero | **2.927 ms** | 2.710 ms |

Reproducer: `src/bench_smp_layout.zig`.

## Zig 0.16 allocator options (and why none is a drop-in fix)

| allocator | single-thread friendly? | usable here? |
|---|---|---|
| `SmpAllocator` | No — built for multi-threading; process-wide singleton, per-thread freelists, 64 KB slabs | current default; the pathology's source |
| `BrkAllocator` | **Yes, explicitly** (`@compileError` unless `builtin.single_threaded`) | **No — Linux/WASM only** (needs an sbrk-like primitive); unavailable on Darwin, and requires a single-threaded *build* |
| `ArenaAllocator` | n/a | tested and rejected (spec 17): bulk free cost exceeded individual frees on M4; lifetime failed the memory gate |
| `MemoryPool` | n/a | single-size pool — a natural shape for uniform 8 KB buffers and arena-backed (ascending addresses), but inherits arena lifetime/peak-memory characteristics |
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

- **Sort the pointers you already hold** before an order-free traversal. Works regardless of
  provenance, other clients, or threads.
- **Allocate from something private** (arena / pool owned by the operation). Ascending order is
  guaranteed **by construction**, immune to other clients. Privacy of allocation matters as much as
  ordering.

**Fragile — depend on shared allocator state:**

- Conditioning the shared allocator so a *later* burst behaves better — e.g. freeing in address order
  hoping to leave address-ordered freelists for the next allocation burst. Other data structures and
  other threads perturb that state between operations, so the effect is not dependable in a library.

> **Benchmark warning.** Fragile remedies **measure well and deploy badly.** A single-purpose
> benchmark process is usually the *only* allocator client, so allocator-state conditioning looks
> excellent there and evaporates in a real program. This is the deployment-side twin of the
> measurement-side trap (allocator process history contaminating benchmark numbers): same cause,
> opposite direction. Before believing any state-conditioning result, ask whether the benchmark is
> the sole allocator client — and if it is, treat the result as unproven.

**Corollary for choosing among order-free traversals:** sorting only helps where visit order is free.
Frees/teardown and independent per-object computations qualify; anything whose output order is
defined by a format or by sorted-key semantics does not.

## Open question

**Which hardware effect** makes the order expensive — hardware prefetching, TLB/page-walk locality,
cache behaviour, or a combination. Notably, **cache set-aliasing is unlikely**: aliasing is
order-independent, and reordering the same addresses fixed the problem. Testable prediction if
TLB/page-walk dominates: reducing *span* (not merely ordering) should help, larger pages should help,
and **neither would move fault counts**.

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
