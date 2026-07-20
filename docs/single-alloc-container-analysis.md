<!-- SPDX-License-Identifier: MPL-2.0 -->

# Single-allocation container prototype: analysis

Date: 07/20/2026

## Executive conclusion

The prototype successfully cut reserved-build and clone allocation calls from two
per array container to one. It nevertheless made the important workloads slower on
both Apple M4/aarch64/macOS and Zen 4/x86-64/Windows.

This is not just noise and it is not evidence that allocation count is irrelevant.
It is evidence that **allocation count alone is the wrong cost model for this
layout**. The current two-allocation representation happens to match Zig 0.16's
`std.heap.smp_allocator` unusually well:

- array payload capacities are powers of two, so payload requests land exactly in
  power-of-two allocator slots;
- the separate 24-byte headers use compact 32-byte slots and remain densely packed;
- combining the header with a power-of-two payload makes the request slightly larger
  than a power of two, pushing it into the next allocator class;
- growth then crosses allocator classes repeatedly, so most attempted in-place
  resizes fail before the code allocates, copies, and frees anyway;
- read-only header scans lose the dense header working set, while payload scans jump
  across larger, partially unused allocator slots.

The proposed design therefore halves calls while increasing allocator-class
footprint, slab pressure, failed-resize work, and address-space dispersion. On the
tested allocator and corpus, those costs dominate.

Recommendation: keep the current production representation and park the proposed
single-block ABI. Keep the prototype as a reproducible experiment. Reopen the design
only around a layout that addresses allocator classes directly, not around the same
header-plus-power-of-two-payload block.

## Experiment

The committed `bench_single_alloc` executable compares four isolated array-container
layouts over 10,000 deterministic containers:

1. current two-allocation layout with 32-byte payload alignment;
2. two-allocation control with 16-byte payload alignment;
3. one allocation with a stored payload slice and 16-byte block alignment;
4. one allocation with a derived payload accessor and 16-byte block alignment.

The corpus is 50% cardinality 1-64, 35% cardinality 256-1024, and 15%
cardinality 3840-4096. Inputs and probe sequences are identical across variants.
Each process reports the median of nine timed trials after one warmup. The decision
uses five fresh processes and a workload-specific noise floor.

The allocator is a counting wrapper over `std.heap.smp_allocator`. It records API
calls and requested/live bytes. Requested bytes are not the allocator's internally
rounded slot bytes; that distinction is central to the result.

## Results

Percentages below compare the stored-slice single-block layout with the current
32-byte-aligned baseline. Positive means slower.

| Workload | Apple M4 | M4 noise | Zen 4 | Zen 4 noise |
|---|---:|---:|---:|---:|
| build, reserved | +2.89% | 3.30% | +7.81% | 0.77% |
| build, growth | +14.09% | 4.59% | +34.83% | 3.39% |
| clone | +56.69% | 31.74% | +21.93% | 2.64% |
| deinit | -41.46% | 46.88% | +2.21% | 6.12% |
| membership | +32.54% | 9.86% | +5.94% | 1.46% |
| iteration | +6.67% | 6.90% | +22.44% | 3.95% |
| cardinality | +233.33% | 25.00% | +434.48% | 11.59% |

The M4 deinit result is inside its large noise floor and does not establish a win.
The Zen 4 run is substantially quieter and independently rejects the design on
reserved build, growth, clone, membership, iteration, and cardinality.

The deterministic allocation counts behaved as designed:

| Variant | Reserved-build allocs | Clone allocs | Growth alloc/free | Resize success/failure |
|---|---:|---:|---:|---:|
| baseline/control | 20,000 | 20,000 | 77,261 / 57,261 | 0 / 0 |
| single, stored | 10,000 | 10,000 | 48,129 / 38,129 | 19,132 / 38,129 |
| single, derived | 10,000 | 10,000 | 57,543 / 47,543 | 9,718 / 47,543 |

The allocation-count premise was correct. Its assumed connection to elapsed time
was not.

## Primary mechanism: allocator size classes

Zig 0.16's `SmpAllocator` computes a size class by rounding the request size and
alignment up to a power of two. Requests below its 64-KB slab size use those slots.
An in-place resize succeeds only when old and new requests remain in the same class.

Rawr's array payload capacity is itself a power of two and each element is two
bytes. The current payload requests are therefore 8, 16, 32, ..., 8192 bytes. Except
for extra alignment on the smallest cases, these fit allocator classes exactly.
The separate 24-byte header occupies a 32-byte slot.

The single-block variants prepend either a 16-byte derived header area or a 32-byte
stored-slice header area. Representative allocator costs are:

| Capacity | Payload | Current requested slots | Derived request -> slot | Stored request -> slot |
|---:|---:|---:|---:|---:|
| 4 | 8 | 32 header + 32 payload | 24 -> 32 | 40 -> 64 |
| 8 | 16 | 32 header + 32 payload | 32 -> 32 | 48 -> 64 |
| 16 | 32 | 32 header + 32 payload | 48 -> 64 | 64 -> 64 |
| 32 | 64 | 32 header + 64 payload | 80 -> 128 | 96 -> 128 |
| 256 | 512 | 32 header + 512 payload | 528 -> 1024 | 544 -> 1024 |
| 1024 | 2048 | 32 header + 2048 payload | 2064 -> 4096 | 2080 -> 4096 |
| 4096 | 8192 | 32 header + 8192 payload | 8208 -> 16384 | 8224 -> 16384 |

For a capacity-4096 array, the current representation consumes one 32-byte header
slot plus one 8192-byte payload slot. Either single block consumes a 16384-byte slot.
The requested-byte counter barely changes, but the allocator-class footprint is
almost doubled. The corpus deliberately includes 1,500 near-threshold containers,
so this is not an edge case.

Larger classes also fit fewer containers in each 64-KB allocator slab. For the
near-threshold example, baseline payloads fit eight per slab; the combined block fits
four. That increases slab acquisition and mapping pressure even though the number of
logical allocations is lower.

This mechanism applies on both tested operating systems because both runs use Zig's
same `smp_allocator` design. The cross-architecture agreement is therefore expected
once allocator classes are considered.

## Why each workload moved

### Reserved build and clone

These are the cleanest two-to-one allocation tests. They still lose because one
larger request is not equivalent to one cheaper request. Combined blocks use larger
classes, consume slabs faster, and leave much more unused space per slot. Clone shows
the same effect without growth-path complexity.

### Growth

The current implementation directly allocates the next exact payload class, copies,
and frees. The prototype first asks `resize` to grow the combined block. Since growth
usually changes its power-of-two class, `smp_allocator` rejects the in-place resize.
The prototype then performs the same allocate/copy/free sequence after paying for the
failed resize attempt.

The stored layout had 38,129 failed resizes; the derived layout had 47,543. This
explains why growth is the largest stable regression and why derived growth is worse
than stored growth. The counting allocator's cumulative requested-byte figure also
includes these failed requests, reaching roughly 76-77 MB for single-block growth
versus 37.6 MB for baseline growth.

### Cardinality and other header-heavy reads

Separate fixed-size headers are not necessarily a pointer-locality defect. The
allocator packs their 32-byte slots densely. A cardinality scan therefore walks a
small, compact header working set.

With a single block, each header begins a much larger slot. Reading one cardinality
field per container now touches cache lines spread across the payload allocation
space. The 3-5x cardinality result is a strong signature of this lost header density.
It is not important by itself because the absolute time is tiny, but it confirms the
locality mechanism.

### Membership and iteration

The expected same-block locality win did not materialize. Membership still performs
a binary search into payload data, while the single-block slots occupy a larger
address footprint. Iteration touches all logical payload bytes but jumps over the
unused remainder of each rounded slot between containers, increasing cache/TLB and
prefetch pressure. Zen 4's stable +5.94% membership and +22.44% iteration results show
that the effect extends beyond allocator-call timing.

## Stored slices versus derived accessors

Derived accessors did not produce the predicted pointer-chase win:

| Read workload | Derived advantage over stored | Required advantage |
|---|---:|---:|
| M4 membership | -3.12% | 19.73% |
| M4 iteration | -1.17% | 13.80% |
| Zen 4 membership | -0.11% | 2.92% |
| Zen 4 iteration | +1.45% | 7.90% |

A stored slice pointer already resides in the loaded header. Replacing that load with
pointer arithmetic is not automatically cheaper, and both layouts fall into the same
allocator classes for nearly all medium and large capacities. Derived accessors save
16 requested header bytes relative to stored slices, but usually save zero rounded
slot bytes.

Derived growth is actively worse because its sequence of block sizes yields fewer
same-class resize successes. It also requires migrating 147 real `.values` and 132
real `.runs` references. There is no measured justification for those 279 changes.

## What the experiment does and does not prove

It proves:

- the proposed single-block layout halves the intended allocation-call count;
- that layout loses its required performance gate with `smp_allocator` on two CPU
  architectures and two operating systems;
- the losses are reproducible and exceed measured noise on Zen 4;
- derived accessors do not recover the loss;
- the current separate-header representation has valuable density and allocator-class
  behavior that the original hypothesis omitted.

It does not prove:

- every possible single-block layout is slower;
- every allocator behaves like `smp_allocator`;
- a bitmap-owned pool, arena, or class-aware capacity scheme cannot win;
- reducing allocation calls is generally unhelpful.

The decision is specifically to reject this production ABI redesign on the evidence
available, not to reject all future allocation work.

## Focused follow-up experiments

If the architecture is revisited, these experiments would isolate the mechanisms in
descending order of value:

1. **Report effective size-class bytes.** Add a capacity histogram and compute
   `smp_allocator` slot bytes for each variant. This should quantify the hidden
   footprint directly instead of inferring it from the allocator source.
2. **Allocator sensitivity matrix.** Run the same prototype over the C allocator,
   an arena/FBA, and `smp_allocator`. A design that wins only with arenas may still be
   useful for arena-owned workloads, but is not a general replacement for a library
   accepting arbitrary allocators.
3. **Remove speculative resize.** Benchmark single-block growth with direct
   alloc/copy/free to measure the cost of the 38K-48K predictable resize failures.
4. **Class-aware capacities.** Choose capacities so `header + payload` fills, rather
   than barely exceeds, an allocator class. For a 16-byte header, capacities such as
   24, 56, 120, 248, 504, 1016, 2040, and 4088 fit 64, 128, 256, ..., 8192-byte
   blocks. This changes growth thresholds and needs separate correctness/performance
   review; capacity 4096 still requires a 16-KB block unless array-to-bitset conversion
   happens slightly earlier.
5. **Hardware-counter validation.** On Linux/WSL, compare cache misses, TLB misses,
   page faults, and mapped bytes for cardinality/iteration. This would validate the
   locality explanation independently of elapsed time.
6. **Different ownership architecture.** If allocation count remains important,
   consider bitmap-owned header slabs or parent-side metadata while preserving exact
   power-of-two payload allocations. That is a different design, not a refinement of
   the moving single-block ABI.

## Architecture recommendation

Do not proceed with specs 13-01 through 13-06 using the measured layout. Retain the
prototype and counting allocator because they provide a useful regression harness and
a concrete test bed for allocator-aware alternatives.

If this work is reopened, require all of the following before production migration:

- a design that does not systematically cross `smp_allocator` power-of-two classes;
- wins on reserved build and clone, not allocation counts alone;
- no beyond-noise regression on growth, membership, or iteration;
- results on both aarch64 and x86-64;
- allocator sensitivity results showing where the design does and does not apply.
