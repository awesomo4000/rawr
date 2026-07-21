<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 20: Lazy-OR construction gap — why is CRoaring faster?

**Diagnosis first.** rawr's lazy-OR **construction** runs ~**2.19x** the reference on the
dense sparse-chunk benchmark, while repair is already at parity. The primary deliverable
is the *answer to why*, measured and attributed — not a fix. A fix is a conditional second
phase, only if the diagnosis says one is reachable without breaking rawr's constraints.

## What is known

- The gap is in **construction, not repair** (repair measured at parity), so it is not the
  popcount/demote path.
- The bench spans **nearly all 65,536 chunk keys**, and for each matched/shared chunk the
  lazy fold builds a **fresh 8 KB bitset** as its accumulator (`bitmap.zig:2073`, `:2115`,
  via `BitsetContainer.init`).
- `BitsetContainer.init` allocates the words with `alignedAlloc` and then **explicitly
  `@memset(words, 0)`** the full 8 KB. There is **no zeroed-allocation path** in rawr —
  every bitset construction pays that memset.

## Leading hypothesis (to confirm or refute, not assume)

CRoaring allocates bitset words with **`calloc`**, whose large-allocation fast path
returns **OS zero-filled pages for free** (fresh `mmap` memory is already zero, so glibc
skips the memset). rawr allocates from a recycled `smp_allocator` size-class slot — dirty
memory — so it **must** memset. Across ~65,536 bitsets that is ~**512 MB of `memset`** rawr
pays and the reference does not. If true, the gap is *zeroing cost*, not allocation count
or bit-setting.

This is a hypothesis. Phase 1 must **attribute the 2.19x** before anyone believes it.

## Phase 1 — Diagnosis (the deliverable)

Attribute lazy-OR construction time across its components, for rawr and the reference, on
the bench of record:

- **allocation** (the `create` + `alignedAlloc` calls),
- **zeroing** (the `@memset`),
- **bit-setting / accumulation** (`lazyAccumulateIntoBitset`).

Microbench or instrument to split these. Report the share of the 2.19x attributable to
each, and specifically **confirm or refute** that the memset (zeroing) is the dominant
term and that the reference avoids it via `calloc` zero-pages. Compare `smp_allocator` vs a
page-backed allocation for the words to isolate the zero-page effect.

**Phase 1 stands alone.** Even if no fix follows, the attributed "why" is the required
output.

## Phase 2 — Can rawr close it? (conditional on Phase 1)

Only if Phase 1 confirms zeroing is the dominant, movable cost. The crux is a real
constraint:

- **rawr is allocator-generic.** `std.mem.Allocator.alloc` returns memory with **undefined
  contents** — there is no "allocate zeroed" in the interface that can exploit OS
  zero-pages the way `calloc` does. So rawr cannot get calloc's free zeroing *through the
  caller's allocator* without either bypassing it or adding machinery.

Candidate directions, each with its tradeoff to measure — not a foregone win:

- **Page-backed zeroed words for bitsets.** Request fresh zero-filled pages (page allocator
  / mmap) for the 8 KB words and skip the memset. Tradeoff: bypasses the caller's allocator
  (breaks the arbitrary-allocator / arena / `Owned` contract) and is slow for the *reused*
  case (syscall per bitset); quantify against the memset it removes.
- **Zero-on-free reusable bitset buffers.** A pool that zeroes words on release so reuse
  skips re-zeroing. Tradeoff: this is pooling — spec 17 showed pooling's lifetime/teardown
  costs can dominate; measure, do not assume.
- **Reduce the number of forced bitset constructions.** Only viable where the accumulator
  is not semantically required to be a bitset; the n-way fold uses a bitset precisely for
  O(1) inserts, so array accumulation would trade zeroing cost for merge cost — measure the
  crossover.
- **Honest fourth outcome:** the gap may be an intrinsic property of libc `calloc`'s
  zero-page fast path that a generic-allocator library cannot match without giving up
  allocator-genericity. If so, **document it as a known, explained gap** — that is a valid
  result, not a failure.

## Constraints

- **Allocator-generic preserved** unless a phase-2 direction explicitly and measurably
  earns an exception (and then it is scoped, not global).
- No correctness change to lazy OR / repair; differential tests stay green.
- Leak-safe; single-threaded parity with the existing bench.

## Measurement / GO

- Authoritative environment: `ReleaseFast`, native CPU, the spec-16 M4 host; five
  independent process runs, median + range; env header recorded.
- **Phase 1 GO:** the 2.19x is attributed to its components with the zeroing hypothesis
  confirmed or refuted. (This is the answer Phase 20 exists to produce.)
- **Phase 2 GO (if attempted):** a candidate closes a material share of the construction
  gap **without** breaking allocator-genericity or correctness, beyond noise. Otherwise
  record the explained gap and stop.

## Estimate

S for Phase 1 (attribution microbench). Phase 2 is M and only attempted if Phase 1 points
to a reachable fix — several candidates are likely tradeoffs, and a documented intrinsic
gap is an acceptable outcome.
