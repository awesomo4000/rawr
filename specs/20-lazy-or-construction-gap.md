<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 20: Lazy-OR construction gap — why is CRoaring faster?

**Diagnosis only, no preselected cause.** rawr's lazy-OR **construction** runs ~**2.19x**
the reference on the dense sparse-chunk benchmark, while repair is at parity. The
deliverable is the *attributed answer to why*. An earlier draft of this spec guessed the
cause (calloc zero-pages); that guess is **refuted** (below), which is exactly why this
spec commits to measurement before any theory. A fix is a separate chunk written only
after the diagnosis names the mechanism.

## What is known (corrected)

- The gap is in **construction, not repair** (repair at parity) — not the popcount/demote
  path.
- Each **shared** chunk key builds a fresh 8 KB bitset accumulator (`bitmap.zig:2073`,
  `:2115`, via `BitsetContainer.init`), which does `alignedAlloc` + an explicit
  `@memset(words, 0)`.
- **The reference does the same kind of work — it is not getting free zeroing.** CRoaring
  allocates the container with `roaring_malloc` and the words with `roaring_aligned_malloc`,
  then **explicitly clears all 8 KB** in `bitset_container_clear` (`vendor/roaring.c:7260`,
  `:7272`). On the authoritative Apple-M4 host that is Apple libc / `posix_memalign`, **not
  glibc `calloc`**. Both implementations pay an explicit memset — zeroing is not a
  free-vs-paid differentiator.
- **Only shared keys** create fresh lazy bitsets; cloned-only keys do not. In this corpus
  `a` covers ~the lower half of the key space and `b` ~the upper three quarters, so the
  overlap is roughly **~16K shared keys, not ~65K** — expected zeroing on the order of
  ~128 MB, not ~512 MB. Phase 1 must **count** these exactly, not estimate.

## What the existing evidence already says

The preserved spec-18 harness measured lazy-OR construction:

| variant | median |
|---|---:|
| rawr, `smp_allocator` | 8.375 ms |
| rawr, libc | 7.574 ms |
| CRoaring, libc | 3.832 ms |

So swapping rawr to libc buys only ~**10%**, and rawr-libc is still ~**1.98x** CRoaring
with the **allocator matched**. The allocator contributes something but **cannot explain
most of the gap.** The dominant cause is therefore elsewhere — candidate territory
includes the accumulation loops, per-bitset overhead, top-level merge/append, or Zig code
generation for `@memset` / the accumulation loops. Phase 1 exists to find which; it does
**not** start from a favored answer.

## Phase 1 (chunk `20-00`) — controlled attribution, the deliverable

Attribute lazy-OR construction across fine-grained components, comparing **rawr-SMP,
rawr-libc, and CRoaring-libc** on the bench of record. Record counters **and** timings for:

- **shared vs cloned-only chunk keys** (exact counts, not estimates);
- **bitsets created and bytes cleared**;
- **container/header allocation** (the `create`);
- **words allocation** (the `alignedAlloc`);
- **zeroing** (the `@memset`);
- **first-source accumulation** and **second-source accumulation** (the two
  `lazyAccumulateIntoBitset` passes) separately;
- **top-level merge / append overhead**.

Plus **inspect the generated code** for the Zig `@memset` and the two accumulation loops —
the remaining gap may be code generation or loop overhead, not memory behavior.

Output: the 2.19x broken down by component with the dominant term identified. Phase 1
stands alone — the attributed "why" is the required result even if no fix follows.

## Phase 2 (chunk `20-01`, written after diagnosis) — do not preselect

`20-01` is authored **around whichever mechanism `20-00` identifies**, not chosen now. No
candidate (pooling, page-backed words, array accumulation, codegen change) is favored in
advance. Whatever is proposed inherits these constraints, established now so they are not
re-litigated:

- **Allocator-genericity preserved.** `std.mem.Allocator.alloc` returns **undefined**
  contents by contract — including `page_allocator`. A "skip the zeroing" experiment may
  **not** assume any `std.mem.Allocator`-provided memory is zero; it must use an explicitly
  known-zero benchmark allocator or a direct OS mapping, and any production form must not
  break the arbitrary-allocator / arena / `Owned` contract.
- **Lifecycle-honest timing.** Construction-only timing can be gamed — e.g. a zero-on-free
  pool merely moves the memset to teardown and *looks* faster while lifecycle cost is
  unchanged. Any candidate must report **both** construction-only timing (for parity with
  the reported 2.19x) **and** full construct + repair + destroy lifecycle timing.
- **No correctness change**; differential tests green; leak-safe.
- **Numeric GO gates** are set in `20-01` against the Phase-1 baseline (e.g. "closes ≥ X%
  of the attributed dominant term without lifecycle regression"), once the dominant term is
  known — not the vague "material share" this draft previously used.

## Measurement environment

`ReleaseFast`, native CPU, the spec-16 M4 host; five independent process runs, median +
range; env header recorded. Same rig for both phases.

## Chunking

- **`20-00`** — diagnosis only (attribution + codegen inspection). Produces the "why".
- **`20-01`** — the fix, authored around `20-00`'s identified mechanism, with its numeric
  gates. Only if `20-00` points to a reachable improvement; an explained-but-intrinsic gap
  (documented, no fix) is a valid outcome.

## Estimate

S for `20-00`. `20-01` sized once the mechanism is known.
