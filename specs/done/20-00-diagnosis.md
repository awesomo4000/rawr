<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 20-00: Lazy-OR construction gap — attribution (diagnosis only)

First chunk of [lazy-OR construction gap](20-lazy-or-construction-gap.md). **Diagnosis
only — no fix, no library change.** Deliverable: the ~2.19x lazy-OR construction gap
attributed to its components, with the dominant term identified and a named residual. This
is the required output even if no fix follows.

## Starting facts (do not re-derive, but verify counts)

- Gap is in **construction, not repair** (repair at parity).
- Each **shared** chunk key builds a fresh 8 KB bitset (`bitmap.zig:2073`, `:2115`), which
  `alignedAlloc`s words + explicit `@memset(0)`. CRoaring does the analogous work and
  **also explicitly memsets** (`vendor/roaring.c:7260`, `:7272`; Apple libc
  `posix_memalign`, not `calloc`) — zeroing is not free on either side.
- Only shared keys build lazy bitsets (`a` ~lower half, `b` ~upper ¾ → ~16K shared, **count
  it exactly**).
- Preserved rawr-libc harness: branch `bench-experiments-17-18`, commit `0599cae` (not on
  `main`). Baseline medians there: rawr-SMP 8.375 ms, rawr-libc 7.574 ms, CRoaring-libc
  3.832 ms — allocator explains only ~10%, so the dominant cause is elsewhere.

## Components to attribute

Across **rawr-SMP, rawr-libc, and CRoaring-libc**:

- shared vs cloned-only chunk keys (exact counts);
- bitsets created, bytes cleared;
- container/header allocation (`create`);
- words allocation (`alignedAlloc`);
- zeroing (`@memset`);
- first-source and second-source accumulation (the two `lazyAccumulateIntoBitset` passes),
  separately;
- top-level merge/append overhead.

## Methodology (mandatory — no per-container clocking)

1. **Untimed counter pass** captures the exact counts (no timing calls in the hot path).
2. **Batched component microbenchmarks** over the **captured shared-container corpus** —
   time each component in isolation across the whole set, never per container.
3. **Whole-operation profiling / controlled variants** (e.g. zeroing elided or accumulation
   elided under an explicitly known-zero benchmark allocator) confirm the microbench
   attribution against real end-to-end time.
4. Do **not** infer component cost from thousands of per-container clock calls.

**CRoaring instrumentation** — explicit and non-invasive; state which is used and leave no
permanent diagnostic edits in `vendor/roaring.c`: a benchmark-only instrumented translation
unit (throwaway copy), sampling/profile of the unmodified reference, and/or controlled C
microbench wrappers from the same upstream functions. If the throwaway translation unit is
chosen, **commit a generator or patch that reproduces it** (not just the transient file),
and any copied CRoaring code **keeps its upstream license and must not receive an MPL
header**.

**Codegen inspection** of the Zig `@memset` and the two accumulation loops records the
**exact build command, symbol/probe, and relevant assembly finding** — reproducible, not
anecdotal.

## Acceptance

- Exact shared/cloned-only key counts, bitsets created, and bytes cleared reported (the
  ~16K/~128 MB figures confirmed or corrected by measurement).
- Construction attributed across the components above for all three variants, as **absolute
  medians + ranges** with **supported attribution bounds** and a **named residual** — not
  percentages forced to total 100%.
- The **dominant term identified** with its bound, and a stated read on whether it is
  allocation, zeroing, accumulation, merge/append, or codegen/loop overhead.
- Any instrumented/replica path shown to **match production rawr after repair and the
  CRoaring oracle** before its timing is accepted.
- Environment: `ReleaseFast`, native CPU, spec-16 M4 host; five independent process runs,
  median + range; env header recorded.
- **Durable artifact:** findings, exact commands, assembly observations, and the final
  attribution are committed to **`docs/lazy-or-construction-analysis.md`** — benchmark
  output under ignored `misc/` alone is not sufficient.
- **Benchmark-only:** no public library behavior change, no committed vendored-source
  change; full build green under `ReleaseSafe` and `ReleaseFast`.

## Result to record (feeds `20-01`)

The dominant construction-cost term and its bound — this decides whether `20-01` (a fix) is
written and around which mechanism. An explained-but-intrinsic gap (documented, no fix) is a
valid terminal outcome, recorded in the analysis doc.

## Estimate

M. Replicas, three allocator/reference variants, non-invasive CRoaring instrumentation,
codegen inspection, five-process runs, validation, and the written attribution — not a
quick microbench.
