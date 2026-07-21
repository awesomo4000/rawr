<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 18: `c_allocator` benchmark diagnostic column

**Diagnostic, not a product change.** Add a third measurement to `bench_croaring` — rawr
run with `std.heap.c_allocator` — alongside the existing rawr-with-`smp_allocator` and
CRoaring columns, across the whole suite. The deliverable is a per-op table that answers
one question cheaply before any allocator is built: **is the whole-result allocator a
broad lever, or was the spec-16 lazyOr result narrow/platform-specific?**

## Why

Spec 17 Phase A falsified the transient-bitset-arena hypothesis and showed the spec-16
libc win was a **whole-result** effect (unmatched-container clones, temp bitsets, index
arrays, repair arrays, destruction), not the transient bitsets. Before committing to
build a persistent segregated heap (the pure-Zig analogue of libc's warmed bins), we need
to know whether swapping the *whole-result* allocator to libc reproduces **broadly and by
how much per op**, or only on lazyOr. That is exactly what a `c_allocator` column measures,
and it costs almost nothing: `bench_croaring` already links libc and CRoaring, so this
adds **no dependency to the rawr library** — only a second allocator instance inside the
benchmark executable.

It also answers the standing "match croaring" question honestly, by reporting both a
representative pure-Zig number and an allocator-matched number.

## Scope

- For **every** rawr operation already in `bench_croaring`, run the rawr side **twice** —
  once with the existing `smp_allocator`, once with `std.heap.c_allocator` — holding the
  corpus, warmup, run count, and timing protocol identical. CRoaring stays as-is.
- Whole-suite, not just the allocation-heavy ops. The point is to see the full tradeoff
  surface: where `c_allocator` helps rawr (expected: clone-/new-result-heavy ops —
  `bitwiseOr`/`bitwiseAnd` sparse, `clone`, `lazyOr+repair`, deserialize) **and where it
  hurts** (earlier measurements show it is slower on some rawr workloads — that regression
  surface is itself a finding).
- Both allocators warmed outside the timed region, matching how `smp_allocator` and the
  CRoaring allocator are already treated.

## Reporting

- Per op, emit three times — `rawr (smp)`, `rawr (c_alloc)`, `CRoaring` — and the ratios
  that matter:
  - `rawr(smp) / CRoaring` (the representative pure-Zig standing, unchanged from today);
  - `rawr(c_alloc) / CRoaring` (allocator-matched);
  - `rawr(c_alloc) / rawr(smp)` (the allocator's own effect on rawr, the diagnostic).
- The env header records target / CPU / features and names which allocator each column
  used, consistent with spec 14's header.

## Constraints

- **Benchmark-only.** No change to rawr library defaults and no `c_allocator` plumbed into
  rawr code paths beyond passing it as the allocator to the bitmaps under test in the
  benchmark. The rawr library keeps no libc dependency.
- Do **not** switch any rawr default to `c_allocator`. It is a diagnostic column here;
  CLAUDE.md warns it hides leaks, and leak checking stays on the GPA-based correctness
  harnesses, not this bench.
- Single-threaded, same as the existing bench.

## Authoritative environment

Numbers of record are `ReleaseFast`, native CPU (`-Dcpu=native`), on the same Apple
M4 / macOS host used for specs 16–17, with target/CPU/features in the header. Other
machines are supporting measurements.

## Acceptance

- `bench_croaring` emits, per op, the three times and the three ratios above, with the env
  header naming the allocators; the deterministic corpus and existing timing protocol
  (warmups + median of timed runs) are unchanged.
- The output makes clear, per op, where `c_allocator` moves rawr relative to `smp` and by
  how much — isolating in particular the clone-/new-result-heavy ops from the in-place and
  read-only ops.
- No rawr library changes and no new library dependency; full build green under
  `ReleaseSafe` and `ReleaseFast`.

## Decision this feeds

The per-op table decides whether the persistent segregated-heap build is worth pursuing:
a **broad** `c_alloc`-vs-`smp` win across the clone/new-result ops argues for it; a win
confined to lazyOr argues against. It also sharpens the separate clone-demand-reduction
track (the 98k-clone finding) by showing how much of each op's cost is allocator-movable
versus demand that must be removed outright.

## Estimate

S. A second allocator pass over existing benchmark ops plus report/columns; no new
corpus, no library change.
