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

- **Only ops that allocate during timing** get a second rawr pass — once with the existing
  `smp_allocator`, once with the libc allocator via `bench_time.cAllocator()` (use the
  helper, not `std.heap.c_allocator` directly — it preserves the OpenBSD libc shim),
  holding corpus, warmup, run count, and timing protocol identical. CRoaring stays as-is.
- **Non-allocating ops get no second pass.** `contains`, `iterate`, `toArray`,
  `cardinality`, `rank`/`select`, `andCardinality`, and range-cardinality do not allocate
  during the timed region; running them twice yields duplicate numbers, not allocator
  evidence. Their `rawr (c_alloc)` time and the allocator-effect ratio are **`N/A`**; keep
  their existing rawr/CRoaring comparison unchanged.
- **Hold input fixtures constant.** For set operations, both rawr variants use the **same
  SMP-allocated immutable inputs** — only the *result* allocator changes. Do not build
  separate libc-backed inputs; that would mix input allocation/layout into a whole-result
  diagnostic. Build-from-empty ops (`add`, `deserialize`) naturally use the selected
  allocator throughout — that is correct for those rows.
- **Existing arena rows stay as-is.** `bitwiseAnd (sparse, arena)`, `bitwiseOr (sparse,
  arena)`, and `deserialize (arena)` already exercise a third allocator; keep them as
  dedicated rows with the allocator-effect column marked `N/A`. Do not reinterpret or
  duplicate them.
- Across the qualifying ops the point is the full tradeoff surface: where the libc
  allocator helps rawr (expected: new-result-heavy ops — `bitwiseOr`/`bitwiseAnd` sparse,
  `lazyOr+repair`, deserialize) **and where it hurts** (earlier measurements show it is
  slower on some rawr workloads — that regression surface is itself a finding).
- Both allocators warmed outside the timed region, matching how `smp_allocator` and the
  CRoaring allocator are already treated.

## Reporting

- Per **qualifying** op, emit three times — `rawr (smp)`, `rawr (c_alloc)`, `CRoaring` —
  and the ratios that matter:
  - `rawr(smp) / CRoaring` (the representative pure-Zig standing, unchanged from today);
  - `rawr(c_alloc) / CRoaring` (allocator-matched — see caveat);
  - `rawr(c_alloc) / rawr(smp)` (the allocator's own effect on rawr, the diagnostic).
- Non-allocating ops and the existing arena rows print `N/A` in the `c_alloc` and
  allocator-effect columns while keeping their rawr/CRoaring numbers.
- **"Allocator-matched" caveat.** CRoaring uses libc internally for bitmap creation, set
  results, and deserialization, but its `serialize` and `toArrayAlloc` wrappers currently
  allocate their **output buffers** with the Zig benchmark allocator
  (`bench_croaring.zig` ~line 508). For those rows `rawr(c_alloc)/CRoaring` is therefore
  *not* fully allocator-matched. Resolve one of two ways and state which in the output:
  (a) switch those CRoaring wrapper output buffers to the C allocator for those rows too,
  or (b) leave them and label those rows "allocator-matched only where CRoaring owns the
  allocation." Either is acceptable; do not leave it ambiguous.
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

Report the M4 result from **at least five independent process runs**, aggregated to
median + range — a single 21-sample process is vulnerable to allocator-state and
benchmark-order effects that are exactly what this diagnostic is probing. Extend
`scripts/run-compare-bench.sh` to launch the runs and aggregate if it does not already.

## Acceptance

- `bench_croaring` emits, per op, the three times and the three ratios above, with the env
  header naming the allocators; the deterministic corpus and existing timing protocol
  (warmups + median of timed runs) are unchanged.
- The output makes clear, per op, where `c_allocator` moves rawr relative to `smp` and by
  how much — isolating in particular the new-result-heavy ops from the in-place and
  read-only ops.
- No rawr library changes and no new library dependency; full build green under
  `ReleaseSafe` and `ReleaseFast`.

## Validation

- Correctness / build green:
  - `zig build test`
  - `zig build -Doptimize=ReleaseSafe`
  - `zig build -Doptimize=ReleaseFast`
- Diagnostic run of record (the `bench-compare` suite is the `bench_croaring` binary):
  - `zig build bench-compare -Dcpu=native` (ReleaseFast target), then
  - `scripts/run-compare-bench.sh` to launch ≥5 independent processes and aggregate
    median + range.
- Confirm the emitted env header names each column's allocator, non-allocating ops and
  arena rows show `N/A` in the `c_alloc`/effect columns, and the "allocator-matched"
  resolution (a or b) is stated in the output.

## Decision this feeds

The per-op table decides whether the persistent segregated-heap build is worth pursuing:
a **broad** `c_alloc`-vs-`smp` win across the new-result ops argues for it; a win
confined to lazyOr argues against. It also sharpens the separate clone-demand-reduction
track (the 98k-clone finding) by showing how much of each op's cost is allocator-movable
versus demand that must be removed outright.

## Estimate

S. A second allocator pass over existing benchmark ops plus report/columns; no new
corpus, no library change.
