<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 18: `c_allocator` benchmark diagnostic column

**Diagnostic, not a product change.** Add a third measurement to `bench_croaring` — rawr
run with `std.heap.c_allocator` — alongside the existing rawr-with-`smp_allocator` and
CRoaring columns, across the whole suite. The deliverable is a per-op table that answers
one question cheaply before any allocator is built: **is the whole-result allocator a
broad lever, or was the spec-16 lazyOr result narrow/platform-specific?**

> **Outcome (2026-07-21) — decisive NO to the allocator track.** Implemented; the M4
> five-run aggregate shows the whole-result libc swap does **not** reproduce broadly once
> inputs are held constant:
> - **lazyOr+repair: only 0.980x** (2% faster) — the spec-16 ~1.07x was narrow /
>   confounded by *fixture* allocation, which this diagnostic holds constant.
> - **Container-heavy ops regress hard on libc:** sparse AND `1.427x`, sparse OR `1.431x`,
>   deserialize `1.782x` (all slower).
> - **libc gains are confined to flat single-buffer allocation:** `toArrayAlloc` 0.77x,
>   `serialize` 0.86x, and a couple of build paths (random add ~0.96x, sequential add
>   0.87x; sequential `addMany` neutral).
>
> Interpretation: rawr's container model is already SMP-optimal (many small power-of-two
> allocs land in classes exactly); libc only wins where one big flat buffer is allocated.
> **Do not build the persistent segregated heap** — it would regress the container-heavy
> set ops that matter. The whole allocator-replacement track (the spec-16 libc lead, the
> segregated-heap idea) is **closed**. The remaining perf lever is reducing allocation
> *demand* — the 98k-clone finding → next spec. Benchmark-only; no library/API change; the
> `c_alloc` column is retained for future allocator questions.

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
  (`bench_croaring.zig` ~line 508). **CRoaring stays unchanged** — changing those buffers
  would move the `rawr(smp)/CRoaring` baseline this suite must keep comparable. So for
  those two rows label `rawr(c_alloc)/CRoaring` as **"allocator-matched only where CRoaring
  owns the allocation"** in the output. The primary diagnostic `rawr(c_alloc)/rawr(smp)`
  is unaffected and remains valid for every qualifying row.
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
`scripts/run-compare-bench.sh` to build once with `-Dcpu=native`, run the binary five
times, aggregate **each timing column independently**, and **recompute the ratios from the
median times** (not by averaging per-run ratios).

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
- Diagnostic run of record (the `bench-compare` step builds the `bench_croaring` binary):
  - `scripts/run-compare-bench.sh` — builds once with `-Dcpu=native` (ReleaseFast), runs
    the binary five times, aggregates each timing column, and recomputes ratios from the
    medians.
  - Manual smoke check: `zig build bench-compare -Dcpu=native` then run the binary once.
- Confirm the emitted env header names each column's allocator, non-allocating ops and
  arena rows show `N/A` in the `c_alloc`/effect columns, and the mandated option-(b)
  allocator-matching caveat is stated in the output.

## Decision this feeds

The per-op table decides whether the persistent segregated-heap build is worth pursuing:
a **broad** `c_alloc`-vs-`smp` win across the new-result ops argues for it; a win
confined to lazyOr argues against. It also sharpens the separate clone-demand-reduction
track (the 98k-clone finding) by showing how much of each op's cost is allocator-movable
versus demand that must be removed outright.

## Estimate

S. A second allocator pass over existing benchmark ops plus report/columns; no new
corpus, no library change.
