<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 32: Compact separate container headers (E1)

> **Outcome (2026-08-06) — RUN GO (major win, shipped); ARRAY NO-GO.** The compact **`RunContainer`**
> header (24 B → 16 B, 32-byte → 16-byte SMP class, payload untouched) was the campaign's biggest
> structural win — run-heavy ops allocate less and traverse denser metadata (locality + allocator
> overhead), closing a whole cluster at once. **M4 rawr/CRoaring:** clone **1.788x → 0.672x**, dense
> AND **1.587x → 0.845x**, dense OR **1.138x → 0.702x**, select **1.387x → 0.864x**, removeRange
> **0.803x → 0.290x** (rawr now faster on all). Zen 4 healthy (select 1.138x → 0.934x). The
> three-way control did its job: the win is the layout, not the refactor. **Array header NO-GO** —
> it made lazy-OR *worse* (rejected). Confirms the "three rows, one root cause" read: the shared
> root was the container header's SMP size class, and structural compaction closed clone / dense-AND
> / dense-OR / select / removeRange together. Records: `docs/parity-measurement.md`. Shipped in
> `d7d357b`.

Campaign: [31-structural-parity-campaign.md](31-structural-parity-campaign.md) (Wave 1). Shrink
`ArrayContainer` and `RunContainer` headers from **24 B → 16 B** so they drop from the **32-byte SMP
size class to the 16-byte class**, **leaving the payload in its own separate allocation with an
unchanged requested length, alignment, and SMP class**. Allocation **count is unchanged**; only the
header slot halves. This is the widest structural lever on the board.

**Targets (M4 SMP):** clone 1.786x, dense-AND 1.570x, select 1.486x (pointer locality), lazy-OR
construction 1.708x (the array-clone share of the 2-way merge).

**Parity is a hard requirement** — a row closes at ≤ 1.10x; a partial is adopted only by owner
judgement (spec-30 policy) and the row stays open.

## Verified structural facts

Current headers (grounded in `src/array_container.zig`, `src/run_container.zig`,
`src/bitset_container.zig`, `src/container.zig`):

| container | current header | SMP slot | compact (`[*]` many-ptr) | slot |
|---|---|---:|---|---:|
| `ArrayContainer` | `values: []align(32) u16` (16) + `cardinality: u16` (2) + `capacity: u16` (2) = 24 | **32** | `[*]align(32) u16` (8) + 2 + 2 = 12→16 | **16** |
| `RunContainer` | `runs: []RunPair` (16) + `n_runs: u16` (2) + `capacity: u16` (2) + `cardinality: i32` (4) = 24 | **32** | `[*]RunPair` (8) + 2 + 2 + 4 = 16 | **16** |
| `BitsetContainer` | `words: *align(64) [1024]u64` (8) + `cardinality: i32` (4) = 12→16 | **16** | already compact | — |

- **Bitset is out of scope** — already 16 B.
- CRoaring's equivalent headers are similarly compact (raw pointer + counts), so this closes a
  structural asymmetry, not a novel layout.

## Not spec 13

Spec 13 co-located header **and** payload in one allocation, so the combined block crossed into the
next SMP size class (8 KB payload + header → 16 KB class). **E1 leaves the payload in its own
separate allocation** — only the small header slot changes (32→16). This is the firewall that makes
E1 legitimate.

**Payload is NOT assumed power-of-two.** Run payload capacities are not consistently power-of-two, so
the firewall is stated as an **assertion per case**, not an assumption: for **every** baseline and
candidate cell, the payload's **requested length, alignment, and resulting SMP-class bytes are
unchanged** vs baseline (host class accounting). Any cell where the candidate shifts the payload
class **fails** — that would be spec 13 in disguise.

## Array and Run are decided independently

They target different rows (Array header → lazy-OR-construction array clones; Run header → clone /
dense-AND / select) and may land differently. **Independent prototype cells and independent GO/NO-GO
decisions.** Do **not** couple the migrations, and do **not** infer Run performance from the Array
prototype — build **real compact-header Run replicas** and measure them on dense clone, dense-AND,
and select.

## Pointer contract (pinned)

- **`cardinality` / `n_runs` bound the readable** values/runs; **`capacity` controls growth and
  deallocation** (the freed length).
- Use sites reconstruct a **temporary slice** (`ptr[0..cardinality]`, `ptr[0..capacity]`) so
  **`ReleaseSafe` bounds checking is restored** at access — no raw unchecked indexing on hot paths.
- **Tagged-pointer alignment stays valid** — the 2-bit tag still fits the container pointer's low
  bits (`@alignOf` of the header ≥ 4).
- **Header `@sizeOf` makes the 24→16 transition** (compile-time asserted); **`@alignOf` is asserted
  separately** (it must stay ≥ 4 for the tag, but it does not itself effect the size transition). The
  **SMP slot class** (32→16) is **allocator behavior, not a compile-time property** — **calculated
  and reported by the benchmark's class accounting on each host**, not asserted from the type.
- **Payload unchanged** — asserted per case: same **requested length, alignment, and SMP-class
  bytes** as baseline (not assumed power-of-two).

## Candidate execution mechanism (how full-row candidates run REAL code)

A standalone compact container replica **cannot** execute the production `lazyOr` / `clone` /
dense-AND / `select` paths — `TaggedPtr` and `Container` cast directly to the concrete production
container layouts. **Duplicating bitmap operations in the diagnostic is forbidden** (it would measure
code that differs from production). The two tiers therefore execute differently:

- **Container-level replica cells** → **standalone replica structs** in the E1-owned diagnostic files
  (no bitmap, no production edit) — these give header-cost attribution and run **concurrently**.
- **Full-bitmap GO rows** → **compile-time layout selection**: a comptime flag switches
  `ArrayContainer` / `RunContainer` to the compact header while the **default build stays unchanged**,
  so the **real** `lazyOr` / `clone` / dense-AND / `select` code runs on the compact layout. Because
  this edits the shared container files, each representation's full-row candidate is **built and
  measured in an isolated diagnostic worktree**; it is **not merged** until adoption (`32-02` /
  `32-03`).

**Three-way comparison (isolate the layout effect from the refactor).** Comparing the compact
candidate directly against the committed baseline **conflates two effects**: the actual 24→16-byte
layout change *and* the source refactor that adds compile-time selection + reconstructed-slice
accessors. Measure three builds:

1. **Committed baseline** — current source.
2. **Candidate source, compact flag OFF** — proves the selection/accessor **infrastructure is timing-
   and instruction-neutral**.
3. **Same candidate source, compact flag ON** — isolates the actual **header-layout** effect.

**GO comparison is `(3) vs (2)`.** Also compare **`(2) vs (1)`** and require it **within noise with
instruction-equivalent relevant kernels** — otherwise the infrastructure movement is separately
accounted for before any layout conclusion.

**Correctness before performance (both flag states).** These worktrees carry a near-complete
representation migration, so **both flag OFF and flag ON** must pass **`zig build test`, `zig build
difftest`, `ReleaseSafe`, and `ReleaseFast`** — validate correctness **before** accepting any
performance number.

**Parallelism consequence:** the container microbenchmarks (`32-00`/`32-01` replica cells) run
concurrently; the **full candidate builds need separate worktrees and cannot be merged
concurrently** (Wave 2 serial adoption already enforces this).

## Phase 1 — diagnostic prototype (benchmark-only, both hosts)

**Module: a new E1-owned diagnostic module** (e.g. `bench_compact_header.zig`). The existing
`bench_single_alloc.zig` is **Array-only** and uses a **1 warmup / 9 timed** protocol — E1 does
**not** edit it; it may reuse its harness patterns but adds a **separate module** covering **both**
Array and Run at the **canonical 3 warmup / 21 timed, five-process-median** protocol. Repository-only
diagnostic; no production default changed. **Full-row candidates use compile-time layout selection in
an isolated worktree** (above), never duplicated bitmap code.

**Pinned diagnostic corpora (assert before timing), tied to the authoritative canonical generators
in `bench_croaring.zig`:**

- **Run corpus (clone / dense-AND / select / build / growth / deinit / membership / iteration Run
  cells):** the canonical dense operands — `a = addRange(0, 499999)` → **8 run containers** (keys
  0–7), `b = addRange(250000, 749999)` (`initRawrDenseBitmaps`). dense-run-AND cell = `a AND b`;
  select cell operates on `a`.
- **select queries:** the canonical set — **1,000,000** queries, each `uintLessThan(u32, 500_000)`,
  from `std.Random.DefaultPrng.init(12345)` (`initTestData`), drawn as the **3rd** per-iteration
  value (after `int(u32)` then a `uintLessThan(500_000)`).
- **Array corpus (build / growth / clone / deinit / membership / iteration Array cells AND the
  targeted lazy-OR-construction cell):** the canonical **sparse** corpus —
  `std.Random.DefaultPrng.init(54321)`, `500_000` values `int(u32)` across full u32 space, sorted +
  deduped to `sparse_len` (`initSparseValues`), then (`initRawrSparseBitmaps`) **`a` =
  `sparse_values[0..half]`, `b` = `sparse_values[half/2..]`** where `half = sparse_len/2` — an
  **overlapping quarter-to-end slice, not a clean half-split**. This populates the many small
  **array containers** the sparse 2-way lazy-OR merge clones.
- **build mode:** `ReleaseFast` for the timing cells; `ReleaseSafe` for the correctness/bounds pass.
- **artifact format:** committed columns per cell — ns, alloc calls, free calls, requested bytes,
  effective SMP-class bytes, teardown.

### Operation matrix, split by representation

Each representation has **two tiers**, and **GO requires movement in the canonical full-bitmap row,
not only the container-level replica microbenchmark** (a replica clone cell can overstate E1's
end-to-end impact):

- **Run cells:**
  - *container-level replicas:* reserved build, growth, clone, deinit, membership, iteration — on
    **real compact-header Run replicas** (never inferred from the Array prototype);
  - *full-bitmap rows (the GO gate):* dense **clone**, **dense run-AND** (`a AND b`), **select** on
    the canonical dense bitmap.
- **Array cells:**
  - *container-level replicas:* reserved build, growth, clone, deinit, membership, iteration on the
    sparse array-container population;
  - *isolated attribution cell:* clone the sparse array-container population (counts the header
    cost) — **attribution only**;
  - *full-bitmap row (the GO gate):* run the **actual canonical `lazyOr(a, b, /*bitset_conversion=*/
    true)` construction path** and **report how many unmatched Array headers it clones** — the
    end-to-end row E1's Array lever must move, not the isolated clone cell alone.

- **Accounting per cell:** allocations, frees, requested bytes, **effective SMP-class bytes** (host
  class accounting), teardown — kept distinct (container instances ≠ allocator calls).
- **Assert:** 16-byte headers (`@sizeOf`), header now in the 16-byte class (host accounting), payload
  requested-length/alignment/class unchanged (per case).
- Named, committed diagnostic artifact (E1 owns its own bench module — no shared-file edits; shared
  `build.zig` / runner / docs are implementer-owned).

## Phase 2 — production migration (conditional, per representation)

If a representation's Phase 1 shows a real M4 improvement with Zen 4 within noise, migrate **that
representation** (Array and/or Run) to the compact header in production:

- **Testable output invariants (outside timing)** — for every op, the compact-header result has the
  **same container kind, same cardinality, same values** as the baseline, and **identical portable
  bytes** where serialization is valid; CRoaring differential across container-type mixes. (Not the
  vaguer "representation-identical output" — these are the checked invariants.)
- **Exhaustive allocation-failure injection** on every changed path (this is an ownership/layout
  change): valid-or-cleanly-errored, inputs untouched, no leak.
- **Board gate + spec-28 layout exception** — full-board before/after, both hosts; untouched-row
  movement is layout only with stable focused timing *and* instruction-identical disassembly.
- **Zen 4 policy (spec 30):** target rows judged within noise by repeated focused timing + range
  overlap; a real regression needs an explicit owner exception.
- **One architecture-neutral shape** per representation.

## Acceptance

- **Phase 1 GO/NO-GO recorded per representation** (Array, Run) with the full accounting, both hosts;
  16 B header + 16-byte class + unchanged payload class asserted; **GO requires movement in the
  canonical full-bitmap row** (Run: clone / dense-AND / select; Array: `lazyOr(...,true)`
  construction) — **not** the container-level replica alone; no production default changed.
- **Phase 2 (per representation, if GO):** the targeted rows improve on M4 SMP with Zen 4 within
  noise, the **output invariants** (same kind / cardinality / values + portable bytes where
  serialize valid) + differential + failure-injection green, board gate held.
  **Rows close at ≤ 1.10x; a beneficial partial is adopted by owner judgement and stays open.**
- `zig build test`; `zig build difftest`; `ReleaseSafe` / `ReleaseFast` green; canonical
  `run-compare-bench.sh` both hosts on any production adoption; `docs/parity-measurement.md` updated.

## Proposed chunk plan (confirm at review)

Diagnostics split by representation (the Run prototype + op matrix is substantial and independently
decidable); production migrations **serialized** (Wave 2 — one change at a time).

- **`32-00`** — **Array** compact-header prototype + measurement, both hosts, assert gate; no
  production change. Array GO/NO-GO.
- **`32-01`** — **Run** compact-header prototype + measurement (real Run replicas), both hosts, assert
  gate; no production change. Run GO/NO-GO.
- **Note:** `32-00` and `32-01` may run concurrently **only if they own separate diagnostic source
  files** (e.g. `bench_compact_header_array.zig` / `bench_compact_header_run.zig`) — if both would
  edit one shared `bench_compact_header.zig`, run them **serially** instead. (Specs 32/33/34 remain
  safe as concurrent separate tracks.)
- **`32-02`** — production migration of the **first** winning representation (conditional): invariants,
  failure injection, board gate.
- **`32-03`** — production migration of the **second** winning representation (conditional): **only
  after `32-02` is adopted, rebased onto, and board-gated** — adopt one, gate it, rebase, then the
  other; never both in one board window.

## Estimate

M each for `32-00` / `32-01` (one representation × the op matrix × two hosts). M each for `32-02` /
`32-03` (core representation change with full correctness + failure injection).
