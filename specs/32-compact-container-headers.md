<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 32: Compact separate container headers (E1)

Campaign: [31-structural-parity-campaign.md](31-structural-parity-campaign.md) (Wave 1). Shrink
`ArrayContainer` and `RunContainer` headers from **24 B → 16 B** so they drop from the **32-byte SMP
size class to the 16-byte class**, **keeping the payload in its existing separate power-of-two
allocation**. Allocation **count is unchanged**; only the header slot halves. This is the widest
structural lever on the board.

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
next SMP size class (8 KB payload + header → 16 KB class). **E1 keeps the payload in its own
existing allocation and class** — only the small header slot changes (32→16). This is the firewall
that makes E1 legitimate.

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
- **Header `@sizeOf` / `@alignOf`** (24→16) are **compile-time asserted**. The **SMP slot class**
  (32→16) is **allocator behavior, not a compile-time property** — **calculated and reported by the
  benchmark's class accounting on each host**, not asserted from the type.
- **Payload class unchanged** — asserted (same power-of-two allocation as baseline).

## Phase 1 — diagnostic prototype (benchmark-only, both hosts)

Add a separate-payload/**compact-header** variant (Array and Run as **separate** cells) to the
existing single-allocation prototype module — a **repository-only diagnostic**, no production default
changed.

- **Per representation (Array, Run), measure both hosts, SMP, canonical protocol** (3 warmup /
  21 timed, five process medians + full range): reserved build, growth, clone, deinit, membership,
  iteration, dense run-AND, select.
- **Accounting per cell:** allocations, frees, requested bytes, **effective SMP-class bytes** (host
  class accounting), teardown — kept distinct (container instances ≠ allocator calls).
- **Assert:** 16-byte headers (`@sizeOf`), header now in the 16-byte class (host accounting),
  payload class unchanged.
- Named, committed diagnostic artifact (E1 owns its own bench module — no shared-file edits).

## Phase 2 — production migration (conditional, per representation)

If a representation's Phase 1 shows a real M4 improvement with Zen 4 within noise, migrate **that
representation** (Array and/or Run) to the compact header in production:

- **Operation-appropriate identity outside timing** — byte-identity via `serialize` where defined,
  set-identity + CRoaring differential elsewhere; representation-identical output across every op.
- **Exhaustive allocation-failure injection** on every changed path (this is an ownership/layout
  change): valid-or-cleanly-errored, inputs untouched, no leak.
- **Board gate + spec-28 layout exception** — full-board before/after, both hosts; untouched-row
  movement is layout only with stable focused timing *and* instruction-identical disassembly.
- **Zen 4 policy (spec 30):** target rows judged within noise by repeated focused timing + range
  overlap; a real regression needs an explicit owner exception.
- **One architecture-neutral shape** per representation.

## Acceptance

- **Phase 1 GO/NO-GO recorded per representation** (Array, Run) with the full accounting, both hosts;
  16 B header + 16-byte class + unchanged payload class asserted; no production default changed.
- **Phase 2 (per representation, if GO):** the targeted rows improve on M4 SMP with Zen 4 within
  noise, representation-identical output, differential + failure-injection green, board gate held.
  **Rows close at ≤ 1.10x; a beneficial partial is adopted by owner judgement and stays open.**
- `zig build test`; `zig build difftest`; `ReleaseSafe` / `ReleaseFast` green; canonical
  `run-compare-bench.sh` both hosts on any production adoption; `docs/parity-measurement.md` updated.

## Proposed chunk plan (confirm at review)

- **`32-00`** — compact-header prototype + measurement (Array and Run as separate cells), both hosts,
  the assert gate; no production change. Produces the per-representation GO/NO-GO.
- **`32-01`** — Array production migration (conditional on `32-00`): identity, failure injection,
  board gate.
- **`32-02`** — Run production migration (conditional on `32-00`): identity, failure injection,
  board gate.

## Estimate

M–L for `32-00` (prototype for two representations × the op matrix × two hosts). M each for `32-01` /
`32-02` (core representation change with full correctness + failure injection).
