<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 32-01: Run compact-header diagnostic

> **Outcome (2026-08-06) — GO (major).** Compact Run header cleared the three-way control and moved
> the full-bitmap rows; adopted in `32-02`. Closed clone / dense-AND / dense-OR / select /
> removeRange on M4.

Toplevel: [32-compact-container-headers.md](32-compact-container-headers.md) (E1). Prototype the
compact **`RunContainer`** header (24 B → 16 B, 32-byte → 16-byte SMP slot, payload untouched) and
measure it. **Benchmark-only; no production default changed.** Produces the **Run GO/NO-GO**.
Independent of `32-00` (separate GO/NO-GO); may run concurrently only with separate source files.

## Prototype (two tiers — see toplevel "Candidate execution mechanism")

- **Container-level replica cells** → **standalone Run replica struct** in an E1-owned diagnostic file
  (e.g. `bench_compact_header_run.zig`). **Real compact-header Run replicas** — never inferred from
  the Array prototype. No bitmap, no production edit.
- **Full-bitmap GO rows** (clone / dense-AND / select) → **compile-time layout selection** switching
  `RunContainer` to the compact header (default build unchanged), built and measured in an **isolated
  diagnostic worktree**. **Do NOT duplicate bitmap ops.**
- **Three-way comparison (see toplevel):** (1) committed baseline; (2) candidate source, **flag OFF**;
  (3) candidate source, **flag ON**. **GO = (3) vs (2)** (isolates the layout effect); **(2) vs (1)**
  must be within noise with instruction-equivalent kernels, else the infrastructure movement is
  separately accounted for. **Correctness before perf:** both flag states pass `zig build test`,
  `zig build difftest`, `ReleaseSafe`, `ReleaseFast` before any number is accepted.
- Compact `RunContainer`: `runs` becomes `[*]RunPair` (8 B) + `n_runs: u16` + `capacity: u16`
  + `cardinality: i32` = 16 B. `n_runs` bounds reads; `capacity` controls growth/dealloc; use sites
  reconstruct a temporary slice so `ReleaseSafe` bounds checks hold.

## Corpus (pinned, assert before timing)

Canonical **dense** operands (`initRawrDenseBitmaps`): `a = addRange(0, 499999)` → **8 run
containers** (keys 0–7), `b = addRange(250000, 749999)`. dense-run-AND = `a AND b`; select operates
on `a` with the canonical 1M `uintLessThan(u32, 500_000)` queries from `DefaultPrng.init(12345)` (3rd
per-iteration draw).

## Cells (both hosts, SMP, canonical 3 warmup / 21 timed, five process medians + full range)

- **Container-level replicas:** reserved build, growth, clone, deinit, membership, iteration.
- **Full-bitmap GO rows:** dense **clone**, **dense run-AND** (`a AND b`), **select** on the
  canonical dense bitmap.

## Accounting + asserts (per cell)

- allocations, frees, requested bytes, **effective SMP-class bytes** (host class accounting),
  teardown — kept distinct.
- **`@sizeOf` 24→16** (compile-time); **`@alignOf` ≥ 4** asserted separately.
- header now in the **16-byte class** (host accounting); payload **requested length / alignment / SMP
  class unchanged per case**.
- `ReleaseFast` for timing cells; `ReleaseSafe` for the correctness/bounds pass.

## Acceptance

- Corpus inventory asserted (8 run containers); all cells run both hosts; the five accounting figures
  reported per cell; 16 B header + 16-byte class + unchanged payload asserted.
- **Run GO requires movement in the canonical full-bitmap rows** (clone / dense-AND / select)
  measured as **`(3) vs (2)`** (flag ON vs flag OFF, same candidate source), not the replica
  microbenchmark alone and not vs the committed baseline; **`(2) vs (1)` within noise** (or
  infrastructure movement separately accounted). Record GO/NO-GO with both hosts' numbers.
- Named, committed diagnostic artifact; **no production default changed** (compile-time layout flag
  defaults off; full-row candidate lives in a worktree, unmerged).
- **Correctness before perf, both flag states:** flag OFF **and** flag ON each pass `zig build test`,
  `zig build difftest`, `ReleaseSafe`, and `ReleaseFast` — this chunk depends specifically on the
  reconstructed-slice `ReleaseSafe` bounds behavior — **before** any performance number is accepted.
  Diagnostic section of `docs/parity-measurement.md` updated.
