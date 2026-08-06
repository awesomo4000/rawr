<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 33-01: orMany production word-major kernel

> **Outcome (2026-08-06) — DONE (shipped).** Seeded word-major `foldManyKey` shipped (`d7d357b`);
> accumulator loaded/stored once, sources ORed in — zero-fill + redundant first pass removed.

Toplevel: [33-nway-or-fusion.md](33-nway-or-fusion.md) (E2). Ship the winning word-major shape into
production `orMany`. **Gated on `33-00`'s direct end-to-end measurement being GO.**

## Change

Implement the pinned **`foldManyKey`** in production `orMany`:

- **total sources == 1** → existing clone fast path unchanged (copy the single source, any type).
- **multiple sources, 0 bitsets** → array/run path unchanged.
- **multiple sources, 1 bitset** → seed path: **copy/clone the bitset into independently owned
  storage** (never adopt borrowed source words), then OR the array/run remainder.
- **multiple sources, ≥2 bitsets** → **word-major N-way OR** into an owned accumulator, then fold
  array/run sources.
- Scratch pointer buffer per `33-00` (one reused buffer, capacity `bitmaps.len`, reset/freed, OOM
  clean).

## Constraints / gates

- **Cardinality contract:** public `orMany` returns a **repaired** result with **known** per-container
  cardinalities equal to baseline (internal accumulator-unknown is transient).
- **Testable output invariants** (kind / cardinality / values + portable bytes where serialize valid)
  + CRoaring differential; the full edge-case set (empty/single, duplicate pointers, 0/1/many bitsets
  per key, mixed types, 4096 boundary).
- **Failure injection:** OOM on the scratch buffer and on a partially constructed result —
  valid-or-cleanly-errored, **inputs untouched**, no leak.
- **Board gate + spec-28 layout exception**; **Zen 4 policy** (spec 30); **one architecture-neutral
  shape**. **OR-specific — `xorMany` untouched.**
- **Rebaseline:** if E1 adopted first (Wave 2), re-measure on the accepted E1 state before this gate.

## Acceptance

- `orMany` closes to **≤ 1.10x M4 SMP** (or a beneficial partial adopted by owner judgement, row
  stays open), Zen 4 within noise, invariants + differential + failure-injection green, board gate
  held.
- `zig build test`; `zig build difftest`; `ReleaseSafe` / `ReleaseFast` green; canonical
  `run-compare-bench.sh` both hosts; `docs/parity-measurement.md` updated.
