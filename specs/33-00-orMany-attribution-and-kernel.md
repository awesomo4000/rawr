<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 33-00: orMany attribution, kernel + ceiling, direct end-to-end candidate

Toplevel: [33-nway-or-fusion.md](33-nway-or-fusion.md) (E2). Attribute the `orMany` 1.248x gap, build
the benchmark-only word-major kernel and ceiling, and — when the projection clears — time a
benchmark-only end-to-end candidate directly. **No production change.**

## Shared corpus helper (do first)

Extract the private `initRawrManyBitmaps` / `addManyPatternRawr` generator from `bench_croaring.zig`
into a **shared repository-only helper** (e.g. `bench_corpus.zig`) imported by **both** the parity
harness and the E2 diagnostic — single source of truth, no drift. **Assert a post-build fingerprint**
(the exact post-`runOptimize` per-key type counts) after the extraction, so the refactor **proves it
did not change the parity row**. (Shared-integration edit — implementer-owned.)

## Attribution

- **Split accumulation time by source type (array / bitset / run).** If the bitset share is small,
  the kernel cannot close the gap — record and stop.
- **Pin the exact post-`runOptimize` per-key type counts** (32 inputs, 6 keys, `base = chunk<<16`);
  assert them (not "~8 bitsets").

## Cells (both hosts, SMP, canonical protocol, `batch_count = 128`)

1. Baseline zero-then-input-major. 2. First-bitset seed (copy the single bitset into owned storage —
never adopt source words). 3. **Word-major N-way OR** per key. 4. Seed + word-major. 5. **Bitset-only
ceiling** (bitset inputs only), measured **before** the full mixed cell.

**Cells 3–5 are always built** (needed to measure the kernel and bound the gain).

## Projection → direct end-to-end (the GO evidence)

- **Projection (build/no-build gate):** bitset-share × ceiling per-share improvement applied to the
  full-row time → projected mixed ratio. If it cannot reach ≤ 1.10x, **do not build the end-to-end
  candidate**; record and stop.
- **Direct end-to-end (GO evidence):** if the projection clears, build a **benchmark-only end-to-end
  implementation of the winning shape** and time it on the **full canonical `orMany` row** —
  including **pointer collection, Array/Run folding, top-level assembly, and repair** — both hosts.
  This directly measured full-row ratio is the GO evidence.

## Scratch + collection

- Per-key bitset-pointer buffer: **one reused buffer**, capacity **`bitmaps.len`** (arbitrary input
  count is the only sound bound; or an explicit prepass), reset (not reallocated) between keys, freed
  once, OOM → clean error. **Collection overhead counted inside the full mixed cell's timing.**

## Correctness (outside timing)

- **Input immutability** (every source unchanged); **cardinality contract** (accumulator unknown
  before repair, but the returned bitmap has known per-container cardinalities equal to baseline);
  **testable output invariants** (kind / cardinality / values + portable bytes where serialize valid)
  + CRoaring differential; **edge cases** (empty/single input, duplicate input pointers, 0/1/many
  bitsets per key, mixed types, 4096 conversion boundary).
- **OR-specific — do not touch `xorMany`.**

## Acceptance

- Shared helper extracted with asserted post-build fingerprint; source-type split + pinned per-key
  counts reported; cells 1–5 timed both hosts; projection computed; and — when it clears — the
  **direct end-to-end full-row measurement** reported both hosts. Input-immutability + differential +
  edge cases green. No production change.
- Decision recorded: proceed to `33-01` only if the **direct end-to-end** number is GO.
- `zig build test`; `zig build difftest` green; diagnostic section of `docs/parity-measurement.md`
  updated.
