<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 34-01: select production kernel

> **Outcome (2026-08-06) — DID NOT RUN.** `34-00` was storage-free NO-GO and select closed via the
> compact Run header (spec 32); no unrolled kernel and no index shipped.

Toplevel: [34-select-kernel-matrix.md](34-select-kernel-matrix.md) (E4). Ship the winning
**storage-free** kernel (2- or 4-container unrolled) into production `select`. **Gated on `34-00`
identifying a storage-free winner that does not regress the mixed-container controls.** If `34-00`
found only a ceiling win, this chunk does **not** run — an index is a separate follow-up spec.

## Change

- Replace the scalar top-level cardinality walk in production `select` with the winning **unrolled
  walk** (2- or 4-container groups + scalar tail), **identical dispatch and cardinality behavior** to
  the scalar walk. **No stored metadata / no index.**
- **Production `select` remains inlineable exactly as before** — the change does **not** add a
  `noinline` boundary to production. The `noinline` measurement boundary belongs only to the **parity
  harness wrapper**, which wraps baseline and candidate identically.

## Constraints / gates

- **Identical results** — `select(n)` returns the identical `?u32` to baseline for all n, including
  the boundary set (`0`, prefix boundaries, `cardinality-1`, `cardinality`, empty, `maxInt(u32)+1`,
  `maxInt(u64)`); CRoaring differential across type mixes.
- **Mixed-container controls held** — ≤ 5% on the 8-Array / 8-Bitset / 8-Mixed / 7-Run-tail controls
  (no non-Run or non-aligned regression).
- **No allocation** in `select` — accounting is timing + branch/disasm.
- **Board gate + spec-28 layout exception**; **Zen 4 policy** (spec 30); **one architecture-neutral
  shape**.
- **Rebaseline:** if E1 adopted first (Wave 2), re-measure on the accepted E1 state before this gate.

## Acceptance

- `select` closes to **≤ 1.10x M4 SMP** (or a beneficial partial adopted by owner judgement, row
  stays open), Zen 4 within noise, identical results, controls ≤ 5%, board gate held.
- **No index added** — a winning ceiling only authorizes a separate follow-up spec.
- `zig build test`; `zig build difftest`; `ReleaseSafe` / `ReleaseFast` green; canonical
  `run-compare-bench.sh` both hosts; `docs/parity-measurement.md` updated.
