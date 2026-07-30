<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 29-00: Dense set-op construction — six-cell diagnostic

> **Outcome (2026-07-30) — DONE.** Attribution delivered: dense **AND** benefits from **exact
> top-level result sizing** (C); dense **OR** benefits from **cloning the full-run identity
> operand** (B). Ruled out (regressed M4 despite less work/allocations, spec-27 trap): pre-sizing
> OR, and bypassing AND scratch allocation. Fed `29-01`/`29-02`.

Attribute the M4 dense-AND (1.911x) and dense-OR (1.167x) gaps across the three construction levers
before any production change. **No production default changes in this chunk** — benchmark-local
variants / gated internal helpers only. Output: a per-host, per-op attribution that tells `29-01`
and `29-02` which levers to ship.

Toplevel: [29-dense-result-construction.md](29-dense-result-construction.md).

## Corpus assertion (must hold before any timing)

Assert this structure; a drift invalidates the attribution and fails the chunk:

- **left operand: 8 run containers; right operand: 9 run containers**;
- **5 matched keys**, all **run/run** pairs, **each with at least one full run**;
- **AND output: 5 containers; OR output: 12 containers**;
- the full-run **identity branch fires exactly 5 times** — asserted **only in the identity-enabled
  cells** (A, B, A+C, B+C), *not* baseline or C.

## The six cells (per operation, AND and OR separately)

| cell | full-run kernel branch (A) | bitmap-level full-run identity (B) | pre-sized top-level (C) |
|---|---|---|---|
| baseline | — | — | — |
| A | ✓ | — | — |
| B | — | ✓ | — |
| C | — | — | ✓ |
| A+C | ✓ | — | ✓ |
| B+C | — | ✓ | ✓ |

(B subsumes A — B bypasses `runIntersectRun`/`runUnionRun` entirely, so A+B ≡ B on this corpus;
that combination is not measured.)

Lever definitions (as pinned in the toplevel):

- **A — full-run kernel identity branches** in `runIntersectRun`/`runUnionRun`: an early identity
  return inside the kernel. Allocation-shape-preserving — allocate the **baseline capacity**
  `min(@as(usize, a.n_runs) + b.n_runs, 65535)` (widen before adding so the `u16` sum cannot
  overflow before the clamp) and copy the identity runs; **no** tighter `clone()`-sized allocation.
- **B — bitmap-level full-run identity**: on a matched full-run pair, skip scratch/kernel and
  produce the owned identity result via the **existing container `clone()`**, with deterministic
  operand selection when both runs are full — AND: `if a is full, clone b; otherwise clone a`; OR:
  `if a is full, clone a; otherwise clone b`. Changes AND allocation shape (bypasses scratch-clone)
  → carries the spec-27 obligation in `29-02`.
- **C — pre-sized top-level storage**: AND `min(self.size, other.size)`; OR
  `min(self.size + other.size, 65536)`. Removes the grow-from-4 cycles.

## Measurement discipline

- **Six rawr-SMP cells per operation**, fresh process each, **M4 and Zen 4**, vs **one CRoaring
  reference per operation per host**. rawr-libc a **conditional control** only.
- Canonical protocol: **3 warmup / 21 timed**, **five process medians** + full range, the
  **canonical batch count** for the dense-AND/OR rows.
- **Construction and teardown measured separately** (26a matched-boundary; teardown
  subtraction-derived, **diagnostic not gated**; no nested timers).
- **Accounting per cell, kept distinct:** dense-AND **scratch** work reported as **stack
  reservations / constructions in the `FixedBufferAllocator`**, *not* allocator allocations —
  separate from **persistent allocator calls + requested bytes** (clone / top-level / payloads).

## Named diagnostic artifacts (reproducible)

- The benchmark-local cell harness and its invocation are named and committed so any cell is
  re-runnable in isolation on either host (mirror the 26a/28-00 diagnostic-artifact convention).
- Per-cell output: construction ns, teardown ns, FixedBufferAllocator reservations/constructions,
  persistent allocator calls + bytes — per host, per op.

## Acceptance (Phase 1 GO)

- Corpus inventory asserted (identity-fires-5× only in identity-enabled cells).
- M4 dense-AND and dense-OR gaps **attributed across the cells** (lever × construction/teardown/
  allocation), per host, on the SMP path.
- Zen 4 measured too (no-regress gate lives in `29-02`, but Zen 4 numbers are recorded here).
- **No production default changed**; diagnostic artifacts named and committed.
- `zig build test` green; `docs/parity-measurement.md` diagnostic section updated.
