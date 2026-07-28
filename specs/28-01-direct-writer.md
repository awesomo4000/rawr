<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 28-01: Serialize — direct fixed-buffer path (conditional)

Second chunk of [serialize direct buffer](28-serialize-direct-buffer.md). Implements the winning
cell from `28-00` in production `serialize()`, keeping `serializeToWriter()` unchanged.

## Gate

- `28-00` complete: the gap attributed to a movable lever (construction and/or output), with the
  winning cell **not** regressing M4 SMP. If `28-00` shows the gap is not in a movable component
  (shared `getCardinality()`-per-container, or variance), this chunk is **not implemented** —
  record the NO-GO here and move to `done/` with the parent.

## Implementation

- Add a **direct fixed-buffer path used only by `serialize()`** implementing the winning levers:
  in-place descriptor/offset construction (if the construction lever won) and/or raw indexed
  stores bypassing `Writer` (if the output lever won). Each lever ships **only on its own
  numbers** — a lever that was M4-SMP-neutral-or-worse in `28-00` is not shipped.
- **`serializeToWriter()` stays exactly as-is** — generic writers can't be indexed; it remains
  their correct implementation and the in-repo byte oracle. No public API change; `serialize()`
  routes internally.
- **Cursor invariant:** the direct path asserts its **final write position == `buf.len`**.

## Correctness

- **Byte-identical output**, checked in order: (1) vs unchanged `serializeToWriter()` (in-repo
  legacy oracle); (2) round-trip through rawr `deserialize`; (3) CRoaring
  `roaring_bitmap_portable_deserialize` equality.
- Differential across container-type mixes (array/bitset/run), run and no-run formats, the empty
  bitmap, and **run-format container counts immediately below and exactly at `NO_OFFSET_THRESHOLD`**
  (the offset-table branch).
- Error semantics unchanged: `serialize()` returns the owned buffer or frees it on error
  (`errdefer`); only the output buffer is allocated on the direct path.

## Acceptance

- **Target is the rawr-SMP path:** serialize reaches **≤ 1.10x** (or a material improvement,
  > 5% with range support) on **M4 SMP**, with **Zen 4 SMP not regressed** (≤ 5%, rerun on
  overlap). The board-regression check still covers **all** rows including serialize-libc (don't
  regress it), vs a fresh pre-change baseline run immediately before the after-run, both hosts —
  but the ≤1.10x goal and gate is SMP, not libc.
- **M4-SMP gate:** the shipped path does not regress M4-SMP serialize even if it removes
  allocations (the spec-27 check) — any lever that does is dropped, documented.
- Byte-identity (legacy oracle + round-trip + CRoaring) and the full differential green; cursor
  invariant asserted.
- A documented partial (one lever shipped, the other NO-GO on M4 SMP) is acceptable.
- `zig build test`; `zig build difftest`; canonical `run-compare-bench.sh` both hosts;
  `ReleaseSafe` / `ReleaseFast` green; `docs/parity-measurement.md` updated.

## Checklist

- [ ] Direct path in `serialize()` only; `serializeToWriter()` untouched; no public API change;
      shares layout helpers with the `28-00` winning cell (no drift)
- [ ] Only levers that won (incl. M4-SMP-non-regressing) in `28-00` shipped
- [ ] Byte-identical vs `serializeToWriter()`, round-trip, and CRoaring; threshold-boundary cases
- [ ] Cursor == `buf.len` asserted
- [ ] M4 **SMP** ≤ 1.10x or material improvement; Zen 4 **SMP** not regressed; board ≤ 5% all
      rows both hosts
- [ ] test / difftest / both-host canonical run / ReleaseSafe / ReleaseFast green; docs updated
