<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 32-03: Production migration — second winning representation

Toplevel: [32-compact-container-headers.md](32-compact-container-headers.md) (E1). Migrate the
**second** GO representation to the compact header — **only after `32-02` is adopted, rebased onto,
and board-gated.** Never both representations in one board window (Wave 2 serial adoption).

## Precondition

- The other representation's diagnostic (`32-00` / `32-01`) was GO **and** `32-02` shipped and passed
  its board gate. This chunk **rebases onto the accepted `32-02` state** and re-measures before its
  own gate (its numbers are stale against the pre-`32-02` baseline).

## Change

- Migrate the second representation (`RunContainer` **or** `ArrayContainer`, whichever `32-02` did not
  do) to the compact many-pointer header in production — same pointer contract as `32-02`.

## Constraints / gates

- **Output invariants**, **exhaustive allocation-failure injection**, **board gate + spec-28 layout
  exception**, **Zen 4 policy**, **one architecture-neutral shape** — identical to `32-02`, measured
  on the post-`32-02` baseline.

## Acceptance

- The second representation shipped compact; its targeted full-bitmap rows improve on M4 SMP vs the
  post-`32-02` baseline, Zen 4 within noise, output invariants + differential + failure-injection
  green, board gate held.
- **Rows close at ≤ 1.10x; a beneficial partial is adopted by owner judgement and stays open.**
- `zig build test`; `zig build difftest`; `ReleaseSafe` / `ReleaseFast` green; canonical
  `run-compare-bench.sh` both hosts; `docs/parity-measurement.md` updated.
