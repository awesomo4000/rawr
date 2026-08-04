<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 32-02: Production migration — first winning representation

Toplevel: [32-compact-container-headers.md](32-compact-container-headers.md) (E1). Migrate the
**first** representation whose diagnostic was GO (Array from `32-00` or Run from `32-01`) to the
compact header in production. Gated on that representation's GO.

## Change

- Migrate the chosen representation (`ArrayContainer` **or** `RunContainer`) to the compact
  many-pointer header in production: `cardinality`/`n_runs` bound reads, `capacity` controls
  growth/dealloc, use sites reconstruct temporary slices so `ReleaseSafe` bounds checks hold, tag
  alignment preserved.
- **Only one representation in this chunk** — the other is `32-03`, after this one is adopted and
  rebased onto.

## Constraints / gates

- **Output invariants (outside timing):** every op's result has the **same container kind,
  cardinality, values** as baseline, and **identical portable bytes** where serialization is valid;
  CRoaring differential across container-type mixes, empty/boundary cases.
- **Exhaustive allocation-failure injection** on every changed path: valid-or-cleanly-errored, inputs
  untouched, no leak.
- **Board gate + spec-28 layout exception** (full-board both hosts; untouched-row movement is layout
  only with stable focused timing *and* instruction-identical disassembly).
- **Zen 4 policy (spec 30):** target rows judged within noise by repeated focused timing + range
  overlap; a real regression needs an explicit owner exception.
- **One architecture-neutral shape.**

## Acceptance

- The chosen representation shipped compact in production; its targeted full-bitmap rows improve on
  M4 SMP, Zen 4 within noise, output invariants + differential + failure-injection green, board gate
  held.
- **Rows close at ≤ 1.10x; a beneficial partial is adopted by owner judgement and stays open.**
- `zig build test`; `zig build difftest`; `ReleaseSafe` / `ReleaseFast` green; canonical
  `run-compare-bench.sh` both hosts; `docs/parity-measurement.md` updated.
