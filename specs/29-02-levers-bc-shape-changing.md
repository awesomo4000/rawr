<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 29-02: Levers B and/or C — allocation-shape-changing

Ship the bitmap-level full-run identity (**B**) and/or the pre-sized top-level storage (**C**) —
whichever `29-00` implicates — to close the dense-AND / dense-OR rows to **≤ 1.10x on M4 SMP**.
Both change allocation shape, so **neither is assumed; each carries the spec-27 measurement
obligation** and is measured **independently for AND and OR**, dropped on any host where it
regresses.

Toplevel: [29-dense-result-construction.md](29-dense-result-construction.md).
Gated on: [29-00](29-00-diagnostic-cells.md) (and [29-01](29-01-lever-a-kernel-identity.md) if A shipped).

## Levers

- **B — bitmap-level full-run identity** (`twoWayAllocatingMerge`, `bitmap.zig:921`/`:927`): on a
  matched full-run pair, skip scratch/kernel and produce the owned result via the **existing
  container `clone()`**, deterministic operand selection when both runs are full — AND: `if a is
  full, clone b; otherwise clone a`; OR: `if a is full, clone a; otherwise clone b`. **Changes AND
  allocation shape** (bypasses the scratch-then-clone at ~`bitmap.zig:1850`/`:1930`).
  Clone-vs-construct-canonical is **not** a "whichever wins" choice — a canonical-full-run
  construction, if ever worth testing, is a **separate subcell** with its own numbers.
- **C — pre-sized top-level storage:** AND `min(self.size, other.size)`; OR
  `min(self.size + other.size, 65536)` (clamp — 16-bit key space). Removes grow-from-4 cycles.
  C's shape differs from clone's (dense **OR** does *multiple* growth cycles), so it may help OR
  even though spec-27 exact-capacity pre-sizing *regressed* clone on M4 SMP — **test independently
  for AND and OR.**

## The spec-27 trap (mandatory)

Spec 27 showed exact-capacity pre-sizing **regressed** M4 SMP clone (~272 → 408–430 ns) despite
fewer allocations — SMP class/behavior effects defy allocation-count intuition. Therefore each of
B and C is **measured on M4 SMP, per op, before shipping**; a lever that regresses its op on any
host is **dropped for that op**, not shipped on allocation-count reasoning.

## Constraints / gates

- **Representation-identical output** (spec 26): byte-identical (via `serialize`) to the current
  merge **and** CRoaring set-parity. Differential across container-type mixes, full/partial runs,
  **empty/disjoint results, and mutation after a zero-capacity result** stays green. Zero-capacity
  cases to cover explicitly: **AND with an empty operand** (`min(0, n) = 0`) and **OR with both
  operands empty** (`min(0 + 0, 65536) = 0`), then **add to that result** (growth from zero
  capacity). A nonempty disjoint AND still reserves `min(a.size, b.size) > 0` — not the
  zero-capacity case.
- **Error semantics — build-then-commit, leak-free:** exhaustive allocation-failure injection on
  the changed paths; on OOM the result is valid or cleanly errored, inputs untouched, no leak.
- **Zen 4 no-regress (hard):** rawr is ahead on both dense ops on Zen 4; the shipped change stays
  within noise (≤ 5%, rerun on overlap).
- **Board gate + tightened layout exception (spec 28):** no canonical row worsens > 5% vs a fresh
  pre-change baseline, both hosts; layout classification requires **both** stable focused timing
  *and* instruction-identical disassembly.

## Acceptance (Phase 2 GO — hard)

- **bitwiseAnd dense and bitwiseOr dense reach ≤ 1.10x on M4 SMP**, Zen 4 not regressed,
  byte-identical output, differential + failure-injection green, board gate held (layout
  exception).
- Ship the **winning combination** (A from `29-01` where implicated + B and/or C here); **each
  lever ships only on its own SMP numbers, per op.**
- **Anything above 1.10x is a partial result and the row stays open** — an attributed-but-material
  residual **reopens with the next lever; it does not close the row.** Parity is hard-required.
- `zig build test`; `zig build difftest`; canonical `run-compare-bench.sh` both hosts;
  `ReleaseSafe` / `ReleaseFast` green; `docs/parity-measurement.md` updated.
