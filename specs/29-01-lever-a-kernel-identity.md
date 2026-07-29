<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 29-01: Lever A — full-run kernel identity branches

Ship the full-run identity fast paths inside `runIntersectRun` / `runUnionRun`
(`src/container_ops.zig:742` / `:608`) **if `29-00` implicates lever A**. A is
allocation-shape-neutral but layout-affected, so it ships on its measured M4/Zen 4 numbers — no
spec-27 gate (it does not change allocation shape), but it does face the board layout gate.

Toplevel: [29-dense-result-construction.md](29-dense-result-construction.md).
Gated on: [29-00](29-00-diagnostic-cells.md).

## Change

- **`runIntersectRun`**: full-run ∩ X = X — early return producing X's runs.
- **`runUnionRun`**: full-run ∪ X = full-run — early return producing the full run.
- **Allocation-shape-preserving:** allocate the **baseline capacity**
  `min(@as(usize, a.n_runs) + b.n_runs, 65535)` (widen before adding so the `u16` sum cannot
  overflow before the clamp) and **copy** the identity runs. Do **not** request a tighter
  `clone()`-sized allocation — that would be allocation-shape-changing and belongs to `29-02`, not
  here.

## Constraints / gates

- **Representation-identical output:** results **byte-identical** (via `serialize`) to the current
  merge **and** CRoaring set-parity — a full-run identity must produce the *same container type*
  the merge would. Differential across container-type mixes, full/partial runs, empty/disjoint,
  chunk-boundary cases stays green.
- **Error semantics — build-then-commit, leak-free:** exhaustive allocation-failure injection on
  the changed kernel paths; on OOM the result is valid or cleanly errored, inputs untouched, no
  leak.
- **Board gate + tightened layout exception (spec 28):** no canonical row worsens > 5% vs a fresh
  pre-change baseline, both hosts. An untouched row's movement is layout (not a regression) **only
  when BOTH** its focused before/after timing is stable *and* its disassembly is
  instruction-identical; CRoaring moving too is *not* sufficient alone.
- **Zen 4 no-regress (hard):** rawr is ahead on both dense ops on Zen 4; stays within noise
  (≤ 5%, rerun on overlap).

## Acceptance

- Lever A shipped **single-implementation on all arches** on its measured M4/Zen 4 numbers, only if
  `29-00` implicated it.
- Byte-identical output; differential + failure-injection green; board gate held (layout
  exception); Zen 4 not regressed.
- Record the contribution toward the ≤ 1.10x target. **If A alone does not close both rows, the
  rows stay open** — `29-02` levers follow; parity is hard-required.
- `zig build test`; `zig build difftest`; canonical `run-compare-bench.sh` both hosts;
  `ReleaseSafe` / `ReleaseFast` green; `docs/parity-measurement.md` updated.
