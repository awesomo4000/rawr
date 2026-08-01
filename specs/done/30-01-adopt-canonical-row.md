<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 30-01: Adopt `removeRangeCopy` into the canonical row

> **Outcome (2026-08-01) — CLOSED (full GO).** Selected the architecture-neutral **fused-default**
> (normal growth) shape — passed both host gates cleanly, no exception needed. Canonical
> `remove-range` row repointed at `removeRangeCopy` (stable `row_id`, operation relabeled),
> copy-vs-copy timing boundary intact. **M4 1.862x → 0.792x, Zen 4 0.412x → 0.187x** — row
> **closes** at well under 1.10x (rawr ahead on both). Board gate held (untouched-row movements were
> variance). Spec 30 moves to `done/`.

Wire the winning fused shape from `30-00` into the canonical `remove-range` parity row (legitimate
copy-vs-copy), rename the row operation, hold the board gate, and make the keep/close decision.

Toplevel: [30-fused-remove-range-copy.md](30-fused-remove-range-copy.md).
Gated on: [30-00](30-00-fused-op-and-measure.md) (the three-cell attribution).

## Winner selection (architecture-neutral)

Choose **one** fused implementation (one capacity policy) — **not** a per-host shape; two different
binaries per architecture is out of scope. Each *lever* (doomed-skip fusion; exact pre-sizing) is
included only if it improves M4 while keeping **Zen 4 within noise** (per the Zen 4 policy below),
yielding a single shipped shape.

**If no single shape keeps Zen 4 within noise** — the only M4-improving shape carries a **real**
Zen 4 regression — that is **not** an automatic "ship neither." It routes to the **keep decision**
below: a single shape with a real Zen 4 cost may be adopted via the **explicit owner exception**, or
declined. **"Ship neither" is the outcome when *either* no single shape improves M4, *or* every
M4-improving shape carries a real Zen 4 / board-gate regression and the owner declines the
exception.**

## Canonical row change

- Point the `remove-range` row's rawr side at the winning **`removeRangeCopy`** shape.
- **Timing boundary unchanged** (30-00): copy + range removal + result teardown inside timing,
  source construction outside; CRoaring `roaring_bitmap_copy` + `roaring_bitmap_remove_range_closed`
  + free, **copy-on-write disabled** — apples-to-apples, both preserving the source.
- **Forbidden:** moving rawr's clone/teardown outside the timed region while CRoaring's copy/free
  stays inside.

## Row rename (stable ID preserved)

- **`row_id` stays `remove-range`** — scripts, historical comparisons, and
  `docs/parity-measurement.md` keep tracking the same row; **no new ID.**
- Update the **display / operation labels** to `removeRangeCopy` ("copy with range removed"):
  `rawr_operation` (currently `"RoaringBitmap.clone plus removeRange"`) and the display name / corpus
  / setup-boundary text as needed. Leaving it named plain `removeRange` would falsely imply the
  **mutating primitive** got faster — it is unchanged and already faster than CRoaring (26a).

## Gates

- **Board gate + tightened layout exception (spec 28):** no canonical row worsens > 5% vs a fresh
  pre-change baseline, both hosts; an untouched row's movement is layout (not a regression) **only
  when BOTH** its focused before/after timing is stable *and* its disassembly is
  instruction-identical. CRoaring also moving is *not* sufficient alone.
- **Zen 4 policy (single, from the toplevel):** rawr is ahead (0.411x). The **target `remove-range`
  row** changes implementation by design, so its disassembly differs — judge a Zen 4 movement as
  **within noise** by **repeated focused timing and process-range overlap** (≤ 5%), **not**
  instruction-identity. (Instruction-identical disassembly stays with the **board gate**, for
  **untouched** rows.) Within noise = **not a regression**, passes. Movement **beyond noise** is a
  **real regression** that **fails by default** and may be adopted **only via the explicit owner
  exception** recorded with the numbers — never silently waived.

## Keep / close decision

- **Close (full GO):** M4 SMP **≤ 1.10x**, Zen 4 within noise (per policy), board gate held (layout
  exception), row renamed → the **row closes** and the spec moves to `specs/done/`.
- **Adopt a partial (row stays open):** the winning shape **moves M4 down** with Zen 4 within noise
  but lands **above 1.10x on M4** → `30-01` **may still ship it** and record the partial (as spec 29
  kept dense-AND 1.479x). **≤ 1.10x is required to *close*, not to *adopt*.** The residual (shared
  M4 SMP per-container-clone cost) keeps the row **open** for the next lever.
- **Human keep/not-keep judgement call:** the numeric gates **inform** the decision, they do not
  automate it. A Zen 4 slip **within noise** needs no exception (it is not a regression). A **real**
  Zen 4 regression against a **large M4 win** may be accepted via the **explicit owner exception**
  (Zen 4 policy), recorded with the numbers. The final call is made on the numbers at hand, gated on
  **no future avenue foreclosed** (in-place `removeRange`, clone, and dense-AND levers stay
  available).
- **Ship nothing** when *either* no fused shape improves M4, *or* every M4-improving shape carries a
  real Zen 4 / board-gate regression the owner declines to except.

## Acceptance

- Winning fused shape adopted into the `remove-range` row (or a reasoned no-ship recorded), stable
  `row_id` preserved, operation relabeled `removeRangeCopy`, timing boundary and copy-vs-copy
  legitimacy intact.
- Board gate held (layout exception); Zen 4 handled per the keep decision.
- Outcome recorded — closed (≤ 1.10x) or partial-adopted (row stays open) or no-ship — with the M4
  and Zen 4 numbers.
- `zig build test`; `zig build difftest`; canonical `run-compare-bench.sh` both hosts;
  `ReleaseSafe` / `ReleaseFast` green; `docs/parity-measurement.md` updated.
