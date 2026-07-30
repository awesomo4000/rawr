<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 30-01: Adopt `removeRangeCopy` into the canonical row

Wire the winning fused shape from `30-00` into the canonical `remove-range` parity row (legitimate
copy-vs-copy), rename the row operation, hold the board gate, and make the keep/close decision.

Toplevel: [30-fused-remove-range-copy.md](30-fused-remove-range-copy.md).
Gated on: [30-00](30-00-fused-op-and-measure.md) (the three-cell attribution).

## Winner selection (architecture-neutral)

Choose **one** fused implementation that passes **both** the M4 and Zen 4 gates — not a per-host
shape. Each *lever* (doomed-skip fusion; exact pre-sizing) is included only if it helps without
regressing either host, yielding a single shipped shape. **If each host favors a different shape
and no single shape passes both gates, ship neither** — that requires a separate
architecture-specific design (out of scope), not an M4-only or Zen4-only binary.

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
- **Zen 4 presumption:** rawr is ahead (0.411x); the change is expected to help or hold. A Zen 4
  regression is the presumption against, **not an automatic veto** (see keep decision).

## Keep / close decision

- **Close (full GO):** M4 SMP **≤ 1.10x**, Zen 4 not regressed, board gate held (layout exception),
  row renamed → the **row closes** and the spec moves to `specs/done/`.
- **Adopt a partial (row stays open):** the winning shape **moves M4 down** and is cross-host-safe
  but lands **above 1.10x on M4** → `30-01` **may still ship it** and record the partial (as spec 29
  kept dense-AND 1.479x). **≤ 1.10x is required to *close*, not to *adopt*.** The residual (shared
  M4 SMP per-container-clone cost) keeps the row **open** for the next lever.
- **Human keep/not-keep judgement call:** the numeric gates **inform** the decision, they do not
  automate it. A **large M4 win against a marginal cross-host cost** (e.g. a ~1% Zen 4 slip) is a
  tradeoff the owner may accept at review. The final call is made on the numbers at hand, gated only
  on **no future avenue foreclosed** (in-place `removeRange`, clone, and dense-AND levers stay
  available).
- **Ship nothing** only if the fused shape fails to improve M4, or regresses a host / the board gate
  beyond what review accepts.

## Acceptance

- Winning fused shape adopted into the `remove-range` row (or a reasoned no-ship recorded), stable
  `row_id` preserved, operation relabeled `removeRangeCopy`, timing boundary and copy-vs-copy
  legitimacy intact.
- Board gate held (layout exception); Zen 4 handled per the keep decision.
- Outcome recorded — closed (≤ 1.10x) or partial-adopted (row stays open) or no-ship — with the M4
  and Zen 4 numbers.
- `zig build test`; `zig build difftest`; canonical `run-compare-bench.sh` both hosts;
  `ReleaseSafe` / `ReleaseFast` green; `docs/parity-measurement.md` updated.
