<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 26-03: Cross-host gate, strategy selection, legacy disposition

Fourth chunk of [direct range ops](26-direct-range-ops.md). Measures the direct paths on both
hosts, selects the shipped strategy per the parent's preference order, settles the legacy
code's fate, and updates the docs.

## Gate

- `26-01` and `26-02` complete (direct paths byte-equal, OOM-covered, behind the seam).

## Baseline of record

Per-host canonical tables in `docs/parity-measurement.md` at commit `190f6d4`:
removeRange **2.167x M4 / 1.078x Zen 4**; flip **1.767x M4 / 0.565x Zen 4** (SMP).

## Deliverables

1. **Measure** `flip` and `removeRange` (direct vs legacy) on the canonical harness, both
   hosts, five fresh processes, median + full range.
2. **Select per the preference order**, per op:
   - single **direct** implementation if neutral-or-better on both hosts (≤ 5% counts as noise,
     rerun on range overlap);
   - **comptime per-arch selection** if direct wins M4 but loses Zen 4 — selector: `aarch64` →
     direct, `x86_64` → legacy, other arches → documented explicit choice (default direct);
     both arms stay tested via the strategy override;
   - direct loses both hosts (unexpected) → keep legacy, record the result.
3. **Legacy disposition** follows the selection: single-direct → remove legacy, preserve the
   byte-equality contract via **pinned serialization fixtures** (or a test-only copy);
   per-arch → both retained and tested. Either way `26-00`'s harness outcome is recorded.
4. **Docs:** update `docs/parity-measurement.md` (new canonical rows + the decision), and note
   the strategy flag if per-arch shipped.

## Acceptance (GO)

- removeRange and flip **≤ 1.10x on M4** — or a statistically supported improvement retained
  with rationale — with **Zen 4 within noise of baseline** (≤ 5% per row, rerun on overlap; an
  M4 win is never bought with an x86 loss), and **no other canonical row worsening > 5%** vs
  the baseline of record.
- Allocation counts confirm the composition allocations (whole-clone + mask bitmaps) are gone
  on the shipped path(s).
- Selection + legacy disposition executed and documented; if per-arch shipped, both arms green
  on the full `26-00` matrix via the override.
- `zig build test`; `zig build difftest`; canonical `run-compare-bench.sh` on both hosts;
  `ReleaseSafe` / `ReleaseFast` green.

## Checklist

- [ ] Direct-vs-legacy measured on both hosts (canonical harness, 5 processes, median + range)
- [ ] Strategy selected per preference order, per op; selector documented
- [ ] Legacy disposition executed (fixtures / test-only / retained-per-arch)
- [ ] M4 gate met (≤ 1.10x or supported improvement); Zen 4 within noise; no row > 5% worse
- [ ] Allocation collapse confirmed on shipped path(s)
- [ ] `docs/parity-measurement.md` updated
- [ ] test / difftest / both-host canonical run / ReleaseSafe / ReleaseFast green
