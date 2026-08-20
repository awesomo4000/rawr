<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 46-01: Measure, decide, and close the campaign

Toplevel: [46-adopt-fused-slotted-construction.md](46-adopt-fused-slotted-construction.md).
Gated on: [46-00](46-00-productize-arm5.md) complete and green.

This chunk decides whether the adoption stands, and — if it does — closes the parity campaign.

## 1. The number is NOT inherited

**1.235x was measured in a five-arm binary on unpushed work. It does not carry over.** Spec 28
established that adding or removing code moves untouched rows — including CRoaring's — with
instruction-identical disassembly. `46-00` deletes four arms and an export; that is exactly such a
change.

**The accepted residual is whatever this chunk measures**, recorded with its date. 1.235x is the
expectation, not the guarantee.

## 2. Protocol

- **Canonical harness only** (`run-compare-bench.sh`): fresh process per cell, 3 warmup / 21 timed, **≥5
  process medians with full ranges**. Never a focused harness — spec 35 read 1.155x where canonical read
  1.727x, from SMP preconditioning earlier in the same process.
- **Both hosts**, all three canonical tuples — rawr/SMP, rawr/libc, CRoaring/libc.
- **Whole board.**
- Report `lazy-or-construction`, `lazyOr+repair`, **and both retained baseline rows**.

## 3. Thresholds

- **M4 SMP construction ≤ 1.30x**, and **materially better than `lazy-or-construction-baseline`** with
  **non-overlapping ranges**, same binary.
- If construction lands worse than ~1.235x by more than measurement noise: **stop and investigate before
  committing.** That would mean the cleanup cost something the diagnostic build was hiding, and shipping
  it without understanding why would bake in a regression nobody chose.
- **Combined `lazyOr+repair` within 5% on median** of `lazy-or-repair-baseline`, same binary.
- **Zen 4: `candidate / baseline ≤ 1.05`** on both rows.
- **Untouched rows: `>5%` triggers a RERUN and targeted inspection, not immediate failure.** The
  comparison is across different binaries and spec 28 layout drift is expected. **Only a repeated,
  attributable regression fails adoption.**
- **libc: reported, not gated** (owner policy). Record the final figure; ~+21.2% is accepted.
- Overlapping ranges on a gated comparison → rerun; still overlapping → **inconclusive → rollback**.

## 4. Rollback

If §3 fails, **restore the previous default**, keep the retained baseline rows or revert the manifest to
40, and record the measured result. **Adoption is not automatic because the diagnostic looked good.**

## 5. Documentation — decided

**Shipped `README.md` and `API.md` unchanged.** Spec 41 removed every performance claim from them one
spec ago; reversing that immediately would be incoherent.

**Record in `docs/parity-measurement.md`** (repo-only, not in `.paths`):

- the **accepted residual** with its measured value and date;
- the final **libc** figure and the owner acceptance;
- the campaign summary and the closed families.

## 6. Language — binding

- **"Accepted residual: N.NNNx on M4"**, with the date.
- **Never** "parity", "row closed", "at parity", or "at parity when enabled" — that phrasing belongs to
  opt-in outcomes like spec 39-01; this is a **default** change carrying a residual.
- The row is **open with an accepted residual**, which is a distinct state from closed. That distinction
  must survive into the umbrella, or in six months it reads as parity.

## 7. Campaign closure — update spec 31

- `lazy-or-construction`: **open with accepted residual**, value and date;
- every other material row: at or under gate;
- **closed families**, with their spec numbers: allocator replacement (18), transient arenas (17),
  header elimination (35), first-touch/residency (36), read-traversal sorting (38), payload-address
  sorting (43), slotted+fused machinery beyond this adoption (44), **per-operation** chunk allocation
  (45);
- **the standing finding**: ordering is worth ~47% of construction time on M4, and **every tested
  vehicle** for obtaining it either regressed or left a residual above the former gate. Any future
  proposal must state how it avoids that.

**Scope the closures precisely.** Spec 45 closes **per-operation chunk allocation** — it does **not**
close a persistent pool amortized across calls, an allocator change, or an upstream Zig `SmpAllocator`
fix. Those are unexplored, and spec 45's data does not speak to them. Overstating this would foreclose
directions the evidence never tested.

## Acceptance

- Both candidate rows and both baseline rows measured, both hosts, all three tuples, ≥5 fresh-process
  medians with full ranges, **in one binary**.
- §3 thresholds evaluated; Zen 4 ratios stated numerically; untouched-row drift handled per the rerun
  procedure.
- libc reported and **explicitly excluded from the decision**.
- **Adopt or roll back, stated with reasoning.**
- On adoption: `docs/parity-measurement.md` updated per §5; shipped docs untouched; language per §6.
- **Spec 31 updated per §7**, including precisely scoped closures.
- All four suites — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
  `check-package`.

## Estimate

**S/M** — no new production logic; the two-host full-board run and the campaign record are the work.
