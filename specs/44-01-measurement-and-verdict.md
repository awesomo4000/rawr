<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 44-01: Two-host canonical measurement, decomposition, and verdict

Toplevel: [44-fused-address-ordered-construction.md](44-fused-address-ordered-construction.md).
Gated on: [44-00](44-00-fused-implementation.md) complete and green.

**No default change in this chunk either.** It produces a decomposition and a verdict. Adoption, if
earned, is a separate spec.

## 1. Measurement protocol

- **Canonical harness only** (`run-compare-bench.sh`): fresh process per cell, 3 warmup / 21 timed, **≥5
  process medians with full ranges**. Never a focused harness — spec 35 read 1.155x where canonical read
  1.727x, purely from SMP preconditioning earlier in the same process.
- **Both hosts.** All three canonical tuples — rawr/SMP, rawr/libc, CRoaring/libc.
- **All four arms in one binary.** Spec 43 established cross-run ratios do not hold (39-00 vs 39-01: rawr
  moved +4.1% while the CRoaring reference moved −7.8%).
- **Whole board** for spec-28 layout noise; sub-~1.2x M4 ratios sit at the measurement floor.

## 2. Timed region

The **complete candidate** is inside: eligible pre-pass, metadata construction, scratch allocation, sort,
zeroing, accumulation, slot assembly, reserved-slot verification, **scratch release**.

**Outside: result teardown only** — matching the canonical construction row, which calls `result.deinit()`
after the clock stops (`bench_croaring.zig:507-512`).

## 3. The decomposition — the durable deliverable

| Quantity | Comparison |
| --- | --- |
| **Batching machinery cost** | arm 2 − arm 1 |
| **Ordering recovery** | arm 3 − arm 2 |
| **Fusion recovery** | arm 4 − arm 3 |
| **Net result** | arm 4 − arm 1 |

**Report all four, on both hosts, whatever the verdict.** Spec 43 measured the batching penalty at
+1.544 ms on M4 but could not say how much was the cold second pass versus pre-pass, scratch, and
deferred assembly. **This chunk answers that**: fusion recovery is the share attributable to the second
pass; whatever remains is machinery.

That answer is the durable result **even on a NO-GO** — it determines whether any future lever should
target cache behaviour or the machinery itself.

## 4. Source-traversal reporting

Report, from the real canonical sparse corpus:

- source container **counts** and **types**;
- **bytes actually read** from sources;
- **source-address travel in key order versus destination order** — sum of absolute address deltas along
  each traversal.

The travel comparison is the load-bearing number. Container types and byte totals alone **cannot** show
whether destination sorting made source traversal pathological, which is this design's main risk. Spec 38
found read traversal wants ascending (M4 1.221x, Zen 4 1.344x) **on large bitsets**; if these sources are
mostly small arrays that may not transfer.

This is what makes a disappointing arm 4 interpretable rather than mysterious.

## 5. Gate

- **Arm 4 beats arm 3 with non-overlapping ranges** — fusion removes a measurable part of the penalty.
- **Arm 4 reaches ≤1.10x vs CRoaring on M4.**
- **libc does not regress — arm 4 vs arm 1, rawr/libc, same binary, ≤5% on median, ranges considered.**
  A libc regression is a **STOP** (spec 43 measured +90%).
- **Zen 4 does not regress — arm 4 vs arm 1, ≤5% on median, ranges considered.** A Zen 4 regression is a
  NO-GO on its own.
- Overlapping ranges → **rerun**; still overlapping → **inconclusive → NO-GO**, never a marginal pass.

## Acceptance

- All four arms measured on **both hosts**, all three canonical tuples, ≥5 fresh-process medians with
  full ranges, in **one binary**.
- Timed region per §2; only result teardown outside.
- **§3 decomposition reported on both hosts** — batching cost, ordering recovery, **fusion recovery**, net
  — with fusion's share of the 1.544 ms stated explicitly.
- **§4 source-traversal reporting complete**, including key-order versus destination-order travel.
- Gate §5 evaluated; **GO/NO-GO stated**, with the reasoning.
- Whole board checked for layout movement beyond the 5% tolerance.
- **Default unchanged regardless of outcome**; canonical board row unmoved.
- All four suites green — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
  `check-package`.
- **Outcome recorded on the umbrella (spec 31).** If the gate fails, record the measured result and the
  decomposition; do **not** report it as closed, and do **not** use "at parity when enabled" — that
  phrasing belongs to opt-in outcomes like spec 39-01, and this spec has no opt-in path.

## Estimate

**S/M** — no new production logic; the two-host canonical run and the reporting are the work.
