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
- **All five arms in one binary.** Spec 43 established cross-run ratios do not hold (39-00 vs 39-01: rawr
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
| **Slotted vehicle delta** | arm 4 − arm 3 |
| **Fusion recovery** | arm 5 − arm 4 |
| **Net result** | arm 5 − arm 1 |

**Report all five, on both hosts, whatever the verdict.**

**Scope each quantity to what it actually measures:**

- **`arm5 − arm4` measures fusion WITHIN the slotted vehicle.** Arms 4 and 5 are identical except for
  pass structure, so this is a clean causal claim. `arm5 − arm3` would bundle fusion with the vehicle and
  is **not** a fusion measurement.
- **`arm5 − arm4` does NOT decompose the historical `arm2 − arm1` (+1.544 ms).** That penalty arose under
  different metadata, traversal, and assembly conditions — different vehicle, different experiment.
  *(An earlier draft claimed arms 4 and 5 "split the 1.544 ms". They do not; do not report it that way.)*
- **`arm4 − arm3` is the slotted vehicle delta** — metadata, direct-slot assembly, reserved handling
  **and the change in source traversal order** together, not a metadata/slot cost in isolation.

The durable result **even on a NO-GO** is whether fusion removes a real cost inside the slotted vehicle,
which tells the campaign whether a future lever should target cache behaviour or machinery.

## 4. Source-traversal reporting

From the real canonical sparse corpus: source container **counts** and **types**, **bytes actually read**,
and **source-address travel in key order versus destination order**.

**Scope: eligible matched pairs ONLY.** Unmatched clones and non-eligible unions are not reordered by the
candidate — they occur in key order in every arm — so including them would dilute the totals with traffic
the experiment does not affect.

**Payload addresses, type-specific** — never `TaggedPtr` or header addresses:

| Container | Address |
| --- | --- |
| array | `values.ptr` |
| bitset | `words` |
| run | `runs` |

**Report SIX travel totals** — three sequences × two orders:

| Sequence | key order | destination order |
| --- | --- | --- |
| **A stream** alone | ✓ | ✓ |
| **B stream** alone | ✓ | ✓ |
| **interleaved `A,B`** as accumulation performs it | ✓ | ✓ |

Accumulate deltas in **`u128`** — summed absolute deltas over ~16k containers can exceed `u64`.

**"Bytes actually read" means live payload bytes, not capacity:**

| Container | Bytes |
| --- | --- |
| array | `cardinality * 2` (`values: []align(32) u16`) |
| bitset | `8192` (1024 × `u64`) |
| run | `n_runs * @sizeOf(RunPair)` (`RunPair` = two `u16`) |

Capacity-based figures would overstate reads for any container with slack.

**Collect AFTER all timed runs, or in a separate diagnostic process.** Walking every source container to
compute travel would **precondition the source caches** and corrupt the measurement this spec depends on —
the same contamination class as spec 35's warmed-context artifact.

The travel comparison is load-bearing: types and byte totals alone cannot show whether destination
sorting made source traversal pathological. Spec 38 found read traversal wants ascending (M4 1.221x,
Zen 4 1.344x) **on large bitsets**; if these sources are mostly small arrays that may not transfer.

## 5. Gate

- **Arm 5 beats arm 4 with non-overlapping ranges** — fusion removes a measurable part of the penalty.
  Arms 4 and 5 differ only by pass structure, so this is a clean causal claim.
- **Arm 5 reaches ≤1.10x vs CRoaring on M4.**
- **libc does not regress — arm 5 vs arm 1, rawr/libc, same binary, ≤5% on median, ranges considered.**
  A libc regression is a **STOP** (spec 43 measured +90%).
- **Zen 4 does not regress — arm 5 vs arm 1, ≤5% on median, ranges considered.** A Zen 4 regression is a
  NO-GO on its own.
- Overlapping ranges → **rerun**; still overlapping → **inconclusive → NO-GO**, never a marginal pass.

## Acceptance

- All **five** arms measured on **both hosts**, all three canonical tuples, ≥5 fresh-process medians with
  full ranges, in **one binary**.
- Timed region per §2; only result teardown outside.
- **§3 decomposition reported on both hosts** — batching cost, ordering recovery, **slotted vehicle
  delta**, **fusion recovery**, net. Fusion recovery derived from `arm5 − arm4`, never `arm5 − arm3`, and
  **not reported as a share of the historical 1.544 ms**.
- **§4 source-traversal reporting complete**: payload addresses per type, three sequences, `u128`
  accumulator, **collected after the timed runs or in a separate process**.
- Gate §5 evaluated; **GO/NO-GO stated**, with the reasoning.
- Whole board checked for layout movement beyond the 5% tolerance.
- **Default unchanged regardless of outcome**; canonical board row unmoved.
- All four suites green — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
  `check-package`.
- **Outcome recorded on the umbrella (spec 31).** If the gate fails, record the measured result and the
  decomposition; do **not** report it as closed, and do **not** use "at parity when enabled" — that
  phrasing belongs to opt-in outcomes like spec 39-01, and this spec has no opt-in path.

## Outcome — NO-GO for adoption. Fusion CONFIRMED. Machinery is now the binding constraint.

**Absolute medians (the deltas below are adjacent-arm differences, not timings):**

| Arm | M4 | Zen 4 |
|---|---:|---:|
| A1 baseline | 5.744 ms | 20.749 ms |
| A2 batched | 7.274 ms | 23.084 ms |
| A3 sorted | 5.063 ms | 18.816 ms |
| A4 slotted | 5.491 ms | 19.605 ms |
| **A5 fused slotted** | **4.179 ms** | **18.570 ms** |

| Decomposition | M4 | Zen 4 |
|---|---:|---:|
| Batching (A2−A1) | +1.530 | +2.335 |
| Ordering (A3−A2) | **−2.211** | **−4.268** |
| Slotted vehicle (A4−A3) | +0.428 | +0.789 |
| **Fusion (A5−A4)** | **−1.312** | **−1.035** |
| Net (A5−A1) | −1.565 | −2.179 |

Re-derived from the absolutes; all five agree.

**A5 is the fastest construction path measured, on both hosts.** But M4 lands at **1.235x** against a
1.10x gate, and **M4 libc regressed +21.2%** (3.787 → 4.591) — an independent STOP. Zen 4 improved and
passed its no-regression gate.

### The three residuals are the same quantity

| Quantity | M4 |
|---|---:|
| Total machinery — batching + slotted + fusion (`A5 − A3` plus the batching cost) | **+0.646 ms** |
| Residual gap to CRoaring (implied 3.384 ms) | **+0.795 ms** |
| libc regression | **+0.804 ms** |

libc is **order-insensitive** (spec 37: 0.011–0.073 ms sorted vs unsorted), so it pays the full machinery
and receives none of the ordering benefit. Its regression is therefore a **direct measurement of the
machinery cost** — and it lands within 0.16 ms of the SMP residual gap.

**Arithmetic, not a measurement:** 4.179 − 0.646 = 3.533 ms = **1.044x**, inside the gate. Treat that as
hypothesis-generating only — subtracting a cost is not the same as removing it. But it says plainly that
**machinery, not ordering, is now what stands between this path and the row.**

### Ordering is worth more than it costs; the vehicle is what fails

Ordering: **−2.211 ms M4, −4.268 ms Zen 4** — the largest single effect measured in this campaign.
Fusion recovers **−1.312 ms** of the batching penalty, confirming the two-pass structure was most of it
*(within the slotted vehicle — this does not causally decompose the historical +1.544 ms)*.

### Source-read order is NOT the limiting factor — spec 38's concern does not transfer

All **32,728 operands were small arrays**, totalling **499,994 live bytes**, and destination ordering
**barely changed** interleaved source travel. Spec 38's read-order penalty was measured on large bitsets
and **does not apply here**. This risk is closed for this path; do not re-raise it.

*(Context, not a lever: 16,364 × 8 KB = **134 MB zeroed to hold 0.5 MB live**, a 268:1 ratio. CRoaring
does the same — spec 31 established identical zero volume — so it is not a parity lever.)*

**Opt-in would not rescue this.** Unlike spec 39-01, the M4 SMP number **fails the gate even when
enabled** (1.235x). There is no "at parity when enabled" outcome available here.

## Estimate

**S/M** — no new production logic; the two-host canonical run and the reporting are the work.
