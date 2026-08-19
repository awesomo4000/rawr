<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 45-02: Two-host canonical measurement and verdict

Toplevel: [45-chunked-payload-arena.md](45-chunked-payload-arena.md).
Gated on: [45-01](45-01-arena-implementation.md) complete and green.

**No default change in this chunk either.** It produces measurements and a verdict. Adoption, if earned,
is a separate spec.

## 1. Protocol

- **Canonical harness only** (`run-compare-bench.sh`): fresh process per cell, 3 warmup / 21 timed, **≥5
  process medians with full ranges**. Never a focused harness — spec 35 read 1.155x where canonical read
  1.727x, from SMP preconditioning earlier in the same process.
- **Both hosts**, all three canonical tuples — rawr/SMP, rawr/libc, CRoaring/libc.
- **Candidate and baseline rows in one binary.** Cross-run ratios do not hold: spec 39-00 vs 39-01 moved
  rawr +4.1% while the CRoaring reference moved −7.8%.
- **Whole board** for spec-28 layout noise; sub-~1.2x M4 ratios sit at the measurement floor.

## 2. Timed boundaries

- **Construction row** — times wrapper `lazyOr` only; retained teardown **outside**, matching
  `bench_croaring.zig:507-512`.
- **Combined row** — times `lazyOr` + `repairAndTake`, including the single chunk-list sort and all
  survivor migration; retained teardown **outside**.

Temporary metadata release is **inside** timing; retained headers, payloads, and chunk list are torn down
**outside**.

## 3. Gates — both required

- **Construction: `lazy-or-construction-arena` ≤ 1.10x vs CRoaring on M4.** Primary.
- **Combined: `lazy-or-repair-arena` within 5% on median** of the baseline combined row, ranges
  considered. **Migration cost lands here** — spec 35 established that gating the aggregate alone
  authorizes work that buys nothing, and this is the mirror case: gating construction alone would hide
  cost pushed into repair.
- **Zen 4: `candidate / baseline ≤ 1.05`** on median, **both rows**, ranges considered.
- Overlapping ranges → **rerun**; still overlapping → **inconclusive → NO-GO**.

**libc: measured and reported on both rows, but NOT a gate** (toplevel §7.1, owner decision). A libc
regression does not stop this spec. It remains diagnostic signal — in spec 44 the libc regression was the
cleanest available measurement of machinery cost, because libc is order-insensitive and so pays overhead
while gaining nothing.

## 4. Memory — new for this spec, and measured OUT OF BAND

Chunking changes the allocation profile, so speed alone is not a sufficient verdict.

**Memory accounting must never contaminate gated timing.** A counting-allocator wrapper **must not**
replace SMP in any timed cell, and **must not run before one in the same process**. Prior specs measured
how badly allocator preconditioning moves this row: spec 35's focused harness read **1.155x** where
canonical read **1.727x**, purely because earlier passes had touched the same population.

**Therefore: collect memory figures in a separate fresh process, or in an explicitly untimed diagnostic
run.** Never in the same process as a gated cell.

- **Requested-byte high-water** during construction and during the combined cycle.
- **Post-repair live bytes.**
- **Tolerance ≤5%** against baseline for both.
- **Hard assertion: retained chunk bytes are zero after a successful repair.** If a chunk survives
  `repairAndTake`, the lifetime rule is broken regardless of what the timings say.

## 5. Reporting

Report, whatever the verdict:

- all rows, both hosts, all three tuples, medians and full ranges;
- **the chunk size selected** and the sweep that chose it (SMP medians, both hosts);
- **construction versus combined split** — how much of any construction win is given back in repair;
- memory figures per §4.

**If the gate fails, the split is still the durable result**: it says whether chunking is unviable, or
viable but defeated by migration — which are different findings pointing at different next steps.

## Acceptance

- Both candidate rows measured on both hosts, all three tuples, ≥5 fresh-process medians with full
  ranges, in **one binary**.
- Timed boundaries per §2.
- **Both §3 gates evaluated**; Zen 4 ratios stated numerically.
- libc reported on both rows and **explicitly excluded from the verdict**.
- **§4 memory reported**, including the zero-retained-chunk-bytes assertion.
- §5 reporting complete, including the construction/combined split.
- **GO / NO-GO stated**, with reasoning. **Default unchanged either way**; canonical board row unmoved.
- All four suites green — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
  `check-package`.
- **Outcome recorded on the umbrella (spec 31).** If the gate fails, record the measured result and the
  split; do **not** report it as closed, and do **not** use "at parity when enabled" — that phrasing
  belongs to opt-in outcomes like spec 39-01, and this spec has no opt-in path.

## Estimate

**S/M** — no new production logic; the two-host canonical run, memory instrumentation, and reporting are
the work.
