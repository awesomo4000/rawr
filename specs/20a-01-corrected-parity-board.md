<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 20a-01: Corrected parity board + harness verdict

Second chunk of [broad-harness residual](20a-broad-harness-residual.md). Re-establishes the
**real isolated** parity gaps for the remaining targets, and — informed by `20a-00`'s cause
— delivers a verdict on whether `bench_croaring`'s numbers can be trusted as-is.
**Benchmark-only.**

The board re-measurement is largely independent of `20a-00` and can run in parallel; the
**harness verdict** depends on `20a-00`'s attributed cause.

## Re-measure the parity board in isolation

For each remaining broad-harness target, report the **real isolated ratio** in a focused
executable — five independent process runs, median + range — versus its broad-harness number:

| target | broad-harness ratio | measure in isolation as |
|---|---:|---|
| sparse AND | 1.28x | **SMP** and **allocation-matched (libc)** — it allocates a result |
| sparse OR | 1.15x | **SMP** and **allocation-matched (libc)** — it allocates a result |
| skewed `andCardinality` | 1.43x | **one rawr number** vs CRoaring — it does **not allocate** in its timed op (SMP-vs-libc is meaningless; optionally confirm input-allocator-invariant timing) |

**Correctness first:** every focused operation is validated — result equal to production rawr
and logically to the CRoaring oracle — **before** its timing is accepted, so a
re-measurement is never of the wrong computation.

Output: a **corrected parity board** — each target's isolated SMP and (where meaningful)
allocation-matched ratio, beside its broad-harness number, showing which gaps survive
isolation and how much each shrinks.

## Harness verdict (uses `20a-00`)

Given `20a-00`'s attributed cause, state whether `bench_croaring`'s numbers can be trusted
as-is or need a methodology fix — e.g. per-op **process isolation**, running each group in a
**fresh process**, or **reporting isolated alongside broad** (not per-group allocator reset,
which `SmpAllocator` cannot do). The goal is that future parity target selection is not
misled by the residual again.

## Acceptance

- Corrected isolated ratios for **sparse AND, sparse OR** (SMP + allocation-matched) and
  **skewed `andCardinality`** (single rawr), five processes each, median + range, each beside
  its broad-harness number.
- Every re-measured op **correctness-validated** (production rawr + CRoaring oracle) before
  its timing counts.
- A written **read on which remaining gaps are real and worth a spec** and which were largely
  harness artifact.
- A concrete **`bench_croaring` methodology recommendation**, consistent with `20a-00`'s
  finding.
- Environment: `ReleaseFast`, native CPU, spec-16 M4 host; env header recorded.
- **Benchmark-only:** no public library behavior change; results committed to the durable
  artifact, not ignored `misc/`.

## Result to record (feeds the next parity spec)

The corrected parity board and the real-vs-artifact verdict per op — this decides which of
sparse AND / sparse OR / skewed `andCardinality`, if any, gets a real optimization spec, so
the next effort targets a confirmed gap rather than a measurement artifact.
