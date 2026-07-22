<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 20a: Broad-harness residual — parity-measurement integrity

Discovered mid-implementation of [spec 20](20-lazy-or-construction-gap.md). **Diagnosis
only, no preselected cause** (same discipline as `20-00`, which caught a wrong premise by
measuring first). Its purpose is to protect the rest of the parity effort from chasing
phantom gaps.

## Why

`20-00` found lazy-OR construction at **2.19x** in the full `bench_croaring` harness but
only **1.71x** (default SMP) / **1.10x** (allocation-matched) in a focused single-op
executable — and the 2.19x did **not** reproduce even after priming each allocator with an
untimed construct+repair cycle. That unexplained delta was named the **broad-harness
context residual**. Its cause is unknown: the larger harness touches several large corpora
and runs allocation-heavy groups first, so candidates include allocator state, executable
code layout, or another cross-group interaction — but none is confirmed.

**The consequence for parity work:** every remaining target — skewed `andCardinality`
(1.43x), sparse AND (1.28x), sparse OR (1.15x) — was measured by the **same broad
harness**. If it inflated lazy-OR from ~1.10x-matched to 2.19x, it may be inflating those
too. We cannot trust broad-harness numbers for target selection until this is understood.
Re-establishing the *real* isolated gaps is the highest-value parity move right now.

## Deliverables

### 1. Reproduce and attribute the residual

Take one op with a known residual (lazy-OR construction) and measure it under controlled
harness conditions to isolate the cause — no favored hypothesis:

- **focused single-op executable** (the `20-00` baseline);
- **full broad harness** (reproduce the inflated number);
- **broad harness with the target group run first** vs **last**;
- **broad harness with other groups removed / order permuted**;
- **with and without a per-group allocator reset** (fresh allocator instance per group).

Candidate mechanisms to test (report attribution + a named residual, not a forced 100%):

- **allocator state** — SMP's internal state (fragmentation, warm/cold size-class pools)
  after earlier allocation-heavy groups changing later-group cost;
- **executable / code layout** — the same op in two different binaries (focused vs broad)
  placing code differently (icache / branch prediction);
- **cache / TLB interaction** — earlier large corpora evicting state the target op needs;
- **protocol differences** — warmup and run counts, group sequencing.

Compare rawr-SMP, rawr-libc, and CRoaring-libc so an allocator-state cause is separable
from a rawr-only layout/interaction cause.

### 2. Re-measure the parity board in isolation

For each remaining target — **skewed `andCardinality`, sparse AND, sparse OR** — report the
**real isolated ratio** in a focused executable, on **default SMP** and **allocation-matched
(libc)**, five independent process runs, median + range. This produces a **corrected parity
board**: which gaps survive isolation, and how much each shrinks versus its broad-harness
number.

### 3. Recommendation on the harness itself

State whether `bench_croaring`'s numbers can be trusted as-is or need a methodology fix
(e.g. per-group allocator reset, group isolation, or reporting isolated alongside broad),
so future parity target selection is not misled again.

## Methodology / constraints

- Same rig as `20-00`: `ReleaseFast`, native CPU, spec-16 M4 host; five independent process
  runs, median + range; env header recorded.
- Prefer isolated single-op executables and controlled variants over instrumenting the hot
  path; do not distort the workload to measure it.
- **Benchmark-only** — no public library behavior change, no committed vendored-source
  change. Findings, commands, and the corrected board are committed to a durable artifact
  (extend `docs/lazy-or-construction-analysis.md` or a sibling `docs/parity-measurement.md`),
  not left in ignored `misc/`.

## Acceptance / output

- The residual is attributed to a cause (allocator state / layout / cache / protocol) with
  bounds and a named residual — or explicitly recorded as not-yet-explained with the
  variants that were ruled out.
- A **corrected parity board** for the three remaining ops: isolated SMP and
  allocation-matched ratios, versus their broad-harness numbers.
- A clear read on which remaining gaps are **real and worth a spec** and which were largely
  harness artifact — so the next parity spec targets a confirmed gap.

## Estimate

M. Controlled harness variants, three ops re-measured in isolation across two allocators,
five-process runs, and the written corrected-board analysis.
