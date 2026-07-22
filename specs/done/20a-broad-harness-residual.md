<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 20a: Broad-harness residual — parity-measurement integrity

Discovered mid-implementation of [spec 20](20-lazy-or-construction-gap.md). **Diagnosis
only, no preselected cause** (same discipline as `20-00`, which caught a wrong premise by
measuring first). Its purpose is to protect the rest of the parity effort from chasing
phantom gaps.

> **Outcome (2026-07-22) — cause found; two of three "gaps" were artifacts.** The matrix
> attributed the residual cleanly: **code layout** ruled out (focused ≈ broad-target-only),
> **unrelated resident data** ruled out (full-init-target-first neutral), **execution
> history** confirmed (lazy OR moved after the allocation-heavy groups went rawr-SMP 4.340 →
> 7.909 ms; libc and CRoaring unchanged), and the discriminator pinned it to **allocator
> state, not cache** (allocator-only prime reproduced part of the penalty; cache-touch-only
> had none). Plus **warmup-dependent allocator state** (2/9 → 3/21 dropped rawr-SMP 6.046 →
> 4.438 ms; libc/CR flat). So `bench_croaring`'s rawr-SMP rows measured *operation cost +
> process-global `smp_allocator` history + warmup-dependent allocator state*; CRoaring (libc)
> escaped it, making the broad ratios structurally asymmetric.
>
> **Corrected isolated board (fresh process), rawr-SMP / CRoaring:**
> - Sparse AND: broad **1.28x → 0.91x** (rawr faster) — artifact.
> - Sparse OR: broad **1.15x → 0.75x** (rawr faster) — artifact.
> - Skewed `andCardinality`: broad **1.43x → 1.46x** — **real** (no allocation → no history
>   penalty; a genuine algorithm/kernel gap).
>
> Allocation-matched (libc) real gaps: sparse AND 1.42x, sparse OR 1.36x — valid **only** as
> explicit libc-matched targets, not as evidence default rawr usage is slower (it is faster).
>
> **Verdict:** keep `bench_croaring` as a broad regression dashboard, but **confirm
> optimization targets in fresh-process focused executables**, and **report rawr-SMP and
> rawr-libc separately** for allocating ops. **The one remaining real parity target is skewed
> `andCardinality`** — not sparse AND or OR.

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

## Anchored baselines (do not depend on branch/ignored state)

`main` no longer contains the three-column diagnostic harness. The numbers `20a` must
reproduce and compare against are fixed here so implementation does not rely on ignored
files or branch state. Lazy-OR construction, M4:

| variant | broad harness | focused (`20-00`) |
|---|---:|---:|
| rawr, SMP (full) | 8.375 ms | 6.162 ms |
| rawr, libc (full) | 7.574 ms | 3.960 ms |
| CRoaring, libc | 3.832 ms | 3.601 ms |

Broad source: branch `bench-experiments-17-18`, commit `0599cae`,
`misc/bench-croaring-20260721-142207-summary.txt`. Focused source: commit `f27e223`.

The residual to explain is the **asymmetry** in each variant's broad-harness penalty
(broad − focused):

- rawr-libc: 7.574 − 3.960 = **3.614 ms** — the **largest** signal;
- rawr-SMP: 8.375 − 6.162 = **2.213 ms**;
- CRoaring-libc: 3.832 − 3.601 = **0.231 ms**.

So the broad harness hammers rawr and barely touches CRoaring — and it penalizes
rawr-**libc** *more* than rawr-SMP. That ordering argues against SMP **result** allocation
being the sole cause: the rawr-libc timed construction routes result allocations through
libc yet takes the larger penalty. But it is the **same process**, its input bitmaps are
built with SMP, and it follows SMP-backed benchmark work — so cross-group SMP state is **not**
fully ruled out. This is exactly why the matrix must separate layout / data-init /
execution-history / protocol rather than assume an allocator cause. The 2.19x is the broad rawr-SMP/CRoaring
ratio (8.375/3.832); focused it is 1.71x (6.162/3.601), 1.10x allocation-matched.

## Deliverables

### 1. Reproduce and attribute the residual — a precise harness matrix

Measure lazy-OR construction under these controlled conditions, **one condition per fresh
process** (Zig's `SmpAllocator` is a global singleton with no reset, and libc has no
portable reset either — process isolation, not per-group allocator reset, is the only clean
control). Five processes per condition, median + range:

1. **focused single-op executable** (the `20-00` baseline);
2. **broad binary, target-only data initialization, running only the target group** — via a
   **runtime group selector** so **all** broad-harness functions stay **linked** (guard
   against dead-code elimination changing the very code layout under test); isolates *code
   layout* from *execution history*;
3. **broad binary, full data initialization, target group run first**;
4. **broad binary, full data initialization, target group run last** (isolates *execution
   history* / prior-group state from *unrelated data initialization*);
5. **protocol swap** — the same op under `2` warmup / `9` timed vs `3` warmup / `21` timed
   (isolates *timing protocol* from everything else).

That matrix separates the four candidate causes cleanly — **code layout** (2 vs 1),
**unrelated data initialization** (3 vs 2), **execution history** (4 vs 3), and **timing
protocol** (5). Run each across rawr-SMP, rawr-libc, and CRoaring-libc so an allocator-state
cause (SMP fragmentation / warm-cold size-class pools after earlier allocation-heavy groups)
is separable from a rawr-only layout/interaction cause. Report attribution with bounds and a
**named residual** — not a forced 100%.

**Follow-up discriminator (required if execution history is material).** Conditions 3 vs 4
detect prior-group influence but cannot on their own separate **allocator state** from
**cache/TLB pollution**. If the 3-vs-4 delta is material, add two minimal priming variants
that isolate each: an **allocator-only prime** (drive the allocator to the prior groups'
state without retaining/reading their data) and a **cache-touch-only prime** (walk the prior
corpora without allocating), and compare their effect on the target.

### 2. Re-measure the parity board in isolation

Report the **real isolated ratio** for each remaining target in a focused executable, five
independent process runs, median + range, versus its broad-harness number:

- **sparse AND** and **sparse OR** — both **default SMP** and **allocation-matched (libc)**
  (they allocate a result, so the allocator matters).
- **skewed `andCardinality`** — **one rawr result**; it does **not allocate** in its timed
  operation, so SMP-vs-libc is meaningless. Optionally confirm that inputs built with either
  allocator produce equivalent timing, but report a single rawr number vs CRoaring.

**Correctness first:** every focused/isolated operation must be validated — result equal to
production rawr and logically to the CRoaring oracle — **before** its timing is accepted, so
a re-measurement is never of the wrong computation.

Output: a **corrected parity board** — which gaps survive isolation and how much each shrinks
versus its broad-harness number.

### 3. Recommendation on the harness itself

State whether `bench_croaring`'s numbers can be trusted as-is or need a methodology fix
(e.g. per-op process isolation, running each group in a fresh process, or reporting isolated
alongside broad — not per-group allocator reset, which `SmpAllocator` cannot do), so future
parity target selection is not misled again.

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
