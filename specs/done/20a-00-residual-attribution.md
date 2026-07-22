<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 20a-00: Reproduce and attribute the broad-harness residual

First chunk of [broad-harness residual](20a-broad-harness-residual.md). **Diagnosis only,
no preselected cause, benchmark-only.** Deliverable: the broad-harness residual attributed
to a cause (or explicitly recorded as not-yet-explained, with the variants that ruled
candidates out).

## Anchored baselines (do not depend on branch/ignored state)

`main` no longer has the three-column diagnostic harness; reproduce and compare against
these fixed numbers. Lazy-OR construction, M4:

| variant | broad harness | focused (`20-00`) | broad penalty (broad − focused) |
|---|---:|---:|---:|
| rawr, libc (full) | 7.574 ms | 3.960 ms | **3.614 ms** (largest) |
| rawr, SMP (full) | 8.375 ms | 6.162 ms | 2.213 ms |
| CRoaring, libc | 3.832 ms | 3.601 ms | 0.231 ms |

Broad source: branch `bench-experiments-17-18`, commit `0599cae`,
`misc/bench-croaring-20260721-142207-summary.txt`. Focused source: commit `f27e223`.

The residual is the asymmetry: the broad harness penalizes rawr heavily and CRoaring barely,
and penalizes rawr-**libc** *more* than rawr-SMP. Note the caveat — rawr-libc routes only
**result** allocation through libc in the **same process** with SMP-built inputs and
SMP-backed surrounding work, so this argues against SMP *result* allocation as the sole
cause but does **not** rule out cross-group SMP state.

## The harness matrix

Measure lazy-OR construction under these conditions, **one condition per fresh process**
(Zig's `SmpAllocator` is a global singleton with no reset; libc has no portable reset either
— process isolation is the only clean control). Five processes per condition, median + range,
each across **rawr-SMP, rawr-libc, and CRoaring-libc**:

1. **focused single-op executable** (the `20-00` baseline);
2. **broad binary, target-only data init, running only the target group** — via a **runtime
   group selector** so **all** broad-harness functions stay **linked** (guard against
   dead-code elimination changing the code layout under test);
3. **broad binary, full data initialization, target group first**;
4. **broad binary, full data initialization, target group last**;
5. **protocol swap** — the same op under `2` warmup / `9` timed vs `3` warmup / `21` timed,
   **both run under the same condition (use condition 2)** so the protocol effect is not
   confounded with binary layout or data initialization.

These separate the four candidate causes: **code layout** (2 vs 1), **unrelated data
initialization** (3 vs 2), **execution history** (4 vs 3), **timing protocol** (the two arms
of 5).

**Follow-up discriminator (required if the 4-vs-3 execution-history delta is material).**
"Material" = the 4-vs-3 **median changes by ≥ 10%** and the two conditions' five-process
**ranges do not substantially overlap**. That delta cannot on its own separate **allocator
state** from **cache/TLB pollution**, so when material add two minimal priming variants and
compare their effect on the target: an **allocator-only prime** (drive the allocator to the
prior groups' state without retaining/reading their data) and a **cache-touch-only prime**
(walk the prior corpora without allocating).

## Acceptance

- All five conditions measured across the three variants, five processes each, **absolute
  medians + ranges** reported; the broad numbers reproduce the anchored baselines within
  their ranges (or the discrepancy is itself explained).
- The residual **attributed to a cause** — code layout / unrelated data init / execution
  history / timing protocol — with bounds and a **named residual**, not a forced 100%. If
  execution history is implicated, the allocator-vs-cache discriminator is run.
- Correctness: any instrumented/replica path validated against production rawr after repair
  and the CRoaring oracle before its timing is accepted.
- Environment: `ReleaseFast`, native CPU, spec-16 M4 host; env header recorded.
- **Benchmark-only:** no public library behavior change, no committed vendored-source
  change; full build green under `ReleaseSafe` and `ReleaseFast`.
- Findings + exact commands committed to a durable artifact (`docs/lazy-or-construction-
  analysis.md` or a sibling `docs/parity-measurement.md`), not left in ignored `misc/`.

## Result to record

The named cause of the residual (or the ruled-out set) — this informs `20a-01`'s
`bench_croaring` methodology verdict and tells us how much to distrust broad-harness numbers
in general.
