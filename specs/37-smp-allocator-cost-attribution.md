<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 37: SMP allocator cost attribution — call time vs induced memory layout

Campaign: [31-structural-parity-campaign.md](31-structural-parity-campaign.md). **Diagnosis only —
no production change, no lever design.** Answer one narrow question:

> The lazy-OR construction gap is **~85% allocator-localized**. Is that cost **inside allocator calls**
> (per-call bookkeeping) or **inside otherwise-identical work operating on allocator-supplied
> addresses** (locality / TLB / cache — i.e. allocator-*induced* layout)?

## What is already settled (do not re-measure)

| finding | source |
|---|---|
| **Zeroing volume identical** — one 8 KB zero per matched pair on both sides; no `calloc` trick, no deferral, no lazy-path container reuse | vendored `roaring.c` read |
| **Zeroing instructions identical** — both emit `mov w1, #0x2000` → `bl _bzero`, same libc stub | `bench_lazy_or_residency` disassembly |
| **Page faults are not the mechanism** — 40 operation faults, 100% page reuse, conditioning gave no material gain | spec 36 |
| **Alignment is not the cause** — rawr/libc requests the same 64-byte alignment and still recovers ~85% | allocator A/B below |

**Explicitly NOT settled:** whether time is spent *inside* `bzero`. Identical instructions on
**differently-located memory** can differ in elapsed time. That is one of the two branches here.

## Anchor evidence (same canonical run)

| variant | construction | alignment |
|---|---:|---|
| rawr/**SMP** | **5.746 ms** | 64 B |
| rawr/**libc** | **3.805 ms** | 64 B |
| CRoaring | **3.456 ms** | 32 B |

Gap **2.290 ms**; swapping **only the allocator** recovers **1.941 ms (~85%)**, residual rawr/libc
**1.101x**. **Recompute the share within whichever single run this spec uses** — CRoaring has read
3.336–3.456 across runs, so the denominator (and thus the share, ~80–85%) is run-dependent. Never mix
runs.

## Phase 1 — sampling attribution on the canonical row (rawr/SMP vs rawr/libc)

**The A/B is the same binary with only the allocator selected at runtime** — identical code, identical
layout, so any difference is the allocator or what it returns. Do not compare across builds.

**Attribution buckets** (symbol-level; these symbols exist in the built binary):

| bucket | symbols |
|---|---|
| **1. SMP allocation machinery** | `heap.SmpAllocator.alloc`, `heap.SmpAllocator.free` |
| **1b. page-mapping fallthrough** | `heap.PageAllocator.map`, `.unmap`, `.alloc`, `.free` — **report separately** |
| **2. zeroing** | the `_bzero` stub |
| **3. accumulation + assembly** | `bitmap.RoaringBitmap.lazyAccumulateIntoBitset*`, container `setList` / `lazyUnionWith` / `setRange`, top-level append/`ensureTotalCapacity`, `memcpy` (clone) |
| **4. everything else** | remainder, reported so buckets sum to 100% |

**Bucket 1b matters:** if SMP is falling through to `PageAllocator.map` (mmap) per allocation, that is
**syscall** cost, a different finding from freelist bookkeeping. **Spec 36's 40-fault result argues
against** per-allocation fresh mappings (fresh mmap pages would fault on touch), so a hot 1b would be
surprising and highly informative; a cold 1b confirms allocations are served from SMP's own free lists.

**Profiling protocol:**

- **Fresh process, canonical corpus, canonical init order, one implementation per process** — the
  spec-35 rule stands: **never profile in a warmed harness.**
- **Profiling runs are for ATTRIBUTION ONLY and are NOT canonical timing numbers.** Sample counts
  require far more work than 21 × ~5.7 ms yields, so the profiled run **may use more iterations
  and/or a higher sample rate** — and its elapsed times **must not be reported as row values**.
  Canonical timing stays with the canonical worker.
- State the tool and rate: Darwin **Instruments/`xctrace` Time Profiler** (or `sample`); Linux
  **`perf record`**. **Zig and libc frames must symbolize** — say how (frame pointers, `-fno-omit-frame-pointer`
  equivalent, dSYM) and report unsymbolized sample fraction.
- Report **absolute sampled time per bucket** for both variants, plus **Δ(SMP − libc) per bucket** and
  each bucket's **share of the 1.941 ms recovery**.

### Decision rule (pre-registered)

- **Extra samples concentrated in buckets 1/1b** → **per-call allocator overhead.**
- **Extra samples concentrated in bucket 2 (`bzero`) and/or 3**, with call-machinery buckets
  comparable → **allocator-induced memory-layout cost** (locality / TLB / cache), since the
  instructions and volume are already proven identical.
- **Both elevated** → report the split; do not force a single story.
- **Neither, i.e. the recovery does not localize** → say so; the profile is then inconclusive and the
  Phase 2 probe becomes the primary evidence.

## Phase 2 — narrow alloc-plus-zero probe (confirms the distinction)

A fresh-process micro-probe that reproduces **exactly** the production allocation population and
nothing else:

- **16,364 retained** (header + 64-byte-aligned 8 KB words) pairs — allocate **all**, keep them live
  (not alloc/free churn), matching production's shape and the spec-36 lifecycle discipline.
- Two arms: **SMP** vs **libc**. Same code, same alignment, same order, same `bzero`.
- **Time the two sub-phases separately:** (i) **allocation only**, (ii) **zeroing only** (a single
  `bzero` pass over the retained buffers, after all allocation completes).
- **Report allocator-supplied address statistics for both arms:** total address-span, number of
  distinct OS pages touched, contiguity/stride of successive words buffers, and page-straddling count
  (runtime page size; Darwin 16 KB).

**Interpretation:**

- **Allocation-only slower on SMP, zeroing-only comparable** → **per-call allocator overhead.**
- **Zeroing-only slower on SMP with identical `bzero` and identical volume** → **allocator-induced
  layout**; the address statistics then say why (span, page count, stride).
- This is the clean isolation: **same instructions, same byte volume, only the supplying allocator
  differs.**

## Conditions and gates

- **Both hosts:** M4 (subject) and Zen 4/WSL2 (OS+arch control, **not** a pure architecture control).
- **Five fresh processes** for the Phase 2 timings, medians + full ranges; Phase 2 **is** a timing
  measurement and follows canonical discipline. Phase 1 is attribution only.
- **A0/C0-style scaffolding check** (spec 36 lesson): confirm the probe harness itself does not shift
  the canonical row — ranges overlap, medians within 5%.
- **rawr/libc is a first-class arm here, not a legacy control** — it is the evidence carrier for this
  spec, notwithstanding that libc remains unsuitable as a global default (spec 18).
- **No production library changes**; `zig build`, `zig build test`, `zig build difftest` green.

## Acceptance

- Phase 1 bucket attribution for **rawr/SMP vs rawr/libc**, same binary, fresh processes, both hosts,
  with per-bucket Δ and share of the recovery; unsymbolized fraction stated; buckets sum to 100%;
  1b reported separately.
- Phase 2 probe: 16,364 retained pairs, SMP vs libc, **allocation-only and zeroing-only timed
  separately**, five fresh processes, both hosts, plus address statistics.
- A single stated verdict: **per-call allocator overhead**, **allocator-induced memory layout**, **both
  (with split)**, or **inconclusive** — per the pre-registered rule, per host.
- Recovered-share arithmetic derived **within one run**, denominator stated.
- No production change; `docs/parity-measurement.md` updated with buckets, probe, and verdict.

## Outcome branches (no lever design in this spec)

- **Per-call allocator overhead** → lever space is reducing **allocator call count** for this pattern
  (note the tension worth recording: spec 18 closed a *global* libc swap because libc **regressed**
  container-heavy ops — sparse AND 1.427x, sparse OR 1.431x, deserialize 1.782x — yet libc **wins by
  1.941 ms here**, so the effect is **op-dependent**; a per-operation or per-allocation-class choice is
  an unexplored shape, to be designed only in its own spec).
- **Allocator-induced layout** → lever space is address locality (batching, contiguity, span
  reduction), which is **also** where a recycling pool could re-enter — but on a *layout* rationale,
  **not** the residency rationale spec 36 refuted.

## Chunk plan

**Single chunk: `37-00`** — Phase 1 and Phase 2 answer one question and should be reported together.

## Estimate

M — profiling setup and symbolization are the bulk; the Phase 2 probe is small and reuses spec 36's
lifecycle discipline.
