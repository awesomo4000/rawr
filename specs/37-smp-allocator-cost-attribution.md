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

**Attribution buckets — SYMMETRIC across arms** (verified present in the ReleaseFast parity worker):

| bucket | rawr/SMP symbols | rawr/libc symbols |
|---|---|---|
| **1. allocation machinery** | `heap.SmpAllocator.alloc`, `.free` | **`_malloc`, `_free`, `_posix_memalign`**, and the Zig **allocator-wrapper/vtable frames** |
| **1b. page-mapping fallthrough** | `heap.PageAllocator.map`, `.unmap`, `.alloc`, `.free` | (libc's equivalent mapping frames, if any) |
| **2. zeroing** | `_bzero` stub | `_bzero` stub (identical) |
| **3. accumulation + assembly** | `bitmap.RoaringBitmap.lazyAccumulateIntoBitset*`, `setList` / `lazyUnionWith` / `setRange`, top-level append/`ensureTotalCapacity`, `_memcpy` (clone) | same |
| **4. everything else** | remainder | remainder |

**The allocation bucket MUST be symmetric.** Listing only SMP symbols would drop the libc arm's
allocation time into "everything else" and **invalidate every delta.** Enumerate the libc-side frames
explicitly.

**Bucket 1b — interpret carefully:** a hot `PageAllocator.map` most likely indicates **SMP slab
replenishment**, **not** one mapping per 8 KB allocation. (Spec 36's 40-fault result already argues
against per-allocation fresh mappings.) Report it separately either way.

**Profiling protocol:**

- **Fresh process, canonical corpus, canonical init order, one implementation per process** — the
  spec-35 rule stands: **never profile in a warmed harness.**
- **DO NOT increase iterations inside a profiling process.** Extra in-process iterations would
  accumulate exactly the allocator process history that specs 20a and 35 identified as the confound.
  **Preserve canonical warmup/timed counts** and obtain sample volume by **aggregating across
  additional fresh processes** instead. Higher sample *rate* is acceptable; more in-process *work* is
  not.
- **Profile ONLY the timed construction call tree.** Whole-process sampling would sweep in validation,
  warmups, result destruction, and allocator frees — and destruction sits **outside** canonical
  timing. Add a **stable profiling-only wrapper / signpost** around the timed `lazyOr` call and
  **classify only samples beneath that call tree.** Note: `SmpAllocator.free` (and `_free`) **should
  not normally appear** inside a successful construction — if they do, say so, it is a finding.
- Tool and rate stated: Darwin **Instruments/`xctrace` Time Profiler** (or `sample`); Linux
  **`perf record`**. **Zig and libc frames must symbolize** — state how (frame pointers, dSYM) and
  report the **unsymbolized sample fraction**.
- **Preflight `perf` availability and symbolization on WSL2 BEFORE relying on it.** If profiling is
  unavailable or unsymbolizable there, **Phase 1 is Darwin-only and Zen 4 contributes Phase 2 only** —
  state that outcome rather than leaving an unsatisfiable acceptance gate.

**Profile arithmetic — normalization (do not mix profile time with canonical time):**

- **Never divide profiler sample time by the canonical 1.941 ms.** They are different measurements.
- Report **either** (preferred, and both if cheap): **(i)** samples **normalized per completed
  construction** (sample time ÷ constructions profiled), so arms are comparable in absolute terms; and
  **(ii)** each bucket's **share of the PROFILED `SMP − libc` delta** — self-consistent within the
  profile.
- **The canonical allocator recovery (~1.941 ms, ~85%) stays a separate timing result**, recomputed
  from **one** canonical run, and is never reconstructed from profile samples.

### Decision rule (pre-registered)

- **Extra samples concentrated in buckets 1/1b** → **per-call allocator overhead.**
- **Extra samples concentrated in bucket 2 (`bzero`) and/or 3**, with call-machinery buckets
  comparable → **allocator-induced memory-layout cost** (locality / TLB / cache), since the
  instructions and volume are already proven identical.
- **Both elevated** → report the split; do not force a single story.
- **Neither, i.e. the recovery does not localize** → say so; the profile is then inconclusive and the
  Phase 2 probe becomes the primary evidence.

## Phase 2 — narrow alloc-plus-zero probe (confirms the distinction)

A fresh-process micro-probe reproducing **exactly** the production allocation population — and
**preserving production's interleaving**, which is itself part of what is under investigation.

**Production order is `header alloc → words alloc → bzero → accumulate`, per container.** A
probe that allocates everything and only then zeroes in one late pass **destroys the
allocate/zero interleaving** being studied. So the cells are:

| cell | per-container sequence | retained? |
|---|---|---|
| **P1 allocation-only** | `header alloc → words alloc` | **yes, all retained** |
| **P2 production-order init** | `header alloc → words alloc → bzero` | **yes, all retained** |
| **P3 split-pass** *(secondary)* | allocate all pairs, **then** one `bzero` pass | yes |

- **`P2 − P1` estimates the induced zeroing cost while preserving production's access order** — this
  is the primary Phase 2 quantity, not a separate late-zeroing pass.
- **P3 is a SECONDARY diagnostic only** — it deliberately breaks the interleave, so it shows what the
  zeroing costs when decoupled from allocation order. Useful for contrast; never the primary number.
- **`N = 16,364` retained** (header + 64-byte-aligned 8 KB words) pairs — all live simultaneously, per
  the spec-36 lifecycle discipline (no alloc/free churn, which would recycle blocks).
- Two arms throughout: **SMP** vs **libc** — same code, same alignment, same order, same `bzero`.

**Report allocator-supplied address statistics for both arms:** total address span, distinct OS pages
covered, contiguity/stride of successive words buffers, page-straddling count (runtime page size;
Darwin 16 KB). **These are CLUES, not proof of physical locality** — virtual contiguity does **not**
establish cache or physical-page locality, and must not be reported as if it does.

**Interpretation:**

- **P1 slower on SMP, `P2 − P1` comparable** → **per-call allocator overhead.**
- **`P2 − P1` slower on SMP**, with identical `bzero` and identical volume → **allocator-induced
  layout**; the address statistics suggest (do not prove) why.
- **Both** → report the split.
- This is the clean isolation: **same instructions, same byte volume, same ordering, only the
  supplying allocator differs.**

## Conditions and gates

- **Both hosts:** M4 (subject) and Zen 4/WSL2 (OS+arch control, **not** a pure architecture control).
- **Five fresh processes** for the Phase 2 timings, medians + full ranges; Phase 2 **is** a timing
  measurement and follows canonical discipline. Phase 1 is attribution only.
- **A0/C0 scaffolding check (spec 36 lesson), with C0 defined concretely:**
  - **A0** = the canonical worker's `lazyOr` construction row, diagnostics absent.
  - **C0** = the **untouched production `lazyOr` row inside the diagnostic executable**, run as **its
    own fresh-process mode, before any probe or profiling work occurs in that process.**
  - **Gate:** A0 and C0 five-process ranges overlap and medians agree within **5%**. Otherwise the
    diagnostic executable itself is shifting the row — report and fix before believing any contrast.
- **rawr/libc is a first-class arm here, not a legacy control** — it is the evidence carrier for this
  spec, notwithstanding that libc remains unsuitable as a global default (spec 18).
- **No production library changes**; `zig build`, `zig build test`, `zig build difftest` green.

## Acceptance

- Phase 1 bucket attribution for **rawr/SMP vs rawr/libc**, **same binary**, **canonical warmup/timed
  counts** (sample volume from **extra fresh processes**, never extra in-process iterations),
  **samples classified only beneath the timed-`lazyOr` signpost**, **symmetric allocation bucket**
  (SMP *and* libc frames enumerated), buckets summing to 100%, 1b reported separately, unsymbolized
  fraction stated.
- Phase 1 arithmetic reported as **samples per completed construction** and/or **share of the profiled
  SMP−libc delta** — **never** profile time divided by the canonical 1.941 ms.
- **Host coverage stated honestly:** if WSL2 `perf`/symbolization preflight fails, **Phase 1 is
  Darwin-only and Zen 4 contributes Phase 2 only** — recorded as the outcome, not left as an
  unsatisfiable gate.
- Phase 2 probe: **16,364 retained** pairs, SMP vs libc, cells **P1 (alloc-only)** and **P2
  (production-order alloc→alloc→bzero)** with **`P2 − P1`** as the primary quantity, **P3 split-pass
  secondary only**, five fresh processes, both hosts, plus address statistics **labelled as clues, not
  proof of physical locality**.
- **A0 vs C0** verified (C0 = untouched production `lazyOr` row in the diagnostic executable, own
  fresh-process mode, before any probe work).
- A single stated verdict **per host**: **per-call allocator overhead**, **allocator-induced memory
  layout**, **both (with split)**, or **inconclusive** — per the pre-registered rule.
- Canonical recovered-share arithmetic derived **within one canonical run**, denominator stated, and
  kept **separate** from profile numbers.
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
