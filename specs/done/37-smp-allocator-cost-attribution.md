<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 37: SMP allocator cost attribution — call time vs induced memory layout

> **Outcome (2026-08-07) — ANSWERED BY A STANDALONE REPRODUCER; not implemented as written.
> Verdict: ALLOCATOR-INDUCED MEMORY LAYOUT, decisively — NOT per-call overhead.**
>
> Reproduced with **no rawr and no CRoaring code at all** — only Zig's `smp_allocator` vs
> `c_allocator`, aligned 8 KB allocations, and `@memset`. That makes it a property of the allocator,
> not of rawr's container model.
>
> | M4 operation | SMP | libc |
> |---|---:|---:|
> | Allocate 8 KB blocks | 0.207 ms | 0.232 ms |
> | Allocate headers + blocks | **0.132 ms** | 0.305 ms |
> | Zero blocks in **allocation order** | **4.482 ms** | 2.753 ms |
> | Zero header-interleaved blocks in allocation order | **5.686 ms** | 2.721 ms |
> | Zero the **same** blocks after **sorting by address** | **2.819 ms** | 2.680 ms |
> | Sort header-interleaved blocks, then zero | **2.927 ms** | 2.710 ms |
>
> **The decisive control:** sorting SMP's *identical* buffers by address before zeroing recovers
> **1.663 ms** (words-only) and **2.759 ms** (header-interleaved) — with **no change to the allocator,
> byte volume, alignment, or zeroing function.** Only traversal **order** changed.
>
> **The asymmetry is the finding:**
> - **libc is order-INSENSITIVE** — sorting changes it by **0.011–0.073 ms.**
> - **SMP is order-SENSITIVE** — sorting changes it by **1.663–2.759 ms.**
> - **Interleaving the 16 B header allocations costs SMP +1.204 ms and libc nothing (−0.032 ms).**
> - **After sorting, SMP ≈ libc** (residual **+0.139 / +0.217 ms**).
> - **SMP's allocation calls are FASTER** (−0.025 ms blocks; **−0.173 ms** headers+blocks) — so the
>   per-call branch of this spec's question is refuted outright.
>
> **Mechanism, partially explained:** Zig 0.16's `SmpAllocator` uses **64 KB slabs with per-size-class
> freelists**; the measured SMP allocation stream has a **median 64 KB stride**, versus libc's **8 KB**.
> So SMP allocates faster but hands back 8 KB blocks in a **poor spatial traversal order** on M4, and
> interleaving the tiny header allocations makes that order worse.
>
> **Ruled out by this result:** per-call allocation overhead, struct packing, first-touch faults
> (spec 36), and any `bzero` implementation difference (both call the same stub).
>
> **REMAINING UNKNOWN (next question, not addressed here):** *which hardware effect* makes that order
> expensive — hardware prefetching, TLB / page-table locality, cache behaviour, or a combination. **No
> lever is proposed until that is established.**
>
> **Bookkeeping:** the Phase 1 profiling apparatus and Phase 2 probe specced below were **not needed** —
> a targeted standalone reproducer answered the question more cleanly and more strongly (it removes
> rawr entirely). Retained as the record of the plan, and as the fallback design if the *hardware-effect*
> question needs profiling. **Reproducer currently at `/tmp/smp_layout_probe.zig`, NOT yet in the
> repository — it is now load-bearing evidence and should be preserved.**

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

**Buckets MUST be mutually exclusive — assign each LEAF sample exactly once.**
`PageAllocator.map` sits **beneath** `SmpAllocator.alloc`, so naïve *inclusive* accounting would
**double-count** it. Rule: classify by the **leaf frame**, walking up to the nearest bucket-owning
symbol, and **subtract 1b from bucket 1** — i.e. bucket 1 is allocation machinery **excluding**
page-mapping descendants. **Apply the identical rule to libc's mapping descendants** (`mmap`/`madvise`
etc. beneath `_malloc`/`_posix_memalign`). Buckets must sum to 100% with no sample counted twice.

**Unsymbolized samples are EXCLUDED from the 100% denominator** and reported as their own line with a
percentage — **not** folded into bucket 4, where measurement failure would masquerade as
"everything else" and could silently carry a real effect. So buckets sum to 100% **of symbolized
samples**, stated as such. **If the unsymbolized fraction exceeds 5% in either arm, the profile is
suspect** — fix symbolization before drawing bucket conclusions.

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
- **Profile ONLY the timed construction call tree, via a MECHANICALLY PINNED wrapper.** Whole-process
  sampling would sweep in validation, warmups, result destruction, and allocator frees — and
  destruction sits **outside** canonical timing. Requirements:
  - a **named, `noinline`, profiling-build-only wrapper function** (e.g.
    `rawr_prof_timed_lazy_or`) that calls `lazyOr` and nothing else;
  - it is used for the **timed invocations ONLY — warmups call the ordinary path**, so warmup samples
    are excluded structurally rather than by post-hoc filtering;
  - attribution keeps **only samples whose stack contains that symbol**, i.e. its descendants.
  - `noinline` matters: an inlined wrapper leaves no frame to filter on.
  - Note: `SmpAllocator.free` (and `_free`) **should not normally appear** beneath it in a successful
    construction — if they do, say so, that is a finding.
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
- **Neither, i.e. the recovery does not localize** → the profile is **INCONCLUSIVE**, and that is the
  reported verdict. **Phase 2 does NOT get promoted to primary evidence** — it is corroborating only
  (it omits the ~49,132 unmatched clones and interleaved work), so an inconclusive Phase 1 means the
  question stays open pending a better-instrumented Phase 1, not that the probe decides it.

## Phase 2 — matched-bitset initialization probe (corroborating, NOT authoritative)

**This is a matched-bitset initialization probe, NOT an exact production reproduction.** Production
additionally interleaves roughly **49,132 unmatched-container clones**, top-level storage growth, and
accumulation **between** the matched-bitset allocations — all of which shape allocator history and
layout. The probe deliberately omits them to isolate one thing, and that omission is a **known
limitation**, not a claim of fidelity.

**Authority:** **Phase 1 governs the canonical verdict.** If Phase 2 **disagrees with Phase 1**, or
fails to reproduce the SMP-vs-libc **direction**, the Phase 2 component is **INCONCLUSIVE** — it is
**not** evidence against the production profile.

It preserves production's **per-container interleaving** for the allocations it does model, which is
itself part of what is under investigation.

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

### `P2 − P1` uncertainty handling (pre-registered)

P1 and P2 run in **independent fresh processes**, so subtracting their medians carries real
uncertainty. Do **not** report a bare difference. Report the **conservative interval**:

```text
P2 − P1  ∈  [ P2_min − P1_max ,  P2_max − P1_min ]
```

**Require the SMP and libc derived intervals to SEPARATE (not overlap)** before declaring a
zeroing/layout difference. **If they overlap, that component is INCONCLUSIVE** — no layout claim.

**The same range-separation rule applies to P1 itself.** Before calling anything **allocator
overhead**, the **P1 SMP and P1 libc five-process ranges must SEPARATE** (not merely differ in
median). Overlapping P1 ranges ⇒ the allocation-only component is **INCONCLUSIVE**, not evidence of
per-call overhead. Same discipline both directions.

**Interpretation:**

- **P1 ranges separated with SMP slower**, `P2 − P1` intervals overlapping → **per-call allocator
  overhead.**
- **`P2 − P1` interval for SMP separated above libc's**, with identical `bzero` and identical
  volume → **allocator-induced layout**; the address statistics suggest (do not prove) why.
- **Both** → report the split.
- Isolation achieved: same instructions, same byte volume, same per-container ordering — only the
  supplying allocator differs. (Within the probe's stated limitation above.)

## Conditions and gates

- **Both hosts:** M4 (subject) and Zen 4/WSL2 (OS+arch control, **not** a pure architecture control).
- **Five fresh processes** for the Phase 2 timings, medians + full ranges; Phase 2 **is** a timing
  measurement and follows canonical discipline. Phase 1 is attribution only.
- **A0/C0 scaffolding check (spec 36 lesson), with C0 defined concretely:**
  - **A0** = the canonical worker's `lazyOr` construction row, diagnostics absent.
  - **C0** = the **untouched production `lazyOr` row inside the diagnostic executable**, run as **its
    own fresh-process mode, before any probe or profiling work occurs in that process.**
  - **Applied INDEPENDENTLY to BOTH rawr arms — rawr/SMP and rawr/libc.** The diagnostic executable
    could perturb one allocator without materially moving the other, and a gate run only on the SMP
    arm would miss exactly that.
  - **Gate:** for **each arm**, A0 and C0 five-process ranges overlap and medians agree within **5%**.
    Otherwise the diagnostic executable is shifting that arm — report and fix before believing any
    contrast involving it.
- **rawr/libc is a first-class arm here, not a legacy control** — it is the evidence carrier for this
  spec, notwithstanding that libc remains unsuitable as a global default (spec 18).
- **No production library changes**; `zig build`, `zig build test`, `zig build difftest` green.

## Acceptance

- Phase 1 bucket attribution for **rawr/SMP vs rawr/libc**, **same binary**, **canonical warmup/timed
  counts** (sample volume from **extra fresh processes**, never extra in-process iterations),
  **samples classified only beneath the named `noinline` timed-only wrapper** (warmups use the ordinary path), **symmetric allocation bucket**
  (SMP *and* libc frames enumerated), **mutually exclusive leaf-assigned** buckets summing to 100% **of symbolized samples** (1b subtracted
  from bucket 1, same rule for libc mapping descendants), 1b reported separately, **unsymbolized samples
  excluded from the denominator and reported as their own line — profile suspect above 5%**.
- Phase 1 arithmetic reported as **samples per completed construction** and/or **share of the profiled
  SMP−libc delta** — **never** profile time divided by the canonical 1.941 ms.
- **Host coverage stated honestly:** if WSL2 `perf`/symbolization preflight fails, **Phase 1 is
  Darwin-only and Zen 4 contributes Phase 2 only** — recorded as the outcome, not left as an
  unsatisfiable gate.
- Phase 2 **matched-bitset initialization probe** (explicitly NOT an exact production reproduction —
  it omits the ~49,132 unmatched clones, top-level growth, and interleaved accumulation): **16,364
  retained** pairs, SMP vs libc, cells **P1 (alloc-only)** and **P2 (production-order
  alloc→alloc→bzero)**, **P3 split-pass secondary only**, five fresh processes, both hosts, plus
  address statistics **labelled as clues, not proof of physical locality**.
- **`P2 − P1` reported as the conservative interval `[P2_min − P1_max, P2_max − P1_min]`**, with SMP
  and libc intervals required to **separate** before any zeroing/layout claim; overlap ⇒ that
  component is **inconclusive**.
- **Phase 1 governs the verdict**; a Phase 2 result that disagrees with Phase 1 or fails to reproduce
  the SMP/libc direction is recorded as **inconclusive**, not as counter-evidence.
- **A0 vs C0 verified INDEPENDENTLY for BOTH rawr arms** (SMP and libc); C0 = untouched production
  `lazyOr` row in the diagnostic executable, own fresh-process mode, before any probe work.
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

L — the cross-platform profiling work (Darwin `xctrace` + Linux/WSL2 `perf`, symbolization of Zig and
libc frames, leaf-assigned bucket classification, the timed-only `noinline` wrapper, and sample
aggregation across extra fresh processes) dominates. The Phase 2 probe itself is small and reuses
spec 36's lifecycle discipline.
