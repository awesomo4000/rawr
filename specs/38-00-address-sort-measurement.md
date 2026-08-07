<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 38-00: Address-sorted traversal — measurement

Toplevel: [38-address-sorted-traversal.md](38-address-sorted-traversal.md). Background:
[allocator-address-order-pathology.md](../docs/allocator-address-order-pathology.md).

**Diagnosis only — no production change.** Decide, per phase, whether address-sorted traversal is worth
shipping, and produce the numbers `38-01` needs. **`38-01` is separately blocked** on two owner
decisions (scope; control mechanism) — this chunk does not depend on either.

## Two phases, unequal confidence

- **Repair (sound target).** Operates on the ~16,364 bitset payloads the pathology was demonstrated on.
  Repair-alone is the largest lazy-OR phase (**8.616 ms**) at **1.069x**.
- **Teardown (lower confidence, and carries a state obligation).** The spec-26a teardown gap (144.9 vs
  48.3 ns) came from the **8-run-container clone corpus** — 8 pointers, below any plausible size gate,
  and not bitsets. It **cannot** motivate sorted teardown at scale.

## Sort key: PAYLOAD, not header

`TaggedPtr` addresses the **header**; the pathology concerns the **8 KB payload**, a *separate*
allocation with an independent order.

- **repair:** sort by **`bc.words`**.
- **teardown:** **type-specific payload key** — `bc.words` / `ac.values` / `rc.runs`. `deinit` frees
  payload then header, so two streams exist and only one can be ordered — **sort by payload**.
- **Required comparison:** **header-key vs payload-key sorting**, to confirm which stream matters.

## Corpora

- **Mass-bitset corpus** (~16k 8 KB payloads) — the diagnosis corpus for **both** phases.
- **Canonical 8-container clone corpus** — retained as a **control**, to confirm the size gate correctly
  excludes it.

## Measurement protocol

- Canonical: **3 warmup / 21 timed**, **five fresh-process medians + full ranges**, **one process per
  `(row, implementation, allocator)` tuple**, implementation-specific init only.
- Arms: **rawr/SMP** and **rawr/libc**; hosts: **M4** and **Zen 4/WSL2**.
- **Sort cost INSIDE the timed region.** Only the **sort-plus-traversal** number may gate a decision;
  the sort-outside-timing figure (which the retained probe also produces) overstates the win and must
  not be used for a verdict.
- **CRoaring references at matched boundaries** — CRoaring **teardown** and **repair**. Without them,
  results are **rawr-internal deltas only** and must be labelled as such; no gap arithmetic.
- **Range separation required** (spec-37 discipline): sorted vs unsorted five-process ranges must
  **separate** before claiming a win; overlap ⇒ inconclusive for that phase.
- **Expect the libc arm to gain little** (libc recovered 0.063–0.180 ms in the probe) — that asymmetry is
  the evidence bearing on the scope decision, so report it prominently.

## Repair specifics

- **Time the COMPLETE user-visible operation:** address-ordered cardinality pass **+ key-ordered
  conversion/compaction**. Report the cardinality sub-phase as **attribution only** — **a
  cardinality-only speedup cannot gate adoption if total `repairAfterLazy` is neutral or slower.**
- **Frees must stay in the key-ordered pass**, unchanged. That is what keeps repair
  allocator-state-neutral; verify it holds in the implementation.
- **Measure scratch BOTH ways — cold and reusable.** Cold = scratch alloc + sort + traversal + scratch
  free. Reusable = steady state. **Do not assume amortization**: the control mechanism `38-01` picks may
  not provide persistent scratch.

## Teardown specifics — THREE stages; stages 1–2 non-negotiable, stage 3 required for a positive verdict

`SmpAllocator.free` pushes onto a **per-size-class LIFO freelist**, so **free order determines the order
later allocations come back.** Sorted teardown is **state conditioning**; a process that exits right
after teardown cannot reveal the downstream effect.

### Sample lifecycle — first-cycle vs steady state (pinned; report BOTH)

**The 3-warmup/21-timed protocol is itself a confound for teardown.** Each teardown iteration
conditions the allocator that the *next* iteration allocates from, so a naive 21-sample run measures an
**arm-specific steady state**, not first-cycle behaviour — and the ascending/descending arms would each
converge to their own different steady state. **Allocator state cannot be normalized within a process**
(`SmpAllocator` is a process-wide singleton with no reset), so:

| mode | protocol | what it answers |
|---|---|---|
| **first-cycle** | **one measured teardown per process**; N ≥ 5 fresh processes per arm (warmups would destroy the property, so **no in-process warmup**) | the one-shot library user: build → teardown once |
| **steady state** | the normal 3 warmup / 21 timed within one process, **labelled as arm-specific steady state** | the looping user: repeated build/teardown cycles |

**Report both, labelled.** They answer different questions and may disagree — a disagreement is a
finding, not an error. **Never present a steady-state number as first-cycle** or vice versa.

**Stage 1 — immediate teardown**, three orders: **unsorted**, **ascending payload**, **descending
payload**.

**Stage 2 — refill and re-measure, with an EXACT boundary.** For each of the three orders, refill the
same allocator and measure the next cycle. LIFO reuse implies **descending frees tend to produce
ascending allocations** (and vice versa), so descending is expected to be the interesting arm — and the
same inversion is a plausible mechanism for the measured words-only case (8 KB stride, descending
allocation order).

**Refill population (pinned):** **16,364** pairs of (16 B header + **8192 B words at 64-byte
alignment**), allocated in **production order** (header then words, per container) and **all retained** —
i.e. the same population and shape the teardown just released.

**Report three numbers separately — a combined figure alone is not acceptable**, because a faster refill
can mask a poisoned traversal order, or a faster traversal can mask a slower refill:

1. **refill/allocation time** (the allocation burst alone),
2. **unsorted payload traversal time** — the traversal **MUST be unsorted**; sorting it would mask the
   very thing being tested (what order the allocator handed back),
3. **combined refill + traversal**.

**Stage 3 — allocator-noise control. MANDATORY for any positive teardown verdict.** Insert defined
noise traffic **between** teardown and refill, then repeat stage 2. This tests whether a downstream
effect survives a realistically **shared** allocator rather than existing only in a rawr-only process.

- **Without stage 3, teardown's verdict ceiling is INCONCLUSIVE.** Teardown may be measured and reported
  without it, but it **cannot advance to `38-01`** on stages 1–2 alone.
- A teardown effect that **disappears under stage 3** is recorded as **not dependable in a library** —
  which is a useful negative result, not a failure of the experiment.
- **A teardown improvement reported without stage 2 does not count at all.**

### Stage 3 noise workload — pinned and reproducible

Generic "unrelated traffic" is **insufficient**: `SmpAllocator` keeps **per-size-class** freelists, so
noise in other classes may never touch the class under test. The workload **must perturb the 8 KB
class specifically.**

| parameter | value |
|---|---|
| seed | `std.Random.DefaultPrng.init(0xA110C)` (fixed; re-seeded per process) |
| primary allocations | **8,192 blocks of exactly 8192 B at 64-byte alignment** — the class under test — interleaved with |
| secondary allocations | **8,192 blocks** drawn from other classes (round-robin over 64 B / 512 B / 4096 B) so the noise is not single-class |
| touch behaviour | write the **first 64 B of each** block (ensures the mapping is real without paying a full 8 KB touch) |
| free behaviour | free a **deterministic 50%** — every other allocation in allocation order |
| retained live | the other **50% stays live across the refill**, so the freelist is *not* simply restored to its pre-noise state |
| ordering | allocation and free order both follow the seeded sequence — **not** address order (this is noise, not another intervention) |

Report the noise workload's own cost separately so it is never confused with teardown or refill time.

**Retained-allocation cleanup rule (required).** The retained 50% is **tens of megabytes per
iteration** (~8,192 × 8 KB ≈ 67 MB plus secondaries). Without an explicit rule it accumulates across 21
samples and both memory and allocator state drift. Pinned:

- **Release the retained set AFTER the refill measurement completes, OUTSIDE any timed region.**
- **Free it in the seeded noise order — NOT address order.** Address-ordered cleanup would be a second
  intervention silently applied between samples.
- **Report peak RSS** for the arm so accumulation, if any, is visible rather than inferred.
- **Because this cleanup itself conditions the freelist for the next sample**, it is a further reason the
  **first-cycle mode must use one measured teardown per fresh process** (see Sample lifecycle) — that is
  the only configuration where cleanup order cannot contaminate the measurement.

## Size-gate derivation

- **Per phase** (teardown and repair get separate thresholds).
- Sweep container counts; at each, require **separated ranges favouring sorted**.
- **Monotonicity required — the first separated win is not enough.** The win must **persist (or at
  minimum not regress) at every LARGER tested size.** A single separated point followed by a regression
  at a larger size is a **crossover**, not a threshold, and must not be shipped as one.
- **Report crossovers separately per allocator and per host** — do not average or collapse them.
- Threshold candidate = **smallest count where sorted wins monotonically thereafter on BOTH hosts**,
  taking the **more conservative (larger)** of the two hosts' values.
- **Do not select the shipping threshold in `38-00`.** Report the candidates and crossovers; the single
  shipping value is chosen **after** the scope and control-mechanism decisions, since a default-on
  mechanism demands a more conservative threshold than an opt-in one.
- **Irreconcilable host disagreement is a finding, not a threshold** — report it; ship nothing.

## Acceptance

- Both phases measured, both arms, both hosts, five fresh processes, sort cost inside timing, range
  separation applied.
- Payload-key vs header-key comparison reported.
- CRoaring teardown/repair references obtained, or results explicitly labelled rawr-internal deltas.
- Repair: complete-operation timing (cardinality as attribution only); **cold and reusable** scratch
  both reported; frees confirmed unmoved from the key-ordered pass.
- Teardown: **sample lifecycle reported BOTH ways** — first-cycle (one measured teardown per fresh
  process, no in-process warmup) **and** arm-specific steady state (3w/21t), each labelled; stage 1
  (three orders) **and** stage 2 with the pinned refill population and **three separate numbers**
  (refill / unsorted traversal / combined); **stage 3 noise control run with the pinned workload —
  without it teardown's verdict is INCONCLUSIVE and it does not advance to `38-01`**; retained-set
  cleanup rule followed (post-refill, untimed, seeded order) with **peak RSS reported**; any
  stage-2-less teardown claim withheld entirely.
- Per-phase size-gate **candidates and crossovers** reported with **monotonicity checked**, per allocator
  and per host; **no shipping threshold selected in this chunk**; host disagreement reported as a finding.
- Results reported as **raw deltas/ratios plus absolute-throughput findings only** — **no parity
  classification** until the owner makes the scope decision.
- **No production change.** `zig build test`, `zig build difftest` green; `docs/parity-measurement.md`
  updated with cells and verdicts; `docs/allocator-address-order-pathology.md` cross-referenced.

## Estimate

M–L — the repair cells are straightforward. The bulk is the teardown three-stage experiment (refill
plus the **pinned noise workload**, which is itself a small harness), the CRoaring matched-boundary
references, and the monotonic size-gate sweep across two allocators × two hosts.
