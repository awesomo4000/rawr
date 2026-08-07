<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 36: Lazy-OR first-touch / page-residency diagnosis

Campaign: [31-structural-parity-campaign.md](31-structural-parity-campaign.md). **Diagnosis only —
no production change, no recycling-pool design.** Confirm or refute the **first-touch / page-residency
hypothesis** for the last material row.

**Target (canonical, fresh-process, 2026-08-06) — M4:** lazy-OR construction **rawr/SMP 5.762 ms vs
CRoaring 3.336 ms = 1.727x**, gap **2.426 ms**. (Repair-only 1.069x; combined 1.190x.)

**Standing evidence:** rawr/SMP construction is **highly sensitive to prior process activity** —
moving validation ahead of timing in the canonical worker took rawr **5.762 → 4.243 ms** while
CRoaring stayed **3.336 → 3.357** (the operation unchanged). The older context modes (allocator prime
5.150 / cache prime 3.890 / target-only 4.139) **were confounded — each also ran
`initAllBenchmarkData()` — so they do NOT establish "allocator state, not cache warmth."** Separating
those two is precisely this spec's job. Leading mechanism: **page residency / first touch on freshly
SMP-allocated 8 KB buffers** — **unconfirmed.**

## Canonical conditions (corrected)

- **One process per `(row, implementation, allocator)` tuple**, exactly as the canonical worker does.
  **Each process initializes ONLY its selected implementation** using that implementation's normal
  setup — **never both rawr and CRoaring in one process**, and never `initAllBenchmarkData`.
- Canonical corpus, timing boundaries, **validation-after-timing**, **five fresh processes**,
  canonical **3 warmup / 21 timed**.
- Per the spec-35 lesson: **no cell takes its production reference from a warmed harness.**

### Two anchors: A0 (canonical) and C0 (diagnostic baseline) — they are NOT the same thing

**C0 is not "untouched."** Every diagnostic cell — C0 included — allocates and prefaults the eviction
buffer and installs fault instrumentation. So:

- **A0 — canonical anchor:** the canonical worker with **all diagnostics disabled** (no eviction
  buffer, no fault instrumentation), run **contemporaneously** with the cells on the same host and
  build.
- **C0 — diagnostic baseline:** diagnostic scaffolding present (eviction buffer allocated +
  prefaulted, counters installed), but **no pre-pass and no eviction**.
- **Validity gate:** **A0 and C0 five-process ranges must overlap, and their medians must agree within
  5%.** If they do not, **the diagnostic scaffolding itself is perturbing the measurement** — report
  that and fix it before any contrast is believed.
- **The historical ranges (M4 rawr `5.701–5.940 ms`, CRoaring `3.270–3.385 ms`) are PROVENANCE, not a
  hard validity gate** — they document what the row measured on 2026-08-06. A0 is the live anchor;
  note any drift from the historical range rather than failing on it.

## The measurement problem

The 8 KB `words` allocations occur **inside** the timed construction, so they cannot literally be
pre-touched outside timing. Instead we **precondition allocator page state immediately before each
timed operation**, holding everything else fixed, and separate **allocator bookkeeping** from **page
residency**.

Existing modes cannot do this: **`allocator_prime` and `cache_prime` both also run
`initAllBenchmarkData()`**, so both are conflated and **`primeCachesOnly` is not reusable**.

## Cell matrix

**Pre-pass timing (corrected — critical):** the pre-pass (and, in C3/C4, the eviction) runs **outside
the timer but immediately before EVERY invocation — each of the 3 warmups and each of the 21 timed runs.** A
single pre-pass before the batch would be overwritten by the warmups, which are exactly the
allocation activity under test.

**A FULL 2×2 FACTORIAL is required (corrected).** With eviction present in only one cell, a
residency-vs-non-residency contrast would also carry eviction of **allocator metadata, TLB state, and
other warmed data**. Residency and cache must therefore vary **independently**:

| cell | residency pre-pass | cache | pre-pass before *every* invocation |
|---|---|---|---|
| **A0** | — | — | **canonical anchor: diagnostics fully DISABLED** (no eviction buffer, no counters) |
| **C0** | — | — | diagnostic baseline: scaffolding present, **no pre-pass, no eviction** |
| **C1** | **unconditioned** | **warm** | bookkeeping-only pre-pass (alloc + free, payload never written) |
| **C2** | **conditioned** | **warm** | alloc + **touch** + free |
| **C3** | **unconditioned** | **evicted** | bookkeeping-only pre-pass, **then evict** |
| **C4** | **conditioned** | **evicted** | alloc + **touch** + free, **then evict** |

**Contrasts (sign convention: `less-conditioned − more-conditioned`, positive = improvement):**

- **C3 − C4** = **residency, cache held cold** ← **PRIMARY**
- **C1 − C2** = **residency, cache held warm** (the warm-cache counterpart)
- **C1 − C3** and **C2 − C4** = the **cache-eviction** effect at each residency level
- **(C1 − C2) vs (C3 − C4)** = **interaction** — if the residency effect differs by cache state, say
  so rather than reporting a single number
- **C0 − C1** = allocator **bookkeeping** effect (diagnostic baseline vs bookkeeping-only pre-pass)
- **A0 vs C0** = **scaffolding check**, not a result — must overlap per the validity gate above

### Allocation SHAPE matches production; the SEQUENCE deliberately does not

**Shape (must match):** production `BitsetContainer.init` does **`allocator.create(Self)` (16 B
header)** then **`allocator.alignedAlloc(u64, .@"64", 1024)` (8 KB words)**; `deinit` frees **`words`
FIRST, then `destroy(header)`**. Pre-passes must replicate **that pair, that alignment, and that free
order** — **plain `alloc(u8, 8192)` is NOT acceptable** (different allocator path / size class).

**Sequence (deliberately differs — this is a feature, not an oversight):** production **interleaves**
per container (allocate → zero → next). The pre-pass instead **allocates all `N` blocks first, then
touches them** (C2/C4). That divergence is **intentional and required**: it holds **allocator
bookkeeping identical between C1 and C2** (both allocate all `N` pairs and free them in the same
order; they differ **only** in whether the payload is touched), which is exactly what makes the
residency contrast clean. Do not "fix" this to match production's interleave — doing so would
reintroduce the confound the factorial exists to remove.

### Pre-pass lifecycle (pinned — must not degenerate into recycling one block)

A naive "allocate a pair, free it, repeat `N` times" would **recycle a single block** and condition
nothing. Required sequence, per invocation:

1. **Allocate and RETAIN all `N` pairs simultaneously** (all live at once).
2. **Touch each words allocation while it is still live** (C2/C4 only).
3. **Then free all pairs**, in the **same cross-container order production frees them** (ascending
   index/key order, matching the repair/deinit traversal).
4. **Within each pair, free `words` then `header`**, as `deinit` does.

### `N` and page geometry (corrected)

- **`N = 16_364`** — the **exact corpus matched-key count**. Do **not** use 16,384; that figure came
  from the confounded `primeAllocatorOnly` diagnostic and carries no authority.
- **"Touch every page" means every RUNTIME OS page intersecting each allocation**, computed from the
  **runtime page size**, not assumed. On **Darwin the page size is 16 KB**, so an 8 KB allocation may
  **straddle two pages** (or share one with a neighbour) depending on alignment — touch every page in
  `[base, base + len)` for each allocation.

## C1 is an assumption to be measured, not asserted (corrected)

**C1 cannot be assumed to leave pages unfaulted** — `free` may write allocator metadata **into the
payload**, faulting the very pages C1 is supposed to leave cold. Therefore:

- **Measure faults during each pre-pass itself** (separately from the timed operation) and **report
  them per cell.** If C1's pre-pass faults ≈ C2's, C1 is not a bookkeeping-only control and the
  bookkeeping/residency separation collapses — say so rather than reporting a false contrast.
- **Page-reuse proof — must NOT pollute the measured process history (corrected).** An explicit
  **pointer / page-overlap check** must demonstrate that production actually **reuses the pages
  conditioned by the pre-pass** (compare pre-pass page addresses against the words buffers production
  receives; report the **overlap fraction**). Run it **either in a separate fresh diagnostic process
  or after ALL timing in the process** — never before or between timed runs, since the check is
  itself allocation activity that would condition the state under test. Implement it with an
  **untimed production construction** whose resulting bitset word addresses are **inspected
  directly**; **do not wrap the timed allocator.**
  **Without demonstrated reuse the entire experiment is void** — we would be conditioning pages
  production never receives.

## Fault counters

- **Boundaries (pinned):** sample **immediately before the operation's internal clock starts** and
  **immediately after it stops, before result teardown** — the samples sit outside the timed span.
- **Per-invocation deltas** for **each of the 21 timed runs**; **warmups discarded.**
- **Aggregation (pinned):** report the **median** per-invocation delta as the headline, plus **min/max
  and the sum across the 21**; then the **five-process median of those medians**, matching the timing
  protocol.
### Host-specific PRIMARY counter (they are not the same quantity)

**Darwin `TASK_EVENTS_INFO.faults` is NOT Linux `ru_minflt`** — do not label it "minor faults" and do
not pool the two. Each host has one **primary** counter, and the 50% gate applies to **that host's
primary**:

| host | primary counter (gate applies here) | also reported |
|---|---|---|
| **Linux (Zen 4/WSL2)** | `getrusage(RUSAGE_SELF)` **`ru_minflt`** | `ru_majflt` |
| **Darwin (M4)** | `TASK_EVENTS_INFO` → `task_events_info` **`faults`** | **`pageins`**, **`cow_faults`** (separately, never merged into "faults") |

`TASK_VM_INFO` reports **residency, not fault counts** — not usable here. `getrusage` may be tried
first on Darwin but is **not trusted**; state which source was actually used.

- **If NO working M4 fault source is found, the M4 mechanism verdict is INCONCLUSIVE.** **Zen 4/WSL2
  counters cannot prove M4 behavior** — different OS *and* architecture.

## Cache eviction must be fully controlled

- The eviction buffer is **allocated and FULLY PREFAULTED identically in every cell** (C0/C1/C2 too,
  even though only C3 and C4 walk it) so no cell differs in allocator or memory state because of it.
- It is allocated **outside the target SMP allocator** (page allocator / direct mapping) so it does
  not perturb the state under test.
- **Size pinned relative to reported LLC** (state the host LLC and the multiple used).
- Prefaulting matters: otherwise **C3 would add both faults and memory pressure**, contaminating the
  cell it is meant to isolate.
- **Do not reuse `primeCachesOnly`** (depends on full benchmark data).

## Allocation counts — do NOT wrap the timed allocator (corrected)

Wrapping the allocator inside the timed region **changes dispatch and codegen** and would corrupt the
timing being measured. Obtain counts from **either** an **untimed duplicate/accounting pass** **or**
**mechanically asserted corpus counts** (matched keys × the known per-key allocation pair). State
which source was used.

## Pre-registered numeric decisions

Fixed **before** data collection; adjust only with an explicit note, never after seeing results.

- **"Materially reduces time"** (residency effect, cache held cold): **`C3 − C4 ≥ 0.5 ms`**
  (≥ ~20% of M4's 2.426 ms gap) **AND** the **five-process ranges of C3 and C4 do not overlap.**
  Report **`C1 − C2`** (warm-cache counterpart) alongside; a residency effect that appears only at one
  cache level is an **interaction**, reported as such, not a clean confirmation.
- **"Materially reduces faults"**: rawr's **host-primary fault-counter median delta falls ≥ 50% from
  C3 to C4** (residency at matched cache state) — **`ru_minflt` on Linux, `TASK_EVENTS_INFO.faults` on
  Darwin**, each judged against itself, never pooled. Report the **C1 → C2** drop alongside, and
  Darwin's `pageins` / `cow_faults` separately. (A residency effect should be dramatic in fault
  counts, not marginal.)
- **"CRoaring unmoved"**: CRoaring's median moves **≤ 2%** across all cells **AND** its five-process
  ranges **overlap**. If a pre-pass moves CRoaring beyond that, the cell is not isolating what we
  think and **that cell's result is void.**

## Host-local recovery accounting (corrected)

- The **2.426 ms denominator is M4-only.** **Zen 4 computes recovered share against its OWN C0 gap.**
- **Zen 4/WSL2 is an OS-plus-architecture control**, not a pure architecture control — differences may
  be Linux-vs-Darwin, not Zen-vs-M4. Do not attribute Zen 4 behavior to architecture alone.

## Confirmation criterion (pre-registered)

**CONFIRMED only if, on M4 under canonical conditions:** the residency pre-pass materially reduces
**BOTH** rawr/SMP **minor faults** and rawr/SMP **construction time** per the thresholds above, with
the effect carried by **residency at matched cache state** (**`C3 − C4`**, corroborated by
**`C1 − C2`**) rather than by the **cache-eviction** contrasts (**`C1 − C3`** / **`C2 − C4`**),
**CRoaring unmoved**, and **page reuse demonstrated**.

- **A timing win with unchanged fault counts does NOT confirm first touch** — it points at allocator
  bookkeeping, size-class behavior, or zeroing codegen.
- **No working M4 fault source ⇒ INCONCLUSIVE on M4**, regardless of timing.

## Outcome branches (no lever design here)

- **CONFIRMED** → a **separate** spec designs a recycling strategy with explicit **memory-retention**,
  **allocator-ownership**, and **thread-safety** gates. (A fixed-size 8 KB **recycling pool** is a
  different shape from spec 17's **bump arena + bulk free**, which lost on teardown — but nothing is
  designed until confirmation.)
- **REFUTED / INCONCLUSIVE** → **cold-page zeroing / codegen diagnosis**: is rawr's 8 KB `@memset`
  itself worse than CRoaring's on cold pages (width, alignment, non-temporal hints, disassembly)?

## Constraints

- **No production library code changes**; cells are gated/benchmark-local.
- **No recycling-pool implementation or design.**
- Diagnostic build passes `zig build`, `zig build test`, `zig build difftest`.
- Board gate not applicable (no production change), but the **A0 vs C0 anchor check** must show the
  canonical rows are unperturbed.

## Acceptance

- All six cells (**A0** canonical anchor + **C0** diagnostic baseline + the **C1–C4** 2×2 factorial), **rawr/SMP and CRoaring as separate per-tuple processes with implementation-specific
  init**, five fresh processes each, on **M4 (subject)** and **Zen 4/WSL2 (OS+arch control)**.
- Pre-pass (and, in **C3 and C4**, the eviction) confirmed to run **before every warmup and every timed invocation**.
- Pre-pass allocation shape verified to match production (header `create` + 64-byte-aligned 1024-word
  `alignedAlloc`; free words-then-header).
- **Pre-pass faults reported per cell**; **page-reuse overlap demonstrated** (else void).
- Fault counters at the pinned boundaries with per-invocation deltas and the pinned aggregation;
  Darwin via `TASK_EVENTS_INFO`, or **M4 INCONCLUSIVE** stated.
- Eviction buffer allocated + prefaulted in **all** cells, outside SMP, size pinned to reported LLC.
- Allocation counts from an **untimed** source (stated), not a wrapped timed allocator.
- All factorial contrasts reported (**C3−C4** primary, **C1−C2**, **C1−C3**, **C2−C4**, **C0−C1**, plus the interaction) with **host-local** recovered shares and the pre-registered thresholds
  applied; **A0 vs C0 overlap (ranges overlap, medians within 5%)** verified, with any drift from the historical provenance ranges noted.
- A clear **CONFIRMED / REFUTED / INCONCLUSIVE** verdict (per host) and the named outcome branch.
- No production change; `docs/parity-measurement.md` updated with cells and verdict.

## Chunk plan

**Single chunk: `36-00`** — one orthogonal experiment, one verdict.

## Estimate

M — six cells × two implementations × two hosts, reusing the canonical worker. The work is
instrumentation and orthogonality discipline (per-invocation pre-passes, Mach fault plumbing,
page-reuse proof), not new algorithms.
