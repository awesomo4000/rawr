<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 36: Lazy-OR first-touch / page-residency diagnosis

Campaign: [31-structural-parity-campaign.md](31-structural-parity-campaign.md). **Diagnosis only —
no production change, no recycling-pool design.** Confirm or refute the **first-touch / page-residency
hypothesis** for the last material M4 row.

**Target gap (canonical, fresh-process, 2026-08-06):** lazy-OR construction **rawr/SMP 5.762 ms vs
CRoaring 3.336 ms = 1.727x**, a **2.426 ms** absolute gap. (Repair-only 1.069x; combined 1.190x.)

**Standing evidence:** rawr/SMP construction moves with **prior allocation activity** and not with
cache warmth (allocator prime 5.150 vs cache prime 3.890 vs target-only 4.139); moving validation
ahead of timing in the canonical worker took rawr **5.762 → 4.243 ms** while CRoaring stayed
**3.336 → 3.357 ms**. The leading mechanism is **page residency / first touch on freshly SMP-allocated
8 KB buffers** — **unconfirmed**; this spec tests it directly.

## Non-negotiable: canonical conditions

Everything is measured **in / as the canonical worker** — same corpus, same timing boundaries,
**validation-after-timing**, **five fresh processes**, canonical warmup/timed protocol. **M4 is the
subject; Zen 4 is the architecture control.** Per the spec-35 lesson, **no cell may take its
production reference from a warmed harness**, and each cell's reference is produced in its own fresh
process.

## The measurement problem this spec must solve

The 8 KB `BitsetContainer.words` allocations happen **inside** the timed construction, so they cannot
literally be "pre-touched outside timing" without changing the operation. Instead we **precondition
the allocator's page state before timing** and hold everything else fixed — and we must separate
**allocator bookkeeping** (free lists, size-class structures) from **page residency** (whether the
pages handed back are already faulted).

Existing modes cannot do this: **`allocator_prime` and `cache_prime` both also run
`initAllBenchmarkData()`**, so they conflate priming with a different allocation history.
**`primeCachesOnly` is therefore not an isolated cache experiment.** Spec 36's cells must be
**orthogonal**.

## Cell matrix (orthogonal; identical init in every cell)

**Fixed in all cells:** the canonical target-only initialization (`initSparseValues` +
`initRawrSparseBitmaps` + `initCRoaringSparseBitmaps`) — **never `initAllBenchmarkData`**; identical
allocation order, zeroing, accumulation, teardown, and output; the **cache-eviction buffer is
allocated in every cell** (even where unused) so allocator state does not differ by cell.

| cell | pre-pass before timing | isolates |
|---|---|---|
| **C0 baseline** | none (production order) | reference |
| **C1 bookkeeping-only** | `N × 8 KB` **alloc + free, payload NEVER touched** | free-list / size-class state, pages left unfaulted |
| **C2 residency** | `N × 8 KB` **alloc + touch every page + free** | bookkeeping **+ faulted pages** (cache also warm) |
| **C3 residency, cold cache** | C2's pre-pass, **then evict caches** | bookkeeping + faulted pages, **cache cold** |

**`N = 16_384`** — matches the corpus demand (~16.4 k matched keys ⇒ ~134 MB of 8 KB buffers) and the
existing `primeAllocatorOnly` block count, so C1 is directly comparable to prior data.

**Contrasts (this is the whole point):**

- **C1 − C0** = allocator **bookkeeping** effect (no page-state change).
- **C2 − C1** = **page-residency / first-touch** effect, cache-warm.
- **C3 − C2** = **cache-warmth** contribution.
- **C3 − C1** = **residency effect isolated from cache** — the primary number.

**Cache eviction must be pure:** walk a **pre-allocated** buffer larger than LLC; it **allocates
nothing and initializes nothing** at eviction time (that buffer is allocated once, up front, in
**every** cell). Do **not** reuse `primeCachesOnly` (it depends on full benchmark data).

## Fault counters (best-effort per host, mandatory where available)

- `getrusage(RUSAGE_SELF)` **`ru_minflt` / `ru_majflt`** sampled **immediately before and after the
  timed region** (outside the timed span), per process, per cell.
- **Darwin caveat:** `getrusage` fault fields on macOS are known to be less reliable than on Linux; if
  they prove unusable, fall back to a **mach `task_info` / `task_vm_info`** sample and **say which
  source was used**. Zen 4 (Linux) is the reliable-counter host — that is part of why it is the
  control. **A missing M4 counter does not block the spec**, but then confirmation rests on the
  timing contrasts plus the Zen 4 fault evidence, and that limitation must be stated.

## Both implementations, every cell

Run **rawr/SMP and CRoaring through every cell.** The hypothesis predicts an **asymmetry** — the
pre-pass conditions `smp_allocator` and should move **rawr** while leaving **CRoaring (libc)**
unchanged, exactly as the validation-order experiment showed. **If a pre-pass moves CRoaring too, the
cell is not isolating what we think it is** and the result is void.

## Reporting

Per cell, per implementation, per host: **minor faults, major faults, elapsed (five-process median +
full range), allocation counts**, and the **recovered share of the 2.426 ms canonical construction
gap**. State the four contrasts above explicitly with their recovered shares.

## Confirmation criterion (pre-registered)

**The mechanism is CONFIRMED only if, under canonical conditions, the residency pre-pass materially
reduces BOTH (a) rawr/SMP minor faults AND (b) rawr/SMP construction time** — with the cache
contribution separated out (**C3 − C1** carrying the effect, not C2 − C3), and **CRoaring unmoved**.

Anything else is a refutation or an inconclusive result, and must be reported as such — **a timing
win with unchanged fault counts does not confirm first touch**, it points elsewhere (allocator
bookkeeping, size-class behavior, or zeroing codegen).

## Outcome branches (decided in advance, no lever design here)

- **CONFIRMED** → a **separate** spec designs a recycling strategy, and it must carry explicit
  **memory-retention**, **allocator-ownership**, and **thread-safety** gates. (Note a fixed-size 8 KB
  **recycling pool** is a different shape from spec 17's **bump arena + bulk free**, which lost on
  teardown — but nothing is designed until this spec confirms.)
- **REFUTED / INCONCLUSIVE** → move to **cold-page zeroing / codegen diagnosis** (is rawr's 8 KB
  `@memset` itself worse than CRoaring's `memset` on cold pages — width, alignment, non-temporal
  hints, disassembly).

## Constraints

- **No production library code changes.** Diagnostic cells are gated/benchmark-local; the shipping
  path is untouched.
- **No recycling-pool implementation or design** in this spec.
- Correctness unaffected (no behavior change), but the diagnostic build still passes `zig build`,
  `zig build test`, and `zig build difftest`.
- **Board gate not applicable** (no production change), but the diagnostic must not perturb the
  canonical rows it reuses — verify the canonical worker still reproduces **5.762 / 3.336** with the
  cells disabled.

## Acceptance

- All four cells run on **M4 (subject)** and **Zen 4 (control)**, five fresh processes each, canonical
  boundaries and validation-after-timing, identical init and eviction-buffer allocation across cells.
- Fault counters reported (or the fallback/limitation stated), for **both** implementations in every
  cell; **CRoaring shown unmoved** by the pre-passes.
- The four contrasts reported with recovered shares of **2.426 ms**.
- A clear **CONFIRMED / REFUTED / INCONCLUSIVE** verdict against the pre-registered criterion, and the
  corresponding outcome branch named.
- Canonical worker verified to still reproduce the baseline row with cells disabled; no production
  change; `docs/parity-measurement.md` updated with the cells and the verdict.

## Proposed chunk plan (confirm at review)

Single chunk — this is one orthogonal experiment with one verdict. If review prefers a split, the
natural seam is **`36-00`** (cells + fault instrumentation on M4) and **`36-01`** (Zen 4 control +
verdict), but the cells are cheap and the verdict needs both hosts, so **one chunk is proposed.**

## Estimate

S–M — four cells × two implementations × two hosts, reusing the canonical worker; the work is
instrumentation and orthogonality discipline, not new algorithms.
