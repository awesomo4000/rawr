<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 43: Address-ordered bitset construction for the lazy-OR path

**Target.** `lazy-or-construction` — the last material open row on the canonical M4 board
(**5.762 / 3.336 = 1.727x**). Gate: **≤1.10x**.

## 1. Why a lever exists now

The campaign rule has been *no lever until the hardware effect is established*, and the mechanism behind
SMP's order-sensitivity (prefetch vs TLB vs cache) is **still unestablished**. This spec argues the rule
is now stricter than the evidence requires.

Spec 37 did not merely localize the cause — it **ran the intervention**:

| Spec 37 measurement (M4, zero rawr/CRoaring code) | Result |
| --- | --- |
| SMP allocation calls themselves | **faster** than libc (0.132 vs 0.305) |
| Zeroing in **allocation order** | 4.482 (SMP) vs 2.753 (libc) |
| Zeroing the **identical SMP buffers, address-sorted first** | **2.819** — SMP ≈ libc |
| libc, sorted vs unsorted | 0.011–0.073 — order-**insensitive** |

Address-sorting the same buffers recovered **1.66–2.76 ms**. That is a controlled intervention on the
independent variable, so **address order is causal** regardless of which microarchitectural effect
converts bad order into stalls.

Everything else is closed off: rawr zeroes **the same 8 KB per matched pair** as CRoaring (no volume
lever), and the zero-fill **codegen is identical** (`mov w1, #0x2000` → `bl _bzero` both sides).

## 2. What the current code forces on the design

Three facts, verified in source, that the first draft did not account for:

- **`BitsetContainer.init` zeroes before any address is observable** (`bitset_container.zig:22`):
  `allocator.create(Self)` → `allocator.alignedAlloc(u64, .@"64", 1024)` → **`@memset(words, 0)`**. The
  zero-fill *is* the work we are trying to reorder, so batching **cannot** be built on `init`.
- **A bitset is TWO allocations** — a header (`create`) and a separate 8 KB payload (`alignedAlloc`).
  Sorting payload addresses therefore **loses the header association** unless a mapping is retained.
- **Matched bitsets and unmatched clones are allocated interleaved** in the merge loop
  (`bitmap.zig:2331`) — clones for unmatched keys, lazy bitsets for matched ones, in key order.

### 2.1 Private pending-allocation path

Add a **private, non-`pub`** uninitialized path — e.g. `BitsetContainer.initPending` — that allocates the
header and payload but **does not zero**. Rules:

- **Zero before publication.** A pending container is never reachable by any read path before its
  `@memset`. State the publication point precisely.
- **Cleanup for partial batches.** If the batch fails midway, every already-allocated pending header and
  payload is freed — including ones not yet zeroed. Pending containers are not valid containers, so
  cleanup cannot route through `deinit` unless `deinit` is safe on an unzeroed body.
- **Not public.** It must not enter the `check-docs` surface or the `check-32` probe as public API.

### 2.2 Scratch for the sort

Sorting needs storage that keeps header↔payload together. Pin all of it:

- **Representation:** an array of `{header_ptr, payload_ptr}` (or an index permutation over a
  pending array), sorted on **payload address as `usize`**. Do not sort slices (§3).
- **Maximum size:** bounded by matched-pair count ≤ `min(a.size, b.size)` ≤ 65,536 entries.
- **Allocator:** the bitmap's allocator. State it explicitly.
- **On scratch allocation failure: fall back to the existing interleaved path and succeed.** The
  optimization is not worth converting a satisfiable request into OOM. Pin this as behaviour, not
  best-effort.

**Correction to the first draft: "allocation count unchanged" is now false and is withdrawn.** The
honest framing: **two allocations per bitset are preserved**, and scratch adds a small, bounded number on
top. Report scratch allocations separately so the arms in §4 stay attributable — this spec still is not
an allocation-count-reduction lever (specs 27 and 35 both regressed M4 SMP that way), and that
distinction is the reason to keep the per-bitset count fixed.

## 3. Sort constraints

- **Sort `usize` addresses with `sortUnstable` (pdq), never `std.mem.sort`.** Spec 38 measured **86.98
  ns/op** sorting `[]u8` **slices**; at ~16,364 buffers that is **~1.4 ms** and would consume the entire
  gain. Raw `usize` is ~8 ns/op → **~0.13 ms**. Spec 38-00 also used stable block sort where pdq was
  intended — **state which sort is used and confirm it in the code.**
- **Do not re-propose residency.** Spec 36 **refuted** first-touch/page-faults on M4 (40 faults across
  ~134 MB, 100% page reuse). Pages are resident; the cost is *touching* them in a bad order.
- **Read-traversal sorting stays NO-GO** (spec 38: M4 1.221x, Zen 4 1.344x). Frees want **descending**,
  reads want **ascending**. This spec is *construction* — do not conflate.

## 4. Three arms — batching and ordering must be separated

Batching changes allocator scheduling **even with no sorting**, so a two-arm test cannot attribute a
result:

| Arm | Purpose |
| --- | --- |
| **1. Baseline** — existing interleaved path | reference |
| **2. Batched, unsorted** | isolates the **batching/scheduling** effect |
| **3. Batched, sorted** | arm 3 vs arm 2 isolates **address ordering** |

**Arm 3 vs arm 2 is the spec's actual claim.** Arm 2 vs arm 1 is a confound that must be measured, not
assumed away — if batching alone moves the row, the story is not ordering.

Run all three as **equivalent fresh-process cells** so prior allocator conditioning cannot decide the
result (spec 35's warmed-context artifact read 1.155x where canonical read 1.727x).

## 5. Failure semantics — explicit gate

A pending pool creates many owners before result assembly, which is exactly where leaks hide.

**Allocation-failure injection required at:** scratch, pending headers, pending payloads, unmatched
clones, and result assembly.

**Every injected failure must:** leave **both inputs untouched**, and leak **nothing** — no assigned
container, no pending container, no scratch. Use a leak-checking GPA, never `c_allocator`.

Scratch failure specifically must **degrade to the existing path and succeed** (§2.2), not propagate OOM.

## 6. Scope — which operations, and how it is selected

`lazyMergeTwo` serves **forced and selective `lazyOr` *and* `lazyXor`** (`bitmap.zig:1128`, `:2344`;
the branch condition includes `op == .xor`). Decide and state:

- whether batching applies to **forced lazy OR only** or to **every lazy-bitset branch**;
- **no-regression coverage for everything sharing the helper** — `lazyXor` must not regress even if it is
  out of scope for the win.

**Selection mechanism — decide before implementing, not after measuring:**

- A **public option** expands the API surface and pulls in `API.md`, the `check-docs` guarded region, and
  the `check-32` probe (spec 41/40-01 rules apply).
- A **default change** affects **every caller-provided allocator**, not only the three canonical tuples —
  the independent reason spec 39-01 rejected default adoption.
- The **negative control** (§7) needs a way to disable sorting that does **not** itself ship as public
  API if the option does not.

## 7. Feasibility prototype — not end-to-end

Extend `src/bench_smp_layout.zig`. **Call it a feasibility prototype:** it does not model clones,
accumulation, or result assembly, and must not be described as end-to-end.

Current gaps to fix in the probe itself: its `sort_zero` cell **allocates before the timed region**
(`bench_smp_layout.zig:167`) and it **sorts slices with stable `std.mem.sort`** (`:233`) — both
misrepresent the candidate.

Cells must time the **chosen production representation** and the **full candidate cost**: scratch
allocation, header **and** payload allocation, sorting, zeroing, and cleanup. Sorting that recovers
1.7 ms and costs 1.4 ms is not a win, and only in-region timing shows it.

**Proceed to production only if the prototype clears the gap with margin.**

## 8. Measurement

- **Canonical harness only** (`run-compare-bench.sh`, fresh process per cell, 3 warmup / 21 timed, ≥5
  process medians + full ranges).
- **Both hosts**, and **all three canonical tuples — rawr/SMP, rawr/libc, CRoaring/libc.** *(The first
  draft said "all three allocators"; there are only two allocator kinds. Corrected.)*
- **Dual stop-gate, construction binding.** Spec 35's combined row improved 0.038 ms while construction
  got *slower*; a combined-only gate would have authorized a large migration to buy nothing. **Gate
  `lazy-or-construction` explicitly.**
- **libc must not regress.** If it does, outcome is opt-in scope per spec 39-01 — and then the canonical
  row does not move, so report "at parity when enabled", never "row closed".
- Whole-board check for spec-28 layout noise; sub-~1.2x M4 ratios are at the measurement floor.

## 9. Acceptance

- Private pending path per §2.1: never publishes unzeroed state, cleans up partial batches, stays
  non-public.
- Scratch per §2.2, including the **fallback-on-scratch-failure** behaviour.
- Prototype (§7) shows net gain with **full candidate cost timed in-region**; recorded.
- **Three arms measured** (§4) as fresh-process cells; **arm 3 vs arm 2** reported as the ordering result
  and **arm 2 vs arm 1** as the batching effect.
- Canonical `lazy-or-construction` **≤1.10x on M4**, or an explicit reasoned stop.
- Combined `lazyOr+repair` does not regress; `lazyXor` does not regress; no other board row moves beyond
  the 5% layout tolerance.
- Zen 4 not regressed; libc not regressed, or opt-in scope with the §8 reporting rule.
- **Failure-injection suite green** (§5) — no leaks, inputs untouched, at every injection point.
- Per-bitset allocation count **still two**; scratch allocations reported separately.
- All four suites green — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
  `check-package`.
- **Negative control on the mechanism:** disable sorting in the production path and confirm the gap
  returns. A win that survives disabling the lever was never the lever.

## 10. Out of scope

- **Contiguous slab.** Only if this spec wins with headroom — it breaks per-container lifetimes
  (containers are individually freed on repair, replacement, `deinit`), and spec 35's comparable design
  implied a ~98-site migration and returned NO-GO. Separate spec.
- Allocator replacement (closed, spec 18); transient arenas (lose, spec 17).
- The microarchitectural attribution question — it would bound the ceiling, not gate the test.

## 11. Estimate

**M/L** — the production change is confined to the lazy path, but the pending-allocation path, failure
injection, and three-arm measurement are each substantial.
