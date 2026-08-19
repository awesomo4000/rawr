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
converts bad order into stalls. Knowing *which* effect would tell us the ceiling; it is not required to
test the lever.

Everything else is closed off: rawr zeroes **the same 8 KB per matched pair** as CRoaring (no volume
lever — both do one fresh 8 KB zero-fill per pair, no `calloc` trick, no reuse), and the zero-fill
**codegen is identical** (`mov w1, #0x2000` → `bl _bzero` on both sides). So order is the only remaining
variable that measurement has implicated.

## 2. Two variants, staged — A first

### Variant A — batch-allocate, address-sort, assign in key order. **No ownership change.**

1. Determine the matched-pair count (upper-bounded by `min(size_a, size_b)`; a pre-pass may sharpen it).
2. Allocate that many 8 KB bitsets up front.
3. **Sort the resulting addresses ascending** — as `usize`, not as slices (see §3).
4. Assign them to containers in key order, so the subsequent zero-fill and accumulation walk memory
   ascending.

Each container remains individually owned and individually freeable. `deinit`, repair, and container
replacement are untouched. This is spec 37's proven intervention moved from *after* traversal to
*before* it.

### Variant B — contiguous slab. **Only if A wins and leaves headroom.**

Carve the bitsets from one contiguous slab so allocation order *is* address order with no sort at all.

**B changes ownership and is therefore gated behind A.** Roaring containers have independent lifetimes —
they are individually freed on repair, replacement, and `deinit`. A slab means an individual container
free cannot return memory, so B needs a slab-lifetime scheme (whole-slab free, or a survivor count).
**Spec 35 is the precedent and the warning:** its transient-lazy-bitset design implied a ~98-site
container-union migration and returned NO-GO. Do not start B on optimism.

## 3. Design constraints — each from a spec that already paid for it

- **Sort `usize` addresses with `sortUnstable` (pdq), never `std.mem.sort`.** Spec 38 measured **86.98
  ns/op** sorting `[]u8` **slices**; at ~16,364 buffers that is **~1.4 ms** and would consume the entire
  gain. Raw `usize` sorting is ~8 ns/op → **~0.13 ms**. The element type is the difference between a win
  and a self-inflicted loss. Also note spec 38-00 accidentally used stable block sort where pdq was
  intended — **state which sort is used and confirm it in the code**.
- **Contiguity is NOT allocation-count reduction — separate them.** Specs 27 and 35 both regressed M4 SMP
  by cutting allocation counts (spec 35 removed 16,364 allocations and got *slower*: 4.026 → 4.109 ms).
  Variant A deliberately keeps the allocation count **identical** and changes only the order in which
  addresses are used. If a future step also reduces count, it needs its own arm, or a win/loss cannot be
  attributed.
- **Do not re-propose residency.** Spec 36 **refuted** first-touch/page-faults on M4: 40 faults across
  ~134 MB in play, 100% page reuse, no material gain from conditioning. Pages are resident; the cost is
  *touching* them in a bad order. Any framing of this spec as pre-touch or residency is wrong.
- **Read-traversal sorting stays NO-GO** (spec 38: M4 1.221x, Zen 4 1.344x). Frees want **descending**
  (LIFO), reads want **ascending** — do not conflate this spec with either. This is *construction*.

## 4. Prototype before production

Extend `src/bench_smp_layout.zig` — the existing zero-rawr-code reproducer — to model variant A end to
end: batch-allocate, sort, then zero in assignment order, **including the sort cost inside the timed
region**. Sorting that recovers 1.7 ms and costs 1.4 ms is not a win, and only in-region timing shows it.

**Proceed to production only if the prototype clears the gap by a margin that survives the sort cost.**

## 5. Measurement

- **Canonical harness only** (`run-compare-bench.sh`, fresh process per cell, 3 warmup / 21 timed, ≥5
  process medians + full ranges). Spec 35's focused harness produced a **warmed-context artifact** —
  1.155x where canonical said 1.727x — because validation and prototype passes preconditioned SMP before
  the production cell. Do not measure this row in a harness that allocated the same population earlier in
  the process.
- **Both hosts**, all three allocators.
- **Dual stop-gate, and construction is the binding one.** Spec 35's combined row improved by 0.038 ms
  while construction got *slower*; a combined-only gate would have authorized a large migration to buy
  nothing. **Gate `lazy-or-construction` explicitly**, not just the combined row.
- **libc must not regress.** If it does, the outcome is opt-in scope per spec 39-01 — and then the
  **canonical row does not move**, so say "at parity when enabled", never "row closed".
- Whole-board check for spec-28 layout noise; sub-~1.2x M4 ratios sit at the measurement floor.

## 6. Acceptance

- Prototype in `bench_smp_layout.zig` shows a net gain **with sort cost timed in-region**; recorded.
- Production variant A implemented with **allocation count unchanged** from baseline — verified by
  counting, not asserted.
- Canonical `lazy-or-construction`: **≤1.10x on M4**, or an explicit, reasoned stop.
- Combined `lazyOr+repair` does not regress; no other board row moves beyond the 5% layout tolerance.
- Zen 4 not regressed; libc not regressed, or scope reduced to opt-in with the reporting rule above.
- All four suites green — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
  `check-package`.
- **Negative control on the mechanism:** run the production path with sorting disabled and confirm the
  gap returns. A win that survives disabling the lever was never the lever.

## 7. Out of scope

- Variant B (slab), unless A wins with headroom — separate spec.
- Allocator replacement (closed, spec 18) and transient arenas (lose, spec 17).
- The microarchitectural attribution question. It would bound the ceiling; it is not needed to test A.

## 8. Estimate

**M** — the prototype is small, the production change is confined to the lazy-OR construction path, and
the measurement protocol is the larger half.
