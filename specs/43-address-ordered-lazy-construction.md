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

### 2.2 Scratch for the sort — one design, decided

```zig
const Pending = struct {
    payload_addr: usize,          // sort key — no dereference in the comparator
    header: *BitsetContainer,     // association preserved alongside the key
};
```

- **Exactly one scratch allocation:** `[]Pending`, length = the **lazy-bitset-eligible pair count**
  (§2.3) — **not** the matched-pair count.
- **Comparator reads `payload_addr` only.** Rejected: `[]*BitsetContainer` sorted by `a.words < b.words`
  — 8-byte elements, but every comparison **dereferences a header**, so ~229k comparisons at 16k buffers
  each risk a cache miss on randomly-ordered headers. A 16-byte element whose key is in-line and
  traversed sequentially is the cheaper shape even though the element is wider. Rejected: index
  permutation — needs a second mapping for no benefit.
- **Sort:** `sortUnstable` (pdq) on `payload_addr`.
- **Size bound:** matched pairs ≤ `min(a.size, b.size)` ≤ 65,536 → ≤ 1 MiB of scratch.
- **Allocator:** the bitmap's allocator (the one passed to `lazyOr`).
- **Ownership:** scratch is owned by the construction routine and freed before it returns, on every path.

**No sort-cost estimate is carried anywhere in this spec.** The prototype (§7) must measure **this exact
struct, this comparator, this sort** and establish the number.

**Scratch allocation failure → retry through the existing interleaved path.** If *that* path also fails,
propagate its error. *(Corrected from "fall back and succeed", which over-promised: a genuinely exhausted
allocator can still legitimately return OOM.)*

### 2.3 Eligible count — a pre-pass, not the matched-pair count

The existing predicate (`bitmap.zig:2344`) is:

```zig
const use_lazy_bitset = op == .xor or bitset_conversion
    or isBitsetContainer(c_a) or isBitsetContainer(c_b);
```

For **forced** lazy OR (`bitset_conversion == true`) every matched pair is eligible, so the two counts
coincide. For **selective** lazy OR they do **not**: only pairs with a bitset operand need a pending
buffer. Sizing off matched pairs would allocate **thousands of unused 8 KB buffers** — up to 512 MB in
the worst case — turning an ordering optimization into a memory-consumption defect.

**Therefore: one pre-pass over the merge walk evaluating the same predicate**, producing the exact
eligible count before any pending allocation. No allocation in the pre-pass.

Its cost is an extra **`O(a.size + b.size)`** scan with container-type checks — the merge walk advances
through *both* key arrays; only the number of *matches* is bounded by `min(a.size, b.size)`. *(An earlier
draft wrote `O(min(a.size, b.size))`, conflating the match count with the walk length.)* That cost must be
**inside the timed region** of both the prototype and production, since it is part of the candidate.

*(Enum corrected throughout: `lazyMergeTwo(comptime op: ManyOp, ...)` (`bitmap.zig:2317`) takes
**`ManyOp = enum { bor, xor }`** (`bitmap.zig:2131`) — **not** `TwoWayOp` at `:1859`, which is a different
enum that also happens to spell one variant `bor`. An earlier draft wrote `.or_`, then cited the wrong
enum.)*

### 2.4 Publication contract

The pending pool **owns** a header and payload until that container has been (1) zeroed, (2) accumulated,
and (3) successfully appended through an ownership-taking helper. **Only on successful append does the
result own it.**

Consequences, which is the point of stating it this way: any failure before the append frees through the
**pool**, any failure after frees through the **result**, and no container is ever owned by both or
neither. Error cleanup becomes mechanical rather than case-by-case.

**Correction to the first draft: "allocation count unchanged" is now false and is withdrawn.** The
honest framing: **two allocations per bitset are preserved**, and scratch adds a small, bounded number on
top. Report scratch allocations separately so the arms in §4 stay attributable — this spec still is not
an allocation-count-reduction lever (specs 27 and 35 both regressed M4 SMP that way), and that
distinction is the reason to keep the per-bitset count fixed.

## 3. Sort constraints

- **Sort `[]Pending` by its inline `usize` key with `sortUnstable` (pdq), never `std.mem.sort`.** The
  reason the element shape matters: spec 38 measured **86.98 ns/op** sorting `[]u8` **slices**, which at
  ~16,364 buffers is **~1.4 ms** — enough to consume the entire gain. §2.2's element is 16 bytes with the
  key in-line and no dereference, which is why it was chosen, but **its cost is not yet measured and no
  estimate is carried here** — the prototype (§7) establishes it. *(An earlier draft quoted ~0.13 ms from
  a flat-`usize` measurement; that figure does not apply to this representation and is withdrawn.)*
  Spec 38-00 also used stable block sort where pdq was intended — **state which sort is used and confirm
  it in the code.**
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

**How they appear in the manifest.** Canonical tuples are keyed by *(implementation, allocator)*, so
three rawr/SMP variants cannot share one row. Express the arms as **three named operation rows**, each
measured against the **same CRoaring/libc tuple**:

| Row | Arm |
| --- | --- |
| `lazy-or-construction` | 1 — existing interleaved baseline (the canonical row; unchanged) |
| `lazy-or-construction-batched` | 2 — batched, unsorted |
| `lazy-or-construction-batched-sorted` | 3 — batched, sorted |

Arm rows are **diagnostic**: they exist to attribute the effect and are not board rows. Only
`lazy-or-construction` gates.

**Mechanism — runtime dispatch, one worker build. DECIDED.** A build option cannot express this: the
parity worker is built **once** (`run-compare-bench.sh` builds `bench-parity-worker -Dcpu=native`) and
selects rows **at runtime**, so a compile-time switch cannot produce three arms from one binary.
Rejected: compiling three worker/library variants — it reintroduces exactly the whole-binary layout noise
spec 28 documented, where adding code moves untouched rows with instruction-identical disassembly. Three
binaries would make arm-to-arm deltas uninterpretable at the ~1.2x measurement floor.

Instead: a **private `ConstructionMode` dispatch** (`.baseline`, `.batched_unsorted`, `.batched_sorted`)
threaded through the lazy construction path, reached by the benchmark through an **internal export**, not
public API. It joins `roaring.zig`'s internal-export manifest with a reason string, so `check-docs`
classifies it and it stays out of `API.md`, the guarded region, and the `check-32` probe.

**Staging — pin it now:**

| Stage | `lazy-or-construction` (canonical, gates) | diagnostic rows |
| --- | --- | --- |
| Before adoption | baseline behaviour | `-batched`, `-batched-sorted` |
| After adoption | **sorted-default** | a separately named row retains the **old baseline** for the §9 negative control |

Without that second step the negative control loses its reference the moment the default changes.

## 5. Failure semantics — explicit gate

A pending pool creates many owners before result assembly, which is exactly where leaks hide.

**Allocation-failure injection required at:** scratch, pending headers, pending payloads, unmatched
clones, and result assembly.

**Every injected failure must:** leave **both inputs untouched**, and leak **nothing** — no assigned
container, no pending container, no scratch. Use a leak-checking GPA, never `c_allocator`.

Scratch failure specifically must **retry through the existing interleaved path** (§2.2) and propagate
only an error arising from *that* path — it must not propagate the scratch OOM directly. A genuinely
exhausted allocator may still legitimately return OOM from the retry.

## 6. Scope and selection — DECIDED

`lazyMergeTwo` serves forced and selective `lazyOr` **and** `lazyXor` (`bitmap.zig:1128`, `:2344`; the
branch condition includes `op == .xor`).

**Scope: lazy OR only — `op == .bor`, both forced and selective. `lazyXor` is excluded.**
The forced/selective flag changes only *how many* pairs take the bitset branch, not the mechanism, so
splitting on it would be arbitrary; splitting on `op` is a clean boundary that keeps XOR's behaviour
identical. The canonical row is `lazyOr(allocator, b, true)` (`bench_croaring.zig:507`), so this scope
covers the target. **`lazyXor` carries no-regression coverage** as a shared-helper obligation.

**Selection: DEFAULT behaviour. No public API in this spec.**

- A public option would pull in `API.md`, the `check-docs` guarded region, and the `check-32` probe
  (spec 41 / 40-01 rules) — real cost for a knob nobody asked for.
- **Default is also the only way the canonical row can close.** An opt-in leaves the canonical default
  untouched, which is exactly why spec 39-01 reports "at parity when enabled" rather than closing a row.
  Choosing opt-in up front would concede the target before measuring.
- **If libc regresses, this spec STOPS.** It does **not** silently become an opt-in spec: opt-in requires
  a public options entry point, which pulls in `API.md`, the `check-docs` guarded region, and the
  `check-32` probe — precisely the surface this spec declined to add. Expanding scope mid-flight to
  rescue a failed default would smuggle that work in under a spec that says "no public API".

  **Outcome in that case:** record the measured result, report the row as **not closed**, and open a
  **follow-up opt-in spec** that owns the API and documentation cost explicitly — as spec 39-01 did,
  where opt-in was a deliberately scoped chunk rather than a contingency.
- **The arm control is internal** — the §4 runtime `ConstructionMode` dispatch reached through
  `roaring.zig`'s internal-export manifest, **not** a build option and not public API. It
  drives the §4 arms and the §9 negative control, and ships as no part of the library surface.

## 7. Feasibility prototype — not end-to-end

Extend `src/bench_smp_layout.zig`. **Call it a feasibility prototype:** it does not model clones,
accumulation, or result assembly, and must not be described as end-to-end.

Current gaps to fix in the probe itself: its `sort_zero` cell **allocates before the timed region**
(`bench_smp_layout.zig:167`) and it **sorts slices with stable `std.mem.sort`** (`:233`) — both
misrepresent the candidate.

Cells must time the **chosen production representation** (§2.2's `Pending` struct, its comparator, its
sort) and the **full candidate cost**: scratch allocation, header **and** payload allocation, sorting,
and zeroing. Sorting that recovers 1.7 ms and costs 1.4 ms is not a win, and only in-region timing shows
it.

**Cleanup splits in two, and the split must match the canonical row:**

- **Scratch release is construction cost — time it inside the region.** It exists only to build the
  result.
- **Retained header/payload teardown is NOT construction — time it outside.** The canonical construction
  row times `lazyOr(...)` alone and calls `result.deinit()` after stopping the clock
  (`bench_croaring.zig:507-512`). A prototype that folds result teardown into the timed region measures a
  different quantity than the row it is trying to predict.

**Proceed to production only if the prototype clears the gap with margin.**

## 8. Measurement

- **Canonical harness only** (`run-compare-bench.sh`, fresh process per cell, 3 warmup / 21 timed, ≥5
  process medians + full ranges).
- **Both hosts**, and **all three canonical tuples — rawr/SMP, rawr/libc, CRoaring/libc.** *(The first
  draft said "all three allocators"; there are only two allocator kinds. Corrected.)*
- **Dual stop-gate, construction binding.** Spec 35's combined row improved 0.038 ms while construction
  got *slower*; a combined-only gate would have authorized a large migration to buy nothing. **Gate
  `lazy-or-construction` explicitly.**
- **libc must not regress. If it does, this spec STOPS** (§6) — record the result, report the row as not
  closed, and open a follow-up opt-in spec that owns the API/docs cost. Do not expand this spec's surface
  mid-flight.

### 8.1 Two gates, in order — the candidate must prove itself before it becomes default

"Only the canonical row gates" created a circular requirement: while the default is baseline, the
candidate lives in a diagnostic row, so it could never demonstrate ≤1.10x *before* adoption, yet adoption
was supposed to be justified by clearing the gate. Split it:

**Gate 1 — promotion (candidate still diagnostic, default unchanged):**

- `lazy-or-construction-batched-sorted` **≤1.10x** against the shared CRoaring/libc reference;
- **arm 3 beats arm 2** — the ordering effect is real and not merely a batching artifact;
- **libc does not regress** (else STOP per §6).

Failing gate 1 means the lever does not work, and **nothing is adopted** — no production default changes,
so the failure costs nothing but the diagnostic rows.

**Gate 2 — adoption (default switched to sorted):**

- **Rerun the canonical row and the whole board**, fresh processes;
- canonical `lazy-or-construction` **≤1.10x**;
- no other board row moves beyond the 5% layout tolerance;
- the retained old-baseline diagnostic row (§4 staging) still available for the §9 negative control.

Gate 2 is a **separate measurement**, not an inference from gate 1: adoption changes which code the whole
binary contains, and spec 28 showed that alone moves untouched rows with instruction-identical
disassembly. A gate-1 pass does not predict the canonical row's post-adoption value.
- Whole-board check for spec-28 layout noise; sub-~1.2x M4 ratios are at the measurement floor.

## 9. Acceptance

- Private pending path per §2.1: never publishes unzeroed state, cleans up partial batches, stays
  non-public. **Publication contract §2.4 holds** — pool owns until successful append, result owns after.
- Scratch per §2.2: **exactly one** `[]Pending` allocation, comparator on `payload_addr`, `sortUnstable`,
  freed on every path; scratch failure retries the existing path and propagates only that path's error.
- **Scope is lazy OR only**; `lazyXor` behaviour byte-identical to baseline.
- Prototype (§7) shows net gain, measuring **§2.2's exact representation**, with scratch release inside
  the timed region and result teardown outside; recorded.
- **Three arms measured** (§4) as fresh-process cells under the named diagnostic rows; **arm 3 vs arm 2**
  reported as the ordering result and **arm 2 vs arm 1** as the batching effect.
- Canonical `lazy-or-construction` **≤1.10x on M4**, or an explicit reasoned stop.
- Combined `lazyOr+repair` does not regress; `lazyXor` does not regress; no other board row moves beyond
  the 5% layout tolerance.
- Zen 4 not regressed; **libc not regressed — a libc regression is a STOP** (§6), not a fallback to
  opt-in scope within this spec.
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

## 10.1 Chunking sketch

Pending review. The two-gate structure maps directly onto three chunks:

- **43-00 — feasibility prototype.** `bench_smp_layout.zig` cells timing §2.2's exact representation and
  the full candidate cost (§7). Stop here if the prototype does not clear the gap with margin.
- **43-01 — diagnostic production path.** Pending path, scratch, eligible pre-pass, ownership contract,
  failure injection, runtime three-arm dispatch, diagnostic rows. **Ends at gate 1**; default unchanged,
  so a gate-1 failure changes no production behaviour.
- **43-02 — adoption.** Switch the default, retain the old-baseline diagnostic row, rerun canonical and
  whole board. **Gate 2.**

## 11. Estimate

**M/L** — the production change is confined to the lazy path, but the pending-allocation path, failure
injection, and three-arm measurement are each substantial.
