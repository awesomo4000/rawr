<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 39: Descending free order for mass bitset release

Campaign: [31-structural-parity-campaign.md](31-structural-parity-campaign.md). Predecessor:
[38-00](38-00-address-sort-measurement.md). Background:
[allocator-address-order-pathology.md](../docs/allocator-address-order-pathology.md),
[smp-free-order.md](../docs/smp-free-order.md).

## Target workload — CORRECTED

`38-00` measured an **all-bitset teardown corpus**, which does **not** correspond to any production step:
in canonical sparse lazy-OR the 16,364 transient bitsets are **demoted and freed inside
`repairAfterLazy`**, and the *resulting* bitmap holds **65,496 array containers**. So `deinit()` of a
lazy-OR result is an **array** teardown, not a bitset teardown, and **a default-off `deinit` option
cannot close the lazy-OR construction gap.**

**The production home of "free ~16,364 8 KB bitsets" is `repairAfterLazy`'s demote path.** That is this
spec's primary target.

### Which canonical row can actually move — and which cannot

| row | can this improve it? |
|---|---|
| `lazy-or-construction` | **NO — that row never calls repair.** |
| `lazy-or-repair-only` | **Probably NO** — the expected benefit lands on the *next* construction, which is **outside that row's timer.** |
| **steady-state `lazyOr+repair`** | **YES — this is the only row that can move**, because one iteration's descending frees condition the next iteration's allocations. |

**Diagnostic sequence (pinned):**

```
repair/free candidate  →  stage-3 noise  →  next lazyOr construction
```

**Report separately:** (i) **current repair cost**, (ii) **following construction cost**, (iii) **full
cycle**. **The hard gate is FULL-CYCLE improvement.** A repair-only improvement that does not show up in
the full cycle is not a win, and a repair-only regression offset by a larger construction gain still is
one — which is exactly why the cycle is the gate.

**Note what `38-00` did and did not test:** it sorted repair's **cardinality read traversal** (regressed)
and separately measured **descending frees on a synthetic bitset corpus** (won, survived stage-3 noise).
It did **not** test **descending demote frees inside repair** — the combination this spec targets.

Two workloads must be benchmarked, and a **representative container mix** used for anything claiming to
be teardown:

| workload | contents | role |
|---|---|---|
| **repair demote frees** | 16,364 transient 8 KB bitsets, released in a **deferred descending pass** (see below) | **primary** — production path |
| **result teardown** | the real lazy-OR result: **~65,496 arrays** (+ any surviving bitsets/runs) | representative teardown; the array-dominated reality |
| all-bitset teardown | 16,364 bitsets, no interleaving | `38-00`'s corpus — retained as a **control only**, not a production claim |

**No workload-shape claims beyond what is measured.** (An earlier draft asserted a "datalog-backend
shape" fit; removed — unsupported.)

## Deferred-free design — PINNED (descending frees cannot stay interleaved)

`repairAfterLazy` today does, per demoted container: **`bitsetToArray` allocation → immediately free the
old bitset.** A *global* descending free order is therefore impossible without restructuring. Two options
existed; **option (a) is selected:**

- **(a) SELECTED — key-order conversion, deferred descending free.** Keep the conversion pass in key
  order (preserving result allocation order and key order exactly as today), **collecting the old bitset
  pointers**; then run a **separate descending free pass**.
- (b) Rejected — reordering the conversion pass itself, which would change result allocation order and
  perturb far more than the frees.

**Consequences that must be measured, not assumed:**

- **Frees are NO LONGER INTERLEAVED with allocations.** That is a deliberate change of shape, not an
  incidental one, and it may itself alter the result independent of ordering — so the deferred-but-
  ascending / deferred-but-unordered case must be measured as a control, isolating *deferral* from
  *direction*.
- **Temporary live-memory peak RISES.** Old bitsets are held while new arrays are allocated: up to
  ~16,364 × 8 KB ≈ **134 MB retained simultaneously** that today is released incrementally. **Measure and
  report peak RSS**; this is a real cost the win must justify.
- **Rung 0 is NOT free on this path.** Deferring requires storing the collected old-bitset pointers —
  **~N pointers of scratch (≈131 KB at n=16,364)** plus the fill. "Reverse iteration" then means reverse
  iteration *of that deferred array*. So rung 0's cost is **scratch allocation + fill, not zero**; it is
  still far cheaper than any sort, but the cost table below must be read with that correction. (Rung 0 is
  genuinely free only for an in-place traversal we already perform — which is the read side, not this
  one.)

## Direction discipline — frees and reads want OPPOSITE orders

- **Frees want DESCENDING.** `SmpAllocator.free` pushes onto a **LIFO** freelist, so descending frees
  hand back **ascending** addresses to the next allocation burst. This is the `38-00` win.
- **Reads want ASCENDING.** A cardinality/zeroing traversal has no LIFO involved; it wants monotone
  ascending order.

**Consequence for the repair retest:** repair's read traversal must be retested **ascending**, which is
the *opposite* direction from the teardown win. And **if rung 0 (reverse iteration) wins for frees, its
read-side equivalent is simply the existing forward order — i.e. it supplies no new repair candidate at
all.** Do not describe the retest as "the winning rung applied to repair" without inverting direction.

## Reorder-mechanism ladder — cheapest sufficient wins

`38-00`'s verdicts were measured with `std.mem.sort`, which Zig 0.16 implements as a **stable block
sort** (not pdq). Stability is unnecessary — payload addresses are unique — and block sort pays for
stability *and* in-place rotation, so **both `38-00` verdicts carried an unquantified, probably large
reorder tax.** Quantify rung 4 so that tax is known.

**⚠ The costs below are PRIOR ESTIMATES from the 1-vCPU Cascade Lake Linux VM** in
`smp-free-order.md` §4.1 — **not** from M4 or the current Zen 4 host, and from a machine that could not
exercise slot rotation. They establish the **relative ordering** of mechanisms only. **Rung selection
must use fresh two-host measurements.**

| rung | mechanism | prior estimate @ n≈16,364 |
|---|---|---:|
| **0** | reverse iteration of the **deferred pointer array** | **scratch alloc + fill only** (not zero — see Deferred-free design) |
| **1** | 1-pass span-adaptive bucket partition | ~0.184 ms |
| **2** | LSD radix on normalized payload addresses | ~0.240 ms |
| **3** | `std.mem.sortUnstable` (pdq) | ~1.423 ms (reference) |
| **4** | `std.mem.sort` (block, stable) — the `38-00` baseline | **unmeasured — measure it** |

**Stop at the cheapest rung that captures the effect.** Report **measured** reorder cost as its own
column at every rung.

### Rung 0 — must be qualified, not assumed

The container array is **fundamentally key-ordered**, not allocation-ordered. In `lazyMergeTwo`
construction and append coincide, so reverse iteration *may* approximate descending payload order there —
but **mutation, container replacement, array↔bitset conversion, and long-lived bitmaps all break that
correlation.** Therefore rung 0 requires **measured address-order quality (travel and/or page-locality)
on the actual target lifecycles**, not only on a fresh synthetic corpus. If the correlation does not hold
on the repair-demote path, rung 0 is out for that path regardless of its cost.

### Rung 1 — bucket partition, specified

- Key: **`(address − min_address) >> shift`**, `min_address` from a first pass.
- **`shift = max(0, @bitSizeOf(usize) − @clz(span) − log2(nbuckets))`** — span-adaptive, and **portable:
  do NOT hardcode 64**, production must compile on 32-bit targets even though the performance gates run
  on M4 and Zen 4. A **fixed** shift is a recorded trap: 256 buckets over a 160 MB span left travel
  **unchanged at 6856x**.
- **`span == 0` (zero or one item) must be pinned:** with `n <= 1` there is nothing to reorder — return
  immediately; `@clz(0)` is `@bitSizeOf(usize)` so the shift formula degenerates safely, but the
  early-return is required so the count/scatter passes are never entered with a degenerate span.
- **Clamp** the bucket index to `nbuckets − 1` (guards the `hi` element and any rounding).
- Steps: **count → prefix-sum → scatter into scratch → copy back**.
- **Within-bucket order is unspecified/arbitrary** (a bucket spans `span/nbuckets`); state that this is
  accepted, and that descending output is produced by **iterating buckets high→low**.
- `nbuckets` default **4096** (2 × 4096 × 4 B counters ≈ 32 KiB — trades L1 residency against
  resolution); sizing recorded.

### Rung 2 — radix, specified

- **Three LSD passes do not sort a full `usize`** — that must be pinned, not implied.
- Key: **normalized `address − min_address`**, so only the **span's** significant bits need sorting.
- **Bits per pass** and **pass count adaptive to the observed span**: passes =
  `ceil(significant_bits / bits_per_pass)` where
  **`significant_bits = @bitSizeOf(usize) − @clz(span)`** — **portable, not hardcoded 64.** State
  `bits_per_pass` (8 or 11) and the resulting count-table size.
- **`span == 0` / `n <= 1`: early-return, no passes.** Same requirement as rung 1.
- **Descending output** produced explicitly (reverse the final scatter, or reverse-iterate the result) —
  do not leave direction implicit.
- **Scratch and count-table sizes** stated; **allocation failure falls back to unreordered**.

## API — pinned before chunking

**Corrections to the previous draft:** `deinit()` always uses the bitmap's **stored** allocator, so the
"per-call allocator can differ" rationale does **not** apply to teardown. And with allocator detection
rejected, **a size gate cannot exclude libc-on-M4** — that has to be a contract, not a mechanism.

**Enforceable contract:**

- **Ordinary `deinit()` is unchanged.** The default path never reorders.
- **An explicit variant / options call** enables descending frees.
- **The caller accepts allocator-specific behaviour** by choosing it — the documented assertion is "my
  allocator benefits from descending free order," not "I am on M4."
- So **"libc-on-M4 excluded" means excluded from the DEFAULT path** — it remains possible to opt into,
  and that is acceptable and documented. It is not, and cannot be, made impossible.

**Surface scope — PINNED (minimal):**

| decision | |
|---|---|
| **ADD** | **`repairAfterLazyWithOptions(...)`** — the only new public entry point |
| **UNCHANGED** | **`repairAfterLazy()`** — default path untouched, no behaviour change |
| **EXCLUDED from `39-01`** | **`deinit`**, **`clearRetainingCapacity`**, **`Roaring64Bitmap`**, **`OwnedBitmap`** |

`OwnedBitmap` is excluded on principle (**its arena owns teardown**, so container-level free order is
irrelevant); the other three are excluded to keep the first shipped surface minimal — they may be
revisited in a later spec on their own evidence, not on this one's.

## Size gate — unit must be pinned

**PINNED: the gate counts the number of BITSETS ACTUALLY BEING DEMOTED** in this repair call — not total
containers, and not all reorderable payloads.

Rationale: `38-00`'s crossovers (**64 M4 / 1,024 Zen 4**) were measured in **8 KB bitset payload count**,
and the demoted-bitset count is exactly the population this mechanism reorders. A total-container gate
would be wrong by ~4× on a lazy-OR result (65,496 containers vs 16,364 demoted bitsets) and is
unjustified without mixed-corpus measurement.

**The array-result teardown remains DIAGNOSTIC EVIDENCE ONLY** — it characterises the array-dominated
reality of a lazy-OR result, but it must **neither select nor block** the bitset-demotion mechanism.

## Gate re-derivation

The 64/1,024 crossovers came from **block sort**. The **selected rung needs its own size sweep**, on both
hosts, with **stage-3 noise re-verified per rung** (survival at rung 4 does not transfer to a cheaper,
lower-quality reorder) and **measured address travel / order quality** reported alongside timing.

## Scope

- **Steady state only.** M4 first-cycle was inconclusive in `38-00`.
- **Not in scope:** compaction/relayout, the private pool (`smp-free-order.md` Idea B), any upstream
  `SmpAllocator` change, and repair's read traversal beyond the single inverted retest.

## Measurement

Canonical protocol; **rawr/SMP and rawr/libc**; **M4 and Zen 4/WSL2**; five fresh-process medians + full
ranges; one process per tuple; reorder cost **inside** the timed region and reported separately; range
separation before any claim; stage-3 noise control per rung.

## Correctness

- **Free order is correctness-invariant:** every container freed **exactly once** at every rung —
  shadow-bitmap verified; no leak, no double-free; failure injection green.
- Rungs 1–2 must be **permutations** (same multiset in and out).
- Reorder-scratch failure **degrades to unreordered**, never propagates an error.
- Repair's demote/compaction results must be **byte-identical** to today regardless of free order.
- `zig build test`, `zig build difftest`, `ReleaseSafe`, `ReleaseFast`.

## Acceptance

- **Primary:** descending demote frees measured **inside `repairAfterLazy`** via the pinned **deferred**
  design, both allocators, both hosts, stage-3 noise verified — plus the **deferred-but-not-descending
  control** isolating deferral from direction, and **peak RSS reported** for the raised temporary live
  footprint.
- **Hard gate is FULL-CYCLE improvement** on steady-state `lazyOr+repair`, with **repair cost, following
  construction cost, and full cycle reported separately**. `lazy-or-construction` and
  `lazy-or-repair-only` are documented as rows this mechanism **cannot** move.
- Representative **result teardown (~65,496 arrays)** measured; the all-bitset corpus reported as a
  control only.
- Rungs 0–2 measured (3 reference, 4 quantified); **rung 0 qualified by measured order quality on the
  real lifecycles**; cheapest sufficient rung selected.
- Repair read-traversal retest run **once, ASCENDING**, with the explicit note that rung 0 offers no
  read-side candidate; verdict recorded either way.
- Size gate counts **demoted bitsets**; threshold **re-derived for the selected rung** on both hosts.
- API is **`repairAfterLazyWithOptions(...)` only**; `repairAfterLazy()` unchanged; `deinit`,
  `clearRetainingCapacity`, `Roaring64Bitmap`, `OwnedBitmap` **excluded from `39-01`**; libc-on-M4
  excluded **from the default path** (opt-in remains possible, and that wording is used).
- Algorithms **portable**: `@bitSizeOf(usize) − @clz(span)` (never hardcoded 64), `span == 0` / `n <= 1`
  early-return pinned, compiles on 32-bit targets.
- Reorder costs reported as **fresh two-host measurements**, with the `smp-free-order.md` figures cited
  only as prior estimates.
- Correctness surface green.

## Chunk plan

- **`39-00`** — measurement: the two production workloads, the ladder, rung-0 order-quality
  qualification, inverted repair retest, gate re-derivation. No production change.
- **`39-01`** — ship the selected rung behind the pinned opt-in.

## Estimate

M for `39-00` — rung 0 is free and rung 1 is small, but the repair-demote instrumentation, the
array-dominated teardown corpus, and per-rung stage-3 re-verification are real work. M for `39-01`.
