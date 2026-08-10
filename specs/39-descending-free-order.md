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
spec's primary target. It sits on the canonical row, so a win there is a real board movement rather than
an opt-in extra.

**Note what `38-00` did and did not test:** it sorted repair's **cardinality read traversal** (regressed)
and separately measured **descending frees on a synthetic bitset corpus** (won, survived stage-3 noise).
It did **not** test **descending demote frees inside repair** — the combination this spec targets.

Two workloads must be benchmarked, and a **representative container mix** used for anything claiming to
be teardown:

| workload | contents | role |
|---|---|---|
| **repair demote frees** | 16,364 transient 8 KB bitsets, freed **interleaved** with `bitsetToArray` allocations | **primary** — production path, canonical row |
| **result teardown** | the real lazy-OR result: **~65,496 arrays** (+ any surviving bitsets/runs) | representative teardown; the array-dominated reality |
| all-bitset teardown | 16,364 bitsets, no interleaving | `38-00`'s corpus — retained as a **control only**, not a production claim |

**No workload-shape claims beyond what is measured.** (An earlier draft asserted a "datalog-backend
shape" fit; removed — unsupported.)

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

| rung | mechanism | cost @ n≈16,364 |
|---|---|---:|
| **0** | reverse iteration of the container array — **no reorder** | **0.000 ms** |
| **1** | 1-pass span-adaptive bucket partition | **0.184 ms** |
| **2** | LSD radix on normalized payload addresses | **0.240 ms** |
| **3** | `std.mem.sortUnstable` (pdq) | 1.423 ms (reference) |
| **4** | `std.mem.sort` (block, stable) — the `38-00` baseline | **unmeasured — measure it** |

**Stop at the cheapest rung that captures the effect.** Report reorder cost as its own column at every
rung.

### Rung 0 — must be qualified, not assumed

The container array is **fundamentally key-ordered**, not allocation-ordered. In `lazyMergeTwo`
construction and append coincide, so reverse iteration *may* approximate descending payload order there —
but **mutation, container replacement, array↔bitset conversion, and long-lived bitmaps all break that
correlation.** Therefore rung 0 requires **measured address-order quality (travel and/or page-locality)
on the actual target lifecycles**, not only on a fresh synthetic corpus. If the correlation does not hold
on the repair-demote path, rung 0 is out for that path regardless of its cost.

### Rung 1 — bucket partition, specified

- Key: **`(address − min_address) >> shift`**, `min_address` from a first pass.
- **`shift = max(0, 64 − clz(span) − log2(nbuckets))`** — span-adaptive. A **fixed** shift is a recorded
  trap: 256 buckets over a 160 MB span left travel **unchanged at 6856x**.
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
  `ceil(significant_bits / bits_per_pass)`, `significant_bits = 64 − clz(span)`. State `bits_per_pass`
  (8 or 11) and the resulting count-table size.
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

**Surface scope — decide at review, then pin:**

| candidate | note |
|---|---|
| `repairAfterLazy` demote frees | **primary target**; internal, so it needs the same opt-in question answered |
| `deinit` only | array-dominated for lazy-OR results (see Target workload) |
| **`clearRetainingCapacity()`** | **arguably the more relevant repeated-use operation** — frees containers while retaining arrays, which is the steady-state reuse shape the win was measured in |
| `Roaring64Bitmap` | in or out? |
| 32-bit `RoaringBitmap` only | narrower first step |
| **`OwnedBitmap`** | **EXPLICITLY EXCLUDED — its arena owns teardown**, so container-level free order is irrelevant |

## Size gate — unit must be pinned

`38-00`'s crossovers (**64 M4 / 1,024 Zen 4**) were measured in **8 KB bitset payload count**, not total
containers. Pin whether the gate counts:

- **bitsets only**, or
- **all reorderable payloads**, or
- **total containers**.

**A total-container gate is not justified without mixed array/run/bitset measurements** — the
65,496-array result teardown is precisely the case where those units diverge by ~4×.

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

- **Primary:** descending demote frees measured **inside `repairAfterLazy`** (interleaved with
  `bitsetToArray`), both allocators, both hosts, stage-3 noise verified.
- Representative **result teardown (~65,496 arrays)** measured; the all-bitset corpus reported as a
  control only.
- Rungs 0–2 measured (3 reference, 4 quantified); **rung 0 qualified by measured order quality on the
  real lifecycles**; cheapest sufficient rung selected.
- Repair read-traversal retest run **once, ASCENDING**, with the explicit note that rung 0 offers no
  read-side candidate; verdict recorded either way.
- Size-gate **unit pinned** and threshold **re-derived for the selected rung**.
- API surface pinned (`OwnedBitmap` excluded); default path unchanged; libc-on-M4 excluded **from the
  default** and that wording used.
- Correctness surface green.

## Chunk plan

- **`39-00`** — measurement: the two production workloads, the ladder, rung-0 order-quality
  qualification, inverted repair retest, gate re-derivation. No production change.
- **`39-01`** — ship the selected rung behind the pinned opt-in.

## Estimate

M for `39-00` — rung 0 is free and rung 1 is small, but the repair-demote instrumentation, the
array-dominated teardown corpus, and per-rung stage-3 re-verification are real work. M for `39-01`.
