<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 45: Chunked payload arena for lazy-OR bitsets

**Target.** `lazy-or-construction` (M4 **1.732x** baseline; best measured **1.235x** via spec 44).
Gate **≤1.10x**.

**Diagnosis first, no default change.** Adoption, if earned, is a separate spec.

## 1. Why this, and why now

Spec 44's decomposition:

| Effect | M4 | Zen 4 |
|---|---:|---:|
| **Ordering** | **−2.211 ms** | **−4.268 ms** |
| Machinery to obtain it | +0.646 net | +1.253 net |

Three M4 numbers were the same quantity: machinery **+0.646 ms**, residual gap **+0.795 ms**, libc
regression **+0.804 ms**. **Ordering pays; the machinery to obtain it does not.**

This spec obtains ordering **without the machinery**: no pre-pass, no eligible count, no scratch array,
**no payload-address sort**, **no per-payload scratch metadata**, no slot assembly, no second pass.
*(Qualified deliberately: the ~128-entry **chunk-list index** IS sorted once before repair per §3.1. What
this spec avoids is sorting 16,364 payload addresses.)* Construction stays as it
is today — fused, key order, one buffer at a time. **Only the source of the 8 KB payload changes.**

*(The chunk list is itself metadata; it is small and bounded, and **its cost belongs inside the timed
region** — see §7.)*

## 2. Mechanism

Payloads are bump-allocated from **chunks**: request a chunk, hand out 8 KB slices sequentially, request
another when exhausted.

Payloads are **ascending within each chunk**. Chunk addresses themselves are whatever the allocator
returns and need **not** be globally ascending. At 1 MiB chunks: **128 payloads per chunk**, so 16,364
payloads occupy **128 chunks with 127 boundary jumps**, versus 16,364 scattered addresses today.

**Headers stay individually allocated** — 16 bytes, cheap, and spec 32 established the header's size
class is load-bearing.

**No pre-pass:** chunks grow on demand, so the eligible count is never needed in advance. That is the
largest cost this design avoids relative to specs 43 and 44.

### 2.1 Chunk size is measured, not guessed

Sweep at minimum **256 KiB, 1 MiB, 4 MiB**; report the ordering benefit at each.

**Selection rule, decided in advance:** using **SMP medians on both hosts**, choose the **smallest** size
within **5%** of the best result on both. **libc is report-only and must not influence selection**
(§7.1). If no single size satisfies both hosts, **report that and stop** — host-specific tuning is a
separate decision requiring explicit sign-off, never a silent default.

## 3. Ownership — an internal wrapper, not a field on `RoaringBitmap`

**Decision: `ChunkedLazyResult { bitmap: RoaringBitmap, arena: ChunkList }`, internal only.**

This resolves three problems at once and is why it is preferred to storing chunks on the bitmap:

**(a) No layout tax.** `RoaringBitmap` gains **no field**. Adding a pointer would change every bitmap's
layout — default paths, `Roaring64Bitmap` buckets, all of it — to serve a diagnostic.

**(b) Transfer is structurally impossible, not merely forbidden.** The wrapper **does not expose the
bitmap** until repair completes. An earlier draft claimed nothing can transfer an unrepaired result —
**that is false today**: `bitwiseOrInPlaceConsume` (`bitmap.zig:1266`) moves right-only container
pointers into another bitmap and empties the source, which would leave arena payloads owned by one
bitmap while the chunk list stayed with another. Those methods are public and have no lazy-state guard.
Documentation cannot fix this; **withholding the bitmap can.**

**(c) No header flag, so no 32-bit problem.** *(An earlier draft added a `bool` to `BitsetContainer`,
claiming it was free. It is free **on 64-bit only** — 16 → 16 bytes, verified. On **32-bit** the header is
`ptr(4) + i32(4) = 8` bytes and the flag makes it **12**, moving it out of the 8-byte SMP size class; an
unconditional `@sizeOf == 16` assertion would also fail `check-32`.)*

**Instead: identify arena payloads by address-range check against the chunk list.** Chunks are few
(~128), sorted once before use (§3.1), so a binary search answers "is this payload arena-backed?" This
runs **once per container during repair or teardown**, never in the construction hot path. No header change, no width
dependence, spec 32 untouched.

### 3.1 Chunk list

Chunk base pointers and lengths in a small growable array owned by the wrapper.

**Policy, decided — append unsorted during construction; sort exactly once before repair or ownership
classification.** *(An earlier draft offered "kept sorted", "sort on insert", and "sort once" as if
interchangeable; they have different construction costs, and construction is the row under test.)* The
sort is over ~128 elements, is **infallible** (no allocation), and **belongs inside combined timing**.

**Reserve chunk-list capacity BEFORE allocating a chunk.** Otherwise a list-growth failure after a
successful chunk allocation **orphans that chunk** — it is allocated, unrecorded, and unfreeable.

Payload alignment is **64 bytes**, matching `alignedAlloc(u64, .@"64", …)`: chunk bases must be 64-byte
aligned and the slice stride must preserve it.

### 3.2 Lifetime

1. `lazyOr` (wrapper form) allocates payloads from chunks owned by the wrapper.
2. `repairAndTake` runs repair. Repair converts sparse bitsets to **arrays** (`bitsetToArray` when
   cardinality ≤ `ArrayContainer.MAX_CARDINALITY`) and drops empty containers; **it does not convert to
   runs** *(an earlier draft said "arrays or runs")*. Converted payloads simply stop being referenced.
3. **Survivors are migrated out**: copy the 8 KB into a normally allocated payload and repoint `words`.
4. **All chunks are freed as a whole**, then the plain `RoaringBitmap` is returned to the caller. **After
   this point no arena-backed payload exists anywhere.**
5. **Wrapper `deinit` without repair** — see §3.2.1; teardown is **not** uniform.
6. **`clearRetainingCapacity` on the wrapper** must free chunks explicitly **and** apply §3.2.1
   classification; retaining bitmap capacity does not retain arena payloads.

### 3.2.1 Mixed ownership — every bitset must be classified

**Not all bitsets in the result are arena-backed.** `lazyMergeTwo` allocates *matched* bitsets from the
arena, but **unmatched bitset containers are cloned with ordinary allocator-owned payloads**. Teardown,
clearing, repair, and failure cleanup must therefore classify **each** container:

| Container | Action |
| --- | --- |
| bitset, **arena** payload (address-range hit) | **destroy header only** — payload lives in a chunk |
| bitset, **normal** payload (no hit) | **normal `BitsetContainer.deinit`** |
| array / run | normal `deinit` |

*(An earlier draft said "free headers individually and chunks wholesale", which would **leak every
ordinary cloned bitset payload**.)*

The address-range check (§3) is what distinguishes them, and it must be applied on **every** cleanup
path, not only the success path.

### 3.2.2 Wrapper API — public repair methods stay unchanged

**Do not modify `RoaringBitmap.repairAfterLazy` / `repairAfterLazyWithOptions`.** They have no chunk-list
context and must never receive the hidden bitmap. *(An earlier draft said the existing methods "must
support arena-backed payloads" — that contradicts the wrapper design, whose whole point is that the
bitmap is not exposed until repair completes.)*

Pin these **wrapper** methods instead:

- `repairAndTake()` → repairs, migrates survivors, frees chunks, returns the plain `RoaringBitmap`;
- `repairAndTakeWithOptions(options)` → same, carrying `repairAfterLazyWithOptions`' options through;
- `cardinality()` → for the required pre-repair check (§9), without exposing the bitmap.

### 3.3 Repair-failure transaction

Migration introduces a new allocation inside repair (`bitmap.zig:1656`). On migration failure:

- **chunks remain owned by the wrapper**;
- **every retained slot remains deinitable** — migrated ones point at normal allocations, unmigrated ones
  still point into chunks, and the wrapper's teardown handles both;
- **repair may be retried**;
- **no already-compacted entry is duplicated.**

**Retry is idempotent by construction:** migration repoints `words` at a normal allocation, so the
address-range check then reports that payload as *not* arena-backed and a retry skips it. Verify this
rather than assume it.

## 4. Deliberately not doing

No pre-pass, eligible count, scratch array, sort, per-payload metadata, or slot assembly. No change to
construction order — ordering comes from the allocator, not from reordering work. No header change. No
`RoaringBitmap` field. Nothing outside lazy-OR; `lazyXor` and all other callers keep normal allocation.

## 5. Alternatives rejected

- **Chunks on `RoaringBitmap`, freed at `deinit`.** Taxes every bitmap's layout, and retains up to 134 MB
  after repair has converted most containers to small arrays.
- **Reference-counted chunks.** Solves lifetime, not retention: one survivor pins a whole chunk.
  Acceptable fallback only if §3.2 migration proves too costly.
- **One contiguous slab sized up front.** Needs the eligible-count pre-pass — the machinery this spec
  exists to avoid.
- **Header flag** — §3(c).

## 6. Prototype before production

Extend `src/bench_smp_layout.zig` (zero rawr code — model the header locally, do **not** import
`BitsetContainer`).

**Four explicit cells**, each timing its **complete** cost — allocation, any metadata, any sort, and
zeroing — with retained teardown **outside** the region:

| Cell | What it does |
| --- | --- |
| `scattered_interleaved` | today's structure: allocate and zero **per buffer**, interleaved |
| `batched_unsorted` | allocate all, then zero in **allocation order** |
| `batched_sorted` | allocate all, **sort payload addresses**, zero in **sorted order** |
| `chunked_<size>` | chunk bump-allocate and zero **per buffer**, interleaved — the candidate |

*(An earlier draft was incoherent: it named a `sorted` cell as the ordering ceiling while also stating
every cell zeroes in allocation order, which would make that cell identical to `batched_unsorted`. It
also required allocation inside timing while the existing probe's sorted cell allocates before it —
`bench_smp_layout.zig:167`, already flagged in spec 43-00.)*

Note the candidate is **interleaved like the baseline**, not batched: chunking gets ordering *without*
deferring the zeroing, which is the entire point.

### 6.1 Stop gate — numeric, decided in advance

**M4, SMP:**

```
available  = batched_unsorted     - batched_sorted        # ordering headroom this probe can see
recovered  = scattered_interleaved - chunked_<size>       # what the candidate actually delivers
```

- **GO requires `recovered >= 0.50 * available`**, with **non-overlapping ranges** between
  `chunked_<size>` and `scattered_interleaved`.
- **Zen 4: `chunked / scattered_interleaved <= 1.05`** on median.
- Overlapping ranges → **rerun**; still overlapping → **inconclusive → NO-GO**.

`available` is measured in the batched world because that is the only place a payload-address sort is
possible; `recovered` is measured against the real baseline structure because that is what the candidate
must actually beat. **They are deliberately different comparisons.**

*(An earlier draft asked for a "comparable share", which is not falsifiable.)* 50% is a screen, not a
prediction: below it the mechanism cannot plausibly cover the residual once production overheads are
added. **Only the §7 gates can decide the row.**

Run both hosts, SMP and libc. **NO-GO here ends the spec** with no production change.

## 7. Gates

- **Construction:** candidate `lazy-or-construction` row **≤1.10x** on M4. **Primary.**
- **Combined:** `lazyOr+repair` **within 5% on median** of baseline, ranges considered. Migration cost
  lands here, and spec 35 established that gating the aggregate alone authorizes work that buys nothing.
- **Zen 4: `candidate / baseline ≤ 1.05`** on median, for **both** rows, ranges considered.
- **Memory:** report **requested-byte high-water** and **post-repair live bytes**, tolerance **≤5%**
  against baseline, plus a hard assertion that **retained chunk bytes are zero after successful repair**.
- Canonical harness only, both hosts, all three tuples, ≥5 fresh-process medians with full ranges.
  Overlapping ranges → rerun; still overlapping → inconclusive → NO-GO.

### 7.1 libc — reported, not a stop *(policy change, owner decision)*

**libc regressions are tolerable and do not stop this spec.** Owner decision, 2026-08-19: SMP is the
performance-relevant allocator for this campaign.

- **Still measured and reported** on both rows — a large libc movement remains diagnostic signal, and in
  spec 44 the libc regression *was* the cleanest measurement of machinery cost.
- **libc improvement is expected here, not contamination.** *(An earlier draft said any substantial libc
  movement invalidates the prototype — backwards. Chunking replaces ~16,364 allocations with ~128 larger
  ones, so libc may legitimately improve. Only an unexplained regression warrants investigation, and even
  then it does not stop the spec.)*

**Two things this policy does not change:**

- **Spec 44 stays NO-GO on its own merits** — M4 SMP was **1.235x**, failing the gate independently of
  libc. Do not reopen it on this policy.
- **The shipped docs make no claim about libc performance.** Spec 41 removed all such claims from
  `README.md` and `API.md`, and the board shows libc winning some rows. This is an internal
  prioritisation, not a documented characterisation.

## 8. Manifest and rows

Two candidate rows, referencing the **existing** CRoaring/libc and baseline references — no duplicate
reference rows:

| Row | Meaning |
| --- | --- |
| `lazy-or-construction-arena` | candidate construction |
| `lazy-or-repair-arena` | candidate combined `lazyOr+repair` |

**Manifest count: 40 + 2 = 42.** Both guards must read exactly **42**:
`src/bench_parity_worker.zig:778` and `scripts/run-compare-bench.sh:72`.

*(An earlier draft said 44 → 46. Wrong: spec 44's diagnostic rows were **never committed to `main`** —
that work remains uncommitted on `spec-43-lazy-construction-diagnostic`. Verified: `main` reads exactly
**40** in both guards. Baseline from `main`, not from the diagnostic branch.)*

**Timed boundaries:** construction times the wrapper `lazyOr` only, teardown outside, matching
`bench_croaring.zig:507-512`. Combined times `lazyOr` + `repairAndTake`, teardown outside.

## 9. Retained requirements

- **Failure injection** at chunk allocation, **chunk-list capacity reservation** (must not orphan a
  chunk — §3.1), header allocation, migration allocation during repair, and the existing clone/union
  sites. Include a case with **both arena-backed and ordinary cloned bitsets present** (§3.2.1), since
  that is where misclassification leaks. Inputs untouched, nothing leaked, leak-checking GPA,
  never `c_allocator`.
- **Correctness:** repaired output **byte-identical** to baseline and CRoaring — forced and selective
  lazy OR, all three container types, disjoint keys, empty inputs.
- **`cardinality()` checked before repair**, not only after — spec 44 established repair-first tests mask
  stale cached state.
- **`lazyXor` byte-identical to baseline.**
- **No public API.** Internal only; outside `API.md`, the `check-docs` guarded region, and the `check-32`
  probe.
- All four suites plus `check-32`, `check-docs`, `check-package`.

## 10. Out of scope

Adoption; sorting/pre-passes/slot assembly/fusion machinery (specs 43–44, measured, they lose);
source-read reordering (spec 44 closed it); the microarchitectural attribution question.

## 11. Estimate

**M** — the allocation change is small; the wrapper, migration path, failure transaction, and dual-gate
measurement are the work.

## 12. Chunking

Not chunked — pending review.
