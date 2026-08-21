<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 45-01: Wrapper, ownership, migration, correctness

Toplevel: [45-chunked-payload-arena.md](45-chunked-payload-arena.md).
Gated on: [45-00](45-00-chunked-prototype.md) returning **GO**.

**No default change.** The arena path ships behind an internal wrapper, reachable only by the diagnostic
rows. This chunk ends when it is correct, leak-free, and selectable — **no measurement verdict here**
(that is `45-02`).

## 1. The wrapper

```zig
ChunkedLazyResult { bitmap: RoaringBitmap, arena: ChunkList }   // internal only
```

**`RoaringBitmap` gains no field.** Adding one would change every bitmap's layout — default paths,
`Roaring64Bitmap` buckets — to serve a diagnostic.

**The wrapper does not expose the bitmap until repair completes.** This is not a convention, it is the
safety mechanism: `bitwiseOrInPlaceConsume` (`bitmap.zig:1266`) moves right-only container pointers into
another bitmap and empties the source, with no lazy-state guard. If an unrepaired arena-backed bitmap were
reachable, its payloads would end up owned by one bitmap while the chunk list stayed with another.
Documentation cannot prevent that; withholding the bitmap can.

**Methods:**

- `repairAndTake()` → repair, migrate survivors, free chunks, return the plain `RoaringBitmap`;
- `repairAndTakeWithOptions(options)` → same, carrying `repairAfterLazyWithOptions`' options;
- `cardinality()` → for the pre-repair check (§5), without exposing the bitmap;
- `deinit()` → §3.2 teardown;
- `clearRetainingCapacity()` → frees chunks explicitly **and** applies §3 classification.

### 1.1 Handoff state — the wrapper and the returned bitmap cannot both own the result

- **On success**, `repairAndTake*` **consumes and invalidates the wrapper**. A subsequent wrapper
  `deinit()` must be **safe and a no-op**, so a caller's `defer wrapper.deinit()` cannot double-free what
  the returned bitmap now owns.
- **On failure**, the wrapper **remains the owner** and remains **retryable** (§3.2).
- **Required test:** `defer` teardown of **both** the wrapper and the returned bitmap in the same scope,
  under a leak-checking GPA, confirming no double free and no leak. That is the exact shape a caller
  will write, so it is the shape that must be proven safe.

**Public `RoaringBitmap.repairAfterLazy` / `repairAfterLazyWithOptions` are NOT modified.** They have no
chunk-list context and must never receive the hidden bitmap.

## 2. Chunk allocation

Payloads bump-allocate from chunks; **headers stay individually allocated** (16 bytes; spec 32 established
the header's size class is load-bearing).

**No pre-pass** — chunks grow on demand, so the eligible count is never needed. Payloads are **ascending
within each chunk**; chunk addresses themselves need not be globally ascending.

**Chunk size: use the size selected by `45-00` §6. No retuning in this chunk** — re-choosing it here
would silently re-open a decision that was made against a pinned rule.

**Chunk list:** append **unsorted** during construction; **sort exactly once** before repair or ownership
classification. The sort is over ~128 elements, is **infallible**, and belongs inside combined timing.

**Reserve chunk-list capacity BEFORE allocating a chunk** — otherwise a growth failure after a successful
chunk allocation **orphans that chunk**: allocated, unrecorded, unfreeable.

Payload alignment **64 bytes**, matching `alignedAlloc(u64, .@"64", …)`; chunk bases 64-byte aligned with
a stride that preserves it.

## 3. Ownership classification — every bitset, every path

**Not all bitsets are arena-backed.** `lazyMergeTwo` allocates *matched* bitsets from the arena, but
**unmatched bitset containers are cloned with ordinary allocator-owned payloads**.

| Container | Action |
| --- | --- |
| bitset, **arena** payload (address-range hit) | **destroy header only** — payload lives in a chunk |
| bitset, **normal** payload (no hit) | **normal `BitsetContainer.deinit`** |
| array / run | normal `deinit` |

Arena membership is a **binary search over the sorted chunk list** (~128 entries) — no header flag, so no
32-bit layout change and spec 32 is untouched. *(A header `bool` is free on 64-bit but takes the 32-bit
header from 8 to 12 bytes, out of the 8-byte SMP class.)*

**This classification must be applied on EVERY cleanup path** — success, failure, `deinit`, and
`clearRetainingCapacity`. Uniform teardown leaks every ordinary cloned bitset payload.

### 3.1 Lifetime

1. Wrapper `lazyOr` allocates payloads from wrapper-owned chunks.
2. `repairAndTake` runs repair. Repair converts sparse bitsets to **arrays** (`bitsetToArray` when
   cardinality ≤ `ArrayContainer.MAX_CARDINALITY`) and drops empty containers. **It does not convert to
   runs.**
3. **Survivors are migrated out**: copy 8 KB into a normally allocated payload, repoint `words`.
4. **All chunks freed as a whole**, then the plain bitmap is returned. **After this no arena-backed
   payload exists anywhere.**

### 3.2 Failure transaction

Migration allocates inside repair (`bitmap.zig:1656`). On migration failure:

- **chunks remain owned by the wrapper**;
- **every retained slot remains deinitable** — migrated ones point at normal allocations, unmigrated ones
  into chunks, and §3 classification handles both;
- **repair may be retried**;
- **no already-compacted entry is duplicated.**

**Retry is idempotent by construction:** migration repoints `words` at a normal allocation, so the
address-range check then reports that payload as *not* arena-backed and a retry skips it. **Verify this
rather than assume it.**

## 4. Failure injection

Inject at:

1. **chunk allocation**;
2. **chunk-list capacity reservation** — must not orphan a chunk (§2);
3. **header allocation**;
4. **migration allocation during repair**;
5. **unmatched clone** allocation;
6. **non-eligible union** allocation.

Use `std.testing.checkAllAllocationFailures`. Every failure: **inputs untouched, nothing leaked**,
leak-checking GPA, never `c_allocator`.

**Include a case with both arena-backed and ordinary cloned bitsets present** — that is where
misclassification leaks, and a corpus with only matched pairs would never exercise it.

## 5. Correctness

- Repaired output **byte-identical** to baseline **and** CRoaring — forced and selective lazy OR, all
  three container types, disjoint keys, empty inputs on either side.
- **`cardinality()` checked BEFORE repair**, not only after. Spec 44 established that repair recomputes
  cardinality, so repair-first tests mask stale cached state entirely.
- **`lazyXor` byte-identical to baseline** — scope stays `op == .bor`.
- **Chunk-boundary case:** a corpus large enough to span **multiple chunks**, so boundary transitions and
  multi-chunk classification are exercised rather than assumed.

## 6. Manifest — 40 → 42 rows

`main` currently reads **40** in both guards — spec 44's diagnostic rows were never committed. Baseline
from `main`.

| Row | Meaning |
| --- | --- |
| `lazy-or-construction-arena` | candidate construction |
| `lazy-or-repair-arena` | candidate combined `lazyOr` + `repairAndTake` |

Reuse the **existing** CRoaring/libc and baseline references — no duplicate reference rows. **Both guards
must read exactly 42:** `src/bench_parity_worker.zig:778`, `scripts/run-compare-bench.sh:72`.

## Acceptance

- Wrapper per §1; **bitmap not exposed before repair**; public repair methods unmodified;
  `RoaringBitmap` gains no field.
- **Handoff per §1.1** — success consumes/invalidates the wrapper, failure retains ownership and stays
  retryable, and the **double-`defer` test passes**.
- Chunk allocation per §2, including **capacity-reserved-before-chunk** and the single pre-repair sort.
- **§3 classification applied on every cleanup path**; no header flag; `BitsetContainer` size unchanged.
- Lifetime and failure transaction per §3.1–3.2; **retry idempotence verified, not assumed**.
- **`checkAllAllocationFailures` green** at every §4 injection site, including the mixed arena/ordinary
  case.
- Correctness per §5, including **pre-repair `cardinality()`** and the **multi-chunk** case.
- **Manifest at 42 rows, both guards updated**; both candidate rows selectable.
- No public API added; internal only; outside `API.md`, the `check-docs` guarded region, and the
  `check-32` probe. `check-docs` green with an empty allow-list.
- Default behaviour unchanged; canonical board row unmoved.
- All four suites green — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
  `check-package`.
- **No measurement verdict claimed in this chunk.**

## Estimate

**M** — allocation change is small; wrapper, classification, migration, and the failure transaction are
the work.
