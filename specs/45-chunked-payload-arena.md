<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 45: Chunked payload arena for lazy-OR bitsets

**Target.** `lazy-or-construction` (M4 **1.732x** baseline; best measured **1.235x** via spec 44).
Gate **≤1.10x**.

**Diagnosis first, no default change.** Adoption, if earned, is a separate spec.

## 1. Why this, and why now

Spec 44 measured the decomposition and it points to one conclusion:

| Effect | M4 | Zen 4 |
|---|---:|---:|
| **Ordering** | **−2.211 ms** | **−4.268 ms** |
| Machinery to obtain it (batching + slotted) | +1.958 | +3.124 |
| Fusion (recovers part of the batching cost) | −1.312 | −1.035 |

And three M4 numbers turned out to be the same quantity: machinery **+0.646 ms**, residual gap to
CRoaring **+0.795 ms**, libc regression **+0.804 ms**. libc is order-insensitive (spec 37), so it pays
the machinery and receives none of the benefit — its regression **is** a measurement of the machinery.

**So: ordering pays, the machinery to obtain it does not.** This spec obtains the ordering **without the
machinery** — no pre-pass, no scratch array, no sort, no metadata, no slot assembly, no second pass.

**Construction stays exactly as it is today** — fused, in key order, one buffer at a time. The *only*
change is **where the 8 KB payload comes from**.

## 2. The mechanism

Payloads for lazy-OR bitsets are bump-allocated from **chunks**: request a chunk from the allocator, hand
out 8 KB slices sequentially, and when a chunk is exhausted request another.

Consecutive payloads are then **ascending by construction**. Traversal jumps only at chunk boundaries —
~134 jumps at a 1 MB chunk size versus 16,364 scattered addresses today.

**Headers stay individually allocated.** They are 16 bytes, cheap, and spec 32 established the header's
size class is load-bearing; leave that path untouched.

**No pre-pass is needed** — chunks grow on demand, so the eligible count never has to be known in
advance. That is the single biggest cost this design avoids compared with specs 43 and 44.

### 2.1 Chunk size is a measured parameter, not a guess

Too small → the ordering benefit disappears into chunk-boundary jumps. Too large → wasted memory and a
large up-front allocation. **The prototype (§6) must sweep it** — at minimum 256 KB, 1 MB, 4 MB — and
report the ordering benefit at each. Pick from the measurement.

## 3. The ownership problem — this is the spec

Today `BitsetContainer.deinit` does `allocator.free(words)` + `allocator.destroy(self)`. A payload inside
a chunk **cannot be freed individually**. This is the difficulty that caused spec 43 to defer the idea,
and it must be answered head-on.

### 3.1 Identifying arena-backed payloads — free, verified

Add a flag to `BitsetContainer`. **Measured: this costs zero bytes.**

```
{ words: *align(64) [1024]u64, cardinality: i32 }               → 16 bytes
{ words: *align(64) [1024]u64, cardinality: i32, arena: bool }  → 16 bytes
```

It fits in existing padding, so **the header stays 16 bytes and spec 32's SMP size-class result is
preserved**. Confirm this with a `comptime` assertion — if a future change pushes the header to 24, spec
32's whole cluster regresses and the build must fail loudly rather than silently.

`deinit` frees `words` only when the flag is clear.

### 3.2 Lifetime — the arena does not outlive the repair

**Rule: arena-backed payloads exist only between `lazyOr` and `repairAfterLazy`.**

1. `lazyOr` allocates payloads from chunks owned by the result bitmap.
2. `repairAfterLazy` already visits every container and converts most bitsets to arrays or runs — those
   payloads simply stop being referenced.
3. **Any bitset that survives repair as a bitset is migrated out**: copy its 8 KB into a normally
   allocated payload and clear the flag.
4. **At the end of repair, every chunk is freed as a whole.** After repair, no arena-backed container
   exists anywhere.
5. **`deinit` without repair frees the chunks too** — a caller may legally discard an unrepaired result.

**Why this is safe:** an unrepaired lazy result is *already* documented as unusable until repaired
(`API.md` footgun, "Lazy Results Must Be Repaired"). So the window in which arena-backed containers exist
is a window in which the bitmap may not be used or have containers extracted from it. **The spec depends
on that contract, so `44-00`-style verification must confirm nothing else can observe the intermediate
state** — in particular that no in-place operation, clone, or container transfer can run on an unrepaired
result.

### 3.3 The migration cost is a real risk

Copying survivors is new work that lands in **repair**, not construction. In spec 44's corpus all 32,728
operands were small arrays, so most results should convert to arrays and survivors should be few — but
that is workload-dependent, and a dense workload could migrate many.

**This is exactly the shape spec 35 warned about**: a change that improves the target row while pushing
cost into a neighbouring one. Hence the dual gate in §7 — **construction and combined `lazyOr+repair`
both gate**, and the combined row is not optional.

## 4. What this deliberately does not do

- **No pre-pass**, no eligible count, no scratch array, no sort, no metadata, no slot assembly.
- **No change to construction order** — still fused, still key order. Ordering comes from the allocator,
  not from reordering the work.
- **No change to the header size**, per §3.1.
- **Nothing outside the lazy-OR path.** `lazyXor` and every other caller keep normal allocation.

## 5. Alternatives considered and rejected

- **Arena survives past repair, freed at bitmap `deinit`.** Simpler, but retains up to 134 MB long after
  repair has converted most containers to small arrays — a large peak-memory regression traded for
  speed. Rejected.
- **Reference-counted chunks, freed when the last payload is released.** Solves lifetime but not
  retention; one long-lived survivor pins an entire chunk. Rejected as the primary design; acceptable as
  a fallback if §3.2's migration proves too costly.
- **One contiguous slab sized up front.** Needs the eligible-count pre-pass — reintroducing the machinery
  this spec exists to avoid.

## 6. Prototype before production

Extend `src/bench_smp_layout.zig` (zero rawr code, per spec 37/43 practice — model the header locally,
do **not** import `BitsetContainer`).

Cells: **scattered allocation** (today) versus **chunked bump allocation** at each candidate chunk size,
zeroing in allocation order in both cases. Time allocation and zeroing together, inside the region.

**The question: does chunked bump allocation recover a comparable share of the ordering benefit that
sorting delivered (−2.211 ms on M4), without any sorting?**

Run on **both hosts, SMP and libc**. libc must show **no material change** — it is order-insensitive, and
this design gives it fewer, larger allocations rather than reordering. If libc moves substantially in
either direction, the prototype is measuring something unintended.

**NO-GO here ends the spec** with no production change.

## 7. Gates

**Both must pass. Neither is optional.**

- **Construction:** `lazy-or-construction` **≤1.10x** on M4.
- **Combined:** `lazyOr+repair` **does not regress** — this is where migration cost lands, and spec 35
  established that gating the aggregate alone authorizes work that buys nothing.
- **libc ≤5% on median**, both rows, ranges considered. Spec 44 died here at +21.2%; this design should
  be neutral-to-positive for libc since it *removes* per-buffer allocations rather than adding
  bookkeeping. **If libc regresses, that is a STOP**, not a fallback to opt-in.
- **Zen 4 ≤5%**, both rows.
- **Peak memory does not regress** — new for this spec, because chunking changes the allocation profile.
  Report peak RSS or allocator high-water for construction and for the combined cycle.
- Canonical harness only, both hosts, all three tuples, ≥5 fresh-process medians with full ranges.
  Overlapping ranges → rerun; still overlapping → inconclusive → NO-GO.

## 8. Retained requirements

- **Failure injection** at chunk allocation, header allocation, migration allocation during repair, and
  the existing clone/union sites. Inputs untouched, nothing leaked, leak-checking GPA, never
  `c_allocator`.
- **Correctness:** repaired output **byte-identical** to baseline and CRoaring, over forced and selective
  lazy OR, all three container types, disjoint keys, empty inputs.
- **`cardinality()` checked before `repairAfterLazy`**, not only after — spec 44 established that
  repair-first tests mask stale cached state.
- **`lazyXor` byte-identical to baseline.**
- **No public API.** Internal only; outside `API.md`, the `check-docs` guarded region, and the `check-32`
  probe.
- **`comptime` assertion that `BitsetContainer` stays 16 bytes.**
- All four suites plus `check-32`, `check-docs`, `check-package`.

## 9. Out of scope

- Adoption — separate spec, only if both gates pass.
- Sorting, pre-passes, slot assembly, fusion machinery — measured in specs 43 and 44; they lose.
- Source-read reordering — spec 44 closed it: all operands were small arrays and destination ordering
  barely moved source travel.
- The microarchitectural attribution question (prefetch vs TLB vs cache).

## 10. Estimate

**M** — the allocation change is small; the ownership rule, migration path, and dual-gate measurement are
the work.

## 11. Chunking

Not chunked — pending review.
