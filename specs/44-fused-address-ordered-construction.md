<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 44: Fused address-ordered lazy-OR construction — diagnosis

**Target.** `lazy-or-construction`, the last material open row (M4 **1.732x** baseline, gate **≤1.10x**).

**Diagnosis only. No default change, no adoption path in this spec.**

**Prior implementation evidence:** branch `spec-43-lazy-construction-diagnostic`, commit **`37d0e8b`** —
three-arm dispatch, pending allocation, scratch, ownership handoff, failure injection. This spec extends
it. **That branch is currently local only; push it or the reference does not resolve.**

## 1. What spec 43 established, and what it did not

| Path | Median | vs CRoaring |
|---|---:|---:|
| CRoaring | 3.402 ms | 1.000x |
| baseline — fused, key order | 5.894 ms | 1.732x |
| batched, unsorted, unfused | 7.438 ms | 2.186x |
| batched, sorted, unfused | 5.222 ms | 1.535x |

**Ordering works in production: −2.216 ms.** **The batching vehicle costs +1.544 ms.** libc +90%.

**NOT established: what the +1.544 ms consists of.** Cold second pass is the **leading explanation**
(~1.4–1.6 ms on the standalone probe), but arm 2 also added the pre-pass, scratch, and deferred assembly.
**Separating them is this spec's first job.**

## 2. The candidate

Baseline is *fused but badly ordered*; spec 43's sorted arm is *well ordered but unfused*. The candidate
is both: **sort by destination address, then zero and accumulate each buffer while it is hot** —
`@memset`, then `lazyAccumulateIntoBitset` for both operands, then write to its output slot. Each payload
is touched once, resident. No second pass.

This needs per-entry metadata, which spec 43 §4.1 forbade. That constraint now has a measured 2.2 ms
benefit to weigh against — hence a new spec.

### 2.1 Metadata — types pinned

```zig
const Pending = struct {
    payload_addr: usize,          // 8 — sort key, no dereference in comparator
    header: *BitsetContainer,     // 8
    src_a: TaggedPtr,             // 8 — packed struct(usize)
    src_b: TaggedPtr,             // 8
    slot: u32,                    // 4 — NOT u16: up to 65,536 output slots
};                                // 36 → 40 bytes with alignment
```

`slot` **must not be `u16`**: the result can hold 65,536 containers, which does not fit. `src_a`/`src_b`
are `TaggedPtr` (8 bytes each), so the ~40-byte figure is dependable rather than assumed.

**Sort `sortUnstable` (pdq) on `payload_addr`; comparator dereferences nothing.**

**The parallel-array variant is DEFERRED ENTIRELY** — a 16-byte `{payload_addr, index}` sort with
metadata alongside is *not* part of this spec. Revisit only if the primary fused result **narrowly misses
the gate**, and then as a follow-up spec. *(An earlier draft said "if sort cost is material", which is
not a threshold.)*

## 3. Four arms — same binary

| Arm | Path | Isolates |
| --- | --- | --- |
| 1 | baseline — fused, key order | reference |
| 2 | batched, **unsorted**, unfused | — |
| 3 | batched, **sorted**, unfused | — |
| 4 | batched, **sorted**, **fused** (new) | — |

| Quantity | Comparison |
| --- | --- |
| **Batching machinery cost** | arm 2 − arm 1 |
| **Ordering recovery** | arm 3 − arm 2 |
| **Fusion recovery** | arm 4 − arm 3 |
| **Net result** | arm 4 − arm 1 |

**Arm 2 is required, not optional.** *(An earlier draft dropped it and labelled sorted-unfused "arm 2",
which makes `arm2 − arm1` a mixture of batching and ordering and destroys the decomposition this spec
exists to produce.)*

All four measured **in one binary** — spec 43 established cross-run ratios do not hold.

**Manifest goes 42 → 43 rows.** Both guards must read exactly **43**:
`src/bench_parity_worker.zig:778` and `scripts/run-compare-bench.sh:72`.

## 4. Ownership — a NEW transactional contract

**Spec 43's contract does not apply and must not be inherited.** It was built around
`appendOwnedContainer` and a sequential cursor; the fused path writes **directly into pre-computed
slots**, so neither is involved.

**Design:**

1. **Allocate scratch before staging any slot**, so scratch OOM leaves the initialized `result`
   untouched and reusable by the baseline fallback.
2. **Initialize `result.containers[0..output_count]` to `.reserved`** tagged values. Verified:
   `Container.deinit` has `.reserved => {}` (`container.zig`), so a reserved slot frees as a no-op, and
   `getCardinality` returns 0.
   **Build reserved entries directly** — `Container.toTagged` on `.reserved` is `unreachable`.
3. **Set `result.size = output_count`** so a plain `errdefer result.deinit()` walks every slot and
   correctly frees exactly the populated ones.
4. **Pending entries own their bitsets until the pointer is written into its destination slot.**
5. **On handoff, advance the sorted pending cursor**; the result owns that slot from then on.
6. **Fill unmatched / non-eligible slots directly.**
7. **Before returning success, verify no `.reserved` slot remains.**

Pending cleanup covers untransferred entries; result cleanup covers assigned slots. **No second ownership
bitmap is needed** — the cursor plus the reserved sentinel carry the whole boundary.

### 4.1 Fallback scope — narrow, and only at the start

**Only the initial scratch allocation failure falls back to the baseline merge loop** (reusing the
untouched initialized `result`).

**Header, payload, metadata, clone, accumulation, and assembly failures all propagate** after
transactional cleanup per §4. *(An earlier draft said "scratch/metadata OOM", which was ambiguous and
would have licensed a mid-flight fallback with buffers already staged.)*

## 5. Cost accounting

Timed region contains the **complete candidate**: eligible pre-pass, metadata construction, scratch
allocation, sort, zeroing, accumulation, slot assembly, reserved-slot verification, and **scratch
release**.

Outside: result teardown only — matching the canonical row, which calls `result.deinit()` after the clock
stops (`bench_croaring.zig:507-512`).

## 6. Corpus and the read-order risk

**Real canonical sparse corpus.** No synthetic population.

**Report, and this is load-bearing:**

- source container **counts** and **types**;
- **bytes actually read** from sources;
- **source-address travel in key order versus destination order** — the sum of absolute address deltas
  along each traversal.

Container type and byte totals alone **cannot** show whether destination sorting made source traversal
pathological; the travel comparison is what does. Spec 38 found read traversal wants ascending (M4
1.221x, Zen 4 1.344x), **but on large bitsets** — if these sources are mostly small arrays that result may
not transfer, and it may also be that source reads are cheap against 8 KB writes.

**Do not combine experiments.** Bucket/radix ordering is out of scope; consider it only if fusion works
and narrowly misses.

## 7. Retained from spec 43 (unchanged, not re-litigated)

- **Failure injection** at scratch, metadata, pending headers, pending payloads, unmatched clones, and
  assembly. Inputs untouched, nothing leaked, leak-checking GPA, never `c_allocator`.
- **Equivalence coverage** driving the fused arm directly: forced and selective, eligible counts of
  zero/partial/all, array/bitset/run combinations, disjoint keys, empty inputs, **repaired output
  byte-identical** to baseline and CRoaring.
- **`lazyXor` byte-identical to baseline**; scope stays `op == .bor`.
- **No public API** — internal export, classified in the manifest, outside `API.md`, the `check-docs`
  guarded region, and the `check-32` probe.
- **Canonical harness only**, both hosts, all three canonical tuples, ≥5 fresh-process medians with full
  ranges.

## 8. Gate

- **Arm 4 beats arm 3 with non-overlapping ranges** — fusion removes a measurable part of the penalty.
- **Arm 4 reaches ≤1.10x vs CRoaring on M4.**
- **libc does not regress — arm 4 vs arm 1, rawr/libc, same binary, ≤5% on median.** A libc regression is
  a **STOP** (spec 43 measured +90%).
- **Zen 4 does not regress — arm 4 vs arm 1, ≤5% on median, ranges considered.** Explicit, not implied:
  the campaign has both hosts and a Zen 4 regression is a NO-GO on its own.
- Overlapping ranges → rerun; still overlapping → **inconclusive → NO-GO**.

**Report the decomposition regardless of outcome** — how much of the 1.544 ms fusion removed. That is the
durable result even on a NO-GO: it tells the campaign whether the residue is cache or machinery.

## 9. Out of scope

- Default adoption — separate spec, only if the gate passes.
- Parallel-array metadata variant (§2.1) and bucket/radix ordering (§6).
- Plain unfused batching — measured, both allocators, it loses.
- The microarchitectural attribution question.

## 10. Chunking

- **[44-00](44-00-fused-implementation.md)** — implementation, ownership, correctness, failure injection,
  43-row manifest.
- **[44-01](44-01-measurement-and-verdict.md)** — two-host canonical measurement, decomposition, verdict.

## 11. Estimate

**M** — vehicle exists at `37d0e8b`; new work is metadata, slot assembly, the fused loop, the
transactional contract, and the measurement.
