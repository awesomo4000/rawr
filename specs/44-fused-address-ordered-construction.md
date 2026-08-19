<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 44: Fused address-ordered lazy-OR construction — diagnosis

**Target.** `lazy-or-construction`, the last material open row (M4 **1.732x** baseline, gate **≤1.10x**).

**Diagnosis only. No default change, no adoption path in this spec.** Adoption, if earned, is a separate
spec.

**Prior implementation evidence:** branch `spec-43-lazy-construction-diagnostic`, commit **`37d0e8b`**.
The three-arm dispatch, pending allocation, scratch, ownership handoff, and failure injection built there
are the starting point — this spec adds a fourth arm rather than rebuilding.

## 1. What spec 43 established, and what it did not

**Established (M4, canonical, same binary):**

| Path | Median | vs CRoaring |
|---|---:|---:|
| CRoaring | 3.402 ms | 1.000x |
| baseline — fused, key order | 5.894 ms | 1.732x |
| batched, unsorted | 7.438 ms | 2.186x |
| batched, sorted | 5.222 ms | 1.535x |

- **Ordering works in production: −2.216 ms** (sorted vs unsorted). Spec 37's premise is confirmed on
  real merge inputs, not just a synthetic probe.
- **The batching vehicle costs +1.544 ms** (unsorted vs baseline) and eats most of the gain.
- libc regressed **+90%**, an independent stop.

**NOT established: what the +1.544 ms actually consists of.** The cold second pass over ~134 MB is the
**leading explanation** (~1.4–1.6 ms against the standalone probe, which brackets it neatly), but arm 2
also introduced the eligible pre-pass, scratch allocation and release, and deferred assembly. Those are
bundled into the same number.

**This spec's first job is to separate them.** If fusion removes the whole 1.544 ms, the cache
explanation is right and the row is within reach. If it removes only part, the remainder is machinery
overhead, and that changes what any future lever should target. **Do not assume; measure.**

## 2. The candidate — fused and address-ordered

Baseline is *fused but badly ordered*. Spec 43's arm 3 is *well ordered but unfused*. The candidate is
both: **sort by destination address, then zero and accumulate each buffer while it is hot.**

Per pending entry, in address order:

1. `@memset` the 8 KB payload,
2. accumulate **both** source operands into it immediately (`lazyAccumulateIntoBitset` ×2),
3. record it into its output slot.

The payload is touched once, while resident. There is no second pass over the population.

**This requires per-entry metadata** — source containers and destination slot — which spec 43 §4.1
explicitly forbade. That constraint existed to stop scratch cost being re-imported for no benefit; it now
has a measured 2.2 ms benefit to weigh against, which is why this is a new spec rather than an amendment.

### 2.1 Metadata and assembly

The pre-pass must compute, for each eligible pair: both source container pointers and the **destination
slot index** in the result. Assembly then writes into pre-computed slots rather than appending
sequentially, so key order is preserved without consuming buffers in key order.

**Element shape is a first-class measurement, not an implementation detail.** Spec 43 established that
the sorted element's size and comparator materially affect cost. Two candidates:

- **(a) One fat sorted struct** — `{payload_addr, header, src_a, src_b, slot}` (~40 B). Sequential access
  during the fused pass; costlier sort.
- **(b) 16-byte `{payload_addr, index}` sorted, metadata in a parallel array** — cheaper sort, but the
  fused pass indexes metadata randomly.

**Default to (a)**, because the fused pass is the hot loop being optimized and interleaving random
metadata lookups into it risks reintroducing exactly the kind of scattered access this spec exists to
remove. **If the measured sort cost is material, measure (b) as a variant** — but do not run both as the
primary comparison, per §5.

## 3. Arms — same binary, four rows

| Arm | Path |
| --- | --- |
| 1 | baseline — fused, key order (existing) |
| 2 | batched, sorted, **unfused** (spec 43 arm 3) |
| 3 | batched, sorted, **fused** (new) |

Arm 2 is retained deliberately: **arm 3 vs arm 2 is this spec's claim** (does fusion remove the second-pass
penalty), and **arm 2 vs arm 1** re-measures the machinery penalty in the same binary so the decomposition
is not carried across runs. Spec 43's own finding was that cross-run ratios do not hold.

Arm 2 unsorted may be kept for continuity but is not required; it is already measured and it loses.

## 4. Cost accounting — everything inside timing

The timed region must contain **the complete candidate**: eligible pre-pass, metadata construction,
scratch allocation, the sort, zeroing, accumulation, assembly into slots, and **scratch release**.

Outside the timed region: result teardown only, matching the canonical construction row
(`bench_croaring.zig:507-512`, `result.deinit()` after the clock stops).

A candidate that looks good only because metadata construction or assembly sits outside the region is not
a result.

## 5. Corpus and the read-order risk

**Use the real canonical sparse corpus** — the same inputs the board row measures. No synthetic
population.

**Report source container types and byte volumes.** This is the load-bearing measurement for the spec's
main risk: accumulating in *destination* address order reads *source* containers in an order unrelated to
their own layout. Spec 38 found read traversal wants ascending (M4 1.221x, Zen 4 1.344x when sorted for
frees) — **but that result involved large bitsets**, and if these sources are mostly small arrays the
penalty may not transfer. It may also be that source reads are cheap enough relative to 8 KB writes that
the trade is clearly worth it.

**Either way it must be reported, not assumed in either direction.** The container-type/byte breakdown is
what makes the result interpretable if arm 3 disappoints.

**Do not combine experiments.** Bucket or radix ordering is out of scope for the first run. Consider it
**only if fusion works and narrowly misses the gate** — combining two changes in one measurement makes
neither attributable, which is the mistake the three-arm design was built to avoid.

## 6. Retained requirements from spec 43

These carry over unchanged and are not re-litigated:

- **Ownership handoff** (`43-01` §5): pool owns until **immediately before** `appendOwnedContainer`;
  cursor advances **before** the call; helper frees on failure or result owns on success; pool cleanup
  covers only entries at or after the cursor. `appendOwnedContainer` takes ownership on entry
  (`bitmap.zig:2094`).
- **Failure injection** at scratch, metadata, pending headers, pending payloads, unmatched clones, and
  assembly. Inputs untouched, nothing leaked, leak-checking GPA, never `c_allocator`.
- **Scratch/metadata OOM** → fall through to the baseline merge loop **reusing the initialized `result`**.
- **Equivalence coverage**: the fused arm must be driven directly, over forced and selective lazy OR,
  eligible counts of zero/partial/all, array/bitset/run combinations, disjoint keys, empty inputs, with
  **repaired output byte-identical** to baseline and CRoaring.
- **`lazyXor` byte-identical to baseline**; scope stays `op == .bor`.
- **No public API.** Internal export, classified in the manifest, outside `API.md`, the `check-docs`
  guarded region, and the `check-32` probe.
- **Manifest guards** updated in **both** `bench_parity_worker.zig:778` and `run-compare-bench.sh:72`.
- **Canonical harness only**, both hosts, all three canonical tuples, ≥5 fresh-process medians with full
  ranges.

## 7. Gate

- **Arm 3 beats arm 2 with non-overlapping ranges** — fusion removes a measurable part of the penalty.
- **Arm 3 reaches ≤1.10x vs CRoaring on M4.**
- **libc does not regress — arm 3 vs arm 1, rawr/libc, same binary, ≤5% on median.** A libc regression is
  a **STOP**, exactly as in spec 43, where it came in at +90%.
- Overlapping ranges → rerun; still overlapping → **inconclusive → NO-GO**.

**Report the decomposition regardless of outcome:** how much of the 1.544 ms fusion actually removed.
That number is the durable result of this spec even if the gate fails — it tells the campaign whether the
residue is cache or machinery.

## 8. Acceptance

- Fused arm implemented per §2, metadata and slot assembly per §2.1, on top of `37d0e8b`.
- **Full candidate cost inside the timed region** per §4; only result teardown outside.
- Real canonical sparse corpus; **source container type and byte breakdown reported** per §5.
- Three arms measured in one binary; **arm 2 vs arm 1 re-measured**, not carried from spec 43.
- **Fusion's share of the 1.544 ms reported explicitly.**
- All §6 requirements met, including failure injection and equivalence coverage.
- Gate §7 evaluated; GO/NO-GO stated. **No default change in this spec either way.**
- All four suites green — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
  `check-package`.

## 9. Out of scope

- Default adoption — a separate spec, only if this one passes its gate.
- Bucket/radix ordering (§5), unless fusion works and narrowly misses.
- Plain unfused batching. Measured, both allocators, it loses — do not re-propose.
- The microarchitectural attribution question (prefetch vs TLB vs cache). Still unestablished, still not
  required.

## 10. Estimate

**M** — the vehicle exists at `37d0e8b`; the new work is metadata, slot assembly, the fused loop, and the
measurement.

## 11. Chunking

Not chunked — pending review. Plausibly single-chunk, since the infrastructure is already built.
