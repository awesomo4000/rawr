<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 46-00: Materialize and productize arm 5

Toplevel: [46-adopt-fused-slotted-construction.md](46-adopt-fused-slotted-construction.md).

**No measurement verdict here** — that is `46-01`. This chunk ends when fused slotted construction is the
production default, both baseline rows still work, the diagnostics are gone, and correctness and failure
coverage are green.

## 0. FIRST — rescue the implementation

**Do this before anything else.**

- `spec-43-lazy-construction-diagnostic` tip (`4b6bec4`) carries **only** baseline / batched / sorted.
- **Arm 5 exists only in stash commit `a1cb8c726686897b2e82dfed879dac540b52c8cd`.** A stash is held by
  `refs/stash` alone and is **reclaimable by `gc` once dropped**. The entire 1.235x result is one
  `git stash drop` from being lost.

**Two separate steps, in order:**

1. **Create a durable local ref at `a1cb8c726686897b2e82dfed879dac540b52c8cd`** (branch or tag) so `gc`
   can no longer reclaim it. **Pin that hash** here and in toplevel §2.0.
2. **Selectively port the arm 5 changes onto current `main`.** **Do NOT apply the stash commit
   wholesale** — it was taken atop the diagnostic branch and carries the multi-arm machinery this spec
   exists to remove. Cherry-picking it in full would reintroduce exactly what §3 deletes.

**Pushing the ref is an explicit owner action, not part of this chunk.** Creating the local ref is what
makes the work safe; publishing it is a separate decision.

## 1. Promote

Fused slotted construction becomes the **default** for `op == .bor`, forced and selective. Diagnostic
provenance does not lower the bar — this is production code now:

- eligible-pair pre-pass (exact eligible count, **not** matched-pair count);
- single `[]Pending` scratch; `sortUnstable` on `payload_addr`; comparator dereferences nothing;
- `initPendingBitset` **file-private in `bitmap.zig`** (Zig privacy is file-level, so it cannot live in
  `bitset_container.zig`); header initialized immediately after a successful payload allocation, before
  any later fallible step;
- `.reserved` slot initialization **built directly** — `Container.toTagged` on `.reserved` is
  `unreachable` and will panic;
- `result.size = output_count`, so `errdefer result.deinit()` **visits every slot**; reserved slots are
  harmless because `Container.deinit` has `.reserved => {}` — a no-op. *(An earlier draft said it "frees
  exactly the populated slots", which misdescribes the mechanism: it visits all of them and the reserved
  ones do nothing.)*
- **`cached_cardinality = -1` before success** — `initCapacity` leaves it `0` and direct-slot writes
  bypass `appendContainer`, so a populated result would otherwise report `cardinality() == 0`;
- fused zero + accumulate per buffer, in address order;
- **scratch-OOM falls through to the baseline loop reusing the initialized `result`**; every other
  fallible site propagates.

### 1.1 Ownership contract — direct slot assignment

The slotted path does **not** use `appendOwnedContainer`. Pin what it actually does:

1. **Assign the tagged pointer into its slot**, then
2. **advance `transferred_count`**, with **no fallible operation between**.

**Pending cleanup owns only the untransferred suffix of the pending array; `result.deinit()` owns the
populated slots corresponding to the transferred pending prefix.**

*(The prefix/suffix split lives in the **sorted pending array**, which is ordered by payload address. The
result slots those entries map to are **keyed by output order and therefore non-contiguous** — so
"`result` owns the transferred prefix" would misdescribe the layout. A cleanup routine that assumed a
contiguous result prefix would free the wrong slots.)*

Every pending buffer is owned by exactly one party at every instant, and `transferred_count` is the single
record of that boundary.

## 2. Retain both baselines

The pre-adoption path stays callable, with **both** rows:

| Retained row | Definition | Variants exposed | CRoaring reference |
| --- | --- | --- | --- |
| `lazy-or-construction-baseline` | **old** construction only | rawr/SMP, rawr/libc | canonical `lazy-or-construction` CRoaring/libc |
| `lazy-or-repair-baseline` | **old construction + repair** — mirrors the canonical *combined* row | rawr/SMP, rawr/libc | canonical `lazy-or-repair` CRoaring/libc |

**Baseline rows carry the two rawr tuples only and borrow the canonical CRoaring cell.** Emitting their
own CRoaring cells would measure identical C code a second time under a new name — pure cost, and a
second number that can drift from the canonical one and invite the wrong comparison.

**So "all three tuples" for a baseline row means:** rawr/SMP + rawr/libc from the baseline row itself,
plus the **referenced canonical CRoaring/libc** cell.

**`lazy-or-repair-baseline` is NOT the existing repair-only row.** It is the old construction path
followed by repair, so it is directly comparable to the candidate's combined row. Naming it after repair
alone would invite exactly the wrong comparison.

Gating the combined row without its own in-binary baseline would force the cross-run comparison spec
43-02 forbids.

**These are permanent, not scaffolding.** The retained implementation and both rows stay after adoption.
**Removing them later requires re-measurement**, because deleting code changes binary layout and spec 28
established that moves untouched rows with instruction-identical disassembly. Anyone tempted to tidy them
away is proposing a measurement, not a cleanup.

**Internal access is narrowed, not removed:** replace the multi-mode dispatch with **one narrowly named
internal baseline export**, and keep a corresponding reason string in `check_docs.zig`'s internal
manifest. The worker must still be able to call the old implementation.

## 3. Remove

- the `ConstructionMode` enum and arms 2, 3, 4 code paths;
- the **mode-selection** export (superseded by §2's single baseline export);
- source-travel and container-type diagnostic instrumentation;
- all diagnostic rows other than the two retained baselines.

**Delete, do not leave dormant.** Dead arms are dead weight that spec 28 layout noise will charge for.

## 4. Manifest — 40 → 42

`main` reads **40**. The two retained baseline rows take it to **42**. Both guards must read exactly
**42**: `src/bench_parity_worker.zig:778`, `scripts/run-compare-bench.sh:72`.

## 5. Correctness — production bar

- **Equivalence, split by what is actually provable:**
  - **vs the previous default: BYTE-EXACT.** Same implementation family, same container-type choices —
    this is the strong test, and it proves the new construction reproduces the old result exactly.
  - **vs CRoaring: SET EQUALITY + CROSS-DESERIALIZATION**, not byte identity.

  *(An earlier draft required byte-identity to **both** simultaneously. That is **not always satisfiable**:
  the same set has multiple valid `RoaringFormatSpec` encodings, and rawr and CRoaring may legitimately
  choose different container representations for some mixed cases. Requiring both would have demanded the
  impossible and invited someone to "fix" a non-defect.)*

  **No coverage is lost.** Byte-exactness against the baseline is the claim that matters for an adoption
  chunk — it pins the candidate to the behaviour being replaced. Cross-implementation agreement is a
  semantic property, and set equality plus cross-deserialization is exactly how `difftest` already
  establishes it.
- coverage spans forced and selective lazy OR, eligible counts of **zero / partial / all**,
  array/bitset/run combinations, disjoint keys, empty inputs on either side;
- **`cardinality()` checked BEFORE `repairAfterLazy`**, not only after — repair recomputes it, so
  repair-first tests mask a stale cache entirely;
- **`lazyXor` byte-identical** to its current behaviour; scope stays `op == .bor`;
- both retained baseline rows produce **identical results** to the pre-adoption implementation.

## 6. Failure injection

`std.testing.checkAllAllocationFailures` at every real fallible site:

1. `Self.initCapacity` — **propagates** (precedes scratch; no result to fall back with);
2. pending `[]Pending` scratch — **the only site that falls back to baseline**;
3. header `create`;
4. `words` payload allocation;
5. unmatched clone allocation;
6. non-eligible union allocation.

Every failure: **inputs untouched, nothing leaked**, leak-checking GPA, never `c_allocator`.

**Additionally assert:** no `.reserved` slot is ever dereferenced, and no slot is freed twice.

**Plus a targeted test** proving only site 2 falls back and sites 1 and 3–6 propagate.

## Acceptance

- **Durable local ref created at the stash hash and pinned** here and in the toplevel; arm 5 **selectively
  ported** onto `main`, not cherry-picked wholesale. *(Pushing the ref is an owner action, outside this
  chunk.)*
- Fused slotted construction is the **default** for `op == .bor`, per §1, with §1.1's ownership contract.
- **Both** baseline rows retained and callable via the single narrowed internal export, with
  `lazy-or-repair-baseline` defined as **old construction + repair**; `check_docs.zig` manifest carries
  its reason string. Retention is **permanent** — later removal requires re-measurement.
- Diagnostics per §3 **deleted**, not dormant.
- **Manifest at 42, both guards updated.**
- §5 correctness green, including **pre-repair `cardinality()`** and baseline-row equivalence.
- §6 failure injection green at all six sites, including the fallback-boundary test and the
  no-reserved-dereference / no-double-free assertions.
- No public API added; internal only; outside `API.md`, the `check-docs` guarded region, and the
  `check-32` probe. `check-docs` green with an empty allow-list.
- All four suites — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
  `check-package`, under `ReleaseSafe` and `ReleaseFast`.
- **No measurement verdict claimed.** Numbers are `46-01`.

## Verification record — implemented, reviewed, ACCEPTED

Verified independently in the working tree:

| Item | Result |
| --- | --- |
| Durable ref | `spec-44-fused-slotted-arm5` → **`a1cb8c726686897b2e82dfed879dac540b52c8cd`** ✓ — the stash-loss risk is closed |
| Manifest guards | **42** in `bench_parity_worker.zig:813` **and** `run-compare-bench.sh:72` ✓ (both, not just the first) |
| Baseline rows | `lazy-or-construction-baseline` and `lazy-or-repair-baseline` both present ✓ |
| `ConstructionMode` in production | **absent** from `bitmap.zig`, `roaring.zig`, `check_docs.zig` ✓ |
| Internal export | exactly one — `lazy_or_construction_baseline`, with reason string `"pre-adoption lazy-OR benchmark baseline"` in the `check_docs.zig` internal manifest ✓ |
| Spec 45 probe | `bench_smp_layout.zig` untouched; its `batched_*` cells are the 45-00 prototype, **not** spec 44 dispatch ✓ |

Reported and accepted: 104 tuples validated; byte-exact candidate-vs-baseline comparisons; CRoaring
interoperability; cardinality-before-repair coverage; exhaustive allocation-failure testing; full
`ReleaseSafe`/`ReleaseFast` matrices across `test`, `test64`, `difftest`, `difftest64`, `check-32`,
`check-docs`, `check-package`.

**One spec defect found and corrected — §5's equivalence requirement.** It demanded byte-identity to the
previous default **and** CRoaring simultaneously, which is not always satisfiable: one set has multiple
valid `RoaringFormatSpec` encodings, and the two implementations may choose different container
representations for some mixed cases. **The implementation was right and the spec was wrong.** Corrected
above; no coverage lost.

**No performance verdict claimed** — that is `46-01`.

## Estimate

**M** — the implementation exists; rescuing it, narrowing the export, deleting the arms, and raising the
test bar to production level are the work.
