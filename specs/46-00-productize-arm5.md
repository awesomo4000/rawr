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

**Give it a real ref** — named branch or tag — **push it**, and **pin the resulting immutable hash** into
this chunk and the toplevel §2.0. Do not productize out of a stash.

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
- `result.size = output_count`, so `errdefer result.deinit()` frees exactly the populated slots;
- **`cached_cardinality = -1` before success** — `initCapacity` leaves it `0` and direct-slot writes
  bypass `appendContainer`, so a populated result would otherwise report `cardinality() == 0`;
- fused zero + accumulate per buffer, in address order;
- **scratch-OOM falls through to the baseline loop reusing the initialized `result`**; every other
  fallible site propagates.

### 1.1 Ownership contract — direct slot assignment

The slotted path does **not** use `appendOwnedContainer`. Pin what it actually does:

1. **Assign the tagged pointer into its slot**, then
2. **advance `transferred_count`**, with **no fallible operation between**.

**Pending cleanup owns only the untransferred suffix; `result.deinit()` owns the populated slots.** Every
pending buffer is owned by exactly one party at every instant, and `transferred_count` is the single
record of that boundary.

## 2. Retain both baselines

The pre-adoption path stays callable, with **both** rows:

| Retained row | Reference for |
| --- | --- |
| `lazy-or-construction-baseline` | the construction gate |
| `lazy-or-repair-baseline` | the **combined** gate |

Gating the combined row without its own in-binary baseline would force the cross-run comparison spec
43-02 forbids.

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

- repaired output **byte-identical** to the previous default **and** CRoaring, across forced and
  selective lazy OR, eligible counts of **zero / partial / all**, array/bitset/run combinations,
  disjoint keys, empty inputs on either side;
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

- **Arm 5 rescued from the stash into a pushed ref; immutable hash pinned** here and in the toplevel.
- Fused slotted construction is the **default** for `op == .bor`, per §1, with §1.1's ownership contract.
- **Both** baseline rows retained and callable via the single narrowed internal export; `check_docs.zig`
  manifest carries its reason string.
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

## Estimate

**M** — the implementation exists; rescuing it, narrowing the export, deleting the arms, and raising the
test bar to production level are the work.
