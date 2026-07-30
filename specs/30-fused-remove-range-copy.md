<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 30: Fused copy-with-range-removed (`removeRangeCopy`)

Close the biggest M4 row on the board: **removeRange (wide, dense) 1.932x** (rawr *ahead* on
Zen 4 — 0.411x — a hard no-regress gate). **Parity is a hard requirement:** the row closes only
at **≤ 1.10x**; anything above stays open for a further lever.

**This is not a range-algorithm fix — the algorithm already wins.** 26a exonerated the mutation
body (rawr 49.8 ns vs CRoaring 78.5 ns); the 1.932x is the **copy+remove workflow**: the canonical
row clones the input to get a fresh mutable copy, then removes. The naive clone dutifully copies
**every** container — including the ones the wide range is about to delete — then frees them. The
lever is a **fused operation that produces the modified copy directly**, never allocating the
doomed containers.

## Why this is the right lever family

The removeRange, `clone`, and dense-AND M4 gaps share a root: `std.heap.smp_allocator` is slower
than libc `malloc` for many small per-container allocations on M4 (arch-specific — Zen 4 rawr
wins). The allocator-replacement track is closed (spec 18) and capacity re-tuning is NO-GO
(spec 27, exact-capacity clone *regressed* M4 SMP). The one family the evidence still favors is
**reducing allocation demand** (fewer clones) — and the wide copy+remove workflow is where demand
is most wasteful.

Critically, **CRoaring pays the same waste**: `roaring_bitmap_copy` copies all 16 containers, then
`roaring_bitmap_remove_range_closed` frees the covered ones. So rawr allocating **9 instead of 16**
containers is how "fewer allocations" **offsets** rawr's slower-per-allocation SMP cost — the
mechanism by which this can reach parity, not just narrow the gap.

## Pinned corpus (assert before any timing)

Dense corpus = `addRange(0, 999_999)` → **16 single-full-run containers** (keys 0–15).
`removeRange(100_000, 650_000)` partitions them:

- **survive untouched (7):** keys 0, 10, 11, 12, 13, 14, 15;
- **fully covered → deleted (7):** keys 2, 3, 4, 5, 6, 7, 8;
- **boundary → partial diff (2):** key 1 (keep `[65536, 99999]`), key 9 (keep `[650001, 655359]`);
- **result: 9 containers**; both boundary results are **single-run containers**.

The `30-00` diagnostic must assert this exact inventory (a drift invalidates the attribution).
**Allocation contrast to assert:** naive copy+remove = **16 allocations + 7 frees**; fused =
**9 allocations, 0 frees**.

## The fused operation

Add a **new owned-result path** (proposed `removeRangeCopy(self: *const Self, allocator, lo, hi)
!Self`) that produces a modified copy with the range removed, **preserving the source**:

1. **Pre-size** the top-level result to the survivor+boundary count from a cheap key-range scan
   (spec-27 gate applies — measure, do not assume).
2. Per container, by disposition:
   - **fully outside `[lo, hi]`** → `clone` the container into the result (untouched survivor);
   - **fully covered** → **skip** (allocate nothing);
   - **boundary** → build the **difference container directly** into the result.
3. **Never mutate `self`.**
4. **Ownership / cleanup:** result is independently owned; on any mid-loop allocation failure the
   partially built result is fully deinited (set `result.size` before returning the error so
   `errdefer` sees the partial containers — the spec-27/`3e27675` clone-leak discipline).

The existing in-place `removeRange` stays unchanged (a separate, exonerated primitive still used by
mutate-in-place callers). This op is **additive**.

## Measurement legitimacy (mandatory — the crux)

The canonical row must compare **rawr `removeRangeCopy`** against **CRoaring
`roaring_bitmap_copy` + `roaring_bitmap_remove_range_closed`** — both produce a modified copy while
**preserving the source**, so the comparison is apples-to-apples. **What is forbidden:** moving
rawr's clone outside the timed region while CRoaring's copy stays inside — that is a measurement
artifact, not an optimization, and is explicitly out of bounds.

## Constraints / gates

- **Representation-identical output** (spec 26): `removeRangeCopy(self)` serializes **byte-identical**
  to `clone(self)`-then-`removeRange` **and** matches CRoaring set-parity. A boundary diff must
  produce the *same container type* the in-place path would. Differential across container-type
  mixes, full/partial coverage, range fully inside one container, range covering all/none,
  empty-source, and chunk-boundary cases stays green.
- **Error semantics — build-then-commit, leak-free:** exhaustive allocation-failure injection on
  the new path; on OOM the **source is untouched**, the result is valid or cleanly errored, no
  leak.
- **Zen 4 no-regress (hard):** rawr is ahead (0.411x); the change stays within noise (≤ 5%, rerun
  on overlap) — it should only help.
- **Spec-27 M4 SMP gate:** the 16→9 allocation reduction and the pre-sizing are **measured on M4
  SMP, per the canonical protocol, before shipping** — a large real reduction is expected to win,
  but it is not assumed from the count.
- **Board gate + tightened layout exception (spec 28):** no canonical row worsens > 5% vs a fresh
  pre-change baseline, both hosts; layout classification requires **both** stable focused timing
  *and* instruction-identical disassembly.

## Scope

- **Closes the removeRange workflow row only.** It does **not** touch the standalone `clone (dense)`
  1.764x row — a full copy has no doomed containers to skip — nor dense-AND. Those remain
  documented allocator residuals.

## Acceptance

- **Phase 1 GO:** corpus + allocation contrast asserted (16+7 → 9+0); focused M4/Zen 4 SMP timing
  of `removeRangeCopy` vs CRoaring copy+remove; byte-identity + differential + failure-injection
  green; no canonical row changed yet.
- **Phase 2 GO — hard:** the canonical **removeRange (wide) row reaches ≤ 1.10x on M4 SMP** (via
  legitimate copy-vs-copy), **Zen 4 not regressed**, board gate held (layout exception). **Anything
  above 1.10x is a partial result and the row stays open** — if a residual remains it is the
  shared M4 SMP per-container-clone cost, which reopens with the next lever, it does not close the
  row.
- `zig build test`; `zig build difftest`; canonical `run-compare-bench.sh` both hosts;
  `ReleaseSafe` / `ReleaseFast` green; `docs/parity-measurement.md` updated.

## Proposed chunk plan (confirm at review)

- **`30-00`** — implement `removeRangeCopy` with full correctness (byte-identity, differential,
  allocation-failure injection, source-preservation), assert the corpus + allocation contrast,
  focused M4/Zen 4 SMP measurement in a named diagnostic; **no canonical row changed**.
- **`30-01`** — adopt into the canonical parity row (legitimate copy-vs-copy), full board gate,
  ship on M4/Zen 4 SMP numbers; hard ≤ 1.10x acceptance (row stays open above it).

## Estimate

M for `30-00` (new path + full correctness surface + failure injection + focused measurement).
S–M for `30-01` (row wiring + board gate).
